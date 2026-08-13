"""Any endpoint that speaks OpenAI's Chat Completions API.

Qwen (Alibaba Model Studio), Kimi (Moonshot), DeepSeek, OpenRouter and a
self-hosted vLLM or LM Studio all expose the same surface, so one module serves
them all and they differ only by `base_url` and key.

Deliberately not a refactor of `providers/openai.py`, which speaks the newer
Responses API. Merging the two would make a single provider answer to two vendor
identities, so someone running Qwen would read `provider=openai` in every
gateway log line and in `.env` - the choice displayed would not be the choice
stored. The ~40 lines of shared HTTP scaffolding are copied on purpose, the same
trade this package already makes for `MODELS_URL` and `SENSITIVITIES`.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from typing import Any

import httpx

from daemon.llm.base import (
    Completion,
    Message,
    ProviderError,
    ToolCall,
    ToolSpec,
    decode_tool_arguments,
    synthesise_call_id,
)

DEFAULT_TIMEOUT = 60.0

BODY_LIMIT = 200
"""How much of an upstream error body reaches the exception message.

Same figure as `daemon/setup.py`'s `BODY_LIMIT`, and for the same reason: enough
of a 4xx body to name the cause, not enough for a stack trace to become the
response body of whatever the endpoint was actually serving."""


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("openai_compatible provider needs OPENAI_COMPATIBLE_API_KEY")
        if not base_url:
            raise ProviderError(
                "openai_compatible provider needs DAEMON_OPENAI_COMPATIBLE_BASE_URL"
            )
        self._api_key = api_key
        """Kept only so `_redact` can strip it back out of an upstream body: a 401
        body often echoes the key that was sent, and that body goes into an
        exception message."""
        self._base_url = base_url.rstrip("/")
        self._chat_url = f"{self._base_url}/chat/completions"
        self._models_url = f"{self._base_url}/models"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> Completion:
        turns = _messages(messages)
        if not turns:
            raise ProviderError("openai_compatible needs at least one message")

        payload: dict[str, Any] = {"model": model, "messages": turns}
        if max_output_tokens is not None:
            # `max_tokens`, not Responses' `max_output_tokens`. Compatible
            # endpoints implement the classic spelling; the newer
            # `max_completion_tokens` is not universal among them.
            payload["max_tokens"] = max_output_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            # Nested under `function`, where Responses puts these fields flat.
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
                for spec in tools
            ]

        data = await self._post(payload)
        try:
            choices = data.get("choices") or []
            message = choices[0].get("message") or {} if choices else {}
            text = _text(message.get("content"))
            calls = _tool_calls(message)
            usage = data.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))
            finish = str(choices[0].get("finish_reason", "")) if choices else ""
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            # Per llm/base.py every failure leaves as ProviderError - including a
            # body that simply is not shaped like the documented one.
            raise ProviderError(
                f"openai_compatible returned an unreadable response: "
                f"{self._redact(repr(data))}"
            ) from exc

        if not text and not calls:
            # A turn that only asked for tools has no text, and that one is fine.
            # Everything else here is a refusal or an empty answer.
            raise ProviderError(
                f"openai_compatible returned no text content: {self._redact(repr(data))}"
            )

        return Completion(
            text=text,
            model=str(data.get("model", model)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            meta={"stop_reason": finish},
            tool_calls=calls,
        )

    async def health(self) -> bool:
        """Cheapest authenticated endpoint that costs no tokens. Optional in the
        compatible spec, so a False here means "could not confirm", not "down"."""
        try:
            response = await self._client.get(self._models_url, headers=self._headers)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _redact(self, text: str) -> str:
        """Remove the key from anything upstream that is about to be quoted.

        Nothing here is raised `from None`: suppressing a context does not remove
        it from `__context__`, so the only real defence is that no secret is in
        the chain to begin with - the key is a header, never a URL.
        """
        return text.replace(self._api_key, "<redacted>")

    def _quoted(self, body: str) -> str:
        """An upstream error body, in the order that makes it safe to quote.

        Redact, collapse, strip, *then* bound - and the order is the whole point.
        Slicing first can cut the key across the boundary, and `_redact`'s
        `.replace()` only matches the key whole, so the remaining prefix survives
        into the raised message. That leaked `HTTP 401 ... key sk-liv` from a real
        401 body; `daemon/setup.py:check_openai_compatible` and `_telegram_said`
        already fixed the same ordering and this module was written before them.

        Whitespace is collapsed so words split across lines do not run together,
        and non-printable characters go because an escape sequence in an upstream
        body would otherwise repaint whatever a terminal prints after this.
        """
        collapsed = " ".join(self._redact(body).split())
        return "".join(char for char in collapsed if char.isprintable())[:BODY_LIMIT]

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exactly one retry (llm/base.py: the gateway decides about
        fallback, providers do not build retry chains)."""
        for attempt in (1, 2):
            may_retry = attempt == 1
            try:
                response = await self._client.post(
                    self._chat_url, json=payload, headers=self._headers
                )
            except httpx.HTTPError as exc:
                if may_retry:
                    continue
                # `exc` names the URL, which carries no secret here.
                raise ProviderError(f"openai_compatible unreachable: {exc}") from exc

            if response.status_code == 429 or response.status_code >= 500:
                if may_retry:
                    continue
                raise ProviderError(f"openai_compatible returned HTTP {response.status_code}")
            if response.status_code >= 400:
                # Permanent - 401 bad key, 402 no credit (OpenRouter's answer for a
                # paid model on an unfunded account), 404 wrong endpoint or model.
                raise ProviderError(
                    f"openai_compatible rejected the request: HTTP {response.status_code} "
                    f"{self._quoted(response.text)}"
                )
            try:
                data: dict[str, Any] = response.json()
            except ValueError as exc:
                raise ProviderError(
                    f"openai_compatible returned a non-JSON body: {exc}"
                ) from exc
            if not isinstance(data, dict):
                # A proxy in front of one of these endpoints can answer with a
                # bare JSON array.
                raise ProviderError(
                    f"openai_compatible returned a JSON {type(data).__name__}, not an object"
                )
            return data

        raise ProviderError("openai_compatible call failed")  # unreachable


def _text(content: object) -> str:
    """The reply text, whether the endpoint sent a string or the multipart form.

    A plain string is the common answer. Some gateways mirror the *request*
    shape back and send a list of typed parts, and treating that as empty would
    turn a perfectly good reply into `no text content`.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    """`tool_calls` off an assistant turn, with arguments already decoded.

    The id is synthesised when the endpoint issues none: a self-hosted server may
    omit it, and `daemon/loop.py` pairs a result back to its request by that id.
    `synthesise_call_id`'s inverse, `call_name`, recovers the function name from
    the shape this produces.
    """
    raw = message.get("tool_calls") or []
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for index, call in enumerate(raw):
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        name = str(function.get("name", ""))
        calls.append(
            ToolCall(
                id=str(call.get("id") or synthesise_call_id(name, index)),
                name=name,
                arguments=decode_tool_arguments(function.get("arguments")),
            )
        )
    return tuple(calls)


def _messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Neutral messages as Chat Completions turns.

    Closer to the neutral shape than Responses is: `role: tool` and
    `tool_call_id` are this API's own spelling, so `daemon/llm/base.py` needs no
    translation for them. The system turn also stays in the array rather than
    being hoisted to a top-level field.
    """
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            turns.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id or "",
                    "content": message.content,
                }
            )
            continue
        if message.role == "assistant" and message.tool_calls:
            turns.append(
                {
                    "role": "assistant",
                    # Explicitly null rather than absent: several endpoints reject
                    # an assistant turn that carries neither content nor the key.
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                # Back to a string: this API sends and expects
                                # JSON text here.
                                "arguments": json.dumps(
                                    call.arguments, ensure_ascii=False
                                ),
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
            )
            continue
        if message.role == "user" and message.images:
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.media_type};base64,"
                        f"{base64.b64encode(img.data).decode()}"
                    },
                }
                for img in message.images
            )
            turns.append({"role": "user", "content": content})
            continue
        turns.append({"role": message.role, "content": message.content})
    return turns
