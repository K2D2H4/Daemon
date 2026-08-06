"""Hosted provider (Gemini `generateContent`).

BYOK: the key is the user's (docs/PLAN.md 7). Raw HTTP with httpx rather than the
vendor SDK, like every other provider here.

The key travels in the `x-goog-api-key` header, never in the documented `?key=`
query parameter. A URL is not a secret-carrying place: httpx logs the request line
at DEBUG and repeats the URL in the message of every transport exception it
raises, so `?key=` leaks the key into logs and stack traces. daemon/setup.py's
`check_gemini` and daemon/voice/gemini_live.py make the same choice.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from daemon.llm.base import (
    Completion,
    Message,
    ProviderError,
    ToolCall,
    ToolSpec,
    call_name,
    decode_tool_arguments,
    synthesise_call_id,
)

logger = logging.getLogger(__name__)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT = 60.0

MALFORMED_FUNCTION_CALL = "MALFORMED_FUNCTION_CALL"
"""A `finishReason` gemini-3 returns when it *tried* to call a tool and produced
nothing parseable - an empty candidate, no parts, no `functionCall`. It is a
sampling glitch, not a bad request: the identical call succeeds on a fresh sample
(measured - 20/20 clean in isolation, yet it struck twice in a row on a real
2900-token turn). So `complete` re-POSTs once rather than failing the whole turn,
which is what surfaced to the owner as "Something went wrong on my side."
"""

ROLES = {"user": "user", "assistant": "model"}
"""`contents` only knows these two. A system turn is a separate field entirely.

A `tool` turn is neither: it becomes a `user` turn carrying a `functionResponse`
part, which is why `_contents` below does not go through this table."""

STANDARD_KEY_HINT = (
    " - if that key is an old Google 'Standard' key, the Gemini API already "
    "refuses unrestricted ones and refuses all of them from September 2026; a key "
    "made at aistudio.google.com/apikey now is an auth key, which is what this needs"
)
"""Appended to 401/403 only, because that is what the rejection looks like. The
wording is duplicated from daemon/setup.py rather than imported: a provider must
not depend on the onboarding wizard."""


class GeminiProvider:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        thinking_level: str = "",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("gemini provider needs GEMINI_API_KEY")
        self._api_key = api_key
        """Kept only so `_redact` can strip it back out of an upstream body
        before that body reaches an exception message."""
        self._thinking_level = thinking_level
        """How hard a Gemini 3 model thinks before answering (`low`/`high`), or empty
        to leave it to the model. `low` roughly a third of the per-call latency of the
        default on a plain tool turn (config.py: DAEMON_GEMINI_THINKING_LEVEL). Sent
        only when set, so an older model that does not know `thinkingConfig` is
        untouched unless someone opts in."""
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._headers = {
            "x-goog-api-key": api_key,
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
        # System turns are their own top-level field here, not a role in the list.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = _contents(messages)
        if not contents:
            raise ProviderError("gemini needs at least one user or assistant message")

        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            # One `tools` entry holding every declaration, not one entry each: the
            # API takes a list of tool *objects*, and a function is a declaration
            # inside one of them.
            payload["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": spec.name,
                            "description": spec.description,
                            "parameters": spec.parameters,
                        }
                        for spec in tools
                    ]
                }
            ]
        config: dict[str, Any] = {}
        if max_output_tokens is not None:
            config["maxOutputTokens"] = max_output_tokens
        if temperature is not None:
            config["temperature"] = temperature
        if self._thinking_level:
            # A Gemini 3 knob. `low` cuts the per-call latency of a plain tool turn to
            # about a third of the default (measured, gemini-3.6-flash) and still makes
            # the call. Sent only when set, so an unconfigured install and older models
            # send no `thinkingConfig` at all.
            config["thinkingConfig"] = {"thinkingLevel": self._thinking_level}
        if config:
            payload["generationConfig"] = config

        # The model id is part of the path. Both `gemini-2.5-flash` and the
        # `models/gemini-2.5-flash` form that Google's own docs print get copied
        # into config, and only one of them makes a valid path.
        url = f"{API_BASE}/models/{model.removeprefix('models/')}:generateContent"

        # One re-sample, reserved for an empty MALFORMED_FUNCTION_CALL (see that
        # constant): a fresh sample fixes it, while everything else - a real reply, a
        # block, an unreadable body - returns or raises on the first pass. This is a
        # second, orthogonal single-retry sitting on top of `_post`'s own transport
        # retry, so a rare transient-then-malformed run can reach up to four POSTs;
        # each axis still fires at most once and then raises, which is what keeps the
        # gateway owning fallback rather than this becoming a chain (llm/base.py).
        for attempt in (1, 2):
            data = await self._post(url, payload)
            try:
                candidate = (data.get("candidates") or [{}])[0]
                parts = (candidate.get("content") or {}).get("parts") or []
                text = "".join(
                    part.get("text", "") for part in parts if isinstance(part, dict)
                )
                calls = _tool_calls(parts)
                usage = data.get("usageMetadata") or {}
                input_tokens = int(usage.get("promptTokenCount", 0))
                # On this API `candidatesTokenCount` already contains the thinking
                # tokens - `thoughtsTokenCount` is a breakdown of it, not an extra
                # charge - so adding the two would double-bill a reasoning model.
                # (Vertex AI splits them; this is generativelanguage, which does not.)
                output_tokens = int(usage.get("candidatesTokenCount", 0))
                finish_reason = str(candidate.get("finishReason", ""))
            except (AttributeError, IndexError, TypeError, ValueError) as exc:
                # Per llm/base.py every failure leaves as ProviderError - including a
                # body that simply is not shaped like the documented one.
                raise ProviderError(
                    f"gemini returned an unreadable response: {self._redact(repr(data))}"
                ) from exc

            if text or calls:
                return Completion(
                    text=text,
                    # `modelVersion` is the id that actually served the call, which is
                    # what Completion.model is for; an alias like `gemini-flash-latest`
                    # resolves to something concrete here.
                    model=str(data.get("modelVersion", model)),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    meta={"stop_reason": finish_reason},
                    tool_calls=calls,
                )

            if finish_reason == MALFORMED_FUNCTION_CALL and attempt == 1:
                # The model tried to call a tool and emitted nothing. Re-sample once;
                # the turn would otherwise die on a glitch a retry clears.
                logger.warning(
                    "gemini returned an empty %s; re-sampling once", MALFORMED_FUNCTION_CALL
                )
                continue

            # A blocked prompt lands here: `promptFeedback.blockReason` is set and
            # there is no candidate at all, which is not a reply. A turn that only
            # asked for tools has no text either, and that one is a reply and already
            # returned above. A MALFORMED that survived its retry lands here too - the
            # honest end is a ProviderError the gateway may still fall back from.
            raise ProviderError(
                f"gemini returned no text content: {self._redact(repr(data))}"
            )

        # Unreachable: the loop returns a reply or raises on every path above.
        raise ProviderError(  # pragma: no cover
            f"gemini returned no usable content: {self._redact(repr(data))}"
        )

    async def health(self) -> bool:
        """Model list: authenticated, and costs no tokens."""
        try:
            response = await self._client.get(f"{API_BASE}/models", headers=self._headers)
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
        the chain to begin with - the key is a header, never a URL, and upstream
        text passes through here.
        """
        return text.replace(self._api_key, "<redacted>")

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exactly one retry (see llm/base.py: the gateway decides
        about fallback, providers do not build retry chains)."""
        for attempt in (1, 2):
            may_retry = attempt == 1
            try:
                response = await self._client.post(url, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                if may_retry:
                    continue
                # `exc` names the URL, which carries no secret because the key is
                # a header. That is the whole reason it is a header.
                raise ProviderError(f"gemini unreachable: {exc}") from exc

            # Transient: RESOURCE_EXHAUSTED, plus UNAVAILABLE/INTERNAL when the
            # model is overloaded.
            if response.status_code == 429 or response.status_code >= 500:
                if may_retry:
                    continue
                raise ProviderError(f"gemini returned HTTP {response.status_code}")
            if response.status_code >= 400:
                # Permanent - 400 API_KEY_INVALID or a malformed request, 401/403
                # a refused key, 404 an unknown model id. A retry fixes none of
                # them; the gateway should fall back instead.
                hint = STANDARD_KEY_HINT if response.status_code in (401, 403) else ""
                raise ProviderError(
                    f"gemini rejected the request: HTTP {response.status_code} "
                    f"{self._redact(response.text[:200])}{hint}"
                )
            try:
                data: dict[str, Any] = response.json()
            except ValueError as exc:
                raise ProviderError(f"gemini returned a non-JSON body: {exc}") from exc
            if not isinstance(data, dict):
                # A proxy in front of the API can answer with a bare JSON array.
                raise ProviderError(f"gemini returned a JSON {type(data).__name__}, not an object")
            return data

        raise ProviderError(f"gemini call to {url} failed")  # unreachable


def _contents(messages: list[Message]) -> list[dict[str, Any]]:
    """Neutral messages as `contents`.

    A tool result is a `functionResponse` part in a **user** turn, and the name -
    not an id - is what pairs it with its request, because this API issues no call
    ids at all. `_tool_calls` below synthesises ids for our own use; they are not
    sent back.
    """
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": call_name(message.tool_call_id),
                                # Must be an object, not a bare string.
                                "response": {"result": message.content},
                            }
                        }
                    ],
                }
            )
            continue
        parts: list[dict[str, Any]] = []
        if message.content:
            parts.append({"text": message.content})
        for call in message.tool_calls:
            # The signature sits on the part beside `functionCall`, and is present
            # only where Gemini issued one - the first of a parallel batch. Sending
            # the key on a call it did not sign is itself a 400, so an absent
            # signature emits no key rather than a null.
            part: dict[str, Any] = {
                "functionCall": {"name": call.name, "args": call.arguments}
            }
            if call.provider_signature:
                part["thoughtSignature"] = call.provider_signature
            parts.append(part)
        if not parts:
            # An empty `parts` is rejected, and an assistant turn with neither text
            # nor calls carries nothing worth sending anyway.
            continue
        contents.append({"role": ROLES[message.role], "parts": parts})
    return contents


def _tool_calls(parts: list[Any]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for index, part in enumerate(parts):
        call = part.get("functionCall") if isinstance(part, dict) else None
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if not isinstance(name, str) or not name:
            continue
        # A sibling of `functionCall` on the part, not a field inside it. Gemini 3
        # requires it echoed back on replay (base.py: `provider_signature`); losing
        # it here is what a dropped tool turn returns HTTP 400 for.
        signature = part.get("thoughtSignature")
        calls.append(
            ToolCall(
                # Synthesised: this API issues none, and the loop needs something
                # to pair a result with its request within the turn.
                id=synthesise_call_id(name, index),
                name=name,
                arguments=decode_tool_arguments(call.get("args")),
                provider_signature=signature if isinstance(signature, str) else None,
            )
        )
    return tuple(calls)
