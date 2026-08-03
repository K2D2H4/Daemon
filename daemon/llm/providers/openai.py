"""Hosted provider (OpenAI Responses API).

BYOK: the key is the user's (docs/PLAN.md 7). Written against the raw HTTP API
with httpx rather than the vendor SDK - one dependency, and the request shape
stays visible next to the provider contract.

Responses rather than Chat Completions: OpenAI points new integrations at it,
`max_output_tokens` and `usage.{input,output}_tokens` land on the fields of
`Completion` without a rename, and the system turn becomes a top-level
`instructions` field - the same shape anthropic.py and gemini.py already have,
so all three hosted providers hoist system turns the same way.
"""

from __future__ import annotations

from typing import Any

import httpx

from daemon.llm.base import Completion, Message, ProviderError

API_URL = "https://api.openai.com/v1/responses"
MODELS_URL = "https://api.openai.com/v1/models"
DEFAULT_TIMEOUT = 60.0


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("openai provider needs OPENAI_API_KEY")
        self._api_key = api_key
        """Kept only so `_redact` can strip it back out of an upstream body: the
        401 body echoes the key that was sent ("Incorrect API key provided:
        ..."), and that body goes into an exception message."""
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
    ) -> Completion:
        # System turns are a top-level field here, not a role in the list.
        instructions = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]
        if not turns:
            raise ProviderError("openai needs at least one user or assistant message")

        payload: dict[str, Any] = {
            "model": model,
            "input": turns,
            # Responses stores every call server-side for 30 days by default and
            # lists it in the dashboard. These are one person's private
            # conversations (docs/PLAN.md 7), so opt out on every request rather
            # than relying on an org-level setting the user may not have.
            "store": False,
        }
        if instructions:
            payload["instructions"] = instructions
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        data = await self._post(payload)
        try:
            # `output` is a list of typed items: reasoning and tool calls sit
            # next to the reply, so the text is the `output_text` parts of the
            # message items and nothing else.
            text = "".join(
                part.get("text", "")
                for item in data.get("output", [])
                if isinstance(item, dict) and item.get("type") == "message"
                for part in item.get("content", [])
                if isinstance(part, dict) and part.get("type") == "output_text"
            )
            usage = data.get("usage") or {}
            input_tokens = int(usage.get("input_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))
            # status is "completed" | "incomplete" | "in_progress"; when it is
            # incomplete the informative half is the reason (max_output_tokens,
            # content_filter), so that wins.
            reason = str((data.get("incomplete_details") or {}).get("reason", ""))
            status = str(data.get("status", ""))
        except (AttributeError, TypeError, ValueError) as exc:
            # Per llm/base.py every failure leaves as ProviderError - including a
            # body that simply is not shaped like the documented one.
            raise ProviderError(
                f"openai returned an unreadable response: {self._redact(repr(data))}"
            ) from exc

        if not text:
            # A refusal lands here too: it comes back as a `refusal` part rather
            # than `output_text`, which leaves no reply to hand back.
            raise ProviderError(
                f"openai returned no text content: {self._redact(repr(data))}"
            )

        return Completion(
            text=text,
            model=str(data.get("model", model)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            meta={"stop_reason": reason or status},
        )

    async def health(self) -> bool:
        """Cheapest reachable authenticated endpoint - costs no tokens."""
        try:
            response = await self._client.get(MODELS_URL, headers=self._headers)
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

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exactly one retry (see llm/base.py: the gateway decides
        about fallback, providers do not build retry chains)."""
        for attempt in (1, 2):
            may_retry = attempt == 1
            try:
                response = await self._client.post(API_URL, json=payload, headers=self._headers)
            except httpx.HTTPError as exc:
                if may_retry:
                    continue
                # `exc` names the URL, which carries no secret here.
                raise ProviderError(f"openai unreachable: {exc}") from exc

            # Transient: rate limit, and anything the far side calls its own fault.
            if response.status_code == 429 or response.status_code >= 500:
                if may_retry:
                    continue
                raise ProviderError(f"openai returned HTTP {response.status_code}")
            if response.status_code >= 400:
                # Permanent - 401 bad key, 403 no access to that model, 404 wrong
                # model or endpoint, 400 malformed request. Retrying burns time
                # and the gateway should fall back instead.
                raise ProviderError(
                    f"openai rejected the request: HTTP {response.status_code} "
                    f"{self._redact(response.text[:200])}"
                )
            try:
                data: dict[str, Any] = response.json()
            except ValueError as exc:
                raise ProviderError(f"openai returned a non-JSON body: {exc}") from exc
            if not isinstance(data, dict):
                # A proxy in front of the API can answer with a bare JSON array.
                raise ProviderError(f"openai returned a JSON {type(data).__name__}, not an object")
            return data

        raise ProviderError("openai call failed")  # unreachable
