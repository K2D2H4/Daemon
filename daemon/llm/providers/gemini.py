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

from typing import Any

import httpx

from daemon.llm.base import Completion, Message, ProviderError

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT = 60.0

ROLES = {"user": "user", "assistant": "model"}
"""`contents` only knows these two. A system turn is a separate field entirely."""

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
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("gemini provider needs GEMINI_API_KEY")
        self._api_key = api_key
        """Kept only so `_redact` can strip it back out of an upstream body
        before that body reaches an exception message."""
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
    ) -> Completion:
        # System turns are their own top-level field here, not a role in the list.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            {"role": ROLES[m.role], "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        if not contents:
            raise ProviderError("gemini needs at least one user or assistant message")

        payload: dict[str, Any] = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        config: dict[str, Any] = {}
        if max_output_tokens is not None:
            config["maxOutputTokens"] = max_output_tokens
        if temperature is not None:
            config["temperature"] = temperature
        if config:
            payload["generationConfig"] = config

        # The model id is part of the path. Both `gemini-2.5-flash` and the
        # `models/gemini-2.5-flash` form that Google's own docs print get copied
        # into config, and only one of them makes a valid path.
        url = f"{API_BASE}/models/{model.removeprefix('models/')}:generateContent"
        data = await self._post(url, payload)
        try:
            candidate = (data.get("candidates") or [{}])[0]
            parts = (candidate.get("content") or {}).get("parts") or []
            text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
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

        if not text:
            # A blocked prompt lands here: `promptFeedback.blockReason` is set and
            # there is no candidate at all, which is not a reply.
            raise ProviderError(
                f"gemini returned no text content: {self._redact(repr(data))}"
            )

        return Completion(
            text=text,
            # `modelVersion` is the id that actually served the call, which is
            # what Completion.model is for; an alias like `gemini-flash-latest`
            # resolves to something concrete here.
            model=str(data.get("modelVersion", model)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            meta={"stop_reason": finish_reason},
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
