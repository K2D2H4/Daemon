"""Hosted provider (Anthropic Messages API).

BYOK: the key is the user's (docs/PLAN.md 7). Written against the raw HTTP API
with httpx rather than the vendor SDK - one dependency, and the request shape
stays visible next to the provider contract.
"""

from __future__ import annotations

from typing import Any

import httpx

from daemon.llm.base import Completion, Message, ProviderError

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_TOKENS = 2048
"""The API requires max_tokens, so an explicit ceiling is unavoidable. It also
doubles as a cost guard for a runaway generation."""


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("anthropic provider needs ANTHROPIC_API_KEY")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._headers = {
            "x-api-key": api_key,
            "anthropic-version": API_VERSION,
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
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]
        if not turns:
            raise ProviderError("anthropic needs at least one user or assistant message")

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens or DEFAULT_MAX_TOKENS,
            "messages": turns,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature

        data = await self._post(payload)
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text:
            raise ProviderError(f"anthropic returned no text content: {data!r}")

        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=str(data.get("model", model)),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            meta={"stop_reason": str(data.get("stop_reason", ""))},
        )

    async def health(self) -> bool:
        """Cheapest reachable authenticated endpoint - costs no tokens."""
        try:
            response = await self._client.get(
                "https://api.anthropic.com/v1/models", headers=self._headers
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

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
                raise ProviderError(f"anthropic unreachable: {exc}") from exc

            # 429 and 529 (overloaded) are the documented transient cases.
            if response.status_code in (429, 529) or response.status_code >= 500:
                if may_retry:
                    continue
                raise ProviderError(f"anthropic returned HTTP {response.status_code}")
            if response.status_code >= 400:
                # Bad key, bad model, malformed request: retrying just burns time.
                raise ProviderError(
                    f"anthropic rejected the request: HTTP {response.status_code} "
                    f"{response.text[:200]}"
                )
            try:
                data: dict[str, Any] = response.json()
            except ValueError as exc:
                raise ProviderError(f"anthropic returned a non-JSON body: {exc}") from exc
            return data

        raise ProviderError("anthropic call failed")  # unreachable
