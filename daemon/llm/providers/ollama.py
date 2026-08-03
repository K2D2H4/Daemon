"""Local provider (Ollama `/api/chat`).

The offline preset runs entirely on this, which is what makes the privacy
promise in docs/PLAN.md 7 true rather than aspirational.

M1a is single-shot: no streaming. Streaming only matters once a UI renders
tokens, and it complicates the token accounting the gateway logs.
"""

from __future__ import annotations

from typing import Any

import httpx

from daemon.llm.base import Completion, Message, ProviderError

DEFAULT_TIMEOUT = 120.0
"""Generous: a 14B model on a cold load is slow. Bounded all the same - an
unbounded wait would hang the conversation loop forever."""


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._base_url = base_url.rstrip("/")

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Completion:
        options: dict[str, Any] = {}
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens
        if temperature is not None:
            options["temperature"] = temperature

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            # No `think` parameter on purpose. Measured on qwen3:4b: `think:
            # false` does not stop a reasoning model from reasoning, it stops
            # Ollama from *separating* the reasoning - so the chain of thought
            # lands in `content` and becomes the reply ("Okay, the user asked
            # ... which translates to"). Faster (11.8 s vs 24.3 s) and useless.
            # `think: "low"` was slower still, and Qwen3's own `/no_think`
            # suffix does not survive Ollama's template. Leaving it unset keeps
            # the thinking in its own field where it is discarded, and the reply
            # clean. The real fix for latency is not to use a reasoning model
            # for conversation - see .env.example.
        }
        if options:
            payload["options"] = options

        data = await self._post(f"{self._base_url}/api/chat", payload)
        try:
            text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"ollama returned no message content: {data!r}") from exc

        return Completion(
            text=text,
            model=str(data.get("model", model)),
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
        )

    async def health(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/tags")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exactly one retry, per the Provider contract in llm/base.py:
        the gateway owns the real recovery decision because only it can see the
        routing table."""
        for attempt in (1, 2):
            may_retry = attempt == 1
            try:
                response = await self._client.post(url, json=payload)
            except httpx.HTTPError as exc:
                if may_retry:
                    continue
                raise ProviderError(f"ollama unreachable at {url}: {exc}") from exc

            if response.status_code == 429 or response.status_code >= 500:
                if may_retry:
                    continue
                raise ProviderError(f"ollama returned HTTP {response.status_code}")
            if response.status_code >= 400:
                # A retry cannot fix a bad model name.
                raise ProviderError(
                    f"ollama rejected the request: HTTP {response.status_code} "
                    f"{response.text[:200]}"
                )
            try:
                data: dict[str, Any] = response.json()
            except ValueError as exc:
                raise ProviderError(f"ollama returned a non-JSON body: {exc}") from exc
            return data

        raise ProviderError(f"ollama call to {url} failed")  # unreachable
