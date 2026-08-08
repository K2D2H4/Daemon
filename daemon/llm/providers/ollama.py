"""Local provider (Ollama `/api/chat`).

The offline preset runs entirely on this, which is what makes the privacy
promise in docs/PLAN.md 7 true rather than aspirational.

M1a is single-shot: no streaming. Streaming only matters once a UI renders
tokens, and it complicates the token accounting the gateway logs.
"""

from __future__ import annotations

import base64
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
        tools: Sequence[ToolSpec] | None = None,
    ) -> Completion:
        options: dict[str, Any] = {}
        if max_output_tokens is not None:
            options["num_predict"] = max_output_tokens
        if temperature is not None:
            options["temperature"] = temperature

        payload: dict[str, Any] = {
            "model": model,
            "messages": [_turn(m) for m in messages],
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
        if tools:
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

        data = await self._post(f"{self._base_url}/api/chat", payload)
        message = data.get("message")
        if not isinstance(message, dict):
            raise ProviderError(f"ollama returned no message content: {data!r}")
        text = message.get("content")
        calls = _tool_calls(message)
        if not isinstance(text, str) or (not text and not calls):
            # A tool-calling turn legitimately has an empty `content`, so absent
            # text is only a failure when the model asked for nothing either.
            raise ProviderError(f"ollama returned no message content: {data!r}")

        return Completion(
            text=text,
            model=str(data.get("model", model)),
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            tool_calls=calls,
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


def _turn(message: Message) -> dict[str, Any]:
    """One neutral Message as Ollama's chat format.

    Ollama follows OpenAI's older chat shape, so the `tool` role survives as
    itself here - unlike Anthropic, where it has to become a user turn.
    """
    turn: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.role == "user" and message.images:
        turn["images"] = [base64.b64encode(img.data).decode() for img in message.images]
    if message.tool_calls:
        turn["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
    if message.role == "tool":
        # Ollama matches results to requests by position and by name, not by id:
        # there is no `tool_call_id` in its schema, and sending one is ignored at
        # best. `tool_name` is what recent versions read - so the *name* goes here,
        # recovered from the synthesised id. Sending the id itself put `read_file-0`
        # where `read_file` belonged, which is the pairing this line exists to get
        # right.
        turn["tool_name"] = call_name(message.tool_call_id)
    return turn


def _tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    """Ollama's tool calls carry no ids, so they are numbered by position.

    That is enough because the ids never leave this process: the loop uses them to
    pair a result with its request within one turn, and `_turn` above sends the
    name back rather than the id.
    """
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        calls.append(
            ToolCall(
                id=str(item.get("id") or synthesise_call_id(name, index)),
                name=name,
                arguments=decode_tool_arguments(function.get("arguments")),
            )
        )
    return tuple(calls)
