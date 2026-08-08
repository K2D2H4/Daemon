"""Hosted provider (Anthropic Messages API).

BYOK: the key is the user's (docs/PLAN.md 7). Written against the raw HTTP API
with httpx rather than the vendor SDK - one dependency, and the request shape
stays visible next to the provider contract.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

import httpx

from daemon.llm.base import Completion, Message, ProviderError, ToolCall, ToolSpec

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
        tools: Sequence[ToolSpec] | None = None,
    ) -> Completion:
        # System turns are a top-level field here, not a role in the list.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = _turns(messages)
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
        if tools:
            # `input_schema`, not `parameters`: the one field name this API spells
            # differently from the other three.
            payload["tools"] = [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.parameters,
                }
                for spec in tools
            ]

        data = await self._post(payload)
        blocks = data.get("content")
        blocks = blocks if isinstance(blocks, list) else []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        calls = tuple(
            ToolCall(
                id=str(block.get("id", "")),
                name=str(block.get("name", "")),
                # Already a decoded object on this API.
                arguments=block.get("input") if isinstance(block.get("input"), dict) else {},
            )
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "tool_use"
        )
        if not text and not calls:
            # `stop_reason: "tool_use"` produces no text at all, which is a
            # complete response rather than a failed one.
            raise ProviderError(f"anthropic returned no text content: {data!r}")

        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=str(data.get("model", model)),
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            meta={"stop_reason": str(data.get("stop_reason", ""))},
            tool_calls=calls,
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


def _turns(messages: list[Message]) -> list[dict[str, Any]]:
    """Neutral messages as Anthropic turns.

    This API has no `tool` role. A tool result is a `tool_result` block inside a
    **user** turn, and consecutive results have to share one turn: the API rejects
    two user turns in a row, so a reply that asked for three tools at once would
    otherwise be answered with three separate turns and a 400.
    """
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": message.content,
            }
            previous = turns[-1] if turns else None
            if previous is not None and previous.get("role") == "user" and _is_results(previous):
                previous["content"].append(block)
            else:
                turns.append({"role": "user", "content": [block]})
            continue
        if message.role == "assistant" and message.tool_calls:
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            content.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            turns.append({"role": "assistant", "content": content})
            continue
        if message.role == "user" and message.images:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.media_type,
                        "data": base64.b64encode(img.data).decode(),
                    },
                }
                for img in message.images
            )
            # A see_screen round is assistant(tool_calls) -> tool -> user(images):
            # the `tool` branch above already opened a user turn for the
            # tool_result. Appending a second user turn here would put two user
            # turns back to back, which this function's docstring says the API
            # rejects - so merge into the preceding one instead, same as the
            # tool-result merge above.
            previous = turns[-1] if turns else None
            if previous is not None and previous.get("role") == "user" and isinstance(
                previous.get("content"), list
            ):
                previous["content"].extend(blocks)
            else:
                turns.append({"role": "user", "content": blocks})
            continue
        turns.append({"role": message.role, "content": message.content})
    return turns


def _is_results(turn: dict[str, Any]) -> bool:
    """A user turn whose content is a list of tool results, so another can join it."""
    content = turn.get("content")
    return isinstance(content, list) and all(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content
    )
