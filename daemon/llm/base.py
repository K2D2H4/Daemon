"""Provider-agnostic LLM contract.

This is a seam: every provider (Ollama, Anthropic, OpenAI, Gemini) implements
`Provider`, and callers only ever talk to `LLMGateway.complete(task, ...)`.

Nothing above this layer may import a provider module directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant", "tool"]
"""'tool' carries the result of one tool call back to the model.

Every provider spells this differently - Anthropic has no tool role at all and
wants a `user` turn holding `tool_result` blocks - so the role exists here, in
the neutral shape, and each provider translates it. Nothing above this layer
should have to know which of the four is answering.
"""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool as the model is told about it.

    `parameters` is a JSON Schema object. It is passed through to the provider
    untouched, which is what lets an MCP server's own schema be forwarded without
    this layer having to understand it (daemon/tools/mcp.py).
    """

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The model asking for one tool to be run."""

    id: str
    """Correlates the result back to the request. Anthropic and OpenAI both issue
    one; Gemini issues none, so its provider synthesises it."""
    name: str
    arguments: dict[str, Any]
    """Already decoded. OpenAI hands these over as a JSON string and Ollama as a
    dict, and that difference stops at the provider boundary."""


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    """Set on an `assistant` turn that asked for tools. Defaulted, so every
    existing constructor keeps working."""
    tool_call_id: str | None = None
    """Set on a `tool` turn: which call this is the result of."""


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    model: str
    """Concrete model id that served the call, e.g. 'qwen3:14b'."""
    input_tokens: int = 0
    output_tokens: int = 0
    meta: dict[str, str] = field(default_factory=dict)
    tool_calls: tuple[ToolCall, ...] = ()
    """Tools the model wants run before it can answer. When this is non-empty
    `text` is often empty - a provider must not treat that as a failed call."""


def decode_tool_arguments(raw: object) -> dict[str, Any]:
    """Normalise whatever a provider calls tool arguments into a dict.

    Three of the four spell this differently - OpenAI sends a JSON string, Ollama
    a dict, Gemini a dict it calls `args` - and a small model will occasionally
    send a string that is not valid JSON at all. That last case is a bad tool call
    and not a broken provider, so it comes back as an empty dict and fails later
    against the tool's own schema, where the error can name the tool.
    """
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    if isinstance(raw, str) and raw.strip():
        import json

        try:
            decoded = json.loads(raw)
        except ValueError:
            return {}
        if isinstance(decoded, dict):
            return {str(key): value for key, value in decoded.items()}
    return {}


class ProviderError(RuntimeError):
    """Raised by a provider on unrecoverable failure. The gateway may fall back."""


@runtime_checkable
class Embedder(Protocol):
    """Text -> vector. Separate from `Provider` because the shape is different
    and because the two are routed independently: chat may be hosted while
    embeddings stay local.

    Vectors must be returned L2-normalised, so recall can use a dot product and
    skip the norm per query.
    """

    name: str
    dimensions: int
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Provider(Protocol):
    """A single LLM backend.

    Implementations must be async, must not retry internally more than once,
    and must raise ProviderError (not provider-specific exceptions) so the
    gateway can decide about fallback.
    """

    name: str
    """Stable identifier used in config and logs, e.g. 'ollama', 'gemini'."""

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> Completion:
        """`tools` offers the model a set of tools it may ask for.

        Offering is not permission: what the model is allowed to *reach* is
        decided deterministically before this call, by filtering the specs
        (daemon/tools/policy.py). A provider only translates.
        """
        ...

    async def health(self) -> bool:
        """Cheap reachability check. Must not raise."""
        ...
