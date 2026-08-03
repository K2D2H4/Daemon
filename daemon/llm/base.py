"""Provider-agnostic LLM contract.

This is a seam: every provider (Ollama, Anthropic, OpenAI, Gemini) implements
`Provider`, and callers only ever talk to `LLMGateway.complete(task, ...)`.

Nothing above this layer may import a provider module directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    model: str
    """Concrete model id that served the call, e.g. 'qwen3:14b'."""
    input_tokens: int = 0
    output_tokens: int = 0
    meta: dict[str, str] = field(default_factory=dict)


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
    ) -> Completion: ...

    async def health(self) -> bool:
        """Cheap reachability check. Must not raise."""
        ...
