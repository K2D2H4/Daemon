"""The one seam between the conversation loop and storage.

The loop must not know whether a message ends up in markdown, sqlite, or both.
It calls `record`, and the memory layer guarantees markdown-first (source of
truth) then sqlite mirror.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

Origin = Literal["owner", "agent", "untrusted", "system"]
SessionKind = Literal["interactive", "voice", "proactive", "reflection"]
Modality = Literal["text", "voice"]


@dataclass(frozen=True, slots=True)
class LoggedMessage:
    ts: datetime
    role: Literal["user", "assistant"]
    content: str
    origin: Origin
    session_kind: SessionKind
    modality: Modality
    channel: str
    sender_id: str | None = None
    external_id: str | None = None


@dataclass(frozen=True, slots=True)
class RecalledItem:
    """One thing worth putting back in front of the model."""

    content: str
    ts: datetime
    role: str
    score: float
    reason: str
    """Why it surfaced - 'keyword', 'vector', 'both'. Goes in the golden-set
    report so a recall failure can be attributed instead of guessed at."""


@runtime_checkable
class Recall(Protocol):
    """Lane 1 (docs/PLAN.md 4.3).

    Hard constraint: **zero LLM calls.** This runs on every turn, including
    voice turns, where the whole round trip has a sub-second budget. An embedder
    call for the query is allowed - that is a local model, measured in
    milliseconds - but nothing that thinks.

    Degrades rather than fails: with no embedder reachable, keyword-only recall
    is a worse answer than no answer at all, but a raised exception in the middle
    of a conversation is worse than both.
    """

    async def search(self, query: str, *, limit: int = 8) -> list[RecalledItem]: ...

    async def index(self, message_id: int, text: str) -> None:
        """Embed and store one message. Called after it is recorded, and must not
        make the caller's turn fail if the embedder is down."""
        ...

    async def backfill(self, limit: int = 500) -> int:
        """Embed whatever has no vector yet, returning how many landed.

        In the protocol rather than left to the implementation because startup
        cannot be correct without it. A rebuilt sqlite file gives every message a
        new id and drops `embeddings` by cascade, so the vector lane would stay
        empty for all history while the health check still said the recall was
        ready. Measured on the golden set, that silent state is a 50% ceiling for
        Korean instead of the hybrid number - the worst kind of regression,
        because nothing fails.

        Reports what it managed rather than raising: a cold embedder at startup
        must not stop the daemon from serving text.
        """
        ...


@runtime_checkable
class MemoryWriter(Protocol):
    async def record(self, message: LoggedMessage) -> None:
        """Append to the markdown log first, then mirror into sqlite.

        Must be safe to call concurrently for the same day's log file.
        """
        ...

    async def seen(self, channel: str, external_id: str) -> bool:
        """Has this channel message already been recorded?

        The markdown is append-only, so a duplicate has to be caught before the
        write rather than reconciled after it.
        """
        ...

    async def recent(self, limit: int = 20) -> list[LoggedMessage]:
        """Most recent messages, oldest first. M1a uses this as the entire
        context window; real recall arrives in M1b."""
        ...
