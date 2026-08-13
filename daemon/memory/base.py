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
    origin: str = "untrusted"
    """Provenance, carried through from the column that keeps it unforgeable.

    Recall replays arbitrary old text, so an item that arrived from elsewhere - a
    forwarded message, an inline-bot result - must not be rendered as something
    the owner said: that is the distinction `messages.origin` exists to protect,
    and rendering it as a plain `user:` line undid it one layer up.

    Defaults closed, not open. It used to default to "owner" - convenient for
    every constructor that always meant to set it, wrong for the one that forgot:
    `daemon/proactivity/candidates.py`'s type E trusts this field to decide
    whether a memory's own words may reach a model prompt at all, so a caller
    that omits the argument should get the value that gets dropped, not the one
    that gets through. `recall.py`'s two real constructors always pass it
    explicitly; only test helpers ever relied on the old default, and they now
    pass "owner" explicitly too."""
    message_id: int | None = None
    """`messages.id`, for a caller that needs a stable identity for this item -
    `daemon/proactivity/candidates.py`'s type E dedups on it. `None` for the
    curated tier (`_curated_item` in recall.py): those rows are `memory_entries`,
    a different id space from `messages`, and stamping a `memory_entries` id in
    here would let two unrelated memories collide on the same dedup key. Defaults
    to `None` so existing constructors keep working."""


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
