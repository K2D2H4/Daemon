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
