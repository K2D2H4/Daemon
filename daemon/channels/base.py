"""Channel contract - how Daemon reaches the user and hears back.

Deliberately two operations only: send one message out, and yield messages in.
Telegram is the first implementation; a local speaker (proactive voice) and a
web UI come later. See docs/PLAN.md 3.1 and 6.3.

Nothing above this layer may import a channel implementation directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class InboundMessage:
    text: str
    sender_id: str
    """Channel-scoped stable id. Used for the allowlist - never trust display names."""
    received_at: datetime
    channel: str
    modality: str = "text"
    """'text' or 'voice'. Recorded on memory rows - see schema.sql."""


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    text: str
    labelable: bool = False
    """True for proactive utterances: the channel should attach thumbs up/down
    controls so the label clock can start (docs/PLAN.md 8.3)."""
    utterance_id: str | None = None
    """Set when labelable, so a label can be attributed back."""


@runtime_checkable
class Channel(Protocol):
    name: str

    async def send(self, message: OutboundMessage) -> None: ...

    def listen(self) -> AsyncIterator[InboundMessage]:
        """Long-running inbound loop. Must filter by allowlist before yielding."""
        ...

    async def close(self) -> None: ...
