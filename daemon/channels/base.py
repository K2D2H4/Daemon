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
    external_id: str | None = None
    """The channel's own id for this message. Telegram only confirms an update
    on the *next* getUpdates call, so a restart in between re-delivers it; this
    is the key that lets a duplicate be recognised before it is appended to the
    markdown, which is append-only and has nothing to reconcile by."""
    modality: str = "text"
    """'text' or 'voice'. Recorded on memory rows - see schema.sql."""
    authored_by_sender: bool = True
    """False when an allowlisted sender relayed someone else's words - a
    forwarded message, an inline-bot result, a quoted third party.

    Without this the channel would launder untrusted text into `origin='owner'`:
    a stranger sends the user a prompt injection, the user forwards it asking
    "what is this?", and the schema's forgery-proof origin column records it as
    something the user said. Reflection then learns it as fact. The channel
    still does not interpret the text - it only refuses to vouch for it."""


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    text: str
    labelable: bool = False
    """True for proactive utterances: the channel should attach thumbs up/down
    controls so the label clock can start (docs/PLAN.md 8.3)."""
    utterance_id: str | None = None
    """Set when labelable, so a label can be attributed back."""
    recipient_id: str | None = None
    """Who this is for. A conversational reply names the sender it answers;
    None means an unsolicited utterance with no request to answer (proactivity),
    which the channel delivers to its configured owner.

    Without it, "who may talk to Daemon" and "who receives everything Daemon
    says" collapse into one list, and widening the first would silently widen
    the second."""


@runtime_checkable
class Cursor(Protocol):
    """Where a channel's inbound stream got to, across restarts.

    A separate protocol so the channel never imports storage: it is handed
    something that can remember a number.
    """

    def load_cursor(self, channel: str) -> int | None: ...

    def save_cursor(self, channel: str, offset: int) -> None: ...


@runtime_checkable
class Channel(Protocol):
    name: str

    async def send(self, message: OutboundMessage) -> None: ...

    def listen(self) -> AsyncIterator[InboundMessage]:
        """Long-running inbound loop. Must filter by allowlist before yielding."""
        ...

    async def close(self) -> None: ...
