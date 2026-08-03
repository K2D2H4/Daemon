"""DM pairing - how a channel learns the owner's id without anyone typing it.

The alternative this replaces: the user opens `@userinfobot`, copies a numeric
id, and pastes it into `.env`. That is a developer's setup step, not a product's.

Instead an unknown sender's DM is **dropped** - it is never handled, never
logged, never shown to a model - and answered with a short code. The owner reads
the code and approves it from their own terminal, so the id is captured from the
message rather than transcribed by hand, and approval happens on a surface a
stranger cannot reach.

The parameters follow OpenClaw's, which has the security details worked out:

  * 8 characters from a 32-symbol alphabet, ambiguous glyphs left out, so a code
    survives being read aloud or retyped (~1.1e12 combinations).
  * Roughly an hour of validity, and **one notice per sender per code lifetime**:
    without that, anyone with a loop turns the bot into an outbound message
    generator.
  * At most `MAX_PENDING` requests waiting per channel, so codes cannot be
    guessed by volume - a fourth stranger is simply ignored.
  * Approval is per-sender and grants nothing but the ability to be heard. The
    **first** approval also bootstraps the owner, once: later approvals add a
    guest, so repeated pairing is not a route to ownership.

This module owns the policy and nothing else. It never touches the network, so
the approval CLI can import it directly; storage arrives through the
`PairingStore` protocol below, keeping `channels/` free of a storage import the
same way `base.Cursor` does.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from daemon import clock

logger = logging.getLogger(__name__)

CODE_LENGTH = 8

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
"""A-Z and 2-9 minus 0/O/1/I. A code is read off one screen and typed into
another, so the pairs that survive neither a font nor a phone call are not worth
the 4 extra symbols: 32**8 is 1.1e12 either way, and `MAX_PENDING` is what
actually bounds guessing."""

CODE_TTL = timedelta(hours=1)
"""How long a code is good for - and, because a live pending row is also the
record that this sender has already been told, exactly how long until they can
be sent another. One constant on purpose: two that had to stay equal would
eventually not."""

MAX_PENDING = 3
"""Codes in flight per channel, once there is an owner. Low deliberately: it is
the real ceiling on guessing by volume, and one person pairing a laptop and a
phone needs two."""

BOOTSTRAP_MAX_PENDING = 12
"""The cap before anyone has been approved, i.e. during first-run onboarding.

With a flat cap of 3, three strangers who happen to find the bot first hold every
slot for an hour and the actual owner's first message gets no code - the one
moment the product cannot afford to be unusable, and unrecoverable without
falling back to pasting a numeric id by hand. A higher ceiling here is close to
free: guessing still has to beat 32**8 within an hour, and nobody but the owner,
reading the terminal, can approve anything. Once an owner exists the tighter cap
applies, because by then a stranger being ignored costs nothing."""

_CREATE_ATTEMPTS = 3
"""Retries when a generated code collides with a live one. Astronomically
unlikely, but this runs inside the inbound poll loop, where an uncaught
IntegrityError would end the generator and leave the daemon permanently deaf."""

PAIRING_NOTICE = (
    "This is someone's private Daemon, and I don't know you yet.\n\n"
    "Pairing code: {code}\n\n"
    "Give it to my owner and they can let you in. It expires in about an hour.\n"
    "\n"
    "이 데몬은 개인용이라 아직 당신을 모릅니다. 위 코드를 오너에게 전달하면 "
    "대화를 시작할 수 있습니다. 약 1시간 후 만료됩니다."
)
"""What a stranger gets instead of an answer.

Deliberately does not name the approval command: the owner already knows it, and
a stranger learning it gains only a way to sound convincing. It also echoes
nothing the sender wrote - their text is dropped, not quoted back.
"""


class PairingError(Exception):
    """An approval could not be made: unknown code, expired code, wrong channel."""


@dataclass(frozen=True, slots=True)
class Decision:
    """What a channel should do with a message from a given sender."""

    allowed: bool
    notice: str | None = None
    """Text to reply with before dropping the message. None means stay silent -
    the sender has already been told, or the pending queue is full."""


ALLOW = Decision(allowed=True)
DENY = Decision(allowed=False)


@dataclass(frozen=True, slots=True)
class PendingRequest:
    sender_id: str
    code: str
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class Approval:
    sender_id: str
    is_owner: bool
    """True only for the very first approval on this channel."""


class _Row(Protocol):
    """Structural view of one `channel_pairing` row. `sqlite3.Row` satisfies it,
    so storage returns its own rows and does not import this module."""

    def __getitem__(self, key: str) -> Any: ...


@runtime_checkable
class PairingStore(Protocol):
    """The storage this policy needs - see `daemon/memory/store.py`."""

    def is_allowed(self, channel: str, sender_id: str) -> bool: ...

    def has_owner(self, channel: str) -> bool: ...

    def create_pairing(
        self,
        channel: str,
        sender_id: str,
        code: str,
        *,
        created_at: datetime,
        expires_at: datetime,
    ) -> bool: ...

    def pairing_by_code(self, channel: str, code: str) -> _Row | None: ...

    def pending_pairings(self, channel: str) -> Sequence[_Row]: ...

    def approve_pairing(
        self, channel: str, sender_id: str, *, approved_at: datetime
    ) -> bool | None: ...

    def expire_pairings(self, channel: str, *, now: datetime) -> int: ...


def generate_code() -> str:
    """A fresh pairing code. `secrets`, not `random`: this is a credential for
    the length of an hour, and the module that guesses it is the same module that
    would guess a session token."""
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


def normalise_code(code: str) -> str:
    """What the owner typed, as it is stored. Codes are uppercase and get pasted
    with stray whitespace, so a lowercase paste must not read as a wrong code."""
    return code.strip().upper()


class Pairing:
    """Pairing policy for one channel.

    Two audiences, deliberately in one place because they share the rules:
    `screen()` is called by the channel on every unknown sender, while
    `pending()` and `approve()` are what the owner's CLI wraps.
    """

    def __init__(
        self,
        store: PairingStore,
        channel: str,
        *,
        now: Callable[[], datetime] = clock.now,
    ) -> None:
        self._store = store
        self._channel = channel
        self._now = now

    # --- channel side -------------------------------------------------------

    def screen(self, sender_id: str) -> Decision:
        """Whether this sender may be heard, and what to reply if not.

        `sender_id` must be the channel's numeric id, never a display name or a
        username - both are attacker-controlled.
        """
        if self._store.is_allowed(self._channel, sender_id):
            return ALLOW

        pending = self._live_pending()
        # Before the first approval this is onboarding, not defence: the owner
        # must not be crowded out of their own first-run by whoever found the bot
        # first. See BOOTSTRAP_MAX_PENDING.
        cap = MAX_PENDING if self._store.has_owner(self._channel) else BOOTSTRAP_MAX_PENDING
        if any(str(row["sender_id"]) == sender_id for row in pending):
            # Their code is still good, so sending another would only be a way to
            # make the bot talk on demand.
            logger.info("pairing: %s id=%s already has a live code", self._channel, sender_id)
            return DENY
        if len(pending) >= cap:
            # Refusing to issue is the whole point: with unbounded pending rows,
            # a guessing attack just needs enough parallel senders.
            logger.warning(
                "pairing: %s has %d requests waiting, ignoring id=%s",
                self._channel,
                len(pending),
                sender_id,
            )
            return DENY

        created_at = self._now()
        expires_at = created_at + CODE_TTL
        for _ in range(_CREATE_ATTEMPTS):
            code = generate_code()
            if self._store.create_pairing(
                self._channel,
                sender_id,
                code,
                created_at=created_at,
                expires_at=expires_at,
            ):
                logger.info("pairing: issued a code on %s to id=%s", self._channel, sender_id)
                return Decision(allowed=False, notice=PAIRING_NOTICE.format(code=code))
        logger.error("pairing: could not store a code on %s for id=%s", self._channel, sender_id)
        return DENY

    # --- owner side ---------------------------------------------------------

    def pending(self) -> list[PendingRequest]:
        """Requests waiting for approval, oldest first. Expired ones are gone."""
        return [
            PendingRequest(
                sender_id=str(row["sender_id"]),
                code=str(row["code"]),
                created_at=clock.parse_iso(row["created_at"]),
                expires_at=clock.parse_iso(row["expires_at"]),
            )
            for row in self._live_pending()
        ]

    def approve(self, code: str) -> Approval:
        """Let the sender who was given `code` through, permanently.

        Raises `PairingError` for a code that is unknown, expired, already spent,
        or issued on a different channel - the lookup is channel-scoped, so a
        code from Telegram cannot approve anyone anywhere else.
        """
        wanted = normalise_code(code)
        row = self._store.pairing_by_code(self._channel, wanted)
        if row is None:
            # Approved rows have their code cleared, so a spent code lands here
            # too rather than approving the same sender twice.
            raise PairingError(f"no pairing request on {self._channel} for code {wanted}")
        if clock.parse_iso(row["expires_at"]) <= self._now():
            self._live_pending()  # sweep it, so the next `pending()` is honest
            raise PairingError(f"code {wanted} has expired; ask them to message again")

        sender_id = str(row["sender_id"])
        is_owner = self._store.approve_pairing(
            self._channel, sender_id, approved_at=self._now()
        )
        if is_owner is None:
            raise PairingError(f"no pairing request on {self._channel} for code {wanted}")
        logger.info(
            "pairing: approved %s id=%s%s",
            self._channel,
            sender_id,
            " as the owner" if is_owner else "",
        )
        return Approval(sender_id=sender_id, is_owner=is_owner)

    def _live_pending(self) -> Sequence[_Row]:
        """Pending rows, after dropping the ones that timed out.

        Expiry deletes rather than marks: `channel_pairing.state` admits only
        'pending' and 'approved' (schema.sql is frozen), and deleting is also what
        makes the row's mere existence mean "this sender has already been told".
        """
        self._store.expire_pairings(self._channel, now=self._now())
        return self._store.pending_pairings(self._channel)
