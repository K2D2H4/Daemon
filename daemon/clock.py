"""The one place Daemon reads the wall clock.

CONTRACTS.md 8 requires ISO-8601 UTC with a literal `Z` and one helper instead
of scattered `datetime.now()` calls. Two reasons beyond tidiness: mixed-offset
timestamps make recency decay (docs/PLAN.md 4.3) quietly wrong, and a single
seam is the only way tests can pin "now".

Two readers, not one: `now`/`to_iso` for anything stored, compared or decayed, and
`local` for anything a human reads. Mixing them is the bug this module prevents.

Precision is milliseconds, not seconds. Two messages in the same turn can land
in the same second, and `MemoryWriter.recent()` orders by timestamp - second
precision would let a reply sort before the message it answers.
"""

from __future__ import annotations

from datetime import UTC, datetime


def now() -> datetime:
    """Current instant, always timezone-aware UTC. Patch this in tests."""
    return datetime.now(UTC)


def to_iso(moment: datetime) -> str:
    """Render as ISO-8601 UTC with `Z`, e.g. `2026-08-03T07:14:00.123Z`.

    A naive datetime is read as UTC: inbound timestamps come from channel
    libraries and one naive value should not be able to kill a turn.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def now_iso() -> str:
    return to_iso(now())


def parse_iso(text: str) -> datetime:
    """Inverse of `to_iso`. Accepts any ISO-8601 form; result is UTC-aware."""
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def local(moment: datetime | None = None) -> datetime:
    """The same instant in the machine's local zone - **display only**.

    `now`/`to_iso` stay the only clock for storage, ordering and recency decay: a
    local timestamp in the database makes `MemoryWriter.recent()` order by a value
    whose meaning moves with the machine, which is the mixed-offset bug this module's
    header exists to prevent. Use this only where a *human* boundary is involved -
    which day it is for the owner, what hour they are reading a sentence - the same
    split `daemon/memory/log.local_date` already makes.

    `.astimezone()` resolves the offset in force on *that date*, which is the only
    form of this that stays correct across a DST boundary.
    """
    return (now() if moment is None else moment).astimezone()
