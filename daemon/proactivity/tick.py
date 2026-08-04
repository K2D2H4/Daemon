"""The five-minute tick: generate, mirror, gate. M3a - it does not speak yet.

The three stages of docs/PLAN.md 6.1 wired together, stopping short of the one
LLM call and the delivery. That boundary is deliberate and it is the order PLAN
6.4 asks for: *"게이트 없이 음성을 켜지 않는다"* - do not turn the voice on
without the gate. An ignored notification costs nothing and a voice in a meeting
is an accident, so the deterministic half ships first and gets checked by a human
reading what it *would* have said.

    generate ──► mirror ──► probe once ──► gate each ──► [M3b: judge, speak]
    no model     sqlite     one Reading    deterministic

## One reading, N candidates

Presence is probed once per tick and the same `Reading` judges every candidate.
Probing per candidate costs more and, worse, lets two candidates in the same tick
disagree about where the user is - so the snapshot recorded against one utterance
would not describe the moment another was suppressed.

## Nothing here catches its own exceptions

`app.py`'s job wrapper does that, once, and logs it. A tick that swallowed its own
failures would keep returning "nothing to say" forever and look exactly like a
quiet week, which is this project's signature defect.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from daemon.clock import now as clock_now
from daemon.config import Settings
from daemon.memory.log import from_iso
from daemon.memory.store import Store
from daemon.proactivity.base import Candidate, Presence, Reading, Verdict
from daemon.proactivity.candidates import generate_candidates
from daemon.proactivity.gate import Gate

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Considered:
    """One candidate and what the gate said about it."""

    candidate: Candidate
    verdict: Verdict


@dataclass(frozen=True, slots=True)
class TickResult:
    """What one tick did. Every number is returned rather than logged, because a
    proactivity loop that has decided to stay silent for a week is
    indistinguishable from one that is broken unless it can say which rule kept
    winning."""

    at: datetime
    reading: Reading
    generated: int = 0
    expired: int = 0
    considered: tuple[Considered, ...] = ()
    disabled: bool = False

    @property
    def allowed(self) -> tuple[Considered, ...]:
        return tuple(item for item in self.considered if item.verdict.allowed)

    @property
    def blocked_by(self) -> dict[str, int]:
        """Which rule blocked how many. The tuning readout: PLAN 6.2's budgets are
        starting values, and this is what says whether they are the thing biting."""
        counts: dict[str, int] = {}
        for item in self.considered:
            if item.verdict.allowed:
                continue
            # `why` carries numbers ("cooldown: last spoke 30m ago, needs 90m");
            # the rule name is the part before the colon.
            rule = item.verdict.why.split(":", 1)[0]
            counts[rule] = counts.get(rule, 0) + 1
        return counts


class ProactiveTick:
    """Runs one round. Holds no state between rounds - everything durable is a row."""

    def __init__(
        self,
        store: Store,
        settings: Settings,
        presence: Presence,
        *,
        gate: Gate | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._presence = presence
        self._gate = gate or Gate(settings, store)

    async def run(self, *, now: datetime | None = None) -> TickResult:
        moment = now or clock_now()
        reading = await self._presence.read()

        if not self._settings.proactive_enabled:
            # Still returns a reading, so `daemon proactive` can show what the
            # probes see before anyone turns this on.
            return TickResult(at=moment, reading=reading, disabled=True)

        # Retire what ran out of time before generating, so a generator's dedup
        # check sees the table in its settled state.
        expired = self._store.expire_candidates(now=moment)
        fresh = generate_candidates(self._store, self._settings, now=moment)
        for candidate in fresh:
            self._store.insert_candidate(
                kind=candidate.kind,
                reason=candidate.reason,
                payload=json.dumps(candidate.payload, ensure_ascii=False),
                now=moment,
                due_at=candidate.due_at,
                expires_at=candidate.expires_at,
                fire_budget=candidate.fire_budget,
                cooldown_secs=candidate.cooldown_secs,
            )

        considered = []
        for row in self._store.due_candidates(now=moment):
            candidate = row_candidate(row)
            considered.append(
                Considered(
                    candidate=candidate,
                    verdict=self._gate.judge(candidate, reading, now=moment),
                )
            )
        return TickResult(
            at=moment,
            reading=reading,
            generated=len(fresh),
            expired=expired,
            considered=tuple(considered),
        )


def row_candidate(row: sqlite3.Row) -> Candidate:
    """A `proactive_candidates` row as a `Candidate`.

    The tick gates candidates read back out of the table rather than the ones it
    just generated, so one that came due while the daemon was asleep is considered
    on the next tick instead of only in the tick that created it.

    A payload that is not an object becomes `{}` rather than raising. The column's
    CHECK proves it is valid JSON, which `[1, 2]` and `null` also are, and a tick
    that dies on one malformed row stops considering every other candidate too.
    """
    payload = json.loads(row["payload"])
    return Candidate(
        kind=row["kind"],
        reason=row["reason"],
        payload=payload if isinstance(payload, dict) else {},
        due_at=_maybe(row["due_at"], from_iso),
        expires_at=_maybe(row["expires_at"], from_iso),
        fire_budget=int(row["fire_budget"]),
        cooldown_secs=int(row["cooldown_secs"]),
        id=int(row["id"]),
    )


def _maybe(value: str | None, parse: Callable[[str], datetime]) -> datetime | None:
    return None if value is None else parse(value)
