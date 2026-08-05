"""The five-minute tick: all three stages of docs/PLAN.md 6.1, wired together.

    generate ──► mirror ──► probe once ──► gate each ──► judge ──► deliver
    no model     sqlite     one Reading    deterministic   1 call   at most once

The judge and the delivery are **optional**, and a tick without them still runs
everything to the left of them. That is not a debugging convenience: PLAN 6.4 asks
for the gate to be trustworthy *before* anything is wired to a speaker - an ignored
notification costs nothing and a voice in a meeting is an accident - so
`daemon proactive` assembles a tick where speaking is structurally impossible and
prints what it *would* have said. `daemon proactive --speak` adds the other two.

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
from daemon.proactivity.base import (
    Candidate,
    Judgement,
    Presence,
    Reading,
    Utterance,
    Verdict,
)
from daemon.proactivity.candidates import generate_candidates
from daemon.proactivity.delivery import Delivered, ProactiveDelivery
from daemon.proactivity.gate import Gate

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Considered:
    """One candidate, what the gate said, and what became of it.

    `utterance` and `delivered` stay `None` when the tick was assembled without a
    judge, and `delivered` alone stays `None` when the judge declined - which is
    the common case and not a failure.
    """

    candidate: Candidate
    verdict: Verdict
    utterance: Utterance | None = None
    delivered: Delivered | None = None


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
    spoke: int = 0
    declined: int = 0
    """Allowed by the gate, and the judge still had nothing worth saying. Counted
    separately because it is the healthy case: docs/CONTRACTS.md 7 makes silence
    the default, and a judge that never declines is one nobody should trust."""

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
        judge: Judgement | None = None,
        delivery: ProactiveDelivery | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._presence = presence
        self._gate = gate or Gate(settings, store)
        # Both or neither. With one of them missing the tick can still gate, which
        # is what `daemon proactive` does to show its verdicts without speaking.
        self._judge = judge
        self._delivery = delivery

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
        spoke = declined = 0
        for row in self._store.due_candidates(now=moment):
            candidate = row_candidate(row)
            verdict = self._gate.judge(candidate, reading, now=moment)
            utterance: Utterance | None = None
            delivered: Delivered | None = None

            if verdict.allowed and self._judge is not None and self._delivery is not None:
                utterance = await self._judge.decide(candidate)
                if utterance:
                    delivered = await self._delivery.deliver(
                        candidate, utterance, verdict, now=moment
                    )
                else:
                    declined += 1

            considered.append(Considered(candidate, verdict, utterance, delivered))
            if delivered:
                spoke += 1
                # One utterance per tick, and the loop stops here. The gate counts
                # the daily budget from rows already stored, so a second delivery
                # in the same tick would read the same pre-tick count and overshoot
                # it - and PLAN 6.2's budget of three is the brake the whole design
                # leans on. Anything still due is reconsidered five minutes later.
                break

        return TickResult(
            at=moment,
            reading=reading,
            generated=len(fresh),
            expired=expired,
            considered=tuple(considered),
            spoke=spoke,
            declined=declined,
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
