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

## Nothing here catches its own exceptions - with one exception

`app.py`'s job wrapper does that, once, and logs it. A tick that swallowed its own
failures would keep returning "nothing to say" forever and look exactly like a
quiet week, which is this project's signature defect.

`_association` (type E) is the one deliberate exception, because it is the only
generator with a network dependency - the embedder - and an unreachable Ollama
must not cost the four generators that need nothing but sqlite. It is narrow (it
wraps only the `association_candidates` call) and loud (logged at warning), so it
cannot decay into the silent failure this section otherwise guards against.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

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
from daemon.proactivity.candidates import (
    AssociativeRecall,
    association_candidates,
    generate_candidates,
)
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
        recall: AssociativeRecall | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._presence = presence
        self._gate = gate or Gate(settings, store)
        # Both or neither. With one of them missing the tick can still gate, which
        # is what `daemon proactive` does to show its verdicts without speaking.
        self._judge = judge
        self._delivery = delivery
        # Optional for the same reason recall is optional everywhere else: a
        # broken embedder must not cost the four generators that need nothing
        # but sqlite. `None` here means type E simply produces nothing.
        self._recall = recall

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
        fresh += await self._association(moment)
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
            judged = False

            if verdict.allowed and self._judge is not None and self._delivery is not None:
                judged = True
                utterance = await self._judge.decide(candidate)
                if utterance:
                    delivered = await self._delivery.deliver(
                        candidate, utterance, verdict, now=moment
                    )
                    if delivered:
                        spoke += 1
                else:
                    declined += 1
                    self._rest(candidate, moment)

            considered.append(Considered(candidate, verdict, utterance, delivered))
            if judged:
                # One judge call per tick, whatever it decided. The gate counts the
                # daily budget from rows already stored, so a second call in the
                # same tick would read the same pre-tick count and a delivery would
                # overshoot it - and PLAN 6.2's budget of three (now five) is the
                # brake the whole design leans on. That covers a second *delivery*,
                # but a decline is a model call too - with `proactive_judge_local=False`,
                # PROACTIVE_JUDGE is hosted and paid for - and a tick with several
                # due candidates used to run the judge on every one of them before
                # this `break`, once per candidate, every five minutes, for as long
                # as each stayed due. `_rest` below is what stops that on the next
                # tick; this `break` is what stops it *within* this one. Anything
                # still due is reconsidered later - a delivered one after the
                # cooldown, a declined one after `_rest`.
                break

        result = TickResult(
            at=moment,
            reading=reading,
            generated=len(fresh),
            expired=expired,
            considered=tuple(considered),
            spoke=spoke,
            declined=declined,
        )
        # The round goes into the audit whether or not anything came of it. A tick
        # that decided against speaking leaves no other trace - no utterance row, no
        # log line - so without this the admin cannot tell a loop that considered
        # 288 candidates and refused them all from one that stopped running.
        self._store.record_proactive_round(
            generated=result.generated,
            expired=result.expired,
            considered=len(result.considered),
            spoke=result.spoke,
            declined=result.declined,
            blocked_by=json.dumps(result.blocked_by, ensure_ascii=False),
            now=moment,
        )
        return result

    def _rest(self, candidate: Candidate, moment: datetime) -> None:
        """Push a declined candidate's `due_at` forward so the next tick does
        not put it back in front of the judge unchanged.

        Nothing about a decline marks the row: no state change, no cooldown -
        `due_candidates` would offer it again five minutes later, the gate
        would allow it again (a decline consumes no budget and sets no
        cooldown), and the judge would run again on the same reason. A
        `silence` candidate carries a 12-hour TTL, so unrested that is up to
        144 calls for one already-answered question.

        The rest is `proactive_cooldown_minutes` - the setting already governs
        the minimum gap between two utterances, so a declined candidate is not
        reconsidered any sooner than the daemon would be allowed to speak
        again anyway. Reusing it means one knob tunes both paces instead of a
        second constant nobody would think to change together with the first.

        **A rest can land past the candidate's own `expires_at`, and then it
        never gets another turn** - `due_candidates` never sees it because
        `expire_candidates` retires it first. That is deliberate (an observation
        too stale to speak on is not worth one more model call), but ADR 0016
        changed what it costs and the change is worth naming. Before the flip
        `silence` declined 20/20, so a rest past expiry lost nothing that was
        ever going to be said. After it, `silence` speaks 30/30, and because its
        dedup key is one-per-episode (see `candidates.py`), a single late decline
        now costs the **whole** silence episode rather than one attempt. Observed
        on the owner's live database, 2026-08-26: the row for the
        `2026-08-25T01:56:03Z` episode was rested at 00:33Z to a `due_at` of
        02:03Z, four minutes past its 01:59Z expiry, and the episode produced
        nothing.

        `silence` is not the only kind with that shape - the permanence comes from
        the dedup key, and `existing_dedup_keys` counts `expired` rows too.
        `emotional` is identical (12h TTL, one key per message) and is the other
        businessless kind PLAN 6.2 is about; `pattern_time` loses only a day but is
        much the likeliest to hit this, since a 90-minute rest covers three
        quarters of its 2-hour life. `topic` is exempt - its key is unique per
        raise, and the rest that lands past `expires_at` *is* its designed
        retirement.

        Left as is. A clamp to `min(moment + rest, expires_at)` would break this
        method's own invariant - that a declined candidate is not reconsidered any
        sooner than the daemon could speak anyway - so it is not free, and
        `TickResult.expired` already counts these where an operator can see them.
        Recorded at the line a debugger would read rather than fixed. One number
        not to over-read on the way past: the 0/30 above is `silence`. Declines are
        not rare everywhere - a `topic` candidate on an install with no `tavily`
        server is dropped before the model call **every** time, and every one of
        those comes through here.
        """
        rest = timedelta(minutes=self._settings.proactive_cooldown_minutes)
        self._store.push_candidate_due(candidate.id, due_at=moment + rest)

    async def _association(self, moment: datetime) -> list[Candidate]:
        """Type E, or nothing. Never raises.

        The one place in this file that swallows an exception, and it is narrow
        on purpose: the module docstring says nothing here catches its own
        failures, because a tick that did would look exactly like a quiet week.
        This is the exception because type E is the only generator with a network
        dependency - the embedder - and an unreachable Ollama must not cost the
        four generators that need nothing but sqlite. Logged at warning so it
        cannot be silent.
        """
        if self._recall is None:
            return []
        try:
            return await association_candidates(self._recall, self._store, now=moment)
        except Exception:
            logger.warning(
                "proactive: type E generator failed; the other four still ran",
                exc_info=True,
            )
            return []


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
