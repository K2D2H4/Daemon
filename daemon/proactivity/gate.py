"""Stage 2 of docs/PLAN.md 6.1: the deterministic gate.

Zero model calls, by contract (CONTRACTS.md non-negotiable 7). Everything here is
arithmetic on settings, counters and one `Reading`, because the thing a model
cannot know is whether the user is in a meeting right now - and the thing it is
bad at is saying "no". So the gate says no, and the model is only ever asked what
to say about a reason that already survived this file.

Two decisions come out of `judge`, and they are not the same decision:

- **whether** to speak at all - quiet hours, cooldown, budgets, audio in use.
  Blocking is cheap: the candidate is still there on the next tick.
- **where** it would go - `Verdict.delivery`. PLAN 6.4's asymmetry lives here: an
  ignored Telegram message costs nothing, a voice out of the speaker during a
  call is an accident. So signals that only bear on *interruption* (the
  foreground app, an unreadable presence probe) downgrade the route to Telegram
  instead of cancelling the utterance.

`Verdict.why` names the rule that decided, with its numbers, because it is stored
in `proactive_utterances.gate_snapshot` and a bad call has to be readable
afterwards rather than reconstructed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, time, timedelta
from typing import Any, Protocol

from daemon import clock
from daemon.config import Settings
from daemon.proactivity.base import Candidate, Delivery, Reading, Verdict

logger = logging.getLogger(__name__)


class SpokenUtterance(Protocol):
    """A `proactive_utterances` row. The gate reads one column off it, `kind`."""

    def __getitem__(self, column: str) -> Any: ...


class UtteranceHistory(Protocol):
    """What the gate needs to read out of `proactive_utterances`.

    Declared here rather than imported so the gate depends on two queries instead
    of on the store; `daemon.memory.store.Store` satisfies it.

    Synchronous for the same reason `Store` is: one LIMIT-ed lookup and one indexed
    range scan against a local file, cheaper than an executor round trip.
    """

    def last_utterance_at(self) -> datetime | None:
        """When it last spoke first, of any kind. `None` if it never has."""
        ...

    def utterances_since(self, *, since: datetime) -> Sequence[SpokenUtterance]:
        """Everything spoken at or after `since`, UTC.

        The gate counts the kinds itself. The store cannot: the budget resets on a
        *local* day and `spoken_at` is UTC, so which rows count is the caller's
        question - which is why only the lower bound is passed.
        """
        ...

    def recent_bad_labels(self, *, since: datetime) -> Sequence[tuple[str, datetime]]:
        """(kind, labeled_at) for every 👎 at or after `since`, newest first.

        Same shape of question as `utterances_since`, and for the same reason:
        `labeled_at` is UTC and the brake's rules are local-day and rolling-window
        ones, so the gate resolves which rows are "today" or "the last 24h"
        itself rather than asking the store to guess.
        """
        ...


FOCUS_APPS: tuple[str, ...] = (
    "zoom",
    "meet",
    "teams",
    "webex",
    "facetime",
    "slack",
    "discord",
    "keynote",
    "powerpoint",
)
"""Foreground apps that take the speaker away, matched as lowercase substrings.

Not exhaustive, and it cannot be - the repo already learned that a blocklist is
never complete (see `daemon/reflection.py`'s path checks). It does not have to be:
matching only downgrades the route to Telegram, so a miss costs a notification the
user would have got anyway and a false positive costs nothing at all. Meetings and
presentations are on it because those are the two moments where a voice from the
laptop is the accident PLAN 6.4 describes.
"""


def _as_utc(moment: datetime) -> datetime:
    """Aware UTC. A naive value is read as UTC, never as local time.

    `naive.astimezone(UTC)` would silently treat it as local - nine hours of skew
    here, which is a cooldown that passes when it should not.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def parse_quiet_hours(raw: str) -> tuple[time, time]:
    """`"23:00-09:00"` -> two local times. Raises `ValueError` if it is not that."""
    start, dash, end = raw.partition("-")
    if not dash:
        raise ValueError(f"quiet hours must be HH:MM-HH:MM, got {raw!r}")
    return _parse_hhmm(start), _parse_hhmm(end)


def _parse_hhmm(raw: str) -> time:
    hour, colon, minute = raw.strip().partition(":")
    if not colon:
        raise ValueError(f"expected HH:MM, got {raw!r}")
    return time(int(hour), int(minute))  # ValueError on anything else, which is wanted


def within_quiet_hours(local: time, start: time, end: time) -> bool:
    """Whether `local` falls in the window, which usually wraps midnight.

    The default is `23:00-09:00`, so wrapping is the ordinary case rather than the
    edge case. `start == end` is an empty window, not a whole day: the setting is
    written as a range, and reading it as "silent forever" would be a way to turn
    the product off by typo.
    """
    if start == end:
        return False
    if start < end:
        return start <= local < end
    return local >= start or local < end


KIND_REST_HOURS = 6
"""Hours one kind rests after a single 👎 against it."""

KIND_REST_REPEAT_HOURS = 24
"""Both the window and the rest length for the second rule: two 👎 against the
same kind inside this many hours rests that kind for this many hours."""

DAY_STOP_LABELS = 3
"""👎 presses in one local day that end the day.

The brake exists because the C rhythm (6-10 a day) needs one, and because the
button is already under every utterance - the gate routes `both` or `telegram`
and never `local_speaker` alone, so a label is always reachable. macOS Focus was
the other candidate and it is unreadable without Full Disk Access (measured
2026-08-11), which is a change to the machine's security settings and not this
project's to ask for.
"""


def local_day_start(moment: datetime) -> datetime:
    """Local midnight of the day `moment` falls in, expressed in UTC.

    The budget resets on a *local* day because "how many times did it speak today"
    is a human question - the same reason `daemon/memory/log.py` files a message
    under `local_date` while the timestamp inside it stays UTC.
    """
    day = _as_utc(moment).astimezone().date()
    return datetime.combine(day, time.min).astimezone(UTC)


class Gate:
    """Deterministic yes/no plus a route, for one candidate at one instant.

    Takes a `Reading` rather than a `Presence`: one tick probes the machine once
    and may judge several candidates against that reading, and probing per
    candidate would both cost more and let two candidates disagree about where the
    user is.
    """

    def __init__(self, settings: Settings, history: UtteranceHistory) -> None:
        self.settings = settings
        self.history = history

    def judge(
        self, candidate: Candidate, reading: Reading, *, now: datetime | None = None
    ) -> Verdict:
        moment = _as_utc(now if now is not None else clock.now())
        settings = self.settings

        if not settings.proactive_enabled:
            return self._blocked("disabled: DAEMON_PROACTIVE_ENABLED is off", reading)

        quiet = self._quiet_hours_block(moment)
        if quiet is not None:
            return self._blocked(quiet, reading)

        cooldown = self._cooldown_block(moment)
        if cooldown is not None:
            return self._blocked(cooldown, reading)

        budget = self._budget_block(candidate, moment)
        if budget is not None:
            return self._blocked(budget, reading)

        brake = self._label_block(candidate, moment)
        if brake is not None:
            return self._blocked(brake, reading)

        delivery, downgrade = self._route(reading)
        why = "ok" if downgrade is None else f"ok - telegram: {downgrade}"
        return Verdict(allowed=True, why=why, reading=reading, delivery=delivery)

    # --- the blocking rules -------------------------------------------------

    def _quiet_hours_block(self, moment: datetime) -> str | None:
        raw = self.settings.proactive_quiet_hours.strip()
        if not raw:
            return None  # emptied on purpose: no quiet window
        try:
            start, end = parse_quiet_hours(raw)
        except ValueError:
            # Staying silent is the safe reading of a broken brake. Treating it as
            # "no quiet hours" would answer a typo with a voice at 03:00.
            logger.warning("gate: quiet hours %r is not HH:MM-HH:MM; staying silent", raw)
            return f"quiet hours: {raw!r} is not HH:MM-HH:MM - fix the setting; nothing is sent"
        local = moment.astimezone().time()
        if within_quiet_hours(local, start, end):
            return f"quiet hours: {raw} local, and it is {local:%H:%M}"
        return None

    def _cooldown_block(self, moment: datetime) -> str | None:
        minutes = self.settings.proactive_cooldown_minutes
        last = self.history.last_utterance_at()
        if last is None:
            return None
        gap = moment - _as_utc(last)
        if gap < timedelta(minutes=minutes):
            # Negative gap means a stored timestamp in the future (clock change).
            # int() floors, so it still reads as "too soon", which it is.
            return f"cooldown: last spoke {int(gap.total_seconds() // 60)}m ago, needs {minutes}m"
        return None

    def _budget_block(self, candidate: Candidate, moment: datetime) -> str | None:
        since = local_day_start(moment)
        kinds = [row["kind"] for row in self.history.utterances_since(since=since)]
        spoken = len(kinds)
        day = since.astimezone().date().isoformat()

        total = self.settings.proactive_daily_budget
        if spoken >= total:
            return f"daily budget: {spoken} of {total} already spoken on {day}"

        # PLAN 6.2: per-kind ceilings, not allocations - they sum to more than the
        # daily total on purpose. A kind with no ceiling here is bound only by the
        # daily budget above; open_loop is the cheap kind to generate, and on equal
        # terms it eats the whole budget, which turns a companion into a reminder app.
        allowed = self.settings.proactive_kind_budgets.get(candidate.kind)
        if allowed is not None:
            used = kinds.count(candidate.kind)
            if used >= allowed:
                return (
                    f"{candidate.kind} budget: {used} of {allowed} already spoken on {day} "
                    f"({total - spoken} of {total} left overall, for other kinds)"
                )
        return None

    def _label_block(self, candidate: Candidate, moment: datetime) -> str | None:
        """What the user said about this kind, recently, with a thumb.

        Deterministic arithmetic on rows, like every other rule here - CONTRACTS
        non-negotiable 7 puts no model in this file, and "the user asked for less
        of this" is exactly the judgement a model would be worst at anyway.

        The lookback passed to the store is `min(day, moment - 24h)`: whichever of
        the two is earlier, so the fetched rows are a superset of both windows this
        method actually needs - the local day (for the three-strikes count below)
        and the trailing 24h (for the per-kind repeat count). `min` guarantees that
        by construction, not by the size of either window on the day in question,
        so it holds across a DST change too. The precise boundary for each rule is
        then re-applied on the Python side (`at >= day`, `moment - at < ...`).
        """
        day = local_day_start(moment)
        recent = self.history.recent_bad_labels(
            since=min(day, moment - timedelta(hours=KIND_REST_REPEAT_HOURS))
        )

        today = [kind for kind, at in recent if at >= day]
        if len(today) >= DAY_STOP_LABELS:
            return (
                f"stopped for the day: {len(today)} thumbs down since "
                f"{day.astimezone().date().isoformat()}"
            )

        mine = [at for kind, at in recent if kind == candidate.kind]
        within_repeat = [at for at in mine if moment - at < timedelta(hours=KIND_REST_REPEAT_HOURS)]
        if len(within_repeat) >= 2:
            return (
                f"thumbs down: {candidate.kind} got {len(within_repeat)} in the last "
                f"{KIND_REST_REPEAT_HOURS}h, resting"
            )
        if any(moment - at < timedelta(hours=KIND_REST_HOURS) for at in mine):
            return f"thumbs down: {candidate.kind} is resting for {KIND_REST_HOURS}h"
        return None

    # --- routing ------------------------------------------------------------

    def _route(self, reading: Reading) -> tuple[Delivery, str | None]:
        """Where it would go, and why not the speaker if not the speaker.

        Ordered cheapest-certainty first, and every rule here only ever *loses*
        the speaker. PLAN 6.4's asymmetry is the whole shape of this method: an
        ignored Telegram message costs nothing, a voice out of the laptop during
        a meeting is an accident, so anything short of "provably safe to speak"
        routes to text and the utterance itself survives.

        `both` rather than `local_speaker` when the user is here: PLAN 6.3 leaves
        the same words in Telegram so nothing is lost when the speaker is not
        heard - and it is what puts the label buttons on every utterance, which
        the 👎 brake depends on.
        """
        if not self.settings.voice_enabled:
            # DAEMON_PROACTIVE_SPEAKER_ENABLED used to be the switch checked here;
            # it is gone, and DAEMON_VOICE_ENABLED now governs the speaker path
            # too - one switch, so "voice on" cannot mean two different things.
            return "telegram", "DAEMON_VOICE_ENABLED is off"

        at_keyboard = reading.at_keyboard
        if at_keyboard is None:
            # PLAN 6.4, and the reason `at_keyboard` is three-valued: unknown is
            # not "present". The expensive failure is the one we refuse to risk.
            return "telegram", f"presence unknown ({', '.join(reading.unknown) or 'no reading'})"
        if not at_keyboard:
            return "telegram", f"user away, idle {reading.idle_seconds:.0f}s"
        if reading.screen_locked is not False:
            # Sitting here with the screen locked is still away, and an unreadable
            # lock state is not proof of presence.
            state = "locked" if reading.screen_locked else "lock state unknown"
            return "telegram", f"screen {state}"
        if reading.output_muted is not False:
            # `say` exits 0 into a muted device and the row would record a line
            # spoken aloud that nobody heard (daemon/proactivity/speaker.py).
            state = "muted" if reading.output_muted else "mute state unknown"
            return "telegram", f"output {state}"
        if reading.mic_busy is not False:
            # Ours is already subtracted (daemon/mic_hold.py), so this is
            # somebody else holding the microphone - which is what a call is.
            state = "in use" if reading.mic_busy else "state unknown"
            return "telegram", f"microphone {state}"
        if reading.output_busy is not False:
            # Deliberately the weak signal. It is `True` for a notification
            # chime, an autoplaying video, and - observed on the development
            # machine - a system-wide audio EQ holding the device all day, so
            # blocking outright on it means someone who plays music is never
            # messaged at all, the dead-bot end of the failure PLAN 6.1's gate is
            # judged on. It costs the speaker and never the utterance, exactly
            # like the foreground app below.
            state = "in use" if reading.output_busy else "state unknown"
            return "telegram", f"output device {state}"
        if focus_app(reading.foreground_app) is not None and reading.headphones is not True:
            # The only rule headphones excuse, and only on an explicit `True` -
            # `is not True` rather than `not reading.headphones` so that `None`
            # reads the same as `False` here, deliberately. A meeting app in
            # front is a reason not to speak *into the room*; on headphones
            # there is no room. But `headphones` has no probe today
            # (presence.py measured it wrong - `Transport: USB` for this
            # machine's own built-in speakers - and deleted it rather than ship
            # a false excuse), so every real `Reading` has `headphones is None`,
            # and unmeasured must not read as "on headphones". Every other block
            # above still applies either way, including the microphone.
            return "telegram", f"{reading.foreground_app} is in the foreground"
        return "both", None

    def _blocked(self, why: str, reading: Reading) -> Verdict:
        return Verdict(allowed=False, why=why, reading=reading, delivery="telegram")


def focus_app(name: str | None) -> str | None:
    """The `FOCUS_APPS` marker `name` matched, if any."""
    if not name:
        return None
    lowered = name.casefold()
    return next((marker for marker in FOCUS_APPS if marker in lowered), None)
