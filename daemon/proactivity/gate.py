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

        # PLAN 6.2: open loops are the cheap kind to generate, and on equal terms
        # they eat the whole budget - which turns a companion into a reminder app.
        if candidate.kind == "open_loop":
            used = kinds.count("open_loop")
            allowed = self.settings.proactive_open_loop_budget
            if used >= allowed:
                return (
                    f"open_loop budget: {used} of {allowed} already spoken on {day} "
                    f"({total - spoken} of {total} left overall, for other kinds)"
                )
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
        if not self.settings.proactive_speaker_enabled:
            return "telegram", "DAEMON_PROACTIVE_SPEAKER_ENABLED is off"

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
