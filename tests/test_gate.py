"""The deterministic gate - docs/PLAN.md 6.1 stage 2.

Every brake is tested firing *and* not firing, because both ways of getting it
wrong look healthy from the outside: stuck on is a dead bot, stuck off is a
stalker, and neither raises anything.

`now` is injected everywhere. Where a boundary is a *local* one - quiet hours, the
day a budget resets on - the test pins TZ the way `tests/test_log.py` does, since
otherwise the runner's timezone decides the result.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from daemon.config import Settings
from daemon.memory.store import Store
from daemon.proactivity.base import Candidate, Presence, Reading
from daemon.proactivity.gate import (
    Gate,
    UtteranceHistory,
    focus_app,
    local_day_start,
    parse_quiet_hours,
    within_quiet_hours,
)

NOW = datetime(2026, 8, 4, 3, 30, tzinfo=UTC)
"""12:30 in Seoul, 03:30 in London. Tests that care which pin TZ themselves."""

SPEAKER_ROUTES = {"local_speaker", "both"}
"""Routes that make a sound in the room. PLAN 6.4's expensive failure."""

OPEN_LOOP = Candidate(kind="open_loop", reason="어제 발표 어떻게 됐는지 안 물어봤다")
EMOTIONAL = Candidate(kind="emotional", reason="힘들다고 한 뒤로 이틀 동안 말이 없다")
ASSOCIATION = Candidate(kind="association", reason="교토 여행 얘기가 문득 떠올랐다")
SILENCE = Candidate(kind="silence", reason="이틀째 대화가 없다")

PRESENT = Reading(
    at=NOW,
    idle_seconds=12.0,
    foreground_app="Terminal",
    mic_busy=False,
    output_busy=False,
    output_muted=False,
    screen_locked=False,
)
AWAY = Reading(
    at=NOW,
    idle_seconds=3_600.0,
    foreground_app="Finder",
    mic_busy=False,
    output_busy=False,
    output_muted=False,
    screen_locked=False,
)
UNREADABLE = Reading(at=NOW, unknown=("idle_seconds: ioreg returned nothing",))

ROUTING_BASE: dict[str, Any] = dict(
    idle_seconds=1.0,
    foreground_app="Warp",
    mic_busy=False,
    output_busy=False,
    output_muted=False,
    screen_locked=False,
    headphones=False,
)
"""Every rule released, for the routing table below - so each row only tests
the one rule its name says."""


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's exported DAEMON_PROACTIVE_QUIET_HOURS must not decide whether
    the quiet-hours tests pass."""
    for name in list(os.environ):
        if name.startswith(("DAEMON_", "TELEGRAM_")) or name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)


@contextmanager
def timezone(name: str) -> Iterator[None]:
    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = previous
        time.tzset()


needs_tzset = pytest.mark.skipif(not hasattr(time, "tzset"), reason="needs a POSIX tzset")


class FakeHistory:
    """`proactive_utterances`, without a database. Records the day boundary it was
    asked about, since that boundary is local and easy to get silently wrong.

    Rows are dicts because the gate reads exactly one column off them; the real
    `Store` hands back `sqlite3.Row`, and
    `test_the_budget_rule_works_against_the_real_store` covers that end.
    """

    def __init__(
        self, *, last: datetime | None = None, counts: Mapping[str, int] | None = None
    ) -> None:
        self.last = last
        self.counts = dict(counts or {})
        self.asked_since: list[datetime] = []
        self.bad: list[tuple[str, datetime]] = []
        """(kind, labeled_at) rows, set directly by tests - the brake's input."""

    def last_utterance_at(self) -> datetime | None:
        return self.last

    def utterances_since(self, *, since: datetime) -> list[dict[str, str]]:
        self.asked_since.append(since)
        return [{"kind": kind} for kind, n in self.counts.items() for _ in range(n)]

    def recent_bad_labels(self, *, since: datetime) -> list[tuple[str, datetime]]:
        return [(kind, at) for kind, at in self.bad if at >= since]


class FakePresence:
    """Stands in for the platform probe. Returns whatever reading it was given."""

    def __init__(self, reading: Reading) -> None:
        self.reading = reading
        self.reads = 0

    async def read(self) -> Reading:
        self.reads += 1
        return self.reading


def settings(**overrides: Any) -> Settings:
    """Proactivity on, quiet hours off, speaker on.

    All three are the opposite of the product defaults, deliberately: with the
    brakes released by default a test that expects a block has to be testing the
    one rule it names. The defaults themselves are asserted in their own tests.

    `voice_enabled=True` under the offline preset used to need a `model_construct`
    escape hatch here, because `Settings` refused the combination outright: the
    preset table decided whether voice was possible, and `offline` carried no
    voice row. ADR 0012 made voice its own axis, so the combination is now
    ordinary - but turning voice on is still an explicit act that names a hosted
    provider, so the voice model and key are supplied here rather than bypassed.
    Constructing a configuration that is actually valid beats constructing one
    that skips its own validator.
    """
    base: dict[str, Any] = {
        "provider": "ollama",
        "proactive_enabled": True,
        "proactive_quiet_hours": "",
        "voice_enabled": True,
        "gemini_api_key": "k",
        "gemini_live_model": "m",
    }
    return Settings(_env_file=None, **{**base, **overrides})


def gate_for(history: UtteranceHistory | None = None, **overrides: Any) -> Gate:
    return Gate(settings(**overrides), history or FakeHistory())


# --- the master switches -----------------------------------------------------


def test_nothing_passes_while_proactivity_is_disabled() -> None:
    verdict = gate_for(proactive_enabled=False).judge(EMOTIONAL, PRESENT, now=NOW)

    assert not verdict.allowed
    assert "DAEMON_PROACTIVE_ENABLED" in verdict.why


def test_an_unobstructed_candidate_passes() -> None:
    """The other direction. Without this the file could pass by blocking always."""
    verdict = gate_for().judge(EMOTIONAL, PRESENT, now=NOW)

    assert verdict.allowed
    assert verdict.why == "ok"


def test_the_speaker_switch_is_separate_from_the_proactivity_switch() -> None:
    """Telegram-only proactivity is a complete product; voice is the addition
    that needs the gate to be trustworthy first (PLAN 6.4). One switch now
    governs both the conversation path and the speaker, but it is still a
    different switch from `proactive_enabled` - turning proactivity on does not
    imply a voice out of the laptop."""
    verdict = gate_for(voice_enabled=False).judge(EMOTIONAL, PRESENT, now=NOW)

    assert verdict.allowed
    assert verdict.delivery == "telegram"
    assert "DAEMON_VOICE_ENABLED" in verdict.why


# --- quiet hours -------------------------------------------------------------


@needs_tzset
def test_quiet_hours_block_across_midnight() -> None:
    """The default window is 23:00-09:00, so wrapping is the ordinary case."""
    with timezone("Asia/Seoul"):
        gate = gate_for(proactive_quiet_hours="23:00-09:00")
        # 20:00Z and 22:00Z are 05:00 and 07:00 in Seoul - both inside the window
        # that started the previous evening.
        for moment in (
            datetime(2026, 8, 3, 20, 0, tzinfo=UTC),
            datetime(2026, 8, 3, 22, 0, tzinfo=UTC),
            datetime(2026, 8, 4, 14, 30, tzinfo=UTC),  # 23:30 the same evening
        ):
            verdict = gate.judge(EMOTIONAL, PRESENT, now=moment)
            assert not verdict.allowed, moment
            assert "quiet hours" in verdict.why
            assert "23:00-09:00" in verdict.why


@needs_tzset
def test_quiet_hours_do_not_block_the_rest_of_the_day() -> None:
    with timezone("Asia/Seoul"):
        gate = gate_for(proactive_quiet_hours="23:00-09:00")
        for moment in (
            datetime(2026, 8, 4, 0, 0, tzinfo=UTC),  # 09:00 local, end is exclusive
            datetime(2026, 8, 4, 3, 30, tzinfo=UTC),  # 12:30 local
            datetime(2026, 8, 4, 13, 59, tzinfo=UTC),  # 22:59 local
        ):
            assert gate.judge(EMOTIONAL, PRESENT, now=moment).allowed, moment


@needs_tzset
def test_a_window_inside_one_day_blocks_only_inside_it() -> None:
    """A user who wants silence during work hours writes a non-wrapping window."""
    with timezone("Asia/Seoul"):
        gate = gate_for(proactive_quiet_hours="09:00-17:00")
        inside = datetime(2026, 8, 4, 3, 30, tzinfo=UTC)  # 12:30 local
        outside = datetime(2026, 8, 4, 11, 0, tzinfo=UTC)  # 20:00 local

        assert not gate.judge(EMOTIONAL, PRESENT, now=inside).allowed
        assert gate.judge(EMOTIONAL, PRESENT, now=outside).allowed


def test_quiet_hours_are_local_not_utc() -> None:
    """03:30Z is the middle of the night in London and lunchtime in Seoul. Getting
    this backwards is silent: it speaks at 03:30 and nothing reports why."""
    if not hasattr(time, "tzset"):
        pytest.skip("needs a POSIX tzset")

    gate = gate_for(proactive_quiet_hours="23:00-09:00")
    with timezone("Europe/London"):
        assert not gate.judge(EMOTIONAL, PRESENT, now=NOW).allowed
    with timezone("Asia/Seoul"):
        assert gate.judge(EMOTIONAL, PRESENT, now=NOW).allowed


def test_unparsable_quiet_hours_stay_silent_and_say_so() -> None:
    """A broken brake is read as engaged. The opposite - treating a typo as "no
    quiet hours" - answers a typo with a voice at 03:00.

    `Settings` now refuses this value at startup, which is the better place to
    catch it: a gate that blocks everything means a typo turns the product off
    silently and stays off. So this builds the object *past* validation with
    `model_construct` - not to work around the check, but to assert the gate does
    not become the second line of failure if such a value ever arrives another way.
    """
    broken = Settings.model_construct(
        **{**settings().model_dump(), "proactive_quiet_hours": "11pm to 9am"}
    )
    verdict = Gate(broken, FakeHistory()).judge(EMOTIONAL, PRESENT, now=NOW)

    assert not verdict.allowed
    assert "11pm to 9am" in verdict.why
    assert "HH:MM-HH:MM" in verdict.why


def test_an_empty_window_switches_quiet_hours_off() -> None:
    assert gate_for(proactive_quiet_hours="").judge(EMOTIONAL, PRESENT, now=NOW).allowed


def test_a_zero_length_window_is_not_a_whole_day() -> None:
    """`23:00-23:00` is an empty range. Reading it as "silent forever" would let a
    typo switch the product off with nothing to show for it."""
    start, end = parse_quiet_hours("23:00-23:00")
    assert not within_quiet_hours(start, start, end)


# --- cooldown ----------------------------------------------------------------


def test_cooldown_blocks_and_reports_both_numbers() -> None:
    history = FakeHistory(last=NOW - timedelta(minutes=30))
    verdict = gate_for(history, proactive_cooldown_minutes=90).judge(EMOTIONAL, PRESENT, now=NOW)

    assert not verdict.allowed
    assert "cooldown" in verdict.why
    assert "30m" in verdict.why and "90m" in verdict.why


def test_cooldown_stops_blocking_once_the_gap_is_wide_enough() -> None:
    history = FakeHistory(last=NOW - timedelta(minutes=91))
    assert gate_for(history, proactive_cooldown_minutes=90).judge(
        EMOTIONAL, PRESENT, now=NOW
    ).allowed


def test_cooldown_does_not_block_the_first_utterance_ever() -> None:
    assert gate_for(FakeHistory(last=None)).judge(EMOTIONAL, PRESENT, now=NOW).allowed


def test_a_naive_stored_timestamp_is_read_as_utc() -> None:
    """Nine hours of skew here is a cooldown that passes when it should not, and
    `naive.astimezone()` would introduce exactly that."""
    history = FakeHistory(last=NOW.replace(tzinfo=None) - timedelta(minutes=10))
    assert not gate_for(history, proactive_cooldown_minutes=90).judge(
        EMOTIONAL, PRESENT, now=NOW
    ).allowed


# --- budgets -----------------------------------------------------------------


def test_the_daily_budget_blocks_once_it_is_spent() -> None:
    history = FakeHistory(counts={"emotional": 2, "association": 1})
    verdict = gate_for(history, proactive_daily_budget=3).judge(EMOTIONAL, PRESENT, now=NOW)

    assert not verdict.allowed
    assert "daily budget" in verdict.why
    assert "3 of 3" in verdict.why


def test_the_daily_budget_allows_the_last_one() -> None:
    # Two *other* kinds, not two of the candidate's own - the daily budget is the
    # rule under test here, and the candidate's own per-kind ceiling (2) would
    # otherwise block it for a second, unrelated reason.
    history = FakeHistory(counts={"open_loop": 1, "association": 1})
    assert gate_for(history, proactive_daily_budget=3).judge(EMOTIONAL, PRESENT, now=NOW).allowed


def test_each_kind_has_its_own_ceiling() -> None:
    """PLAN 6.2: the cheap kind to generate eats the budget on equal terms, and
    then the companion is a reminder app. Type E makes that a live race."""
    history = FakeHistory(counts={"association": 3})  # the default association ceiling
    verdict = gate_for(history).judge(ASSOCIATION, PRESENT, now=NOW)

    assert not verdict.allowed
    assert "association budget" in verdict.why


def test_a_kind_at_its_ceiling_does_not_block_the_others() -> None:
    history = FakeHistory(counts={"open_loop": 2})  # the default open_loop ceiling
    assert gate_for(history).judge(ASSOCIATION, PRESENT, now=NOW).allowed


def test_the_overall_budget_still_wins() -> None:
    """Per-kind ceilings sum to 9 against a total of 8, deliberately - they are
    ceilings, not allocations."""
    history = FakeHistory(
        counts={
            "silence": 1,
            "pattern_time": 1,
            "open_loop": 2,
            "emotional": 2,
            "association": 2,
        }
    )
    verdict = gate_for(history).judge(ASSOCIATION, PRESENT, now=NOW)

    assert not verdict.allowed
    assert "daily budget" in verdict.why


def test_a_kind_absent_from_the_table_is_bound_only_by_the_daily_total() -> None:
    """`pattern_time`'s own ceiling is 1; a kind with no entry at all - the shape a
    future generator would arrive in before anyone adds it a line here - must not
    silently inherit a limit of zero and be blocked forever.

    The count stays below `proactive_daily_budget`'s default (5, task-4) with
    headroom to spare - this test is about the missing per-kind ceiling, not
    about the daily total, which `test_the_budget_rule_works_against_the_real_store`
    and friends already cover."""
    history = FakeHistory(counts={"unlisted_kind": 2})
    candidate = Candidate(kind="unlisted_kind", reason="아직 표에 없는 유형")

    assert gate_for(history).judge(candidate, PRESENT, now=NOW).allowed


@needs_tzset
def test_the_budget_counts_a_local_day_not_a_utc_one() -> None:
    """17:00Z on the 3rd is already 02:00 on the 4th in Seoul - the same split
    `daemon/memory/log.py` files that message under. So the day to count is 15:00Z
    the 3rd to 15:00Z the 4th; the UTC day would count the one before it and reset
    the budget in the middle of a Seoul evening."""
    history = FakeHistory()
    with timezone("Asia/Seoul"):
        gate_for(history).judge(EMOTIONAL, PRESENT, now=datetime(2026, 8, 3, 17, 0, tzinfo=UTC))

    assert history.asked_since == [datetime(2026, 8, 3, 15, 0, tzinfo=UTC)]


@needs_tzset
def test_the_local_day_starts_at_local_midnight_across_a_dst_change() -> None:
    """London's clocks go back on 2026-10-25, so that day starts at 23:00Z the day
    before. Truncating the UTC clock to midnight would be an hour out."""
    with timezone("Europe/London"):
        assert local_day_start(datetime(2026, 10, 25, 12, 0, tzinfo=UTC)) == datetime(
            2026, 10, 24, 23, 0, tzinfo=UTC
        )
    with timezone("Asia/Seoul"):
        assert local_day_start(datetime(2026, 8, 3, 17, 0, tzinfo=UTC)) == datetime(
            2026, 8, 3, 15, 0, tzinfo=UTC
        )


@needs_tzset
def test_the_budget_rule_works_against_the_real_store(db: sqlite3.Connection) -> None:
    """The fake reads one dict key; `Store` hands back `sqlite3.Row` out of real SQL,
    and the boundary it filters on is the one this rule depends on. Yesterday's
    open loop must not count against today's sub-cap.
    """
    store = Store(db)
    now = datetime(2026, 8, 3, 17, 0, tzinfo=UTC)  # 02:00 on the 4th in Seoul

    def spoke(kind: str, at: datetime) -> None:
        store.insert_utterance(
            utterance_id=f"{kind}-{at.isoformat()}",
            candidate_id=None,
            kind=kind,
            text="발표 어떻게 됐어?",
            route="telegram",
            gate_snapshot="{}",
            now=at,
        )

    with timezone("Asia/Seoul"):
        spoke("open_loop", now - timedelta(hours=6))  # 20:00 on the 3rd, local
        # Cooldown off, so the rule under test is the only one that can block.
        gate = gate_for(
            store,
            proactive_daily_budget=3,
            proactive_kind_budgets={"open_loop": 1},
            proactive_cooldown_minutes=0,
        )

        # Same UTC day, previous local day: it is 02:00 and the sub-cap is clear.
        assert gate.judge(OPEN_LOOP, PRESENT, now=now).allowed

        spoke("open_loop", now - timedelta(minutes=30))  # 01:30, today
        verdict = gate.judge(OPEN_LOOP, PRESENT, now=now)
        assert not verdict.allowed
        assert "open_loop budget" in verdict.why


# --- the 👎 brake (Task 16) ---------------------------------------------------
# label_counts() had one reader before this - a number in `daemon doctor`.
# Pressing 👎 changed nothing. These are the three rules that make it a brake,
# tested firing and not firing for the reason the module docstring gives: stuck
# on and stuck off both look healthy from the outside.


def test_one_thumbs_down_rests_that_kind_for_six_hours() -> None:
    history = FakeHistory()
    history.bad = [("association", NOW - timedelta(hours=2))]

    rested = gate_for(history).judge(ASSOCIATION, PRESENT, now=NOW)
    assert not rested.allowed
    assert "thumbs down" in rested.why

    others = gate_for(history).judge(EMOTIONAL, PRESENT, now=NOW)
    assert others.allowed, "one kind resting must not silence the rest"


def test_the_rest_expires() -> None:
    history = FakeHistory()
    history.bad = [("association", NOW - timedelta(hours=7))]

    assert gate_for(history).judge(ASSOCIATION, PRESENT, now=NOW).allowed


def test_two_in_a_day_rests_that_kind_for_twenty_four_hours() -> None:
    history = FakeHistory()
    history.bad = [
        ("association", NOW - timedelta(hours=7)),
        ("association", NOW - timedelta(hours=20)),
    ]

    verdict = gate_for(history).judge(ASSOCIATION, PRESENT, now=NOW)
    assert not verdict.allowed
    assert "thumbs down" in verdict.why


def test_a_single_thumbs_down_per_kind_does_not_trigger_the_repeat_rule() -> None:
    """Two different kinds, one 👎 each - the repeat rule counts per kind, not
    per label, so this must read as two singles, not one pair."""
    history = FakeHistory()
    history.bad = [
        ("association", NOW - timedelta(hours=2)),
        ("emotional", NOW - timedelta(hours=2)),
    ]

    verdict = gate_for(history).judge(ASSOCIATION, PRESENT, now=NOW)
    assert not verdict.allowed
    assert "resting for 6h" in verdict.why  # the single-press rule, not the repeat one


@needs_tzset
def test_three_in_a_day_stops_everything() -> None:
    """The owner's "be quiet" switch. Three presses and the day is over - no new
    setting, no new UI, on the button that is already under every utterance.

    TZ pinned on purpose: whether three timestamps up to 5h apart share a local
    day depends on the local day's boundary, and a machine running this suite in
    UTC disagrees with one running it in Seoul about how many of the three are
    "today" unless the boundary is nailed down here.
    """
    history = FakeHistory()
    history.bad = [
        ("association", NOW - timedelta(hours=1)),
        ("emotional", NOW - timedelta(hours=3)),
        ("open_loop", NOW - timedelta(hours=5)),
    ]

    with timezone("Asia/Seoul"):
        verdict = gate_for(history).judge(SILENCE, PRESENT, now=NOW)

    assert not verdict.allowed
    assert "stopped for the day" in verdict.why


@needs_tzset
def test_the_day_stop_is_checked_before_the_per_kind_rules() -> None:
    """A candidate whose own kind has never been thumbed down still stops once the
    day total reaches three - the day-stop message, not silence about its own
    kind having nothing against it.

    TZ pinned for the reason `test_three_in_a_day_stops_everything` is."""
    history = FakeHistory()
    history.bad = [
        ("association", NOW - timedelta(hours=1)),
        ("association", NOW - timedelta(hours=2)),
        ("emotional", NOW - timedelta(hours=3)),
    ]

    with timezone("Asia/Seoul"):
        verdict = gate_for(history).judge(SILENCE, PRESENT, now=NOW)
    assert not verdict.allowed
    assert "stopped for the day" in verdict.why


def test_a_thumbs_down_outside_every_window_does_not_block() -> None:
    """Two days old: outside the 24h repeat window and outside today, so none of
    the three rules should fire."""
    history = FakeHistory()
    history.bad = [("association", NOW - timedelta(hours=48))]

    assert gate_for(history).judge(ASSOCIATION, PRESENT, now=NOW).allowed


# --- the six-signal routing table (Task 6) ------------------------------------


@pytest.mark.parametrize(
    ("name", "override", "expected"),
    [
        ("nothing in the way", {}, "both"),
        ("away from the keyboard", {"idle_seconds": 600.0}, "telegram"),
        ("presence unknown", {"idle_seconds": None}, "telegram"),
        ("screen locked", {"screen_locked": True}, "telegram"),
        ("muted", {"output_muted": True}, "telegram"),
        ("somebody else's mic", {"mic_busy": True}, "telegram"),
        ("output device in use", {"output_busy": True}, "telegram"),
        ("a meeting app in front", {"foreground_app": "zoom.us"}, "telegram"),
        # Every tri-valued field again, unmeasured rather than busy. `is not
        # False` is what makes these route to telegram instead of `both` - a
        # "simplification" to `is True` would flip every one of these from safe
        # to speaking, and nothing above would notice.
        ("microphone state unmeasured", {"mic_busy": None}, "telegram"),
        ("output device state unmeasured", {"output_busy": None}, "telegram"),
        ("output mute state unmeasured", {"output_muted": None}, "telegram"),
        ("screen lock state unmeasured", {"screen_locked": None}, "telegram"),
    ],
)
def test_routing_table(name: str, override: dict[str, Any], expected: str) -> None:
    """One row per rule in `_route`, so a missing rule shows up as a missing row
    rather than as a gap nobody wrote a test for.

    Every row also asserts `allowed`: none of these signals may block the
    utterance outright, only downgrade where it goes (PLAN 6.4's asymmetry -
    the reason `audio_busy=True` was demoted from a block to a route in the
    first place, per this file's own docstring). Confirmed against `judge()`
    rather than assumed: with `gate_for()`'s defaults (quiet hours off, no
    prior utterance, budget unspent) none of the rules that *can* block -
    quiet hours, cooldown, budget - depend on the `Reading` at all, so every
    row here is allowed regardless of which routing rule it exercises.
    """
    reading = Reading(at=NOW, **{**ROUTING_BASE, **override})
    verdict = gate_for().judge(EMOTIONAL, reading, now=NOW)
    assert verdict.allowed, name
    assert verdict.delivery == expected, name


def test_headphones_excuse_only_the_foreground_app() -> None:
    """A meeting app in front is a reason not to speak *into the room*. On
    headphones there is no room. Every other block still applies."""
    on_cans = {**ROUTING_BASE, "foreground_app": "zoom.us", "headphones": True}
    assert gate_for().judge(EMOTIONAL, Reading(at=NOW, **on_cans), now=NOW).delivery == "both"

    still_blocked = {**on_cans, "mic_busy": True}
    verdict = gate_for().judge(EMOTIONAL, Reading(at=NOW, **still_blocked), now=NOW)
    assert verdict.delivery == "telegram"


def test_headphones_unknown_does_not_excuse_the_foreground_app() -> None:
    """`Reading.headphones` has no probe today - presence.py measured that this
    machine's default output answers `Transport: USB` for its own built-in
    speakers, and deleted the probe rather than ship a false "headphones" excuse.
    So every real `Reading` has `headphones is None`, and unlike the other
    tri-valued fields, `None` here must NOT read as permission: only an explicit
    `True` may widen what the speaker may do."""
    meeting = {**ROUTING_BASE, "foreground_app": "zoom.us", "headphones": None}
    verdict = gate_for().judge(EMOTIONAL, Reading(at=NOW, **meeting), now=NOW)
    assert verdict.delivery == "telegram"


def test_our_own_speech_does_not_block_the_next_utterance() -> None:
    """`output_busy` is the weak signal on purpose: it is True for a chime, an
    autoplaying video, and the audio EQ installed on the development machine.
    It costs the speaker and never the utterance."""
    reading = Reading(at=NOW, **{**ROUTING_BASE, "output_busy": True})
    verdict = gate_for().judge(EMOTIONAL, reading, now=NOW)
    assert verdict.delivery == "telegram"
    assert "output device" in verdict.why


def test_a_meeting_in_the_foreground_downgrades_to_telegram() -> None:
    """PLAN 6.4's asymmetry as policy: a text notification during a meeting is
    ignorable, a voice out of the laptop is the accident. So the foreground app
    moves the route, and does not cancel the utterance."""
    meeting = Reading(
        at=NOW,
        idle_seconds=5.0,
        foreground_app="zoom.us",
        mic_busy=False,
        output_busy=False,
        output_muted=False,
        screen_locked=False,
    )
    verdict = gate_for().judge(EMOTIONAL, meeting, now=NOW)

    assert verdict.allowed
    assert verdict.delivery == "telegram"
    assert verdict.delivery not in SPEAKER_ROUTES
    assert "zoom.us" in verdict.why


def test_an_ordinary_foreground_app_keeps_the_speaker() -> None:
    editor = Reading(
        at=NOW,
        idle_seconds=5.0,
        foreground_app="Obsidian",
        mic_busy=False,
        output_busy=False,
        output_muted=False,
        screen_locked=False,
    )
    assert gate_for().judge(EMOTIONAL, editor, now=NOW).delivery == "both"


def test_focus_apps_match_however_the_os_spells_them() -> None:
    assert focus_app("Google Meet") == "meet"
    assert focus_app("Microsoft Teams") == "teams"
    assert focus_app("Obsidian") is None
    assert focus_app(None) is None


# --- presence routing (PLAN 6.3, 6.4) ----------------------------------------


def test_at_the_keyboard_uses_the_speaker_and_telegram_together() -> None:
    """PLAN 6.3 sends both, so nothing is lost when the speaker is not heard."""
    verdict = gate_for().judge(EMOTIONAL, PRESENT, now=NOW)

    assert verdict.allowed
    assert verdict.delivery == "both"
    assert verdict.delivery in SPEAKER_ROUTES


def test_away_from_the_keyboard_routes_to_telegram() -> None:
    verdict = gate_for().judge(EMOTIONAL, AWAY, now=NOW)

    assert verdict.allowed
    assert verdict.delivery == "telegram"
    assert verdict.delivery not in SPEAKER_ROUTES


def test_unknown_presence_never_reaches_the_speaker() -> None:
    """The asymmetry `Reading.at_keyboard` is three-valued for: a failed probe is
    not a present user, and the failure it would risk is the expensive one."""
    verdict = gate_for().judge(EMOTIONAL, UNREADABLE, now=NOW)

    assert verdict.allowed
    assert verdict.delivery == "telegram"
    assert verdict.delivery not in SPEAKER_ROUTES
    assert "presence unknown" in verdict.why
    assert "ioreg" in verdict.why  # which probe failed, for the snapshot


async def test_a_reading_from_a_presence_probe_flows_through_the_gate() -> None:
    """The gate takes a `Reading`, not a `Presence`: one tick probes once and may
    judge several candidates, and probing per candidate would let two of them
    disagree about where the user is."""
    probe = FakePresence(PRESENT)
    assert isinstance(probe, Presence)

    gate = gate_for()
    reading = await probe.read()
    verdicts = [gate.judge(c, reading, now=NOW) for c in (EMOTIONAL, OPEN_LOOP)]

    assert probe.reads == 1
    assert {v.delivery for v in verdicts} == {"both"}


# --- what ends up in gate_snapshot -------------------------------------------


def test_a_blocked_verdict_serialises_with_the_rule_that_decided() -> None:
    """`proactive_utterances.gate_snapshot` is CHECK (json_valid(...)), and the
    column exists so a wrong call can be read back rather than guessed at."""
    history = FakeHistory(last=NOW - timedelta(minutes=5))
    verdict = gate_for(history, proactive_cooldown_minutes=90).judge(
        EMOTIONAL, UNREADABLE, now=NOW
    )

    snapshot = json.loads(json.dumps(verdict.as_snapshot()))
    assert snapshot["allowed"] is False
    assert "cooldown" in snapshot["why"]
    assert snapshot["unknown"] == ["idle_seconds: ioreg returned nothing"]


def test_the_gate_makes_no_model_call() -> None:
    """CONTRACTS.md non-negotiable 7. Asserted structurally: the gate is
    constructed from settings and a history, and there is nowhere for a provider,
    a gateway or an httpx client to hide."""
    gate = gate_for()

    assert set(vars(gate)) == {"settings", "history"}
