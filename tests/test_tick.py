"""The proactivity tick: generate, gate, and - when wired for it - speak.

The judge and the speaker are fakes here. What is under test is the orchestration:
how many model calls happen, how many utterances a single tick can produce, and
what a decline costs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from daemon.channels.base import OutboundMessage
from daemon.config import Settings
from daemon.memory.base import LoggedMessage
from daemon.memory.store import Store
from daemon.memory.writer import FileMemoryWriter
from daemon.proactivity.base import Candidate, Reading, Utterance
from daemon.proactivity.delivery import ProactiveDelivery
from daemon.proactivity.tick import ProactiveTick

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)


class FakePresence:
    def __init__(self, reading: Reading | None = None) -> None:
        self.reading = reading or Reading(
            at=NOW, idle_seconds=5.0, foreground_app="Warp", mic_busy=False, output_busy=False
        )
        self.reads = 0

    async def read(self) -> Reading:
        self.reads += 1
        return self.reading


class FakeJudge:
    """Records every call, so "exactly one model call per candidate" is checkable."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies) or ["발표 어떻게 됐어?"]
        self.asked: list[Candidate] = []

    async def decide(self, candidate: Candidate) -> Utterance:
        self.asked.append(candidate)
        text = self.replies.pop(0) if self.replies else ""
        return Utterance(text=text) if text else Utterance(why_not="할 말이 없다")


class FakeChannel:
    name = "fake"

    def __init__(self) -> None:
        self.sent: list[OutboundMessage] = []

    async def send(self, message: OutboundMessage) -> None:
        self.sent.append(message)


def settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "preset": "offline",
        "proactive_enabled": True,
        "proactive_quiet_hours": "",
    }
    # `voice_enabled=True` under the offline preset used to need a
    # `model_construct` escape hatch here, because `Settings` refused it outright
    # (the switch was overloaded to also mean "the hosted conversation route
    # exists," which offline never satisfies). That routed-session check now
    # lives at session start instead, so the offline preset legitimately allows
    # `voice_enabled=True` - it is what lets `/usr/bin/say` run without a route -
    # and this builds like any other `Settings` call.
    return Settings(_env_file=None, **{**base, **overrides})


@pytest.fixture
def store(db: Any) -> Store:
    return Store(db)


def add_candidate(store: Store, kind: str = "silence", reason: str = "조용하다") -> int:
    return store.insert_candidate(kind=kind, reason=reason, payload="{}", now=NOW)


def tick_for(
    store: Store,
    data_dir: Path,
    *,
    speaking: bool = True,
    judge: FakeJudge | None = None,
    **over: Any,
) -> tuple[ProactiveTick, FakeJudge, FakeChannel]:
    channel = FakeChannel()
    used = judge or FakeJudge()
    delivery = ProactiveDelivery(store, FileMemoryWriter(data_dir, store), channel=channel)
    return (
        ProactiveTick(
            store,
            settings(**over),
            FakePresence(),
            judge=used if speaking else None,
            delivery=delivery if speaking else None,
        ),
        used,
        channel,
    )


# --- the dry half still works ------------------------------------------------


async def test_without_a_judge_it_gates_and_says_nothing(store: Store, data_dir: Path) -> None:
    """`daemon proactive` runs exactly this: the deterministic half, checkable on
    its own, which is why M3a shipped before anything could speak."""
    add_candidate(store)
    tick, judge, channel = tick_for(store, data_dir, speaking=False)

    result = await tick.run(now=NOW)

    assert len(result.considered) == 1
    assert result.considered[0].verdict.allowed
    assert result.considered[0].utterance is None
    assert (result.spoke, judge.asked, channel.sent) == (0, [], [])


async def test_presence_is_probed_once_for_the_whole_tick(
    store: Store, data_dir: Path
) -> None:
    """Probing per candidate costs more and lets two candidates in one tick
    disagree about where the user is."""
    add_candidate(store, "silence")
    add_candidate(store, "emotional")
    presence = FakePresence()
    tick = ProactiveTick(store, settings(), presence)

    await tick.run(now=NOW)

    assert presence.reads == 1


async def test_a_disabled_daemon_generates_nothing_but_still_reads_presence(
    store: Store, data_dir: Path
) -> None:
    """So `daemon proactive` can show what the probes see before anyone turns it
    on, and so a month switched off leaves no backlog to dump."""
    tick, _, _ = tick_for(store, data_dir, proactive_enabled=False)

    result = await tick.run(now=NOW)

    assert result.disabled
    assert result.reading.foreground_app == "Warp"
    assert result.generated == 0


# --- type E, optional ---------------------------------------------------------


async def test_a_tick_without_recall_still_runs_the_other_four(
    store: Store, data_dir: Path
) -> None:
    """Recall is optional everywhere else in this codebase - a broken embedder
    must not cost the conversation loop - and it is optional here for the same
    reason. Four generators is a worse tick, not a dead one."""
    add_candidate(store)
    tick = ProactiveTick(store, settings(), FakePresence(), recall=None)

    result = await tick.run(now=NOW)

    assert result.disabled is False


async def test_a_failing_recall_does_not_kill_the_tick(store: Store, data_dir: Path) -> None:
    """An embedder that cannot be reached is an ordinary Tuesday. The other four
    generators do not depend on it and must still be considered."""

    class _Broken:
        async def associate(self, *args: Any, **kwargs: Any) -> list[Any]:
            raise RuntimeError("ollama is down")

    add_candidate(store)
    tick = ProactiveTick(store, settings(), FakePresence(), recall=_Broken())

    result = await tick.run(now=NOW)  # must not raise

    assert result.disabled is False


# --- the one model call ------------------------------------------------------


async def test_the_judge_is_asked_only_about_candidates_the_gate_allowed(
    store: Store, data_dir: Path
) -> None:
    """docs/CONTRACTS.md 7: exactly one LLM call, and only for a candidate that
    already passed. A blocked candidate must not cost a model call."""
    add_candidate(store)
    tick, judge, _ = tick_for(store, data_dir, proactive_quiet_hours="00:00-23:00")

    result = await tick.run(now=NOW)

    assert not result.considered[0].verdict.allowed
    assert judge.asked == []


async def test_one_allowed_candidate_costs_exactly_one_call(
    store: Store, data_dir: Path
) -> None:
    add_candidate(store)
    tick, judge, _ = tick_for(store, data_dir)

    await tick.run(now=NOW)

    assert len(judge.asked) == 1


async def test_a_tick_speaks_at_most_once(store: Store, data_dir: Path) -> None:
    """The gate counts the daily budget from stored rows, so a second delivery in
    the same tick reads the same pre-tick count and overshoots PLAN 6.2's three."""
    add_candidate(store, "silence")
    add_candidate(store, "emotional")
    add_candidate(store, "pattern_time")
    tick, judge, channel = tick_for(
        store, data_dir, judge=FakeJudge("첫 마디", "둘째 마디", "셋째 마디")
    )

    result = await tick.run(now=NOW)

    assert result.spoke == 1
    assert len(channel.sent) == 1
    assert len(judge.asked) == 1  # the loop stopped, so no wasted calls either


async def test_what_was_not_reached_is_still_due_next_tick(
    store: Store, data_dir: Path
) -> None:
    add_candidate(store, "silence")
    add_candidate(store, "emotional")
    tick, _, _ = tick_for(store, data_dir)

    await tick.run(now=NOW)

    # One fired; the other is still waiting, and the global cooldown is what stops
    # it going out immediately - not the absence of a candidate.
    assert len(store.due_candidates(now=NOW)) == 1


# --- declining is the healthy case -------------------------------------------


async def test_a_decline_sends_nothing_and_spends_nothing(
    store: Store, data_dir: Path
) -> None:
    """Silence is the default (non-negotiable 7). A judge that never declines is
    one nobody should trust, so the path has to cost nothing but a rest."""
    add_candidate(store)
    tick, _, channel = tick_for(store, data_dir, judge=FakeJudge(""))

    result = await tick.run(now=NOW)

    assert (result.declined, result.spoke) == (1, 0)
    assert channel.sent == []
    assert store.utterances_since(since=NOW) == []
    # Finding 1: an untouched candidate is exactly the bug. It is not fired,
    # cancelled or expired - a better moment can still use it - but it is
    # resting, so the very next tick does not put it back in front of the
    # judge for the same already-answered question.
    assert len(store.due_candidates(now=NOW)) == 0
    rest = timedelta(minutes=settings().proactive_cooldown_minutes)
    assert len(store.due_candidates(now=NOW + rest)) == 1


async def test_an_unrested_decline_would_cost_a_call_every_tick(
    store: Store, data_dir: Path
) -> None:
    """The regression finding 1 describes: without the rest, five minutes later
    `due_candidates` returns the same declined candidate and the judge runs on it
    again. Reproduced here by running two ticks five minutes apart on the
    hosted-shaped setup (`silence`'s TTL is long enough to still be live)."""
    add_candidate(store, "silence")
    tick, judge, _ = tick_for(store, data_dir, judge=FakeJudge("", ""))

    await tick.run(now=NOW)
    await tick.run(now=NOW + timedelta(minutes=5))

    # One call, not two: the rest from the first decline pushed `due_at` past
    # the second tick's `now`, so the candidate was not offered to the judge
    # again five minutes later.
    assert len(judge.asked) == 1


async def test_a_decline_still_stops_the_tick(store: Store, data_dir: Path) -> None:
    """Finding 1's other half: the loop used to `break` only after a *delivery*,
    so several due candidates in one tick each cost a judge call before a decline
    stopped it - under the `quality` preset PROACTIVE_JUDGE is hosted, so that is
    a paid call per candidate, not a free one."""
    add_candidate(store, "silence")
    add_candidate(store, "emotional")
    tick, judge, channel = tick_for(store, data_dir, judge=FakeJudge(""))

    result = await tick.run(now=NOW)

    assert len(judge.asked) == 1
    assert (result.declined, result.spoke) == (1, 0)
    assert channel.sent == []


async def test_a_decline_is_counted_separately_from_a_block(
    store: Store, data_dir: Path
) -> None:
    """Two different things: the gate said no, or the gate said yes and there was
    nothing worth saying. Collapsing them hides which brake is actually biting."""
    add_candidate(store)
    tick, _, _ = tick_for(store, data_dir, judge=FakeJudge(""))

    result = await tick.run(now=NOW)

    assert result.declined == 1
    assert result.blocked_by == {}


# --- reporting ---------------------------------------------------------------


async def test_blocked_by_names_the_rule_that_won(store: Store, data_dir: Path) -> None:
    """A loop that has stayed silent for a week is indistinguishable from a broken
    one unless it can say which rule kept winning."""
    add_candidate(store)
    tick, _, _ = tick_for(store, data_dir, proactive_quiet_hours="00:00-23:00")

    result = await tick.run(now=NOW)

    assert result.blocked_by == {"quiet hours": 1}


async def test_expiry_runs_before_generation(store: Store, data_dir: Path) -> None:
    """So a generator's dedup check sees the table in its settled state."""
    store.insert_candidate(
        kind="silence",
        reason="오래된 것",
        payload="{}",
        now=NOW - timedelta(days=2),
        expires_at=NOW - timedelta(days=1),
    )
    tick, _, _ = tick_for(store, data_dir, speaking=False)

    result = await tick.run(now=NOW)

    assert result.expired == 1


async def test_a_malformed_payload_does_not_take_the_tick_down(
    store: Store, data_dir: Path
) -> None:
    """The column's CHECK proves the payload is valid JSON, which `null` also is. A
    tick that died on one row would stop considering every other candidate."""
    store.insert_candidate(kind="silence", reason="조용하다", payload="null", now=NOW)
    tick, _, _ = tick_for(store, data_dir, speaking=False)

    result = await tick.run(now=NOW)

    assert result.considered[0].candidate.payload == {}


# --- what the tick writes ----------------------------------------------------


async def test_a_spoken_tick_logs_the_utterance_as_proactive(
    store: Store, data_dir: Path
) -> None:
    """Hygiene rule 1 again, at the level that matters: the tick that spoke must not
    have made itself look like a conversation."""
    add_candidate(store)
    tick, _, _ = tick_for(store, data_dir)

    await tick.run(now=NOW)

    row = store.conn.execute("SELECT session_kind FROM messages").fetchone()
    assert row["session_kind"] == "proactive"
    assert store.last_conversation_at() is None


async def test_the_utterance_carries_its_candidate_and_kind(
    store: Store, data_dir: Path
) -> None:
    cid = add_candidate(store, "open_loop", "발표 시각이 지났다")
    tick, _, _ = tick_for(store, data_dir)

    await tick.run(now=NOW)

    row = store.utterances_since(since=NOW)[0]
    assert (row["candidate_id"], row["kind"]) == (cid, "open_loop")


async def test_the_recorded_route_is_telegram_when_no_speaker_is_wired(
    store: Store, data_dir: Path
) -> None:
    add_candidate(store)
    tick, _, _ = tick_for(store, data_dir, voice_enabled=True)

    await tick.run(now=NOW)

    assert store.utterances_since(since=NOW)[0]["route"] == "telegram"


def test_a_logged_message_is_what_the_writer_stores(data_dir: Path) -> None:
    """Guard on the fixture rather than the product: if `LoggedMessage` grows a
    required field, the tests above should fail loudly here instead of drifting."""
    assert LoggedMessage(
        ts=NOW,
        role="assistant",
        content="x",
        origin="agent",
        session_kind="proactive",
        modality="text",
        channel="telegram",
    ).session_kind == "proactive"


async def test_the_one_per_tick_stop_is_load_bearing_at_zero_cooldown(
    store: Store, data_dir: Path
) -> None:
    """Found by mutation-checking: with the default 90-minute cooldown, replacing
    the tick's `break` with `continue` changes nothing, because the gate blocks the
    second candidate anyway. The stop only becomes the sole defence at
    `proactive_cooldown_minutes=0`, which is a legal setting - and there the daily
    budget is counted from stored rows, so every candidate in the tick would read
    the same pre-tick count and the budget of three would be spent at once.
    """
    for kind in ("silence", "emotional", "pattern_time"):
        add_candidate(store, kind)
    tick, _, channel = tick_for(
        store,
        data_dir,
        proactive_cooldown_minutes=0,
        judge=FakeJudge("첫 마디", "둘째 마디", "셋째 마디"),
    )

    result = await tick.run(now=NOW)

    assert result.spoke == 1
    assert len(channel.sent) == 1
