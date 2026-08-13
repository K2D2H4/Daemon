"""Delivering a proactive utterance, and what it records.

The interesting cases are the failures. A half-delivered utterance decides whether
the day's budget was spent, whether the user can label it, and whether the daemon's
own voice becomes evidence for the next time it decides to speak.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from daemon.channels.base import OutboundMessage
from daemon.memory.base import LoggedMessage
from daemon.memory.store import Store
from daemon.memory.writer import FileMemoryWriter
from daemon.proactivity.base import Candidate, Reading, Utterance, Verdict
from daemon.proactivity.delivery import ProactiveDelivery

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)
SAID = Utterance(text="발표 어떻게 됐어?")


class FakeChannel:
    name = "fake"

    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[OutboundMessage] = []
        self.fail = fail

    async def send(self, message: OutboundMessage) -> None:
        if self.fail:
            raise RuntimeError("telegram is unreachable")
        self.sent.append(message)


class FakeSpeaker:
    def __init__(self, *, works: bool = True) -> None:
        self.said: list[str] = []
        self.works = works

    async def say(self, text: str) -> bool:
        self.said.append(text)
        return self.works

    async def aclose(self) -> None:
        return None


def reading() -> Reading:
    return Reading(
        at=NOW, idle_seconds=5.0, foreground_app="Warp", mic_busy=False, output_busy=False
    )


def verdict(delivery: str = "telegram") -> Verdict:
    return Verdict(allowed=True, why="ok", reading=reading(), delivery=delivery)  # type: ignore[arg-type]


@pytest.fixture
def store(db: Any) -> Store:
    return Store(db)


def candidate(store: Store) -> Candidate:
    cid = store.insert_candidate(
        kind="open_loop", reason="발표 시각이 지났다", payload="{}", now=NOW
    )
    return Candidate(kind="open_loop", reason="발표 시각이 지났다", id=cid)


def delivery_for(
    store: Store, data_dir: Path, **kwargs: Any
) -> tuple[ProactiveDelivery, FileMemoryWriter]:
    memory = FileMemoryWriter(data_dir, store)
    return ProactiveDelivery(store, memory, **kwargs), memory


# --- the happy paths --------------------------------------------------------


async def test_a_telegram_utterance_is_recorded_and_labelable(
    store: Store, data_dir: Path
) -> None:
    channel = FakeChannel()
    delivery, _ = delivery_for(store, data_dir, channel=channel)

    result = await delivery.deliver(candidate(store), SAID, verdict(), now=NOW)

    assert result.route == "telegram"
    sent = channel.sent[0]
    assert sent.text == "발표 어떻게 됐어?"
    # Without both of these the label clock never starts (docs/PLAN.md 8.1).
    assert sent.labelable is True
    assert sent.utterance_id == result.utterance_id
    # Unsolicited: it answers no request, so the channel picks its owner.
    assert sent.recipient_id is None


async def test_both_routes_speak_and_send_the_same_words(
    store: Store, data_dir: Path
) -> None:
    """docs/PLAN.md 6.3: answering a speaker on a keyboard is awkward without a
    thread to answer in, and a line nobody heard is lost otherwise."""
    channel, speaker = FakeChannel(), FakeSpeaker()
    delivery, _ = delivery_for(store, data_dir, channel=channel, speaker=speaker)

    result = await delivery.deliver(candidate(store), SAID, verdict("both"), now=NOW)

    assert result.route == "both"
    assert speaker.said == ["발표 어떻게 됐어?"]
    assert [m.text for m in channel.sent] == ["발표 어떻게 됐어?"]


async def test_the_gate_snapshot_is_stored_with_the_utterance(
    store: Store, data_dir: Path
) -> None:
    """The column exists so a bad call can be diagnosed later instead of guessed
    at, which means the reading has to survive the trip."""
    import json

    delivery, _ = delivery_for(store, data_dir, channel=FakeChannel())

    await delivery.deliver(candidate(store), SAID, verdict(), now=NOW)

    row = store.utterances_since(since=NOW)[0]
    snapshot = json.loads(row["gate_snapshot"])
    assert snapshot["why"] == "ok"
    assert snapshot["foreground_app"] == "Warp"
    assert snapshot["idle_seconds"] == 5.0


async def test_the_candidate_is_marked_fired(store: Store, data_dir: Path) -> None:
    delivery, _ = delivery_for(store, data_dir, channel=FakeChannel())
    target = candidate(store)

    await delivery.deliver(target, SAID, verdict(), now=NOW)

    assert store.due_candidates(now=NOW) == []


# --- the hygiene guarantee --------------------------------------------------


async def test_the_utterance_is_logged_so_the_next_turn_has_context(
    store: Store, data_dir: Path
) -> None:
    """"잘 됐어" answering a question that is not in the history reads as a non
    sequitur."""
    delivery, memory = delivery_for(store, data_dir, channel=FakeChannel())

    await delivery.deliver(candidate(store), SAID, verdict(), now=NOW)

    recent = await memory.recent()
    assert [m.content for m in recent] == ["발표 어떻게 됐어?"]


async def test_the_daemons_own_utterance_cannot_become_its_own_evidence(
    store: Store, data_dir: Path
) -> None:
    """Hygiene rule 1 (docs/PLAN.md 4.2) filters on `session_kind`, so this is the
    field that stops speaking from being its own excuse to speak: it must not reset
    the silence clock, and reflection must not read it back as something the user
    said.
    """
    delivery, _ = delivery_for(store, data_dir, channel=FakeChannel())

    await delivery.deliver(candidate(store), SAID, verdict(), now=NOW)

    row = store.utterances_since(since=NOW)[0]
    assert row["kind"] == "open_loop"
    # The three reads the generators use, and the one reflection uses.
    assert store.last_conversation_at() is None
    assert store.conversation_between(NOW, NOW) == []
    assert store.conversation_times(NOW) == []
    assert store.messages_for_day("2026-08-04") == []


async def test_a_spoken_utterance_is_logged_as_voice(store: Store, data_dir: Path) -> None:
    """The column records that a paralinguistic channel was involved (PLAN 4.2),
    so `both` counts as voice - it used the speaker too."""
    delivery, _ = delivery_for(
        store, data_dir, channel=FakeChannel(), speaker=FakeSpeaker()
    )

    await delivery.deliver(candidate(store), SAID, verdict("both"), now=NOW)

    row = store.conn.execute("SELECT modality FROM messages").fetchone()
    assert row["modality"] == "voice"


# --- half-delivered --------------------------------------------------------


async def test_a_failed_speaker_keeps_the_telegram_copy(store: Store, data_dir: Path) -> None:
    """`base.Speaker` returns False rather than raising for exactly this: a silent
    speaker must not cost the message that went with it."""
    channel, speaker = FakeChannel(), FakeSpeaker(works=False)
    delivery, _ = delivery_for(store, data_dir, channel=channel, speaker=speaker)

    result = await delivery.deliver(candidate(store), SAID, verdict("both"), now=NOW)

    assert result.route == "telegram"
    assert len(channel.sent) == 1
    # The stored route says what happened, not what was planned.
    assert store.utterances_since(since=NOW)[0]["route"] == "telegram"


async def test_a_failed_channel_keeps_the_spoken_copy(store: Store, data_dir: Path) -> None:
    channel, speaker = FakeChannel(fail=True), FakeSpeaker()
    delivery, _ = delivery_for(store, data_dir, channel=channel, speaker=speaker)

    result = await delivery.deliver(candidate(store), SAID, verdict("both"), now=NOW)

    assert result.route == "local_speaker"
    assert speaker.said == ["발표 어떻게 됐어?"]
    assert store.utterances_since(since=NOW)[0]["route"] == "local_speaker"


async def test_nothing_delivered_leaves_no_record_and_no_spent_budget(
    store: Store, data_dir: Path
) -> None:
    """An utterance that reached nobody was not said. Keeping the row would spend
    the day's budget on silence and put an unlabelable message into the precision
    numbers M3's own gate is judged on.
    """
    delivery, memory = delivery_for(store, data_dir, channel=FakeChannel(fail=True))
    target = candidate(store)

    result = await delivery.deliver(target, SAID, verdict(), now=NOW)

    assert result.route is None
    assert not result
    assert store.utterances_since(since=NOW) == []
    assert await memory.recent() == []
    # And still live, so the next tick tries again.
    assert len(store.due_candidates(now=NOW)) == 1


async def test_a_speaker_route_with_no_speaker_does_not_crash(
    store: Store, data_dir: Path
) -> None:
    """A mismatch between the gate's route and what was assembled must not lose the
    other half - or, here, take the process down."""
    delivery, _ = delivery_for(store, data_dir, channel=FakeChannel())

    result = await delivery.deliver(candidate(store), SAID, verdict("both"), now=NOW)

    assert result.route == "telegram"


async def test_a_log_failure_does_not_lose_the_utterance(
    store: Store, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The words already reached the user. Losing the log costs context on the next
    turn; raising here would also lose the label row."""

    async def boom(message: LoggedMessage) -> None:
        raise OSError("disk full")

    delivery, memory = delivery_for(store, data_dir, channel=FakeChannel())
    monkeypatch.setattr(memory, "record", boom)

    result = await delivery.deliver(candidate(store), SAID, verdict(), now=NOW)

    assert result.route == "telegram"
    assert store.utterances_since(since=NOW)[0]["text"] == "발표 어떻게 됐어?"


# --- guards ----------------------------------------------------------------


async def test_delivering_a_decline_is_a_programming_error(
    store: Store, data_dir: Path
) -> None:
    """A judge that declined has nothing to deliver, and silently sending an empty
    message would be worse than the exception."""
    delivery, _ = delivery_for(store, data_dir, channel=FakeChannel())

    with pytest.raises(ValueError):
        await delivery.deliver(candidate(store), Utterance(why_not="할 말 없음"), verdict())


async def test_the_row_exists_before_the_message_leaves(store: Store, data_dir: Path) -> None:
    """The id is on the button, so a fast tap has to resolve. If the row were
    written after the send, the user would be told their label was stale."""
    seen: list[bool] = []

    class WatchingChannel(FakeChannel):
        async def send(self, message: OutboundMessage) -> None:
            assert message.utterance_id is not None
            seen.append(
                store.label_utterance(message.utterance_id, "good", now=NOW)
            )
            await super().send(message)

    delivery, _ = delivery_for(store, data_dir, channel=WatchingChannel())
    await delivery.deliver(candidate(store), SAID, verdict(), now=NOW)

    assert seen == [True]
