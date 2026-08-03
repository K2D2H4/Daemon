"""The voice conversation loop. No network, no key, no microphone, no speaker.

The session and the hardware are both scripted fakes, because the two seams fail
for different reasons (daemon/voice/base.py) and because a test that needs an API
key is a broken test.

Timing is not slept on. The fake session advances only once the prefetch watcher
has actually peeked at the in-progress transcript, so "recall started while the
user was still talking" is asserted from call order rather than from a sleep long
enough to usually work.
"""

from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from daemon.memory.base import LoggedMessage, RecalledItem
from daemon.voice import conversation as conversation_module
from daemon.voice.base import AudioIO, Transcript, VoiceSession
from daemon.voice.conversation import VoiceConversation

TICK = 0.005
"""Prefetch interval for the tests. Small because nothing waits on it: the fake
session blocks until a peek happens."""


# --- script steps ------------------------------------------------------------


@dataclass(frozen=True)
class Says:
    """A transcription delta. Accumulates in the session and is not yielded -
    exactly what Gemini does (docs/PLAN.md 6.5)."""

    role: str
    text: str


@dataclass(frozen=True)
class Turn:
    """A turn boundary: the accumulated transcript is released, final."""


@dataclass(frozen=True)
class Hang:
    """The user has stopped and the turn has not ended."""


# --- the fakes ---------------------------------------------------------------


class FakeSession:
    """A scripted `VoiceSession`, plus the two accessors the real one grew.

    `partial_transcripts` is how recall gets a head start and `pending_transcripts`
    is how a cancelled turn is not lost; neither is in the frozen protocol, so both
    are duck-typed here the same way the conversation looks for them.
    """

    name = "fake-live"

    def __init__(self, *script: Any, events: list[str] | None = None) -> None:
        self.script = list(script)
        self.events = events if events is not None else []
        self.sent: list[bytes] = []
        self.texts: list[str] = []
        self.interrupts = 0
        self.entered = False
        self.closed = False
        self.ended: str | None = None
        self.peeked = ""
        self._said: dict[str, list[str]] = {"user": [], "assistant": []}
        self._peek_happened = asyncio.Event()

    async def __aenter__(self) -> FakeSession:
        self.entered = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True

    async def send_audio(self, chunk: bytes) -> None:
        self.sent.append(chunk)

    async def send_text(self, text: str) -> None:
        self.texts.append(text)

    async def interrupt(self) -> None:
        self.interrupts += 1
        self.events.append("interrupt")

    async def receive(self) -> Any:
        for step in self.script:
            if isinstance(step, bytes):
                yield step
            elif isinstance(step, Transcript):
                yield step
            elif isinstance(step, Says):
                self._said[step.role].append(step.text)
                await self._observed()
            elif isinstance(step, Turn):
                for transcript in self._drain(final=True):
                    yield transcript
            elif isinstance(step, Hang):
                await asyncio.Event().wait()
            elif isinstance(step, BaseException):
                raise step
        self.ended = "the script ran out"

    def partial_transcripts(self) -> list[Transcript]:
        self.peeked = self._state()
        self._peek_happened.set()
        return self._snapshot(final=False)

    def pending_transcripts(self) -> list[Transcript]:
        return self._drain(final=True)

    async def _observed(self) -> None:
        """Block until the watcher has peeked and seen *this* text.

        Waiting for any peek would race: an earlier, empty one would satisfy it and
        the script would run on before the conversation had noticed anything.
        """
        while self.peeked != self._state():
            self._peek_happened.clear()
            await self._peek_happened.wait()
        # One more turn of the loop so the search the watcher just started can run.
        await asyncio.sleep(0)

    def _state(self) -> str:
        return "|".join("".join(self._said[role]).strip() for role in ("user", "assistant"))

    def _snapshot(self, *, final: bool) -> list[Transcript]:
        found = []
        for role in ("user", "assistant"):
            text = "".join(self._said[role]).strip()
            if text:
                found.append(Transcript(text=text, role=role, final=final))
        return found

    def _drain(self, *, final: bool) -> list[Transcript]:
        released = self._snapshot(final=final)
        self._said = {"user": [], "assistant": []}
        return released


class BareSession(FakeSession):
    """A provider that offers no view of the turn in progress - OpenAI Realtime
    reports partial and final separately, so this is not hypothetical."""

    partial_transcripts = None  # type: ignore[assignment]
    pending_transcripts = None  # type: ignore[assignment]

    async def _observed(self) -> None:
        await asyncio.sleep(0)


class FakeAudio:
    """The hardware seam. Records what happened rather than making a sound."""

    sample_rate = 16_000
    playback_sample_rate = 24_000

    def __init__(self, *blocks: bytes, events: list[str] | None = None) -> None:
        self.blocks = list(blocks)
        self.events = events if events is not None else []
        self.played: list[bytes] = []
        self.stops = 0
        self.closed = False
        self.recording = False

    async def record(self) -> Any:
        self.recording = True
        try:
            for block in self.blocks:
                yield block
            # A real microphone stays open until somebody closes it.
            await asyncio.Event().wait()
        finally:
            self.recording = False

    async def play(self, chunk: bytes) -> None:
        self.played.append(chunk)
        self.events.append("play")

    async def stop_playback(self) -> None:
        self.stops += 1
        self.events.append("stop_playback")

    async def close(self) -> None:
        self.closed = True


class FakeMemory:
    def __init__(self, events: list[str] | None = None) -> None:
        self.records: list[LoggedMessage] = []
        self.events = events if events is not None else []

    async def record(self, message: LoggedMessage) -> None:
        self.records.append(message)
        self.events.append(f"record:{message.role}")

    async def seen(self, channel: str, external_id: str) -> bool:
        return False

    async def recent(self, limit: int = 20) -> list[LoggedMessage]:
        return list(self.records[-limit:])


def _item(content: str = "치과 예약은 8월 5일 오후 3시") -> RecalledItem:
    return RecalledItem(
        content=content,
        ts=datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
        role="user",
        score=0.9,
        reason="both",
    )


class FakeRecall:
    def __init__(
        self,
        *items: RecalledItem,
        events: list[str] | None = None,
        delay: float = 0.0,
        fail: bool = False,
    ) -> None:
        self.items = list(items)
        self.events = events if events is not None else []
        self.delay = delay
        self.fail = fail
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int = 8) -> list[RecalledItem]:
        self.queries.append(query)
        self.events.append(f"search:{query}")
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("the embedder is down")
        return list(self.items)

    async def index(self, message_id: int, text: str) -> None: ...

    async def backfill(self, limit: int = 500) -> int:
        return 0


def conversation(
    session: FakeSession,
    audio: FakeAudio | None = None,
    memory: FakeMemory | None = None,
    **kwargs: Any,
) -> VoiceConversation:
    return VoiceConversation(
        session,
        audio or FakeAudio(),
        memory or FakeMemory(),
        prefetch_interval=kwargs.pop("prefetch_interval", TICK),
        **kwargs,
    )


RUN_LIMIT = 5.0
"""Wall-clock ceiling for a scripted conversation. Nothing here should approach
it; it is there so a hang fails the test instead of the suite."""


async def run(conv: VoiceConversation) -> None:
    async with asyncio.timeout(RUN_LIMIT):
        await conv.run()


# --- the fakes are the real protocols ----------------------------------------


def test_the_fakes_satisfy_the_protocols() -> None:
    """Otherwise this whole file could pass against something the product cannot
    talk to - the defect class tests/test_reachable.py exists for."""
    assert isinstance(FakeSession(), VoiceSession)
    assert isinstance(FakeAudio(), AudioIO)


def test_the_conversation_knows_nothing_concrete() -> None:
    """It must work against either provider and against no hardware at all, so a
    concrete import here would be the layering rule (CONTRACTS.md 4) breaking."""
    source = pathlib.Path(conversation_module.__file__).read_text(encoding="utf-8")
    for concrete in (
        "gemini_live",
        "GeminiLiveSession",
        "SoundDeviceAudio",
        "sounddevice",
        "websockets",
    ):
        assert concrete not in source, f"conversation.py reaches for {concrete}"


# --- recording ---------------------------------------------------------------


async def test_a_voice_turn_is_recorded_as_voice_in_every_column() -> None:
    """Provenance is columns, not prose (CONTRACTS.md 3). If voice turns land as
    `interactive`/`text` then reflection cannot tell what was spoken from what was
    typed, and nothing ever fails to say so."""
    session = FakeSession(
        Says("user", "오늘 저녁에 김치찌개 먹었어"),
        b"\x01\x02",
        Says("assistant", "맛있었어?"),
        Turn(),
    )
    memory = FakeMemory()
    await run(conversation(session, memory=memory))

    user, assistant = memory.records
    assert user.content == "오늘 저녁에 김치찌개 먹었어"
    assert user.role == "user" and user.origin == "owner"
    assert assistant.content == "맛있었어?"
    assert assistant.role == "assistant" and assistant.origin == "agent"
    for record in memory.records:
        assert record.session_kind == "voice"
        assert record.modality == "voice"
        assert record.channel == "voice"
        assert record.ts.tzinfo is not None


async def test_a_partial_transcript_is_never_recorded() -> None:
    """A delta recorded as an utterance leaves a syllable in the log for ever
    (daemon/voice/base.py). The flag is the only thing standing between the two."""
    session = FakeSession(
        Transcript(text="김치", role="assistant", final=False),
        Transcript(text="김치찌개 어때", role="assistant", final=True),
    )
    memory = FakeMemory()
    await run(conversation(session, memory=memory))

    assert [record.content for record in memory.records] == ["김치찌개 어때"]


async def test_an_empty_transcript_is_not_recorded() -> None:
    session = FakeSession(Transcript(text="   ", role="user", final=True))
    memory = FakeMemory()
    await run(conversation(session, memory=memory))

    assert memory.records == []


async def test_korean_survives_the_whole_round_trip() -> None:
    korean = "내일 아침에 우산 챙겨. 비 온다더라 ☔"
    session = FakeSession(Says("assistant", korean), Turn())
    memory = FakeMemory()
    audio = FakeAudio(b"\xaa\xbb")
    conv = conversation(session, audio, memory)
    await run(conv)

    assert [record.content for record in memory.records] == [korean]
    assert session.sent == [b"\xaa\xbb"], "the microphone did not reach the session"


# --- the audio path ----------------------------------------------------------


async def test_model_audio_reaches_the_speaker_and_the_microphone_the_session() -> None:
    # The `Says` step is what lets the microphone task get a turn at all: the
    # receive loop is otherwise pure synchronous handoff.
    session = FakeSession(b"\x01", b"\x02", Says("assistant", "응"), Turn())
    audio = FakeAudio(b"mic-1", b"mic-2")
    await run(conversation(session, audio))

    assert audio.played == [b"\x01", b"\x02"]
    assert session.sent == [b"mic-1", b"mic-2"]


async def test_the_microphone_is_closed_when_the_conversation_ends() -> None:
    """An input stream nobody reads is a recording light left on."""
    session = FakeSession(Turn())
    audio = FakeAudio(b"mic")
    await run(conversation(session, audio))

    assert not audio.recording


async def test_the_session_is_closed_when_the_conversation_ends() -> None:
    """Live sessions bill per minute, so an idle open connection is pure cost
    (docs/PLAN.md 6.5)."""
    session = FakeSession(Turn())
    await run(conversation(session))

    assert session.entered and session.closed


async def test_a_session_that_hears_nothing_is_closed_rather_than_held() -> None:
    session = FakeSession(Hang())
    conv = conversation(session, idle_timeout=0.05)
    await run(conv)

    assert session.closed
    assert conv.ended is not None and "nothing heard" in conv.ended


async def test_why_the_session_ended_is_carried_out_of_it() -> None:
    """A `goAway` reads exactly like a finished turn unless somebody says
    otherwise."""
    session = FakeSession(Turn())
    conv = conversation(session)
    await run(conv)

    assert conv.ended == "the script ran out"


async def test_a_session_failure_reaches_the_caller() -> None:
    """A voice turn that cannot run has to surface so the caller can fall back to
    text, rather than looking like a conversation that simply ended."""
    session = FakeSession(RuntimeError("the socket died"))
    with pytest.raises(RuntimeError, match="socket died"):
        await run(conversation(session))

    assert session.closed


# --- barge-in ----------------------------------------------------------------


async def test_a_barge_in_stops_the_stream_and_the_speaker_both() -> None:
    """Either one alone leaves the daemon talking: the session refusing to hand
    over more audio does not empty the buffer, and emptying the buffer does not
    stop the stream (daemon/voice/base.py)."""
    events: list[str] = []
    session = FakeSession(
        b"\x01",  # the daemon is talking
        Says("user", "아니 잠깐만"),  # and the user starts talking over it
        Turn(),
        events=events,
    )
    audio = FakeAudio(events=events)
    conv = conversation(session, audio, FakeMemory(events=events))
    await run(conv)

    assert conv.interruptions == 1
    assert session.interrupts == 1, "the abandoned turn kept being handed over"
    assert audio.stops == 1, "the speaker kept talking out of its buffer"
    assert events.index("interrupt") < events.index("record:user")


async def test_the_user_speaking_first_is_not_a_barge_in() -> None:
    """Nothing is playing, so there is nothing to interrupt - and interrupting a
    silence is what dropped a whole answer while still recording it."""
    session = FakeSession(Says("user", "오늘 일정 뭐야"), Turn(), b"\x01")
    audio = FakeAudio()
    conv = conversation(session, audio)
    await run(conv)

    assert conv.interruptions == 0
    assert session.interrupts == 0
    assert audio.stops == 0
    assert audio.played == [b"\x01"]


# --- recall ------------------------------------------------------------------


async def test_recall_starts_before_the_user_has_finished_speaking() -> None:
    """The requirement docs/PLAN.md 4.3.1 pins: the embedder round trip is 117 ms
    at p50 (105 ms of it fixed overhead), which is free while the user is still
    talking and unaffordable afterwards. So the search has to be under way before
    the utterance ends."""
    events: list[str] = []
    session = FakeSession(
        Says("user", "어제 얘기한"),
        Turn(),
        b"\x01",
        events=events,
    )
    recall = FakeRecall(_item(), events=events)
    conv = conversation(session, FakeAudio(events=events), FakeMemory(events=events), recall=recall)
    await run(conv)

    assert recall.queries[0] == "어제 얘기한", "recall waited for the final transcript"
    assert events.index("search:어제 얘기한") < events.index("record:user")
    assert events.index("search:어제 얘기한") < events.index("play"), (
        "the search started after the model had already answered, which is the "
        "latency this design exists to avoid"
    )


async def test_a_prefetch_that_covers_the_utterance_is_reused() -> None:
    """The whole point: when the finished utterance is what the prefetch already
    asked about, the turn pays nothing at all."""
    session = FakeSession(
        Says("user", "치과 예약 언제였"),
        Says("user", "지"),
        Turn(),
    )
    # Slow enough that the prefetch is still in flight when the user stops, which
    # is the case that matters - the result is awaited, not re-requested.
    recall = FakeRecall(_item(), delay=0.05)
    conv = conversation(session, recall=recall)
    await run(conv)

    assert recall.queries == ["치과 예약 언제였"], "the covered prefetch was thrown away"
    assert conv.recalled == [_item()]


async def test_a_prefetch_that_missed_the_point_is_thrown_away() -> None:
    """Korean puts the question at the end, so a prefetch from the opening words is
    a different query. Correctness wins over the 117 ms."""
    session = FakeSession(
        Says("user", "아 맞다"),
        Says("user", " 그때 말한 치과 예약 언제였는지 알려줄 수 있어?"),
        Turn(),
    )
    recall = FakeRecall(_item(), delay=0.05)
    conv = conversation(session, recall=recall)
    await run(conv)

    assert recall.queries == [
        "아 맞다",
        "아 맞다 그때 말한 치과 예약 언제였는지 알려줄 수 있어?",
    ]
    assert conv.recalled == [_item()]


async def test_a_provider_with_no_partial_transcripts_still_recalls() -> None:
    """Degrade, do not fail: recall then costs the turn a round trip, which is the
    behaviour before the prefetch existed."""
    session = BareSession(Says("user", "치과 예약 언제였지"), Turn())
    recall = FakeRecall(_item())
    conv = conversation(session, recall=recall)
    await run(conv)

    assert recall.queries == ["치과 예약 언제였지"]
    assert conv.recalled == [_item()]


async def test_a_syllable_is_not_worth_an_embedder_call() -> None:
    session = FakeSession(Says("user", "어"), Turn())
    recall = FakeRecall(_item())
    conv = conversation(session, recall=recall)
    await run(conv)

    assert recall.queries == ["어"], "the prefetch fired on one syllable"


async def test_recall_failing_does_not_cost_the_turn() -> None:
    """Lane 1 degrades rather than fails (daemon/memory/base.py). In voice mode an
    exception is silence, which is the worst possible answer."""
    session = FakeSession(Says("user", "치과 예약 언제였지"), Turn(), b"\x01")
    memory = FakeMemory()
    audio = FakeAudio()
    conv = conversation(session, audio, memory, recall=FakeRecall(delay=0.02, fail=True))
    await run(conv)

    assert [record.content for record in memory.records] == ["치과 예약 언제였지"]
    assert audio.played == [b"\x01"]
    assert conv.recalled == []


async def test_the_assistants_own_words_do_not_trigger_a_search() -> None:
    session = FakeSession(Says("assistant", "여덟 시에 만나자"), Turn())
    recall = FakeRecall(_item())
    conv = conversation(session, recall=recall)
    await run(conv)

    assert recall.queries == []


# --- cancellation ------------------------------------------------------------


async def test_a_cancelled_conversation_still_records_what_was_said() -> None:
    """The audit finding. The transcript is the only record voice mode produces and
    it is accumulated until the turn boundary, so a shutdown arriving first left the
    utterance in neither the markdown nor the mirror."""
    session = FakeSession(Says("user", "치과 예약 언제였지"), Hang())
    memory = FakeMemory()
    conv = conversation(session, memory=memory)

    task = asyncio.create_task(conv.run())
    for _ in range(200):
        await asyncio.sleep(0.005)
        if session.peeked.startswith("치과"):
            break
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert [record.content for record in memory.records] == ["치과 예약 언제였지"], (
        "the utterance was lost between the microphone and the log"
    )
    assert memory.records[0].session_kind == "voice"
    assert session.closed


async def test_a_cancelled_conversation_closes_the_microphone() -> None:
    session = FakeSession(Hang())
    audio = FakeAudio(b"mic")
    conv = conversation(session, audio)

    task = asyncio.create_task(conv.run())
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert not audio.recording
    assert session.closed


async def test_a_provider_with_nothing_pending_is_not_a_problem() -> None:
    session = BareSession(Hang())
    memory = FakeMemory()
    conv = conversation(session, memory=memory)

    task = asyncio.create_task(conv.run())
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert memory.records == []
