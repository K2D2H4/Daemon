"""The voice conversation loop. No network, no key, no microphone, no speaker.

The session and the hardware are both scripted fakes, because the two seams fail
for different reasons (daemon/voice/base.py) and because a test that needs an API
key is a broken test.

Timing is not slept on. The fake session advances only once the prefetch watcher
has actually handled the in-progress transcript, so "recall started while the
user was still talking" is asserted from call order rather than from a sleep long
enough to usually work.
"""

from __future__ import annotations

import asyncio
import base64
import json
import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from test_loop import Ids
from websockets.exceptions import ConnectionClosedOK
from websockets.frames import Close

from daemon.companion import Companion
from daemon.llm.base import ToolCall
from daemon.memory.base import LoggedMessage, RecalledItem
from daemon.memory.store import Store
from daemon.tools.base import Registry, ToolResult
from daemon.tools.builtin import builtin_tools
from daemon.tools.policy import ToolPolicy
from daemon.tools.runner import ToolRunner
from daemon.voice import conversation as conversation_module
from daemon.voice.base import AudioIO, Interrupted, Transcript, VoiceSession
from daemon.voice.conversation import VoiceConversation
from daemon.voice.gemini_live import GeminiLiveSession

POLL = 0.005
"""How often a test that has to wait for another task looks again. Only the
cancellation tests need it: everything else is ordered by the fake session, which
does not advance until the conversation has reacted."""


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
class Cuts:
    """The provider's activity detection says the user talked over the answer.

    A script step rather than something the fake infers, because inferring it is
    the bug: `serverContent.interrupted` is the only thing that knows, and the
    conversation used to guess from transcript growth instead - which fired on
    every turn, since this provider delivers the user's transcript in the same
    event as the answer's first audio (daemon/voice/base.py)."""


class Calls:
    """The model asks for one or more tools, mid-turn.

    A script step rather than something inferred, because a blocking tool call is
    exactly that: the `toolCall` arrives before any audio, the model waits, and the
    conversation has to answer it from inside `receive()` before anything else
    comes back (daemon/voice/gemini_live.py). Yielded as the neutral `ToolCall` the
    tool layer runs - the same dataclass the text path uses, so the same runner and
    the same policy serve both."""

    def __init__(self, *calls: ToolCall) -> None:
        self.calls = calls


@dataclass(frozen=True)
class Does:
    """Run something at this exact point in the script, then let the loop breathe.

    Exists because the thing under test is *timing*: recall has to land while the
    model is mid-answer, and a search that resolves on its own does so before the
    first audio chunk - which is how three tests here passed against the very bug
    they were written for."""

    action: Any


@dataclass(frozen=True)
class Hang:
    """The user has stopped and the turn has not ended."""


# --- the fakes ---------------------------------------------------------------


class FakeSession:
    """A scripted `VoiceSession`, protocol-complete.

    All eight methods are the protocol's now, including the three the audit added,
    `send_tool_response`, and `send_frame` (ADR 0009), so the conversation calls
    them rather than hunting for them - and a fake that lacked one would fail
    `test_the_fakes_satisfy_the_protocols` instead of quietly exercising a fallback
    the product does not have.

    One `receive()` is one turn: the script is consumed across calls and a `Turn`
    step ends the iterator, which is what the real session does at
    `turnComplete` (daemon/voice/base.py).
    """

    name = "fake-live"

    def __init__(self, *script: Any, events: list[str] | None = None) -> None:
        self.script = list(script)
        self.events = events if events is not None else []
        self.sent: list[bytes] = []
        self.frames: list[bytes] = []
        self.texts: list[str] = []
        self.contexts: list[str] = []
        self.sent_while_generating: list[str] = []
        self.tool_responses: list[list[ToolResult]] = []
        self.generating = False
        self.interrupts = 0
        self.entered = False
        self.closed = False
        self.ended: str | None = None
        self.turns = 0
        self.peeked = ""
        self._said: dict[str, list[str]] = {"user": [], "assistant": []}
        self._partials: asyncio.Queue[Transcript | None] = asyncio.Queue()
        self._peek_happened = asyncio.Event()

    async def __aenter__(self) -> FakeSession:
        self.entered = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True
        self._partials.put_nowait(None)

    async def send_audio(self, chunk: bytes) -> None:
        self.sent.append(chunk)

    async def send_frame(self, jpeg: bytes) -> None:
        self.frames.append(jpeg)

    async def send_text(self, text: str) -> None:
        self.texts.append(text)

    async def send_context(self, text: str) -> None:
        self.contexts.append(text)
        self.events.append("context")
        # Recorded, because "did this interrupt the answer" is a question about
        # *when* it was sent. `clientContent` mid-generation is what the Live API
        # documents as interrupting, so a fake that only kept the text could not
        # tell the fixed behaviour from the bug.
        if self.generating:
            self.sent_while_generating.append(text)

    async def send_tool_response(self, results: Sequence[ToolResult]) -> None:
        self.tool_responses.append(list(results))
        # The answer is what starts the model's post-tool generation (measured:
        # a blocking call produces nothing before the response and 13.69s of audio
        # after it - gemini_live.py's tool notes). From here until the turn
        # boundary a `send_context` is the documented interrupt, and the live
        # symptom of sending one was the server cancelling the tool call - so the
        # fake counts it the same way it counts one sent over audio.
        self.generating = True
        self.events.append("tool_response")

    async def interrupt(self) -> None:
        self.interrupts += 1
        self.events.append("interrupt")

    async def receive(self) -> Any:
        self.turns += 1
        while self.script:
            step = self.script.pop(0)
            if isinstance(step, bytes):
                self.generating = True
                yield step
            elif isinstance(step, Transcript):
                yield step
            elif isinstance(step, Says):
                self._said[step.role].append(step.text)
                if step.role == "user":
                    self._partials.put_nowait(
                        Transcript(text=self._text("user"), role="user", final=False)
                    )
                    await self._observed()
                else:
                    # The assistant's deltas are not offered as partials, so there
                    # is nothing to wait for. One turn of the loop, so the
                    # microphone pump gets to run.
                    await asyncio.sleep(0)
            elif isinstance(step, Does):
                step.action()
                # Several turns, not one: the prefetch task has a search to finish
                # and a `send_context` to make after being released.
                for _ in range(20):
                    await asyncio.sleep(0)
            elif isinstance(step, Cuts):
                yield Interrupted()
            elif isinstance(step, Calls):
                # A blocking call: the model waits, so nothing else is scripted
                # after this until the conversation has answered it. The consumer
                # runs each and sends the result back before the next step.
                for call in step.calls:
                    yield call
            elif isinstance(step, Turn):
                self.generating = False
                for transcript in self._drain(final=True):
                    yield transcript
                return  # the turn ended; the session did not
            elif isinstance(step, Hang):
                await asyncio.Event().wait()
            elif isinstance(step, BaseException):
                raise step
        self.ended = "the script ran out"

    async def partial_transcripts(self) -> Any:
        while True:
            partial = await self._partials.get()
            if partial is None:
                return
            yield partial
            # Recorded after the consumer has handled it, which is what `_observed`
            # waits on: peeking is not the point, acting on the peek is.
            self.peeked = partial.text
            self._peek_happened.set()

    def pending_transcripts(self) -> list[Transcript]:
        return self._drain(final=True)

    async def _observed(self) -> None:
        """Block until the watcher has handled *this* text.

        Waiting for any partial would race: an earlier, shorter one would satisfy it
        and the script would run on before the conversation had noticed the rest.
        The assistant's own deltas are not offered as partials at all, so there is
        nothing to wait for - one turn of the loop, so the microphone pump and any
        search already started get to run.
        """
        while self.peeked != self._text("user"):
            self._peek_happened.clear()
            await self._peek_happened.wait()
        # One more turn of the loop so the search the watcher just started can run.
        await asyncio.sleep(0)

    def _text(self, role: str) -> str:
        return "".join(self._said[role]).strip()

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
    """A provider that has the seams and puts nothing through them.

    Not hypothetical: the protocol requires the methods, but nothing can require a
    provider to transcribe mid-utterance, and OpenAI Realtime reports partial and
    final separately. The conversation has to degrade - recall then costs the turn a
    round trip - rather than fail.
    """

    async def partial_transcripts(self) -> Any:
        return
        yield  # pragma: no cover - an empty stream still has to be one

    def pending_transcripts(self) -> list[Transcript]:
        return []

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
        # Explicit rather than the dataclass default, which is deliberately the
        # closed value ("untrusted") - this helper is for ordinary owner recall.
        origin="owner",
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
        self.indexed: list[tuple[int, str]] = []

    async def search(self, query: str, *, limit: int = 8) -> list[RecalledItem]:
        self.queries.append(query)
        self.events.append(f"search:{query}")
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("the embedder is down")
        return list(self.items)

    async def index(self, message_id: int, text: str) -> None:
        self.indexed.append((message_id, text))
        self.events.append(f"index:{text}")

    async def backfill(self, limit: int = 500) -> int:
        return 0


class BlockingRecall:
    """A `Recall` that hangs until released, so a test decides when the search lands.

    The whole point of the deferral is what happens when it lands *during* an answer,
    and no amount of scripting the session can arrange that if the search resolves
    by itself on the next loop turn."""

    def __init__(self, *items: RecalledItem) -> None:
        self.items = list(items)
        self.gate = asyncio.Event()
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int = 6) -> list[RecalledItem]:
        self.queries.append(query)
        await self.gate.wait()
        return list(self.items)

    async def index(self, message_id: int, text: str) -> None: ...

    async def backfill(self, limit: int = 500) -> int:
        return 0


NO_DATA_DIR = pathlib.Path("/nonexistent-voice-test-data-dir")
"""What the companion's `data_dir` is pointed at here.

Voice does not read the persona through the companion - `daemon/app.py` reads it
once and hands it to the session as its system instruction - so nothing in this file
should touch the filesystem for it. Pointing it at nothing keeps that true rather
than assumed, and keeps a developer's own `seed.md` out of these assertions.
"""


def companion_for(
    memory: FakeMemory | None = None,
    *,
    recall: Any = None,
    recall_limit: int = 6,
    resolve_id: Any = None,
    tools: ToolRunner | None = None,
) -> Companion:
    return Companion(
        memory or FakeMemory(),
        data_dir=NO_DATA_DIR,
        recall=recall,
        recall_limit=recall_limit,
        resolve_id=resolve_id,
        tools=tools,
    )


def conversation(
    session: FakeSession,
    audio: FakeAudio | None = None,
    memory: FakeMemory | None = None,
    *,
    recall: Any = None,
    recall_limit: int = 6,
    resolve_id: Any = None,
    tools: ToolRunner | None = None,
    **kwargs: Any,
) -> VoiceConversation:
    return VoiceConversation(
        session,
        audio or FakeAudio(),
        companion_for(
            memory,
            recall=recall,
            recall_limit=recall_limit,
            resolve_id=resolve_id,
            tools=tools,
        ),
        **kwargs,
    )


def tool_runner(
    db: Any, roots: pathlib.Path, *, mode: str = "allowlist", allowlist: Any = ()
) -> tuple[ToolRunner, Store]:
    """The real tool layer over a scratch filesystem, for the voice tool loop.

    The store is handed back too: CONTRACTS rule 12 says every executed call leaves
    an audit row, and a spoken call is no exception - so the tests read it back the
    way the text-loop tests do."""
    store = Store(db)
    registry = Registry()
    for tool in builtin_tools(roots=[roots]):
        registry.register(tool)
    return ToolRunner(registry, ToolPolicy(store, mode=mode, allowlist=allowlist), store), store


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


async def test_both_sides_of_a_voice_turn_are_embedded_during_the_session() -> None:
    """The defect that came of two endpoints doing the same work by hand.

    Voice wrote the markdown and the mirror and nobody ever called `recall.index`,
    so a spoken turn had no vector until the next restart's backfill reached it -
    and for Korean the vector lane is not a refinement: keyword-only recall is 50%
    where the hybrid is 93% (daemon/CLAUDE.md). Nothing failed, nothing logged, and
    "what did I say about the dentist" simply did not find it.

    Both sides, because a vector index holding only the questions makes "what did
    you suggest?" unanswerable while looking like it works.
    """
    session = FakeSession(
        Says("user", "치과 예약은 8월 5일 오후 3시"),
        Says("assistant", "그날 오후는 비워둘게"),
        Turn(),
    )
    recall = FakeRecall()
    await run(conversation(session, recall=recall, resolve_id=Ids()))

    assert recall.indexed == [
        (100, "치과 예약은 8월 5일 오후 3시"),
        (101, "그날 오후는 비워둘게"),
    ]


async def test_an_embedder_that_is_down_does_not_cost_the_conversation() -> None:
    """The markdown is the source of truth and the vector is an index, so losing one
    must not take the turn with it - in voice mode a raised exception is silence."""

    class Broken(FakeRecall):
        async def index(self, message_id: int, text: str) -> None:
            raise RuntimeError("ollama is not running")

    session = FakeSession(Says("user", "치과 예약 언제였지"), b"\x01", Turn())
    memory, audio = FakeMemory(), FakeAudio()
    await run(
        conversation(session, audio, memory, recall=Broken(), resolve_id=Ids())
    )

    assert [record.content for record in memory.records] == ["치과 예약 언제였지"]
    assert audio.played == [b"\x01"]


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


async def test_a_conversation_is_as_many_turns_as_the_session_gives() -> None:
    """`receive()` ends at the turn boundary (daemon/voice/base.py), so a
    conversation is a loop over calls. Held as one call it delivered the first answer
    and then blocked until the server cut the idle session - measured, and the reason
    both M0 spike runs died.
    """
    session = FakeSession(
        Says("user", "안녕"),
        Says("assistant", "안녕! 오늘 어땠어?"),
        Turn(),
        Says("user", "좋았어"),
        Says("assistant", "잘됐네"),
        Turn(),
    )
    memory = FakeMemory()
    conv = conversation(session, memory=memory)
    await run(conv)

    assert session.turns >= 2, "the second turn was never asked for"
    assert [record.content for record in memory.records] == [
        "안녕",
        "안녕! 오늘 어땠어?",
        "좋았어",
        "잘됐네",
    ]
    assert session.closed


async def test_a_session_that_stops_producing_turns_is_not_looped_on() -> None:
    """A provider whose `receive()` returns at once and never says the session ended
    would otherwise be called in a tight loop - no await, so not even the idle
    timeout could fire."""

    class Mute(FakeSession):
        async def receive(self) -> Any:
            self.turns += 1
            return
            yield  # pragma: no cover

    session = Mute()
    conv = conversation(session)
    await run(conv)

    assert session.turns <= 3, "an empty turn was looped on"
    assert conv.ended is not None
    assert session.closed


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
        Says("user", "아니 잠깐만"),  # the user starts talking over it
        Cuts(),  # and the provider's own activity detection says so
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


# --- tools --------------------------------------------------------------------
# A spoken tool call is the same event as a typed one (daemon/voice/base.py): the
# model asks, `receive()` yields a `ToolCall`, and the conversation runs it through
# the same `Companion.run_tools` the text loop uses and hands the result back with
# `send_tool_response`. Voice follows the owner's `DAEMON_TOOLS_MODE` but degrades
# `ask` to `allowlist` in `daemon/app.py`, so the only two outcomes at call time are
# run-it and refuse-it - there is nowhere in a spoken turn to ask, so nothing is ever
# parked.


def _read_file_call(path: pathlib.Path, call_id: str = "1") -> ToolCall:
    return ToolCall(id=call_id, name="read_file", arguments={"path": str(path)})


async def test_a_spoken_tool_call_runs_and_its_result_goes_back(
    db: Any, tmp_path: pathlib.Path
) -> None:
    """The whole point of PR-2b: the model can reach the machine over voice, and the
    answer to what it asked for goes back on the same socket, paired by call id."""
    (tmp_path / "notes.md").write_text("발표는 목요일")
    runner, _store = tool_runner(db, tmp_path)
    session = FakeSession(
        Calls(_read_file_call(tmp_path / "notes.md")),
        b"\x01",
        Says("assistant", "목요일이라고 적혀 있어"),
        Turn(),
    )
    await run(conversation(session, tools=runner))

    (results,) = session.tool_responses
    (result,) = results
    assert result.call_id == "1"
    assert result.ok
    assert "발표는 목요일" in result.content


async def test_the_answer_arrives_after_the_tool_call(
    db: Any, tmp_path: pathlib.Path
) -> None:
    """A blocking call generates nothing while it waits (daemon/voice/gemini_live.py),
    so the audio comes after the response goes back - and the turn is still one turn,
    recorded like any other."""
    (tmp_path / "notes.md").write_text("hi")
    runner, _store = tool_runner(db, tmp_path)
    memory = FakeMemory()
    audio = FakeAudio()
    session = FakeSession(
        Says("user", "메모 뭐라고 돼 있어"),
        Calls(_read_file_call(tmp_path / "notes.md")),
        b"\x01\x02",
        Says("assistant", "별거 없어"),
        Turn(),
    )
    await run(conversation(session, audio, memory, tools=runner))

    assert audio.played == [b"\x01\x02"], "the answer after the tool call was lost"
    assert [r.content for r in memory.records] == ["메모 뭐라고 돼 있어", "별거 없어"]


async def test_recall_waits_out_a_tool_answer_instead_of_killing_it(
    db: Any, tmp_path: pathlib.Path
) -> None:
    """A blocking tool call arrives before any audio, so nothing had set
    `_generating` when the user's transcript settled recall - and the recall block's
    `clientContent` landed right after the tool response, which the live server
    answers by cancelling the call: `tool.ran ok=True` then `the server cancelled
    tool calls [...]` on every single call, and the daemon opened Finder and said
    nothing about it (measured on the owner's Mac). Recall now waits for the turn
    boundary here exactly as it does mid-audio."""
    (tmp_path / "notes.md").write_text("hi")
    runner, _store = tool_runner(db, tmp_path)
    session = FakeSession(
        Calls(_read_file_call(tmp_path / "notes.md")),
        # The user's final transcript arrives with the tool call, before any audio -
        # the exact window the live cancellations came from.
        Transcript(text="메모 좀 열어봐 줄래", role="user", final=True),
        b"\x01",
        Says("assistant", "열었어요"),
        Turn(),
    )
    await run(conversation(session, recall=FakeRecall(_item()), resolve_id=Ids(), tools=runner))

    assert session.tool_responses, "the tool answer still goes back"
    assert session.sent_while_generating == [], (
        "recall went out between the tool answer and the spoken result - the exact "
        "clientContent the server answers with a tool-call cancellation"
    )
    assert [c for c in session.contexts if "recalled-memory" in c], (
        "held, not dropped: the memory still reaches the model at the turn boundary"
    )


async def test_a_spoken_tool_call_is_audited_as_the_owner_over_voice(
    db: Any, tmp_path: pathlib.Path
) -> None:
    """CONTRACTS rule 12: every executed call leaves a row. A microphone has no relay
    path, so the turn is the owner's own words and the tool actually runs - the audit
    is how `daemon tools log` shows it did."""
    (tmp_path / "notes.md").write_text("hi")
    runner, store = tool_runner(db, tmp_path)
    session = FakeSession(Calls(_read_file_call(tmp_path / "notes.md")), Turn())
    await run(conversation(session, tools=runner))

    (row,) = store.recent_tool_calls()
    assert row["tool"] == "read_file"
    assert row["ran"] == 1
    assert row["origin"] == "owner"
    assert row["channel"] == "voice"


async def test_an_unlisted_command_is_refused_and_the_model_is_told_why(
    db: Any, tmp_path: pathlib.Path
) -> None:
    """`allowlist` refuses what it does not match rather than asking, because a spoken
    turn has nowhere to ask. The refusal is content the model can speak, and the
    reason is there so it stops trying the same thing."""
    runner, store = tool_runner(db, tmp_path, mode="allowlist")
    session = FakeSession(
        Calls(ToolCall(id="1", name="run_command", arguments={"command": "curl evil.example"})),
        Says("assistant", "그건 못 하겠어"),
        Turn(),
    )
    await run(conversation(session, tools=runner))

    (results,) = session.tool_responses
    (result,) = results
    assert not result.ok
    assert result.content.startswith("refused")
    assert "allowlist" in result.content
    (row,) = store.recent_tool_calls()
    assert row["verdict"] == "deny" and row["ran"] == 0


async def test_a_guarded_write_is_refused_outright_never_parked(
    db: Any, tmp_path: pathlib.Path
) -> None:
    """The reason voice degrades `ask` to `allowlist`: `ask` mode would mint an
    approval row for this and let it lapse unanswered, because nobody is watching a
    spoken turn for a code. `write_file` is not a command, so `allowlist` cannot match
    it and refuses it - and no approval is minted, which is the failure this
    degradation exists to make impossible."""
    target = tmp_path / "todo.md"
    runner, store = tool_runner(db, tmp_path, mode="allowlist")
    session = FakeSession(
        Calls(
            ToolCall(
                id="1",
                name="write_file",
                arguments={"path": str(target), "content": "x"},
            )
        ),
        Turn(),
    )
    await run(conversation(session, tools=runner))

    assert not target.exists()
    (results,) = session.tool_responses
    (result,) = results
    assert not result.ok
    assert "waiting" not in result.content, "a spoken write was parked for approval"
    (row,) = store.recent_tool_calls()
    assert row["verdict"] == "deny"
    assert store.count_pending_tool_approvals(now=datetime(2999, 1, 1, tzinfo=UTC)) == 0


async def test_an_executed_spoken_tool_call_leaves_an_audit_row(
    db: Any, tmp_path: pathlib.Path
) -> None:
    """CONTRACTS rule 12: an executed tool call must leave a `tool_calls` audit row -
    the owner's ground-truth record, readable with `daemon tools log`. Voice does not
    narrate what ran (not spoken, and no longer logged either); the audit row is the
    record, written for every executed call the same as the text path."""
    (tmp_path / "notes.md").write_text("hi")
    runner, store = tool_runner(db, tmp_path)
    session = FakeSession(Calls(_read_file_call(tmp_path / "notes.md")), Turn())
    await run(conversation(session, tools=runner))

    (row,) = store.recent_tool_calls()
    assert row["tool"] == "read_file" and row["ran"] == 1


async def test_a_refused_spoken_tool_call_is_not_recorded_as_run(
    db: Any, tmp_path: pathlib.Path
) -> None:
    """The audit reflects what actually ran, not what was asked: a refused call leaves
    a deny row with ran=0, never a run - recording a refusal as a run is exactly the
    misleading state the audit guards against."""
    runner, store = tool_runner(db, tmp_path, mode="allowlist")
    session = FakeSession(
        Calls(ToolCall(id="1", name="run_command", arguments={"command": "curl evil.example"})),
        Turn(),
    )
    await run(conversation(session, tools=runner))

    (row,) = store.recent_tool_calls()
    assert row["verdict"] == "deny" and row["ran"] == 0


async def test_an_allowlisted_command_runs_over_voice(
    db: Any, tmp_path: pathlib.Path
) -> None:
    """The other side of the gate: a command the shared allowlist covers just runs,
    the same entry a text `/approve CODE always` would have written."""
    runner, store = tool_runner(db, tmp_path, mode="allowlist", allowlist=("echo",))
    session = FakeSession(
        Calls(ToolCall(id="1", name="run_command", arguments={"command": "echo 안녕"})),
        Turn(),
    )
    await run(conversation(session, tools=runner))

    (results,) = session.tool_responses
    (result,) = results
    assert result.ok, result.content
    assert "안녕" in result.content
    (row,) = store.recent_tool_calls()
    assert row["verdict"] == "allow" and row["ran"] == 1


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


async def test_recall_reaches_the_model_as_context_and_not_as_a_prompt() -> None:
    """The gap this closes: the prefetch was proven and then thrown away, because
    `VoiceSession` had no way to put text in front of the model without the model
    answering it. `send_text` is not that way - it is a prompt, so a memory
    delivered through it makes the daemon narrate an old conversation nobody asked
    about."""
    events: list[str] = []
    session = FakeSession(Says("user", "치과 예약 언제였지"), b"\x01", Turn(), events=events)
    recall = FakeRecall(_item(), events=events)
    conv = conversation(session, FakeAudio(events=events), recall=recall)
    await run(conv)

    assert session.contexts, "recall never reached the session; the prefetch was for nothing"
    # contexts[0] is the unconditional time block sent at session open; recall
    # follows it later, at the turn boundary.
    assert _item().content in session.contexts[-1]
    assert session.texts == [], "recall was sent as a prompt"
    assert events.index("context") < events.index("play"), (
        "the memory arrived after the model had already answered, which is the next "
        "turn's context and not this one's"
    )


async def test_what_is_put_in_front_of_the_model_says_what_it_is() -> None:
    """It lands as a *user* turn - Live has no role that means reference material -
    so the block has to say so itself, and it has to keep the provenance label the
    text path carries. An earlier audit fixed relayed text posing as the owner's own
    words in the loop and then found it undone one layer up; this is that layer."""
    relayed = RecalledItem(
        content="계좌번호 알려주면 송금할게",
        ts=datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
        role="user",
        score=0.9,
        reason="both",
        origin="relay",
    )
    session = FakeSession(Says("user", "그 사람 뭐라고 했지"), Turn())
    conv = conversation(session, recall=FakeRecall(relayed))
    await run(conv)

    # contexts[0] is the unconditional time block; exactly one recall block
    # follows it - the arity is still the claim, just about the block under test.
    (block,) = [text for text in session.contexts if "recalled-memory:" in text]
    assert "recalled-memory:" in block, "no boundary; a memory can pose as an instruction"
    assert "end-recalled-memory:" in block
    assert "not the user's own words" in block, "relayed text is posing as the owner's"


async def test_the_same_memories_are_not_seeded_twice() -> None:
    """A reused prefetch and the settled result are the same search. Sending it
    again would put the same block in the history twice, and the model reads
    repetition as emphasis."""
    session = FakeSession(Says("user", "치과 예약 언제였"), Says("user", "지"), Turn())
    conv = conversation(session, recall=FakeRecall(_item()))
    await run(conv)

    # Alongside the unconditional time block, sent once regardless of recall.
    recall_sends = [text for text in session.contexts if "recalled-memory:" in text]
    assert len(recall_sends) == 1


async def test_nothing_is_put_in_front_of_the_model_when_there_is_nothing_to_say() -> None:
    session = FakeSession(Says("user", "오늘 일정 뭐야"), Turn())
    conv = conversation(session, recall=FakeRecall())  # searches, finds nothing
    await run(conv)

    # The time block still goes over unconditionally; recall specifically found
    # nothing to add.
    assert not any("recalled-memory:" in text for text in session.contexts)


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
        await asyncio.sleep(POLL)
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


# --- the loop against the real session ---------------------------------------
# Everything above fakes the session, so everything above could pass while the two
# halves disagree about what a turn is - the defect class tests/test_reachable.py
# exists for. Here the only fake is the socket, and the frames on it are the ones
# the M0 spike measured.


class Socket:
    """A scripted websocket.

    Every read suspends, the way a real one does. Without that the receive loop
    never yields, the prefetch watcher never runs, and the test would prove the
    opposite of what it claims.
    """

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def close(self) -> None:
        self.closed = True

    def frames(self, key: str) -> list[dict[str, Any]]:
        return [message[key] for message in self.sent if key in message]

    def __aiter__(self) -> Socket:
        return self

    async def __anext__(self) -> str:
        await asyncio.sleep(0)
        if not self.script:
            raise StopAsyncIteration
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return json.dumps(item)


async def test_the_conversation_and_the_real_session_agree_about_a_turn() -> None:
    socket = Socket(
        {"setupComplete": {}},
        # Transcription arrives as deltas, so the first one is a query the prefetch
        # can start on while the user is still speaking.
        {"serverContent": {"inputTranscription": {"text": "치과 예약"}}},
        {"serverContent": {"inputTranscription": {"text": " 언제였지"}}},
        {
            "serverContent": {
                "modelTurn": {
                    "parts": [{"inlineData": {"data": base64.b64encode(b"pcm").decode("ascii")}}]
                }
            }
        },
        {"serverContent": {"outputTranscription": {"text": "8월 5일 오후 3시야"}}},
        # Both, in the order the API sends them: the turn ends on the first and the
        # second is read as a turn with nothing in it.
        {"serverContent": {"generationComplete": True}},
        {"serverContent": {"turnComplete": True}},
        ConnectionClosedOK(Close(1000, "bye"), None),
    )

    async def connect(url: str, **kwargs: Any) -> Socket:
        return socket

    live = GeminiLiveSession(
        "AIzaSy-fake-key-value", "gemini-3.1-flash-live-preview", connect=connect
    )
    audio = FakeAudio(b"mic")
    memory = FakeMemory()
    recall = FakeRecall(_item())
    conv = VoiceConversation(
        live, audio, companion_for(memory, recall=recall), idle_timeout=1.0
    )

    async with asyncio.timeout(RUN_LIMIT):
        await conv.run()

    assert [record.content for record in memory.records] == [
        "치과 예약 언제였지",
        "8월 5일 오후 3시야",
    ]
    assert audio.played == [b"pcm"]
    assert conv.ended is not None, "the conversation ended without saying why"
    assert socket.closed, "a session that bills per minute was left open"
    # Recall started from the in-progress transcript - the first query is a prefix
    # nobody ever finished saying - and went back as context rather than as a
    # prompt: `clientContent` with `turnComplete: false`.
    assert recall.queries == ["치과 예약", "치과 예약 언제였지"]
    # Two clientContent frames now: the unconditional time block at session open,
    # then recall at the turn boundary - filter to the one carrying the memory and
    # unpack a one-tuple, so "exactly one" is still asserted, just about that block.
    (seeded,) = [
        frame
        for frame in socket.frames("clientContent")
        if _item().content in frame["turns"][0]["parts"][0]["text"]
    ]
    assert seeded["turnComplete"] is False
    assert not any("text" in frame for frame in socket.frames("realtimeInput")), (
        "recall reached the model as a prompt; the daemon will read it aloud"
    )
    assert any("audio" in frame for frame in socket.frames("realtimeInput")), (
        "the microphone never reached the session"
    )


async def test_a_provider_with_nothing_pending_is_not_a_problem() -> None:
    session = BareSession(Hang())
    memory = FakeMemory()
    conv = conversation(session, memory=memory)

    task = asyncio.create_task(conv.run())
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert memory.records == []


# --- what the session reports about itself -----------------------------------
# Every number here used to exist and never leave the object. The one that
# mattered - `interruptions` - is how the daemon cutting itself off would have
# been visible, and it was counted for three milestones and printed nowhere.


async def test_a_conversation_reports_what_it_did() -> None:
    session = FakeSession(
        Says("user", "오늘 일정 뭐야"),
        b"\x01\x02",
        Says("assistant", "회의가 두 개 있어요"),
        Turn(),
    )
    audio = FakeAudio()
    conv = conversation(session, audio)
    await run(conv)

    stats = conv.stats
    assert stats.turns == 1
    assert stats.interruptions == 0
    assert stats.first_audio_seconds is not None
    assert stats.played_seconds == pytest.approx(2 / (24_000 * 2))


async def test_a_barge_in_is_visible_in_the_report() -> None:
    """The whole reason the report exists. An interruption on every turn with
    almost no audio played is the signature of the daemon cutting itself off, and
    nothing used to say so - it went unnoticed for three milestones."""
    session = FakeSession(b"\x01", Says("user", "아니 잠깐만"), Cuts(), Turn())
    conv = conversation(session)
    await run(conv)

    assert conv.stats.interruptions == 1
    assert "1 interruption(s)" in conv.stats.describe()


async def test_a_session_that_never_spoke_says_so_rather_than_zero() -> None:
    """A silent session and an instant one are different failures. Reported as
    0.00s they would read the same, and the silent one is the one worth finding."""
    session = FakeSession(Says("user", "여보세요"), Turn())
    conv = conversation(session)
    await run(conv)

    assert conv.stats.first_audio_seconds is None
    assert conv.stats.played_seconds == 0.0
    assert "never spoke" in conv.stats.describe()


async def test_played_seconds_is_seconds_at_the_playback_rate() -> None:
    """Not the capture rate. 24 kHz out against 16 kHz in is the chipmunk bug's
    other half (daemon/voice/base.py), and getting it wrong here would report
    every session as 50% longer than it was."""
    audio = FakeAudio()
    session = FakeSession(b"\x00" * 48_000, Turn())
    conv = conversation(session, audio)
    await run(conv)

    assert conv.stats.played_seconds == pytest.approx(1.0)


async def test_the_answer_survives_the_transcript_of_the_question() -> None:
    """The regression this file exists for now.

    Gemini delivers `inputTranscription` at the turn boundary - measured, in the
    same server event as the answer's first audio chunk. The conversation used to
    read that as the user talking over the answer, so *every* turn was ruled a
    barge-in: against the live API a complete reply ("안녕하세요! 반갑습니다. 오늘
    어떤 대화를 나누고 싶으세요?") was generated and 0.0s of it was played.

    So: audio, then the user's own transcript, and no `Cuts`. Nothing may be
    interrupted."""
    session = FakeSession(
        b"\x01\x02",
        Says("user", "안녕"),
        Says("assistant", "안녕하세요! 반갑습니다"),
        Turn(),
    )
    audio = FakeAudio()
    conv = conversation(session, audio)
    await run(conv)

    assert conv.interruptions == 0, (
        "the question's own transcript was read as a barge-in against its answer"
    )
    assert session.interrupts == 0
    assert audio.stops == 0
    assert audio.played == [b"\x01\x02"], "the answer was thrown away unheard"
    assert conv.stats.played_seconds > 0


async def test_the_report_survives_a_conversation_that_was_cancelled() -> None:
    """`run_voice` reports from a `finally`, so this is read on the failure path
    more often than the happy one."""
    session = FakeSession(Says("user", "잠깐"), Hang())
    conv = conversation(session)

    task = asyncio.create_task(conv.run())
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert conv.stats.turns == 0
    assert "never spoke" in conv.stats.describe()


# --- reconnecting a conversation the server cut -------------------------------
# Until `app._voice_attempts` existed, a dropped socket simply ended `daemon voice`:
# mid-sentence, with a shell exit code, and nothing said about whether it was the
# network or the key.


class Cut(Exception):
    """Stands in for `GeminiLiveError`, with the flag that decides a retry."""

    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


def _attempts(*sessions: FakeSession, **kwargs: Any) -> Any:
    """Drive `app._voice_attempts` over a queue of scripted sessions."""
    from daemon import app as app_module

    queue = list(sessions)
    built: list[FakeSession] = []

    def new_session() -> FakeSession:
        session = queue.pop(0) if len(queue) > 1 else queue[0]
        built.append(session)
        return session

    return app_module, new_session, built, companion_for()


async def _drive(*sessions: FakeSession, audio: FakeAudio | None = None) -> tuple[int, list[Any]]:
    app_module, new_session, built, companion = _attempts(*sessions)
    code = await app_module._voice_attempts(
        new_session, audio or FakeAudio(), companion, Cut
    )
    return code, built


async def test_a_conversation_that_simply_ends_is_not_reconnected() -> None:
    """An idle timeout is the conversation being over. Reconnecting into one bills
    per minute for nothing."""
    code, built = await _drive(FakeSession(Says("user", "안녕"), Turn()))

    assert code == 0
    assert len(built) == 1


async def test_a_transient_close_is_picked_back_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """1011, or the 1008 that is really an idle timeout. The user is standing in the
    middle of a conversation, so it is resumed rather than reported."""
    from daemon import app as app_module

    monkeypatch.setattr(app_module, "VOICE_RECONNECT_BACKOFF_SECONDS", 0.0)
    failing = FakeSession(Cut("connection closed 1011: internal error"))
    working = FakeSession(Says("user", "다시 들려?"), Turn())

    code, built = await _drive(failing, working)

    assert code == 0
    assert len(built) == 2, "the dropped conversation was not picked back up"


async def test_a_permanent_failure_is_not_retried() -> None:
    """A bad key or a wrong model id. Retrying leaves the process alive,
    healthy-looking and mute."""
    code, built = await _drive(FakeSession(Cut("api key not valid", permanent=True)))

    assert code == 1
    assert len(built) == 1, "a permanent failure was retried"


async def test_retries_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sessions bill per minute, so this is deliberately not a reconnect-forever
    loop."""
    from daemon import app as app_module

    monkeypatch.setattr(app_module, "VOICE_RECONNECT_BACKOFF_SECONDS", 0.0)
    # One fresh failure per attempt, plus a spare: a reused session would hand the
    # second attempt an exhausted script, which ends cleanly and would make this
    # test pass for the wrong reason.
    code, built = await _drive(
        *(
            FakeSession(Cut("connection closed 1011: internal error"))
            for _ in range(app_module.VOICE_RECONNECT_ATTEMPTS + 1)
        )
    )

    assert code == 1
    assert len(built) == app_module.VOICE_RECONNECT_ATTEMPTS


async def test_a_go_away_is_reconnected_even_though_nothing_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured trap this closes: `goAway` means the server is ending the
    *session*, not the turn, and it arrives looking exactly like a conversation that
    finished normally - so the daemon used to stop mid-conversation with a clean exit
    code."""
    from daemon import app as app_module

    monkeypatch.setattr(app_module, "VOICE_RECONNECT_BACKOFF_SECONDS", 0.0)
    leaving = FakeSession(Says("user", "잠깐"), Turn())
    leaving.going_away = True
    resumed = FakeSession(Says("user", "계속하자"), Turn())

    code, built = await _drive(leaving, resumed)

    assert code == 0
    assert len(built) == 2


async def test_each_attempt_gets_a_fresh_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reconnecting means starting clean. The old session carries a half-flushed
    transcript and a partial queue nobody will read again."""
    from daemon import app as app_module

    monkeypatch.setattr(app_module, "VOICE_RECONNECT_BACKOFF_SECONDS", 0.0)
    first = FakeSession(Cut("connection closed 1011: internal error"))
    second = FakeSession(Says("user", "다시"), Turn())

    _code, built = await _drive(first, second)

    assert built[0] is not built[1]


async def test_what_each_attempt_did_is_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """One report per attempt, numbered. A single line covering three attempts would
    hide which of them actually held a conversation."""
    from daemon import app as app_module

    monkeypatch.setattr(app_module, "VOICE_RECONNECT_BACKOFF_SECONDS", 0.0)
    with caplog.at_level("INFO"):
        await _drive(
            FakeSession(Cut("connection closed 1011: internal error")),
            FakeSession(Says("user", "다시"), b"\x01", Turn()),
        )

    assert "voice session:" in caplog.text
    assert "voice session (attempt 2):" in caplog.text


# --- recall must not interrupt the answer it was fetched for -------------------
# The Live API: "a message here will interrupt any current model generation."
# Measured against it - `interrupted` arrived 90 ms after the recall block went out,
# and one conversation delivered 2.2s of audio with recall on against 46.7s with it
# off. The daemon was killing its own answers with its own memory.


async def test_recall_is_not_sent_while_the_model_is_answering() -> None:
    """The measured bug. `clientContent` mid-generation is documented as interrupting
    the answer, and against the live API `interrupted` came back 90 ms later."""
    recall = BlockingRecall(_item())
    session = FakeSession(
        Says("user", "치과 언제였지"),  # the prefetch starts, and blocks
        b"\x01\x02",  # the answer starts arriving: generating
        Does(recall.gate.set),  # the search lands *now*, mid-answer
        Turn(),
    )
    conv = conversation(session, recall=recall)
    await run(conv)

    assert recall.queries, "the prefetch never ran, so this proves nothing"
    assert session.sent_while_generating == [], (
        "recall was sent while the model was generating - the Live API documents "
        "that as interrupting the answer, and it cost 2.2s of audio against 46.7s"
    )
    assert session.contexts, "recall was held back and then never sent at all"


async def test_held_recall_is_sent_at_the_turn_boundary() -> None:
    """Held, not dropped - and this is the case where only the flush can deliver it.

    The prefetch covers the finished utterance, so `_settle_recall` reuses it instead
    of searching again; with nothing redoing the offer, a missing flush means a memory
    that was searched for and silently discarded."""
    recall = BlockingRecall(_item())
    session = FakeSession(
        Says("user", "치과 언제였지"),
        b"\x01",
        Does(recall.gate.set),
        Turn(),
    )
    conv = conversation(session, recall=recall)
    await run(conv)

    # Alongside the unconditional time block, sent once at the turn boundary.
    recall_sends = [text for text in session.contexts if "치과" in text]
    assert len(recall_sends) == 1, "the held recall never reached the model"
    assert session.sent_while_generating == []


async def test_the_same_turn_is_not_seeded_twice() -> None:
    """Two blocks for one turn would hand the model the same facts again, and each
    send is another chance to interrupt."""
    recall = BlockingRecall(_item())
    session = FakeSession(
        Says("user", "치과"),
        b"\x01",
        Does(recall.gate.set),
        Says("user", " 언제였지 그리고 약은"),
        Turn(),
    )
    conv = conversation(session, recall=recall)
    await run(conv)

    # The unconditional time block is one more context; recall itself must still
    # go over at most once.
    recall_sends = [text for text in session.contexts if "치과" in text]
    assert len(recall_sends) <= 1, "the same turn was seeded more than once"
    assert session.sent_while_generating == []


# --- the utterance that opened the session, and the cue that invites the next ---
# The gate used to consume "루시 뭐 해", match on the alias, and throw the sound away,
# so the session opened deaf to the question it was opened for. Measured on a real
# run: 14.79s from the wake word to the first audio out, most of it a person saying
# the same thing twice.


async def test_the_opening_utterance_is_the_first_thing_the_session_hears() -> None:
    """First, not merely eventually: live microphone audio arriving ahead of it would
    put the question behind the noise of a person settling down.

    Asserted as "the first chunk", not "before `record()` is called", because the
    latter is not a real guarantee - `record()` only builds the generator and nothing
    flows until the pump task runs. A mutation swapping those two lines changes
    nothing, and a test that failed on it would be pinning noise."""
    opening = b"\x11\x22" * 400
    audio = FakeAudio(b"live-block")
    session = FakeSession(Says("user", "뭐 해"), b"\x01", Turn())
    conv = conversation(session, audio, opening_audio=opening)
    await run(conv)

    assert session.sent, "the opening utterance never reached the session"
    assert session.sent[0] == opening, (
        "something else was fed to the session before the utterance that opened it"
    )


async def test_no_opening_audio_is_the_ordinary_case() -> None:
    """`daemon voice` run by hand has no wake segment, and neither does a session
    opened by a provider with nothing to offer."""
    audio = FakeAudio(b"live-block")
    session = FakeSession(Says("user", "안녕"), Turn())
    conv = conversation(session, audio)
    await run(conv)

    assert b"" not in session.sent, "an empty opening was sent as a chunk"


async def test_a_session_that_refuses_the_opening_still_holds_the_conversation() -> None:
    """One repeated sentence is the cost of losing it. Raising here would cost the
    whole turn, which is worse."""

    class Refuses(FakeSession):
        async def send_audio(self, chunk: bytes) -> None:
            if not self.sent:
                self.sent.append(chunk)
                raise RuntimeError("the socket hiccuped on the first chunk")
            await super().send_audio(chunk)

    session = Refuses(Says("user", "안녕"), b"\x01", Turn())
    conv = conversation(session, opening_audio=b"\x11\x22")
    await run(conv)

    assert conv.stats.played_seconds > 0, "a refused opening took the answer with it"


# --- the ready cue -------------------------------------------------------------


def test_the_ready_cue_is_audible_brief_and_starts_at_silence() -> None:
    """A bare sine starts on a discontinuity, which is an audible click and exactly
    the kind of noise a VAD notices.

    The lower bound is the one that earns its keep: an acknowledgement too quiet to
    notice is why the owner said the wake word again, and on this provider each repeat
    interrupts the answer being generated, which the server discards and restarts -
    the measured "same sentence three times" loop. Loud enough to be heard the first
    time, short enough not to delay the turn, quiet enough not to startle."""
    import numpy as np

    from daemon.app import READY_CUE_MS, ready_cue

    pcm = ready_cue(24_000)
    samples = np.frombuffer(pcm, dtype="<i2")
    peak = int(np.abs(samples).max())

    assert len(samples) == pytest.approx(24_000 * READY_CUE_MS / 1000, rel=0.02)
    assert samples[0] == 0 and samples[-1] == 0, "the cue clicks"
    assert peak > 8000, "an acknowledgement nobody hears is one the owner talks over"
    assert peak < 16000, "the cue is loud enough to startle"
    assert READY_CUE_MS <= 250, "an ack that delays the conversation is not an ack"


def test_the_ready_cue_follows_the_devices_rate() -> None:
    """Synthesised rather than shipped as a wav precisely so this holds: a cue at the
    wrong rate is the chipmunk bug in miniature."""
    from daemon.app import ready_cue

    assert len(ready_cue(48_000)) == 2 * len(ready_cue(24_000))


def test_the_ready_cue_rises() -> None:
    """Rising reads as "go ahead"; falling reads as "finished"."""
    import numpy as np

    from daemon.app import ready_cue

    samples = np.frombuffer(ready_cue(24_000), dtype="<i2")
    half = len(samples) // 2

    def hz(part: np.ndarray) -> float:
        crossings = int(((part[:-1] >= 0) != (part[1:] >= 0)).sum())
        return crossings / 2 / (len(part) / 24_000)

    assert hz(samples[half:]) > hz(samples[:half])


async def test_the_cue_plays_and_a_speaker_that_refuses_it_is_not_fatal() -> None:
    """The cue is a courtesy. A conversation that failed because it could not be
    played would be a worse trade than no cue at all."""
    from daemon.app import play_ready_cue, ready_cue

    audio = FakeAudio()
    await play_ready_cue(audio)
    assert audio.played, "the cue never reached the speaker"
    # At the playback rate, not the capture rate. FakeAudio deliberately has two
    # different ones (16k in, 24k out), so a cue built at the wrong one arrives the
    # wrong length - the chipmunk bug in miniature (daemon/voice/base.py).
    assert audio.played[0] == ready_cue(audio.playback_sample_rate)
    assert audio.played[0] != ready_cue(audio.sample_rate)

    class Deaf(FakeAudio):
        async def play(self, chunk: bytes) -> None:
            raise RuntimeError("no speaker here")

    await play_ready_cue(Deaf())  # must not raise


async def test_an_unanswered_opening_survives_a_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An attempt cut down before it answers never got the utterance in front of a
    model, so dropping it on the retry would reopen the failure the opening exists to
    fix: the owner saying the wake phrase twice."""
    from daemon import app as app_module

    monkeypatch.setattr(app_module, "VOICE_RECONNECT_BACKOFF_SECONDS", 0.0)
    opening = b"\x11\x22" * 200
    failing = FakeSession(Cut("connection closed 1011: internal error"))
    working = FakeSession(Says("user", "다시"), b"\x01", Turn())
    app_mod, new_session, built, companion = _attempts(failing, working)

    code = await app_mod._voice_attempts(
        new_session, FakeAudio(), companion, Cut, opening_audio=opening
    )

    assert code == 0
    assert built[1].sent and built[1].sent[0] == opening, (
        "the utterance that opened the session was lost on the reconnect"
    )


async def test_an_answered_opening_is_not_asked_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half: once something has been said back, re-sending it would have
    the daemon answer the same question twice."""
    from daemon import app as app_module

    monkeypatch.setattr(app_module, "VOICE_RECONNECT_BACKOFF_SECONDS", 0.0)
    opening = b"\x11\x22" * 200
    answered = FakeSession(Says("user", "안녕"), b"\x01" * 4800, Turn())
    answered.going_away = True  # forces a reconnect after a turn that did answer
    resumed = FakeSession(Says("user", "계속"), Turn())
    app_mod, new_session, built, companion = _attempts(answered, resumed)

    await app_mod._voice_attempts(
        new_session, FakeAudio(), companion, Cut, opening_audio=opening
    )

    assert opening not in built[1].sent, "the answered question was asked again"


# --- run_voice wires the tools into the session ------------------------------
# tests/test_reachable.py has a blind spot it names itself: `GeminiLiveSession` is
# already constructed by app.py, so nothing there can tell whether run_voice passes
# it `tools=`. Written, tested and unreachable is the defect this whole repo guards
# against, so the wiring gets a test that drives run_voice and reads what the
# session was actually offered - the only fakes are the hardware and the socket.


def _voice_settings(tmp_path: pathlib.Path, **overrides: str) -> Any:
    from daemon.config import Settings

    base = {
        "_env_file": None,
        "DAEMON_PROVIDER": "gemini",
        "DAEMON_OLLAMA_MODEL": "gemma3:4b",
        "DAEMON_DATA_DIR": str(tmp_path),
        "TELEGRAM_BOT_TOKEN": "123456:AAHfake-token-value",
        "GEMINI_API_KEY": "k",
        "DAEMON_VOICE_ENABLED": "true",
        "DAEMON_GEMINI_LIVE_MODEL": "gemini-3.1-flash-live-preview",
        "DAEMON_GEMINI_MODEL": "gemini-3.5-flash",
        "DAEMON_TOOLS_ENABLED": "true",
        "DAEMON_TOOLS_ROOTS": str(tmp_path),
    }
    base.update(overrides)
    return Settings(**base)


async def _run_voice_capturing(
    settings: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Run `run_voice` once with a fake session and spy on the mode `_build_tools`
    is asked for and the kwargs the session is built with."""
    from daemon import app as app_module

    captured: dict[str, Any] = {}

    def capturing_session(**kwargs: Any) -> FakeSession:
        captured.update(kwargs)
        return FakeSession(Turn())  # one empty turn, then the conversation ends

    seen: dict[str, Any] = {}
    real_build_tools = app_module._build_tools

    async def spy_build_tools(s: Any, store: Any, **kw: Any) -> Any:
        seen["mode"] = kw.get("mode")
        return await real_build_tools(s, store, **kw)

    monkeypatch.setattr(app_module, "build_voice_audio", lambda: FakeAudio())
    monkeypatch.setattr(app_module, "_build_tools", spy_build_tools)
    monkeypatch.setattr("daemon.voice.gemini_live.GeminiLiveSession", capturing_session)

    code = await app_module.run_voice(settings)
    return code, seen, captured


async def test_run_voice_follows_the_owners_tool_mode(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Voice honours the owner's configured `DAEMON_TOOLS_MODE` - `full` here, the
    install default - rather than pinning `allowlist` and silently refusing guarded
    tools the owner asked for by voice. A microphone has no relay path, so a spoken
    turn is the owner's own words and the origin gate is the real boundary; running
    `full` there is the same authority the text path already has (memory: tools
    default to full). Also checks the specs reach the session and the tool contract
    rides with them (daemon/companion.py, TOOL_CONTRACT)."""
    from daemon.companion import TOOL_CONTRACT

    settings = _voice_settings(tmp_path, DAEMON_TOOLS_MODE="full")
    code, seen, captured = await _run_voice_capturing(settings, monkeypatch)

    assert code == 0
    assert seen.get("mode") == "full", "voice did not follow the owner's tool mode"
    specs = captured.get("tools")
    assert specs, "the session was offered no tools, so the model can never call one"
    assert {spec.name for spec in specs} >= {"read_file", "run_command"}
    assert TOOL_CONTRACT in (captured.get("system_instruction") or ""), (
        "the model was handed tools but not the rules for using them"
    )


# --- live screen share is gated on the provider (PR #79 review, Finding 1) ----
# `OpenAIRealtimeSession.send_frame` is a deliberate no-op - no realtime video
# input channel - so registering the live-share start/stop tools for it would
# let the model tell the owner "I'm watching your screen now" while every frame
# is silently dropped (ADR 0009 forbids exactly this). These drive the real
# `run_voice` assembly and read what the session was actually offered, the same
# way `test_run_voice_follows_the_owners_tool_mode` does.


async def test_run_voice_openai_drops_live_share_tools_but_keeps_see_screen(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _voice_settings(
        tmp_path,
        DAEMON_VOICE_PROVIDER="openai",
        OPENAI_API_KEY="k",
        DAEMON_OPENAI_REALTIME_MODEL="gpt-realtime",
        DAEMON_SCREEN_ENABLED="true",
    )
    from daemon import app as app_module

    captured: dict[str, Any] = {}

    def capturing_session(**kwargs: Any) -> FakeSession:
        captured.update(kwargs)
        return FakeSession(Turn())

    monkeypatch.setattr(app_module, "build_voice_audio", lambda: FakeAudio())
    monkeypatch.setattr(
        "daemon.voice.openai_realtime.OpenAIRealtimeSession", capturing_session
    )

    code = await app_module.run_voice(settings)

    assert code == 0
    names = {spec.name for spec in (captured.get("tools") or ())}
    assert "see_screen" in names, (
        "the still-image tool is a different path and must stay on"
    )
    assert "start_screen_share" not in names, (
        "OpenAI's send_frame is a no-op; offering this tool lets the model "
        "claim a screen-watching capability it cannot deliver"
    )
    assert "stop_screen_share" not in names


async def test_run_voice_gemini_keeps_live_share_tools(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half: Gemini Live's session does take video frames
    (`realtimeInput.video`, ADR 0009), so the gate must not cost it the
    live-share tools it can actually back."""
    settings = _voice_settings(tmp_path, DAEMON_SCREEN_ENABLED="true")
    from daemon import app as app_module

    captured: dict[str, Any] = {}

    def capturing_session(**kwargs: Any) -> FakeSession:
        captured.update(kwargs)
        return FakeSession(Turn())

    monkeypatch.setattr(app_module, "build_voice_audio", lambda: FakeAudio())
    monkeypatch.setattr(
        "daemon.voice.gemini_live.GeminiLiveSession", capturing_session
    )

    code = await app_module.run_voice(settings)

    assert code == 0
    names = {spec.name for spec in (captured.get("tools") or ())}
    assert {"see_screen", "start_screen_share", "stop_screen_share"} <= names


# --- screen-share lifecycle (Task 2.3) ----------------------------------------
#
# The pump needs a live VoiceSession, so the conversation binds it once the
# session is open and stops+unbinds it when the conversation ends - a share must
# never outlive its session. These tests drive the real `ScreenShareController`
# and the real `start_screen_share`/`stop_screen_share` tools through the same
# `_run_tool_call` path a spoken call actually takes, with a fake pump standing in
# for `ScreenSharePump` so nothing here touches a real screen or socket.


class _FakeContextSession:
    """Records `send_context` calls, standing in for the live session the
    pump would otherwise hold (Finding 3: the framing seed on share start)."""

    def __init__(self) -> None:
        self.context_sent: list[str] = []

    async def send_context(self, text: str) -> None:
        self.context_sent.append(text)


class _FakePump:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.session = _FakeContextSession()

    def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


async def test_screen_share_binds_a_fresh_pump_when_the_session_opens() -> None:
    from daemon.voice.screen_share import ScreenShareController

    controller = ScreenShareController()
    created: list[_FakePump] = []

    def factory(session: Any) -> _FakePump:
        assert session is not None
        pump = _FakePump()
        created.append(pump)
        return pump

    session = FakeSession(Turn())
    conv = conversation(session, screen_share=controller, screen_pump_factory=factory)
    await run(conv)

    assert len(created) == 1, "the conversation must bind exactly one pump per session"


async def test_screen_share_start_stop_tools_drive_the_bound_pump(
    db: Any,
) -> None:
    """The full spoken path: `start_screen_share` flips the bound pump on through
    `_run_tool_call`, exactly as a real tool call from the model would."""
    from daemon.tools.screen import screen_share_tools
    from daemon.voice.screen_share import ScreenShareController

    controller = ScreenShareController()
    created: list[_FakePump] = []

    def factory(session: Any) -> _FakePump:
        pump = _FakePump()
        created.append(pump)
        return pump

    store = Store(db)
    registry = Registry()
    for tool in screen_share_tools(controller):
        registry.register(tool)
    runner = ToolRunner(registry, ToolPolicy(store, mode="allowlist", enabled=True), store)

    session = FakeSession(
        Calls(ToolCall(id="1", name="start_screen_share", arguments={})),
        Turn(),
    )
    conv = conversation(
        session, screen_share=controller, screen_pump_factory=factory, tools=runner
    )
    await run(conv)

    assert len(created) == 1
    assert created[0].started is True, "the tool call must have started the bound pump"
    # The conversation ended, so `stop_and_unbind` must have run in `finally` -
    # a share must never outlive its session.
    assert created[0].stopped is True
    assert controller.active is False


async def test_screen_share_is_stopped_and_unbound_when_the_conversation_ends() -> None:
    """Even with no `start_screen_share` call at all: an active share left over
    from an earlier turn (or a controller reused across a reconnect) must not
    survive past this session's `run()`."""
    from daemon.voice.screen_share import ScreenShareController

    controller = ScreenShareController()
    pump = _FakePump()
    controller.bind(pump)
    await controller.start()
    assert controller.active is True

    session = FakeSession(Turn())
    conv = conversation(session, screen_share=controller, screen_pump_factory=lambda s: pump)
    await run(conv)

    assert pump.stopped is True
    assert controller.active is False


async def test_screen_share_unbinds_even_when_run_is_cancelled() -> None:
    """`run()`'s teardown is a `finally`, so cancellation must not leak a pump -
    the same guarantee the transcript-recording teardown already has."""
    from daemon.voice.screen_share import ScreenShareController

    controller = ScreenShareController()
    created: list[_FakePump] = []

    def factory(session: Any) -> _FakePump:
        pump = _FakePump()
        created.append(pump)
        return pump

    session = FakeSession(Hang())
    conv = conversation(session, screen_share=controller, screen_pump_factory=factory)
    task = asyncio.ensure_future(conv.run())
    for _ in range(20):
        await asyncio.sleep(0)
    # The share is on when the cancellation lands - the case that actually risks
    # a leak, since `stop_and_unbind` only calls `pump.stop()` while active.
    await controller.start()
    assert created[0].started is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=RUN_LIMIT)

    assert len(created) == 1
    assert created[0].stopped is True
    assert controller.active is False


async def test_conversation_without_screen_share_never_touches_the_controller() -> None:
    """The default (`screen_share=None`) - the text path never even constructs a
    controller, and this is the voice case with screen sharing off entirely."""
    session = FakeSession(Turn())
    conv = conversation(session)  # no screen_share, no screen_pump_factory
    await run(conv)  # must not raise


async def test_run_voice_degrades_ask_to_allowlist(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one mode a spoken turn cannot honour is `ask`: it has nowhere to surface an
    approval, so `ask` would mint rows that lapse unanswered - the silent degradation
    this repo calls the dangerous failure. Voice degrades `ask` to `allowlist` (run
    what is listed, refuse the rest, never park), so `full` reaches voice but `ask`
    never does."""
    settings = _voice_settings(tmp_path, DAEMON_TOOLS_MODE="ask")
    code, seen, _captured = await _run_voice_capturing(settings, monkeypatch)

    assert code == 0
    assert seen.get("mode") == "allowlist", "voice did not degrade `ask` to `allowlist`"


# --- session-start continuity -------------------------------------------------
# Every session used to start empty - the server holds voice history and a new
# socket has none - so calling the daemon twice in three minutes met a stranger
# the second time. The tail of the recent conversation now rides in on
# `send_context` before the opening audio (daemon/companion.py, continuity_block).


def _spoke(text: str, role: str = "user", *, minutes_ago: float = 1.0) -> LoggedMessage:
    from datetime import timedelta

    from daemon import clock

    return LoggedMessage(
        ts=clock.now() - timedelta(minutes=minutes_ago),
        role=role,  # type: ignore[arg-type]
        content=text,
        origin="owner" if role == "user" else "agent",
        session_kind="interactive",
        modality="voice",
        channel="voice",
    )


async def test_the_recent_conversation_rides_in_before_the_first_turn() -> None:
    session = FakeSession(Says("user", "이어서 하자"), b"\x01", Turn())
    memory = FakeMemory()
    memory.records.extend(
        [
            _spoke("면접 준비 도와줘", minutes_ago=3),
            _spoke("좋아요, 어디 회사부터?", "assistant", minutes_ago=2),
        ]
    )

    await run(conversation(session, FakeAudio(), memory))

    continuity = [text for text in session.contexts if "recent-conversation" in text]
    assert len(continuity) == 1, "the tail goes over exactly once per session, at the start"
    assert "면접 준비 도와줘" in continuity[0]
    assert "어디 회사부터" in continuity[0]
    assert continuity[0] not in session.sent_while_generating, (
        "sent before any generation - mid-generation clientContent kills the answer"
    )


async def test_the_session_learns_the_date_before_the_conversation_tail() -> None:
    """Order matters: the tail is read *in* the present, so the present goes first.

    Both go over at open - the one point where nothing is generating, and
    mid-generation `clientContent` kills the answer.
    """
    session = FakeSession(Says("user", "이어서 하자"), b"\x01", Turn())
    memory = FakeMemory()
    memory.records.extend(
        [
            _spoke("면접 준비 도와줘", minutes_ago=3),
            _spoke("좋아요, 어디 회사부터?", "assistant", minutes_ago=2),
        ]
    )

    await run(conversation(session, FakeAudio(), memory))

    assert session.contexts[0].startswith("[현재 시각] ")
    tail = next(i for i, text in enumerate(session.contexts) if "recent-conversation" in text)
    assert tail > 0, "the date is established before the tail that is read against it"
    assert session.contexts[0] not in session.sent_while_generating, (
        "sent before any generation - mid-generation clientContent kills the answer"
    )


async def test_a_past_commitment_reaches_the_session_over_send_context() -> None:
    """`Companion.time_block` is the only way `[약속 상태]` can reach a voice
    session - its history lives server-side, so there is no `list[Message]` for the
    text path's `_assemble` to splice a block into, and this `send_context` call is
    the whole handoff. Modelled on
    `test_the_session_learns_the_date_before_the_conversation_tail`: the seeded
    message is well outside the 120-minute continuity window, so the only new
    content in `session.contexts[0]` beyond `[현재 시각]` is the commitment block
    itself (verified by mutation: deleting the `timesense.commitments(...)` call
    from `time_block` fails this assertion; restoring it passes).
    """
    session = FakeSession(Says("user", "안녕"), b"\x01", Turn())
    memory = FakeMemory()
    memory.records.append(_spoke("오늘 오후 2시에 면접 있어", minutes_ago=60 * 24 * 2))

    await run(conversation(session, FakeAudio(), memory))

    assert session.contexts[0].startswith("[현재 시각] ")
    assert "[약속 상태]" in session.contexts[0]
    assert "이미 지났습니다" in session.contexts[0]
    assert not any("recent-conversation" in text for text in session.contexts), (
        "the seeded message is two days old - well outside the continuity window"
    )


async def test_a_quiet_stretch_still_tells_the_session_what_day_it_is() -> None:
    """The complement of `test_a_quiet_stretch_means_no_continuity_block_at_all`: no
    tail is correct after two hours of silence, but a session opening after a quiet
    night is exactly the one that most needs the date."""
    session = FakeSession(Says("user", "안녕"), b"\x01", Turn())

    await run(conversation(session, FakeAudio(), FakeMemory()))

    assert session.contexts[0].startswith("[현재 시각] ")
    assert not any("recent-conversation" in text for text in session.contexts)


async def test_a_quiet_stretch_means_no_continuity_block_at_all() -> None:
    """Two hours of silence is a finished conversation: greeting afresh is correct,
    recall is the tool for older context, and an empty block must not become an
    empty `send_context` frame on the wire."""
    session = FakeSession(Says("user", "안녕"), b"\x01", Turn())
    memory = FakeMemory()
    memory.records.append(_spoke("어제 얘기", minutes_ago=60 * 5))

    await run(conversation(session, FakeAudio(), memory))

    assert [text for text in session.contexts if "recent-conversation" in text] == []


async def test_an_empty_history_sends_nothing_either() -> None:
    session = FakeSession(Says("user", "안녕"), b"\x01", Turn())

    await run(conversation(session, FakeAudio(), FakeMemory()))

    assert [text for text in session.contexts if "recent-conversation" in text] == []


async def test_the_microphone_holds_while_a_tool_answer_is_pending() -> None:
    """Between "the model asked for a tool" and its first audio back, mic sound is
    read by the server's activity detection as the user interrupting, and it cancels
    the pending call - the daemon ran the tool and never spoke the result (measured
    live: `tool.ran ok=True` then `the server cancelled tool calls`, zero real
    interruptions in the session). Driven directly rather than through a scripted
    session because the property is about interleaving: which chunks cross while the
    flags are in each state."""
    session = FakeSession()
    conv = conversation(session)

    async def mic() -> Any:
        yield b"m1"  # ordinary conversation: forwarded
        conv._answering_tool = True
        yield b"m2"  # tool pending, nothing playing: held - this is the fix
        conv._generating = True
        yield b"m3"  # the answer is playing: forwarded, so barge-in still works

    await conv._forward_microphone(session, mic())

    assert session.sent == [b"m1", b"m3"], (
        "a chunk crossed while the tool answer was pending - the exact audio the "
        "server answers with a tool-call cancellation"
    )


async def test_half_duplex_holds_the_microphone_while_the_daemon_speaks() -> None:
    """DAEMON_VOICE_BARGE_IN=false: while the daemon is speaking (or a tool answer is
    pending), the microphone yields entirely, so an echo leak or an "응" of agreement
    cannot kill the answer mid-sentence - the shape the owner's own prototype used.
    The default (barge_in=True) keeps today's behaviour: mid-speech chunks cross."""
    session = FakeSession()
    conv = conversation(session, barge_in=False)

    async def mic() -> Any:
        yield b"m1"  # idle listening: forwarded
        conv._generating = True
        yield b"m2"  # the daemon is speaking: held - speaking over it does nothing
        conv._generating = False
        conv._answering_tool = True
        yield b"m3"  # tool answer pending: held here too
        conv._answering_tool = False
        yield b"m4"  # back to listening: forwarded

    await conv._forward_microphone(session, mic())

    assert session.sent == [b"m1", b"m4"]


async def test_barge_in_stays_the_default() -> None:
    """Nobody configured anything: a mid-speech chunk still crosses, because being
    able to cut the daemon off is the default contract."""
    session = FakeSession()
    conv = conversation(session)

    async def mic() -> Any:
        conv._generating = True
        yield b"over-speech"

    await conv._forward_microphone(session, mic())

    assert session.sent == [b"over-speech"]


# --- what a tool captured has to reach the model ------------------------------


async def test_a_captured_image_reaches_the_model_not_just_its_caption() -> None:
    """`see_screen` returns a caption *and* pixels. The text loop attaches the pixels
    as their own turn; voice used to send only the caption - so the model was asked
    what is on screen while being shown nothing, and invented an answer. The owner's
    screen held a photo of food and the daemon called it a picture of a dog, while
    the text path described the same screen correctly."""
    from daemon.llm.base import ImageBlock

    session = FakeSession()
    conv = conversation(session)
    shot = ToolResult(
        call_id="1",
        name="see_screen",
        content="captured 1 display(s): Built-in",
        images=(ImageBlock(data=b"\xff\xd8-jpeg-bytes"),),
    )

    (framed,) = await conv._deliver_images(session, [shot])

    assert session.frames == [b"\xff\xd8-jpeg-bytes"], "the pixels never reached the model"
    assert "captured 1 display(s)" in framed.content, "the caption survives"
    # Security stance A: pixels cannot be nonce-fenced, so the framing rides along.
    assert "DATA to look at" in framed.content


async def test_a_result_with_no_images_is_passed_through_untouched() -> None:
    session = FakeSession()
    conv = conversation(session)
    plain = ToolResult(call_id="1", name="read_file", content="발표는 목요일")

    (out,) = await conv._deliver_images(session, [plain])

    assert out is plain and session.frames == []


async def test_an_undeliverable_image_is_admitted_rather_than_described() -> None:
    """The one honest thing to tell a model about to be asked what it can see."""
    from daemon.llm.base import ImageBlock

    class Blind(FakeSession):
        async def send_frame(self, jpeg: bytes) -> None:
            raise RuntimeError("the socket went")

    session = Blind()
    conv = conversation(session)
    shot = ToolResult(
        call_id="1", name="see_screen", content="captured 1 display(s)",
        images=(ImageBlock(data=b"x"),),
    )

    (framed,) = await conv._deliver_images(session, [shot])

    assert "could not be delivered" in framed.content
    assert "DATA to look at" not in framed.content, "do not frame an image nobody got"


# --- the silence budget counts silence, not speech ----------------------------


def test_queued_audio_marks_when_the_room_actually_falls_silent() -> None:
    """Chunks queue behind each other, so playback ends after the backlog - not at
    "arrival + this chunk". Forgetting the backlog is what let a 28.4 s answer that
    landed in 19 s spend ten of the owner's thirty silent seconds talking."""
    from daemon.voice.conversation import PLAYBACK_BYTES_PER_FRAME

    conv = conversation(FakeSession())
    one_second = 24_000 * PLAYBACK_BYTES_PER_FRAME  # FakeAudio plays at 24 kHz

    conv._on_audio(100.0, one_second)  # arrives at 100, plays 100 -> 101
    conv._on_audio(100.1, one_second)  # arrives while the first is still playing

    assert conv._playback_until == pytest.approx(102.0), "the backlog was forgotten"


async def test_a_session_is_not_closed_while_the_daemon_is_still_speaking() -> None:
    """The measured complaint: "데몬이 말하고 있는 시간도 nothing heard에 포함되나?" It
    was. The model generates faster than real time, so the whole answer arrived, the
    budget started running on arrival, and the owner was cut off mid-reply while the
    log claimed nothing had been heard for 30 s."""
    from daemon.voice.conversation import PLAYBACK_BYTES_PER_FRAME

    quarter_second = 24_000 // 4 * PLAYBACK_BYTES_PER_FRAME
    session = FakeSession(b"\x00" * quarter_second, Hang())
    conv = conversation(session, idle_timeout=0.05)

    started = asyncio.get_running_loop().time()
    async with asyncio.timeout(5):
        await conv.run()
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed > 0.25, (
        "the session closed while the speaker still had audio queued - the budget "
        "counted the daemon's own voice as silence"
    )


async def test_being_called_by_name_is_answered_rather_than_met_with_silence() -> None:
    """The wake-word-only handover. Audio was tried and misheard ("벨라" -> "별로",
    answered as a sentence nobody said); sending nothing was tried next and left the
    owner calling into silence, waiting ten seconds, and calling again. The name goes
    over as text - settled words, nothing left to mishear - and `send_text` is a
    prompt, so the model answers being called the way a person does."""
    session = FakeSession(Says("assistant", "네?"), Turn())

    await run(conversation(session, opening_text="벨라"))

    assert session.texts == ["벨라"], "the model was never told it had been called"
    assert session.sent == [], "no audio was sent - that is the misheard path"


async def test_a_question_in_the_same_breath_still_goes_over_as_audio() -> None:
    """When the segment carries more than the name, the audio is what has the
    question in it - text would throw the question away."""
    session = FakeSession(Turn())

    await run(conversation(session, opening_audio=b"pcm"))

    assert session.sent == [b"pcm"] and session.texts == []


# --- the room stays out of the socket until the wake word is answered ---------


async def test_the_microphone_is_held_until_the_answer_to_the_wake_word_starts() -> None:
    """The same opening text is answered in 1.1 s against a session with no
    microphone and took 11.77 s in the resident, where the mic streams the room the
    moment the session is up: audio arriving as a turn begins reads as the user
    speaking, so the server cancels the answer and waits for an utterance that never
    comes - without emitting `interrupted`, which is why the session reported zero
    barge-ins while the owner sat through eleven seconds of silence."""
    session = FakeSession()
    conv = conversation(session, opening_text="벨라")
    await conv._send_opening(session)

    async def mic() -> Any:
        yield b"room"          # the answer has not started: held
        conv._on_audio(asyncio.get_running_loop().time(), 480)  # first audio arrives
        yield b"after"         # the room is the owner's again

    await conv._forward_microphone(session, mic())

    assert session.sent == [b"after"], "the room was streamed into the opening turn"


async def test_a_model_that_never_answers_gives_the_microphone_back() -> None:
    """Bounded, because holding the microphone for a model that is not coming back
    would turn a slow turn into a deaf one."""
    session = FakeSession()
    conv = conversation(session, opening_text="벨라")
    await conv._send_opening(session)
    # The hold has expired without any audio ever arriving.
    conv._opening_answer_until = asyncio.get_running_loop().time() - 0.01

    async def mic() -> Any:
        yield b"speak"

    await conv._forward_microphone(session, mic())

    assert session.sent == [b"speak"]


async def test_an_audio_opening_does_not_hold_the_microphone() -> None:
    """The hold is for the text opening, where the owner has demonstrably finished
    speaking - the gate captured the whole phrase. An audio opening carries their
    question and the turn is already under way."""
    session = FakeSession()
    conv = conversation(session, opening_audio=b"pcm")
    await conv._send_opening(session)

    async def mic() -> Any:
        yield b"more"

    await conv._forward_microphone(session, mic())

    assert session.sent == [b"pcm", b"more"]


async def test_half_duplex_holds_the_microphone_until_the_speaker_runs_dry() -> None:
    """`_generating` clears when the last audio chunk *arrives*, not when it is
    *heard*, so half-duplex used to reopen the microphone while the speaker was
    still working through the backlog - and the room it then recorded was the
    daemon's own voice.

    Measured in the owner's log (2026-08-19): every leaked turn is the *tail* of
    the line before it. "당연하죠! 저도 응원하고 있을게요. 잘하고 오세요!" came back as
    a user turn reading "원할 때 있을게요. 잘하고 오세요." - the opening clause missing
    because the microphone opened partway through playback, and the rest mangled
    because what leaks past echo cancellation is a distorted residual. The daemon
    then answered itself, and the reply parroted its own last sentence.

    The gap is not small: `_on_audio` records 28.4 s of audio landing in about
    19 s. `_playback_until` already knows when the speaker runs dry - it was only
    ever read by the idle budget.
    """
    from daemon.voice.conversation import PLAYBACK_BYTES_PER_FRAME

    audio = FakeAudio()
    session = FakeSession()
    conv = conversation(session, audio, barge_in=False)
    loop = asyncio.get_running_loop()
    # One second of playback, arriving now: heard until now + 1.
    one_second = audio.playback_sample_rate * PLAYBACK_BYTES_PER_FRAME

    async def mic() -> Any:
        yield b"m1"  # idle listening: forwarded
        conv._generating = True
        conv._on_audio(loop.time(), one_second)
        yield b"m2"  # the answer is arriving: held
        conv._generating = False  # last chunk received - the speaker is not done
        yield b"m3"  # still playing: held, and this is the fix
        conv._playback_until = loop.time() - 1.0  # the speaker has run dry
        yield b"m4"  # the room is the owner's again: forwarded

    await conv._forward_microphone(session, mic())

    assert session.sent == [b"m1", b"m4"], (
        "a chunk crossed while the speaker was still playing - that chunk is the "
        "daemon's own voice, and it lands in memory as something the owner said"
    )


async def test_barge_in_still_hears_the_room_during_playback() -> None:
    """The drain hold is half-duplex only. With barge-in on (the default), the
    microphone streams throughout - being able to cut the daemon off mid-answer is
    that mode's whole contract, and it does not end when the socket goes quiet."""
    audio = FakeAudio()
    session = FakeSession()
    conv = conversation(session, audio)
    loop = asyncio.get_running_loop()

    async def mic() -> Any:
        conv._generating = False
        conv._playback_until = loop.time() + 30.0
        yield b"over-playback"

    await conv._forward_microphone(session, mic())

    assert session.sent == [b"over-playback"]


async def test_the_session_is_told_what_it_keeps_saying() -> None:
    """Voice is where the owner noticed it: a wake word, and the same opener again.

    The tic list rides in with the tail it was computed from, before any generation
    - `send_context` mid-generation kills the answer, which is the rule the whole
    session-start sequence is built around."""
    session = FakeSession(Says("user", "어"), b"\x01", Turn())
    memory = FakeMemory()
    memory.records.extend(
        [
            _spoke("무슨 재미난 얘기라도?", "assistant", minutes_ago=9),
            _spoke("아니", minutes_ago=8),
            _spoke("오늘은 재미난 일 없었어요?", "assistant", minutes_ago=7),
            _spoke("없어", minutes_ago=6),
            _spoke("재미난 얘기 좀 해봐요.", "assistant", minutes_ago=5),
        ]
    )

    await run(conversation(session, FakeAudio(), memory))

    (tics,) = [text for text in session.contexts if text.startswith("[verbal-tics]")]
    assert '"재미난"' in tics
    assert tics not in session.sent_while_generating


async def test_a_session_with_no_habit_to_report_sends_no_tic_block() -> None:
    session = FakeSession(Says("user", "어"), b"\x01", Turn())
    memory = FakeMemory()
    memory.records.extend([_spoke("그건 몰랐네요.", "assistant", minutes_ago=2)])

    await run(conversation(session, FakeAudio(), memory))

    assert not [text for text in session.contexts if text.startswith("[verbal-tics]")]
