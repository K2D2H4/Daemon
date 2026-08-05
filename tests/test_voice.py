"""Gemini Live session tests. No network, no audio hardware, no API key.

A fake connection object stands in for the websocket; nothing here constructs a
real `websockets` connection, so a machine with no route to Google runs the whole
file.

The two tests that matter most are the ones guarding what the design rests on:
setup must enable transcription in both directions (docs/PLAN.md 6.5 - without it
voice mode still talks and silently stops remembering), and the API key must not
reach a log, an exception, or an exception's context.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import ssl as ssl_module
from collections.abc import AsyncIterator
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, InvalidStatus
from websockets.frames import Close

from daemon.voice import gemini_live
from daemon.voice.base import Interrupted, Transcript, VoiceSession
from daemon.voice.gemini_live import GeminiLiveError, GeminiLiveSession

KEY = "AIzaSy-fake-key-value"  # fake shape; no test may need a real one
MODEL = "gemini-3.1-flash-live-preview"

SETUP_COMPLETE = {"setupComplete": {}}


def audio_frame(pcm: bytes) -> dict[str, Any]:
    return {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "audio/pcm;rate=24000",
                            "data": base64.b64encode(pcm).decode("ascii"),
                        }
                    }
                ]
            }
        }
    }


def said(role: str, text: str) -> dict[str, Any]:
    key = "inputTranscription" if role == "user" else "outputTranscription"
    return {"serverContent": {key: {"text": text}}}


def closed(code: int, reason: str = "") -> ConnectionClosedError:
    return ConnectionClosedError(Close(code, reason), None)


class FakeConnection:
    """Scripted websocket. Each scripted item is either a dict to be delivered as
    JSON, or an exception to be raised from the receive loop."""

    def __init__(self, *scripted: Any) -> None:
        self.scripted = list(scripted)
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send(self, raw: str) -> None:
        if self.closed:
            raise closed(1000)
        self.sent.append(json.loads(raw))

    async def close(self) -> None:
        self.closed = True

    def messages(self, key: str) -> list[dict[str, Any]]:
        return [m[key] for m in self.sent if key in m]

    def __aiter__(self) -> FakeConnection:
        return self

    async def __anext__(self) -> str:
        if not self.scripted:
            raise StopAsyncIteration
        item = self.scripted.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, str) else json.dumps(item)


class Hanging(FakeConnection):
    """A connection that delivers its script and then goes quiet, exactly as the
    real one did: the answer arrives, and nothing after it ever does.

    This is the shape that hung the M0 spike twice. A test whose fake runs out of
    messages proves nothing about a turn ending, because StopAsyncIteration ends the
    iterator for it.
    """

    async def __anext__(self) -> str:
        if self.scripted:
            return await super().__anext__()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")  # pragma: no cover


def connector(*connections: Any) -> Any:
    """A stand-in for `websockets.connect`: hands out scripted connections, or
    raises a scripted exception instead of connecting.

    `ssl` is in the signature because the real one is called with it - a trust
    store we choose rather than the library default. A fake that quietly accepted
    **kwargs would have let that argument be dropped again unnoticed.
    """
    queue = list(connections)

    async def connect(
        url: str, *, additional_headers: dict[str, str], ssl: ssl_module.SSLContext
    ) -> Any:
        connect.calls.append((url, additional_headers, ssl))  # type: ignore[attr-defined]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item

    connect.calls = []  # type: ignore[attr-defined]
    return connect


def session(*connections: Any, **kwargs: Any) -> GeminiLiveSession:
    return GeminiLiveSession(KEY, MODEL, connect=connector(*connections), **kwargs)


async def drain(live: GeminiLiveSession) -> list[bytes | Transcript]:
    """One turn's worth of output. `receive()` ends at the turn boundary, so this
    returns rather than blocking - and the timeout is what fails the test if that
    ever stops being true."""
    async with asyncio.timeout(5):
        return [item async for item in live.receive()]


async def _collect(stream: AsyncIterator[Transcript], into: list[Transcript]) -> None:
    async for item in stream:
        into.append(item)


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff delays instead of waiting them out."""
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(gemini_live, "_sleep", fake_sleep)
    return delays


def test_satisfies_voice_session_protocol() -> None:
    assert isinstance(session(FakeConnection()), VoiceSession)


def test_missing_credentials_are_rejected_before_any_connection() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiLiveSession("", MODEL)
    with pytest.raises(ValueError, match="DAEMON_GEMINI_LIVE_MODEL"):
        GeminiLiveSession(KEY, "")


# --- setup ------------------------------------------------------------------


async def test_setup_enables_transcription_in_both_directions() -> None:
    """docs/PLAN.md 6.5 rests on this: transcripts are what keep memory and
    persona evolution working in voice mode."""
    connection = FakeConnection(SETUP_COMPLETE)
    async with session(connection):
        pass

    (setup,) = connection.messages("setup")
    assert setup["inputAudioTranscription"] == {}
    assert setup["outputAudioTranscription"] == {}


async def test_setup_asks_for_audio_and_names_the_model_as_a_resource() -> None:
    connection = FakeConnection(SETUP_COMPLETE)
    async with session(connection):
        pass

    (setup,) = connection.messages("setup")
    assert setup["model"] == f"models/{MODEL}"
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    # A language code would be rejected: native-audio models choose the language.
    assert "languageCode" not in json.dumps(setup)


async def test_model_id_already_prefixed_is_not_prefixed_twice() -> None:
    connection = FakeConnection(SETUP_COMPLETE)
    async with GeminiLiveSession(KEY, f"models/{MODEL}", connect=connector(connection)):
        pass

    assert connection.messages("setup")[0]["model"] == f"models/{MODEL}"


async def test_persona_voice_and_system_instruction_reach_setup() -> None:
    connection = FakeConnection(SETUP_COMPLETE)
    live = session(connection, voice_name="Kore", system_instruction="너는 조용한 편이다")
    async with live:
        pass

    (setup,) = connection.messages("setup")
    voice = setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
    assert voice["voiceName"] == "Kore"
    assert setup["systemInstruction"]["parts"][0]["text"] == "너는 조용한 편이다"


async def test_an_unconfigured_session_lets_the_server_decide_turn_boundaries() -> None:
    """No `realtimeInputConfig` at all, which is what every session sent before
    these settings existed. Sending a partly-filled one would replace the server's
    ~800 ms default with three numbers nobody chose."""
    connection = FakeConnection(SETUP_COMPLETE)
    async with session(connection):
        pass

    assert "realtimeInputConfig" not in connection.messages("setup")[0]


async def test_endpointing_settings_reach_setup_in_the_wire_spelling() -> None:
    """`low` is what a person writes in .env; `START_SENSITIVITY_LOW` is what the
    server accepts, and anything else closes the socket with 1007."""
    connection = FakeConnection(SETUP_COMPLETE)
    live = session(
        connection,
        start_sensitivity="low",
        end_sensitivity="high",
        prefix_padding_ms=100,
        silence_duration_ms=400,
    )
    async with live:
        pass

    detection = connection.messages("setup")[0]["realtimeInputConfig"][
        "automaticActivityDetection"
    ]
    assert detection == {
        "startOfSpeechSensitivity": "START_SENSITIVITY_LOW",
        "endOfSpeechSensitivity": "END_SENSITIVITY_HIGH",
        "prefixPaddingMs": 100,
        "silenceDurationMs": 400,
    }


async def test_only_the_settings_that_were_set_are_sent() -> None:
    """An omitted field is the server's default; a field sent as 0 is not."""
    connection = FakeConnection(SETUP_COMPLETE)
    async with session(connection, silence_duration_ms=400):
        pass

    detection = connection.messages("setup")[0]["realtimeInputConfig"][
        "automaticActivityDetection"
    ]
    assert detection == {"silenceDurationMs": 400}


async def test_a_zero_is_a_choice_and_is_sent() -> None:
    """None means "leave it alone" and 0 means "no padding at all". Collapsing the
    two would make the more aggressive setting unreachable."""
    connection = FakeConnection(SETUP_COMPLETE)
    async with session(connection, prefix_padding_ms=0):
        pass

    detection = connection.messages("setup")[0]["realtimeInputConfig"][
        "automaticActivityDetection"
    ]
    assert detection == {"prefixPaddingMs": 0}


async def test_a_misspelled_sensitivity_fails_here_rather_than_on_the_wire() -> None:
    """The server's answer is a 1007 close, which `_permanent_close` classifies as
    permanent - so left to the wire a typo does not fail the setting, it ends voice
    mode with a message about an unknown name."""
    with pytest.raises(ValueError, match="sensitivity"):
        GeminiLiveSession(KEY, MODEL, start_sensitivity="LOW")
    with pytest.raises(ValueError, match="sensitivity"):
        GeminiLiveSession(KEY, MODEL, end_sensitivity="medium")


async def test_api_key_travels_as_a_header_not_in_the_url() -> None:
    connect = connector(FakeConnection(SETUP_COMPLETE))
    async with GeminiLiveSession(KEY, MODEL, connect=connect):
        pass

    (url, headers, _ssl) = connect.calls[0]
    assert headers == {"x-goog-api-key": KEY}
    assert KEY not in url


async def test_no_setup_complete_means_no_usable_session() -> None:
    """A socket that opens and never confirms setup would otherwise be billed
    while doing nothing."""
    with pytest.raises(GeminiLiveError, match="before setupComplete"):
        async with session(FakeConnection(), max_attempts=1):
            pass  # pragma: no cover


async def test_send_before_open_is_a_clear_error() -> None:
    with pytest.raises(GeminiLiveError, match="not open"):
        await session(FakeConnection()).send_text("hi")


# --- receive ----------------------------------------------------------------


async def test_receive_separates_audio_from_transcripts() -> None:
    connection = FakeConnection(
        SETUP_COMPLETE,
        said("user", "오늘 뭐 먹을까"),
        audio_frame(b"\x01\x02"),
        audio_frame(b"\x03\x04"),
        said("assistant", "김치찌개 어때"),
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        received = await drain(live)

    assert [item for item in received if isinstance(item, bytes)] == [b"\x01\x02", b"\x03\x04"]
    assert [item for item in received if isinstance(item, Transcript)] == [
        Transcript(text="오늘 뭐 먹을까", role="user", final=True),
        Transcript(text="김치찌개 어때", role="assistant", final=True),
    ]


async def test_transcript_fragments_are_never_yielded_before_the_turn_ends() -> None:
    """Gemini streams transcription a few syllables at a time. Handing those out
    would let a caller record half a word as an utterance."""
    connection = FakeConnection(
        SETUP_COMPLETE,
        said("assistant", "김치"),
        said("assistant", "찌개 "),
        said("assistant", "어때"),
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        received = await drain(live)

    assert received == [Transcript(text="김치찌개 어때", role="assistant", final=True)]
    assert all(item.final for item in received)


async def test_generation_complete_flushes_and_does_not_duplicate_at_turn_complete() -> None:
    connection = FakeConnection(
        SETUP_COMPLETE,
        said("assistant", "다 했어"),
        {"serverContent": {"generationComplete": True}},
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        received = await drain(live)
        # The turn ended on `generationComplete`, so the `turnComplete` behind it
        # reads as one more turn with nothing in it. Cheaper than guessing which of
        # the two a given model sends - as long as it stays empty.
        assert await drain(live) == []

    assert [item.text for item in received] == ["다 했어"]


# --- the turn boundary -------------------------------------------------------
# The measured defect. `receive()` delivered the final transcript 2.6 s in and then
# blocked forever; the server cut the idle session and reported it as close 1008
# "The operation was aborted." Both M0 spike runs died there, and only when consumed
# to the end - breaking out after five items hid it completely.


async def test_receive_ends_at_turn_complete_rather_than_blocking_forever() -> None:
    connection = Hanging(
        SETUP_COMPLETE,
        said("assistant", "안녕하세요! 반갑습니다. 무슨 재미있는 이야기 있으세요?"),
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        try:
            async with asyncio.timeout(1.0):
                received = [item async for item in live.receive()]
        except TimeoutError:
            pytest.fail(
                "receive() did not end at turnComplete: `async for` blocked until the "
                "server aborted the session, which is the M0 spike failure"
            )

    assert [item.text for item in received] == [
        "안녕하세요! 반갑습니다. 무슨 재미있는 이야기 있으세요?"
    ]
    assert live.ended is None, "the turn ended, not the session; the caller may take another"


async def test_receive_ends_at_generation_complete_too() -> None:
    connection = Hanging(
        SETUP_COMPLETE,
        audio_frame(b"\x01"),
        {"serverContent": {"generationComplete": True}},
    )
    async with session(connection) as live:
        try:
            async with asyncio.timeout(1.0):
                assert [item async for item in live.receive()] == [b"\x01"]
        except TimeoutError:
            pytest.fail("receive() did not end at generationComplete")


async def test_the_next_turn_is_the_next_call_on_the_same_session() -> None:
    """What the loop above buys: a conversation is turns, and one socket serves all
    of them - sessions bill per minute, so reopening one per turn would be paying
    twice for the same minute."""
    connection = Hanging(
        SETUP_COMPLETE,
        said("assistant", "첫 번째"),
        {"serverContent": {"turnComplete": True}},
        said("assistant", "두 번째"),
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        async with asyncio.timeout(1.0):
            first = [item async for item in live.receive()]
            second = [item async for item in live.receive()]

    assert [item.text for item in first] == ["첫 번째"]
    assert [item.text for item in second] == ["두 번째"]
    assert live.ended is None


async def test_a_dropped_connection_still_yields_what_was_said() -> None:
    connection = FakeConnection(
        SETUP_COMPLETE,
        said("user", "잠깐만"),
        closed(1011, "internal error"),
    )
    async with session(connection) as live:
        received: list[Any] = []
        with pytest.raises(GeminiLiveError, match="1011"):
            async for item in live.receive():
                received.append(item)

    assert received == [Transcript(text="잠깐만", role="user", final=True)]


async def test_clean_close_ends_the_stream_without_an_error() -> None:
    connection = FakeConnection(
        SETUP_COMPLETE,
        said("assistant", "잘 자"),
        ConnectionClosedOK(Close(1000, "bye"), None),
    )
    async with session(connection) as live:
        assert await drain(live) == [Transcript(text="잘 자", role="assistant", final=True)]


async def test_malformed_frames_do_not_end_the_stream() -> None:
    connection = FakeConnection(
        SETUP_COMPLETE,
        "not json at all",
        json.dumps([1, 2, 3]),  # valid JSON, wrong shape
        {"serverContent": {"modelTurn": {"parts": [{"inlineData": {"data": "!!!not base64"}}]}}},
        audio_frame(b"\x09"),
    )
    async with session(connection) as live:
        assert await drain(live) == [b"\x09"]


async def test_go_away_is_reported_rather_than_looking_like_a_bug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connection = FakeConnection(SETUP_COMPLETE, {"goAway": {"timeLeft": "9.5s"}})
    with caplog.at_level(logging.INFO):
        async with session(connection) as live:
            assert await drain(live) == []

    assert "9.5s" in caplog.text


async def test_a_session_ending_is_distinguishable_from_a_turn_ending() -> None:
    """The audit finding: Live sends `goAway` before the session limit and the
    stream then just stops, which reads exactly like a finished turn. The caller
    kept sending audio into a socket that was gone, and from the user's side the
    daemon stopped answering mid-conversation."""
    connection = FakeConnection(
        SETUP_COMPLETE,
        said("assistant", "잠깐만"),
        {"serverContent": {"turnComplete": True}},
        {"goAway": {"timeLeft": "1s"}},
    )
    async with session(connection) as live:
        received = await drain(live)
        # The turn ended and said nothing about the session, which is the point: the
        # `goAway` is still ahead, and it arrives in the turn the caller takes next.
        assert live.ended is None
        assert await drain(live) == []

    assert [item.text for item in received] == ["잠깐만"]
    assert live.going_away
    assert live.ended is not None and "goAway" in live.ended


async def test_a_stream_that_simply_ends_says_so_too() -> None:
    """The other side of the same property: no goAway, so the reason must not
    claim one."""
    connection = FakeConnection(SETUP_COMPLETE, audio_frame(b"\x01"))
    async with session(connection) as live:
        await drain(live)

    assert not live.going_away
    assert live.ended is not None and "goAway" not in live.ended


async def test_the_reason_a_session_ended_is_available_even_when_it_raises() -> None:
    connection = FakeConnection(SETUP_COMPLETE, closed(1011, "internal error"))
    async with session(connection) as live:
        with pytest.raises(GeminiLiveError):
            await drain(live)

    assert live.ended is not None and "1011" in live.ended


# --- transcripts the caller can still get at ---------------------------------


async def test_a_cancelled_receive_does_not_lose_what_was_said() -> None:
    """The audit finding this exists for: the transcript is the *only* record voice
    mode produces, and it is accumulated until `turnComplete`. A shutdown or an
    upper-layer timeout arriving first left the utterance in neither the markdown
    nor the mirror - and an async generator cannot yield from its own `finally`, so
    the accumulation has to be reachable from outside.
    """
    connection = Hanging(SETUP_COMPLETE, said("user", "치과 예약 언제였지"))
    async with session(connection) as live:
        stream = live.receive()

        async def pull() -> None:
            async for _ in stream:  # pragma: no cover - nothing is yielded
                pass

        task = asyncio.create_task(pull())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert live.pending_transcripts() == [
            Transcript(text="치과 예약 언제였지", role="user", final=True)
        ]
        # Destructive, so a later flush cannot record the same words twice.
        assert live.pending_transcripts() == []


async def test_abandoning_the_stream_mid_flush_loses_neither_transcript() -> None:
    """A turn releases the user's words and then the assistant's, and `yield` is a
    suspension point - so the drain has to be one role at a time. Taken together
    they are the exchange; either alone is a monologue in the log."""
    connection = FakeConnection(
        SETUP_COMPLETE,
        said("user", "질문"),
        said("assistant", "대답"),
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        stream = live.receive()
        assert await stream.__anext__() == Transcript(text="질문", role="user", final=True)
        await stream.aclose()

        assert live.pending_transcripts() == [
            Transcript(text="대답", role="assistant", final=True)
        ]


async def test_a_turn_abandoned_mid_flush_does_not_cut_the_next_one_short() -> None:
    """Walking away at a boundary leaves the boundary flag set. Read on the way in
    rather than the way out, or the next turn ends after its first event with the
    rest of the answer still on the socket."""
    connection = Hanging(
        SETUP_COMPLETE,
        said("user", "질문"),
        {"serverContent": {"turnComplete": True}},
        audio_frame(b"\x01"),
        said("assistant", "대답"),
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        stream = live.receive()
        assert await stream.__anext__() == Transcript(text="질문", role="user", final=True)
        await stream.aclose()

        async with asyncio.timeout(1.0):
            assert await drain(live) == [
                b"\x01",
                Transcript(text="대답", role="assistant", final=True),
            ]


async def test_partial_transcripts_arrive_while_the_user_is_still_talking() -> None:
    """Recall has to start embedding before the utterance ends - 117 ms is free
    while the user is talking and is silence afterwards (docs/PLAN.md 4.3.1). Each
    item is the utterance so far and `final=False`, which is what stops it being
    recorded as one."""
    connection = FakeConnection(
        SETUP_COMPLETE,
        said("user", "어제 얘기한"),
        said("user", " 치과 예약"),
        audio_frame(b"\x01"),
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        seen: list[Transcript] = []
        partials = live.partial_transcripts()
        watch = asyncio.create_task(_collect(partials, seen))
        received = await drain(live)
        await asyncio.sleep(0)

        assert seen == [
            Transcript(text="어제 얘기한", role="user", final=False),
            Transcript(text="어제 얘기한 치과 예약", role="user", final=False),
        ]
        assert not any(item.final for item in seen)
        # Reading them does not consume the turn: it still flushes normally.
        assert received == [
            b"\x01",
            Transcript(text="어제 얘기한 치과 예약", role="user", final=True),
        ]
        watch.cancel()
        await asyncio.gather(watch, return_exceptions=True)


async def test_the_assistants_own_words_are_not_offered_as_partials() -> None:
    """A partial exists so recall can embed what the user is asking. The model
    answering itself is not a query, and one that reached the prefetch would search
    for the daemon's own sentence."""
    connection = FakeConnection(
        SETUP_COMPLETE,
        said("assistant", "여덟 시야"),
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        seen: list[Transcript] = []
        watch = asyncio.create_task(_collect(live.partial_transcripts(), seen))
        await drain(live)
        await asyncio.sleep(0)

        assert seen == []
        watch.cancel()
        await asyncio.gather(watch, return_exceptions=True)


async def test_the_partial_stream_ends_when_the_session_does() -> None:
    """Otherwise the consumer is a task parked forever on a socket that is gone -
    the same defect as a `receive()` that never ends, one seam over."""
    connection = FakeConnection(SETUP_COMPLETE, said("user", "거기"), closed(1011, "internal"))
    async with session(connection) as live:
        seen: list[Transcript] = []
        watch = asyncio.create_task(_collect(live.partial_transcripts(), seen))
        with pytest.raises(GeminiLiveError):
            await drain(live)

        async with asyncio.timeout(1.0):
            await watch  # ends on its own, or this test times out
        assert [item.text for item in seen] == ["거기"]


# --- sending ----------------------------------------------------------------


async def test_send_audio_uses_the_input_rate_and_base64() -> None:
    connection = FakeConnection(SETUP_COMPLETE)
    async with session(connection) as live:
        await live.send_audio(b"\x00\x01\x02\x03")

    (realtime,) = connection.messages("realtimeInput")
    assert realtime["audio"]["mimeType"] == "audio/pcm;rate=16000"
    assert base64.b64decode(realtime["audio"]["data"]) == b"\x00\x01\x02\x03"
    assert gemini_live.INPUT_SAMPLE_RATE != gemini_live.OUTPUT_SAMPLE_RATE


async def test_send_text_speaks_without_any_user_audio() -> None:
    """The proactive path: nothing has been recorded, and something still has to
    come out of the session."""
    connection = FakeConnection(SETUP_COMPLETE, audio_frame(b"\xaa"))
    async with session(connection) as live:
        await live.send_text("자기 전에 물 한 잔 마셔")
        received = await drain(live)

    assert connection.messages("realtimeInput") == [{"text": "자기 전에 물 한 잔 마셔"}]
    assert not any("audio" in m for m in connection.messages("realtimeInput"))
    assert received == [b"\xaa"]


async def test_send_context_seeds_history_without_asking_for_an_answer() -> None:
    """Measured, and the measurement is the whole design: `clientContent` with
    `turnComplete: true` came back with 138 kB of audio and a transcript, and the
    same payload with `false` came back with nothing at all. That is what makes it
    the only way to hand recall to a voice turn without the daemon reading old
    conversations aloud."""
    memory = "[recalled-memory:ab12] 치과 예약은 8월 5일 오후 3시 [end-recalled-memory:ab12]"
    connection = Hanging(SETUP_COMPLETE)  # the server answers nothing, ever
    async with session(connection) as live:
        try:
            async with asyncio.timeout(1.0):
                await live.send_context(memory)
        except TimeoutError:
            pytest.fail("send_context waited for a response that the protocol never sends")

    (client,) = connection.messages("clientContent")
    assert client["turnComplete"] is False, "turnComplete: true makes the daemon answer itself"
    assert client["turns"] == [{"role": "user", "parts": [{"text": memory}]}]
    # Not the prompt path: `realtimeInput.text` triggers generation.
    assert connection.messages("realtimeInput") == []


async def test_recall_and_a_prompt_do_not_travel_the_same_way() -> None:
    """Both are text and only one is a request. If they arrived as the same message
    the daemon would narrate a memory the user never asked about."""
    connection = FakeConnection(SETUP_COMPLETE)
    async with session(connection) as live:
        await live.send_context("어제 치과 얘기를 했다")
        await live.send_text("자기 전에 물 한 잔 마셔")

    assert len(connection.messages("clientContent")) == 1
    assert connection.messages("realtimeInput") == [{"text": "자기 전에 물 한 잔 마셔"}]


async def test_empty_audio_text_and_context_are_not_sent() -> None:
    connection = FakeConnection(SETUP_COMPLETE)
    async with session(connection) as live:
        await live.send_audio(b"")
        await live.send_text("   ")
        await live.send_context("  \n ")

    assert connection.messages("realtimeInput") == []
    assert connection.messages("clientContent") == []


async def test_korean_text_round_trips_through_a_turn() -> None:
    korean = "내일 아침에 우산 챙겨. 비 온다더라 ☔"
    connection = FakeConnection(
        SETUP_COMPLETE,
        said("assistant", korean),
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        await live.send_text(korean)
        received = await drain(live)

    assert connection.messages("realtimeInput")[0]["text"] == korean
    assert received == [Transcript(text=korean, role="assistant", final=True)]


# --- interruption -----------------------------------------------------------


async def test_interrupt_drops_audio_from_the_abandoned_turn() -> None:
    connection = FakeConnection(
        SETUP_COMPLETE,
        audio_frame(b"heard"),
        audio_frame(b"still generating"),
        {"serverContent": {"turnComplete": True}},
        audio_frame(b"next turn"),
    )
    async with session(connection) as live:
        stream = live.receive()
        assert await stream.__anext__() == b"heard"
        await live.interrupt()
        # Everything queued behind the interruption is dropped, and the turn ends
        # where it would have ended anyway.
        assert [item async for item in stream] == []
        # The next turn is the next call, and it is heard normally.
        assert await drain(live) == [b"next turn"]


async def test_interrupting_a_silence_does_not_mute_the_next_answer() -> None:
    """The audit finding: `_dropping` outlived the turn it was set for. An
    interrupt arriving while nothing was being generated dropped the *whole* next
    answer's audio - and its transcript still accumulated and still flushed, so
    memory held a reply the user never heard. That is worse than either half."""
    connection = FakeConnection(
        SETUP_COMPLETE,
        audio_frame(b"the answer"),
        said("assistant", "여덟 시야"),
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        # Nobody is talking, and nothing is being generated.
        await live.interrupt()
        received = await drain(live)

    assert [item for item in received if isinstance(item, bytes)] == [b"the answer"], (
        "the next turn went mute while its transcript was still recorded"
    )
    assert [item.text for item in received if isinstance(item, Transcript)] == ["여덟 시야"]


async def test_server_side_interruption_drops_the_rest_of_the_turn() -> None:
    """The server confirms a barge-in with `interrupted`, sometimes in the same
    event as more audio - which must not slip out ahead of the flag."""
    connection = FakeConnection(
        SETUP_COMPLETE,
        audio_frame(b"early"),
        {
            "serverContent": {
                "interrupted": True,
                "modelTurn": {"parts": [{"inlineData": {"data": base64.b64encode(b"late")
                                                        .decode("ascii")}}]},
            }
        },
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        # The marker is handed over as well as acted on: dropping the rest of the
        # stream is only half a barge-in, and the caller owns the speaker whose
        # buffer is the other half.
        assert await drain(live) == [b"early", Interrupted()]


async def test_an_interruption_after_generation_finished_is_ignored() -> None:
    """Measured against the live API, four runs: `interrupted` arrives about 0.25s
    after `generationComplete` on a turn nobody interrupted - every time. Handed on,
    it makes the caller drop a complete answer out of the speaker mid-playback."""
    connection = FakeConnection(
        SETUP_COMPLETE,
        audio_frame(b"the whole answer"),
        {"serverContent": {"generationComplete": True}},
        {"serverContent": {"interrupted": True}},
        {"serverContent": {"turnComplete": True}},
    )
    async with session(connection) as live:
        first = await drain(live)
        assert Interrupted() not in first, (
            "an interruption with nothing being generated reached the caller, and "
            "the caller's answer to it is to empty the speaker"
        )
        assert b"the whole answer" in first


# --- failure and backoff ----------------------------------------------------


async def test_transient_connect_failures_back_off_then_succeed(
    no_real_sleep: list[float],
) -> None:
    live = GeminiLiveSession(
        KEY,
        MODEL,
        connect=connector(
            OSError("no route to host"),
            TimeoutError("handshake timed out"),
            FakeConnection(SETUP_COMPLETE),
        ),
    )
    async with live:
        pass

    assert no_real_sleep == [1.0, 2.0]


async def test_connect_failures_are_bounded(no_real_sleep: list[float]) -> None:
    """A voice turn that cannot open must fail so the caller can fall back to
    text - not retry into a per-minute-billed connection forever."""
    with pytest.raises(GeminiLiveError, match="could not connect"):
        async with session(OSError("still down"), max_attempts=3):
            pass  # pragma: no cover

    assert no_real_sleep == [1.0, 2.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_auth_failure_is_raised_immediately_and_never_retried(
    status: int, no_real_sleep: list[float]
) -> None:
    class Response:
        status_code = status
        headers: dict[str, str] = {}

    with pytest.raises(GeminiLiveError) as caught:
        async with session(InvalidStatus(Response()), max_attempts=10):  # type: ignore[arg-type]
            pass  # pragma: no cover

    assert caught.value.permanent
    assert no_real_sleep == []  # not one retry


@pytest.mark.parametrize(
    ("code", "reason", "permanent"),
    [
        # Measured, both of them, against the real API - and 1008 means the opposite
        # thing in each. Classifying on the code alone got the second one wrong in
        # the direction that leaves the daemon mute.
        (1008, f"models/{MODEL} is not found for API version v1beta", True),
        (1008, "The operation was aborted.", False),
        (1008, "", False),
        (1008, "API key not valid. Please pass a valid API key.", True),
        (1007, 'Unknown name "responseModalities"', True),
        (1011, "internal error", False),
        (1006, "", False),
    ],
)
async def test_close_codes_are_classified_by_code_and_reason(
    code: int, reason: str, permanent: bool
) -> None:
    connection = FakeConnection(closed(code, reason))
    with pytest.raises(GeminiLiveError) as caught:
        async with session(connection, max_attempts=1):
            pass  # pragma: no cover

    assert caught.value.permanent is permanent


async def test_an_idle_abort_is_retried_and_a_missing_model_is_not(
    no_real_sleep: list[float],
) -> None:
    """The behavioural half of the classification. "The operation was aborted." is
    an idle timeout wearing a policy-violation code: reconnecting works. A model
    that is not found is config, and no number of attempts will find it."""
    with pytest.raises(GeminiLiveError, match="1008"):
        async with session(closed(1008, "The operation was aborted."), max_attempts=3):
            pass  # pragma: no cover
    assert no_real_sleep == [1.0, 2.0], "an idle abort was treated as permanent"

    no_real_sleep.clear()
    missing = f"models/{MODEL} is not found for API version v1beta"
    with pytest.raises(GeminiLiveError, match="is not found") as caught:
        async with session(closed(1008, missing), max_attempts=3):
            pass  # pragma: no cover

    assert caught.value.permanent
    assert no_real_sleep == [], "a wrong model id was retried"


async def test_backoff_does_not_overflow_after_a_thousand_failures(
    no_real_sleep: list[float],
) -> None:
    """telegram.py had exactly this bug: min() bounded the delay but the exponent
    kept growing, and after about 1024 failures `2 ** n` raised OverflowError from
    inside the retry handler - killing the loop when the network came back."""
    for failures in (1, 10, 1_000, 10_000, 2**20):
        assert gemini_live._backoff_delay(failures) <= gemini_live.BACKOFF_MAX_SECONDS

    with pytest.raises(GeminiLiveError):
        async with session(OSError("down"), max_attempts=1_500):
            pass  # pragma: no cover

    assert len(no_real_sleep) == 1_499
    assert max(no_real_sleep) == gemini_live.BACKOFF_MAX_SECONDS


async def test_send_after_the_server_hangs_up_surfaces_as_a_session_error() -> None:
    connection = FakeConnection(SETUP_COMPLETE)
    async with session(connection) as live:
        await connection.close()
        with pytest.raises(GeminiLiveError, match="connection closed"):
            await live.send_text("아직 있어?")


async def test_exiting_the_session_closes_the_connection() -> None:
    """Sessions are billed per minute, so a session left open is pure cost
    (docs/PLAN.md 6.5)."""
    connection = FakeConnection(SETUP_COMPLETE)
    async with session(connection):
        assert not connection.closed
    assert connection.closed


async def test_a_failed_handshake_does_not_leave_a_connection_open() -> None:
    connection = FakeConnection()  # opens, never sends setupComplete
    with pytest.raises(GeminiLiveError):
        async with session(connection, max_attempts=1):
            pass  # pragma: no cover

    assert connection.closed


# --- TLS trust ---------------------------------------------------------------
# Voice failed 100% of the time for anyone who installed Python from python.org:
# those framework builds do not read the system keychain, so
# `ssl.create_default_context()` holds zero CAs until "Install
# Certificates.command" is run - and websockets uses exactly that default. Text
# worked throughout, because httpx bundles certifi. Measured on a fresh 3.13
# framework build: 0 CAs in the default context, 121 in certifi.


@pytest.fixture(autouse=True)
def fresh_ssl_cache() -> Any:
    """The context is cached per CA bundle, so a test that changes the bundle must
    not inherit or leave one."""
    gemini_live._ssl_context.cache_clear()
    yield
    gemini_live._ssl_context.cache_clear()


async def test_the_connection_never_relies_on_the_default_trust_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A default context is not merely unhelpful here, it is empty. So assert that
    every context we build names a CA file - an argument-less call is the bug."""
    empty = ssl_module.create_default_context()
    empty.load_default_certs()  # whatever this machine has; possibly nothing
    calls: list[Any] = []

    def spy(*args: Any, **kwargs: Any) -> ssl_module.SSLContext:
        calls.append(kwargs.get("cafile"))
        return real(*args, **kwargs)

    real = ssl_module.create_default_context
    monkeypatch.setattr(ssl_module, "create_default_context", spy)

    connect = connector(FakeConnection(SETUP_COMPLETE))
    async with GeminiLiveSession(KEY, MODEL, connect=connect):
        pass

    assert calls and all(cafile for cafile in calls), (
        "a context was built with no CA file - the python.org build has none"
    )
    (_url, _headers, context) = connect.calls[0]
    assert isinstance(context, ssl_module.SSLContext)
    assert context.get_ca_certs(), "the trust store we handed websockets is empty"
    assert context.verify_mode == ssl_module.CERT_REQUIRED


def test_the_trust_store_is_certifi_unless_the_environment_names_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read from the variables a proxy user has already set for httpx, requests or
    curl, rather than inventing a Daemon-only setting to get wrong separately."""
    import certifi

    for name in gemini_live.CA_BUNDLE_ENV:
        monkeypatch.delenv(name, raising=False)
    assert gemini_live._ca_bundle() == certifi.where()

    monkeypatch.setenv(gemini_live.CA_BUNDLE_ENV[0], "/etc/corp/ca.pem")
    assert gemini_live._ca_bundle() == "/etc/corp/ca.pem"


async def test_an_injected_context_is_the_one_that_reaches_the_socket() -> None:
    """A corporate proxy that re-signs TLS needs its own CA, and it must not need a
    switch that turns verification off to get one."""
    mine = ssl_module.create_default_context()
    connect = connector(FakeConnection(SETUP_COMPLETE))
    async with GeminiLiveSession(KEY, MODEL, connect=connect, ssl_context=mine):
        pass

    (_url, _headers, context) = connect.calls[0]
    assert context is mine


def test_the_context_is_built_once_and_reused() -> None:
    """Building one parses a ~200 kB PEM file, and a session is opened per
    proactive utterance."""
    first = gemini_live._ssl_context(gemini_live._ca_bundle())
    assert gemini_live._ssl_context(gemini_live._ca_bundle()) is first


async def test_a_tls_failure_says_what_to_do_about_it() -> None:
    """"could not connect" sent people looking at their own network. The cause is a
    trust store and the fix is naming a different one, so the message has to say
    both - and still not say the key."""
    refused = ssl_module.SSLCertVerificationError(
        f"[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get "
        f"local issuer certificate while sending x-goog-api-key: {KEY}"
    )
    with pytest.raises(GeminiLiveError) as caught:
        async with session(refused, max_attempts=1):
            pass  # pragma: no cover

    message = str(caught.value)
    assert "TLS verification failed" in message
    assert "cacert.pem" in message or "ca.pem" in message, "the bundle in use is not named"
    assert gemini_live.CA_BUNDLE_ENV[0] in message, "no actionable next step"
    assert KEY not in message
    assert KEY not in repr(caught.value)
    assert caught.value.__context__ is None


# --- the API key -------------------------------------------------------------


async def test_api_key_never_appears_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    """websockets logs the handshake headers at DEBUG, and the key is a header."""
    leaky = OSError(f"handshake failed, sent x-goog-api-key: {KEY}")
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("websockets.client").debug("> x-goog-api-key: %s", KEY)
        with pytest.raises(GeminiLiveError):
            async with session(leaky, max_attempts=1):
                pass  # pragma: no cover

    assert KEY not in caplog.text
    assert "<key>" in caplog.text


async def test_api_key_never_appears_in_an_exception_or_its_context() -> None:
    leaky = OSError(f"cannot reach wss://...?key={KEY}")
    with pytest.raises(GeminiLiveError) as caught:
        async with session(leaky, max_attempts=1):
            pass  # pragma: no cover

    exc = caught.value
    assert KEY not in str(exc)
    assert KEY not in repr(exc)
    # __cause__ alone is not enough: anything that walks the chain - an error
    # reporter, traceback.format_exception, a pytest failure report - would print
    # __context__ and the key with it.
    assert exc.__cause__ is None
    assert exc.__context__ is None


async def test_api_key_never_appears_in_a_close_reason() -> None:
    connection = FakeConnection(closed(1008, f"API key {KEY} was reported as leaked"))
    with pytest.raises(GeminiLiveError) as caught:
        async with session(connection, max_attempts=1):
            pass  # pragma: no cover

    assert KEY not in str(caught.value)
    assert KEY not in repr(caught.value)
    assert caught.value.__context__ is None


async def test_closing_removes_the_log_filter_it_installed() -> None:
    """A filter left on the root handlers would outlive the session and keep a
    reference to the key."""
    root = logging.getLogger()
    before = [len(h.filters) for h in root.handlers]
    async with session(FakeConnection(SETUP_COMPLETE)):
        pass

    assert [len(h.filters) for h in root.handlers] == before


async def test_a_session_that_never_opens_still_removes_its_log_filter() -> None:
    """__aexit__ does not run when __aenter__ raises, so this is the path where a
    filter holding the key would be left behind."""
    root = logging.getLogger()
    before = [len(h.filters) for h in root.handlers]
    with pytest.raises(GeminiLiveError):
        async with session(OSError("down"), max_attempts=1):
            pass  # pragma: no cover

    assert [len(h.filters) for h in root.handlers] == before
