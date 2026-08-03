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
from typing import Any

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, InvalidStatus
from websockets.frames import Close

from daemon.voice import gemini_live
from daemon.voice.base import Transcript, VoiceSession
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


def connector(*connections: Any) -> Any:
    """A stand-in for `websockets.connect`: hands out scripted connections, or
    raises a scripted exception instead of connecting."""
    queue = list(connections)

    async def connect(url: str, *, additional_headers: dict[str, str]) -> Any:
        connect.calls.append((url, additional_headers))  # type: ignore[attr-defined]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item

    connect.calls = []  # type: ignore[attr-defined]
    return connect


def session(*connections: Any, **kwargs: Any) -> GeminiLiveSession:
    return GeminiLiveSession(KEY, MODEL, connect=connector(*connections), **kwargs)


async def drain(live: GeminiLiveSession) -> list[bytes | Transcript]:
    async with asyncio.timeout(5):
        return [item async for item in live.receive()]


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


async def test_api_key_travels_as_a_header_not_in_the_url() -> None:
    connect = connector(FakeConnection(SETUP_COMPLETE))
    async with GeminiLiveSession(KEY, MODEL, connect=connect):
        pass

    (url, headers) = connect.calls[0]
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

    assert [item.text for item in received] == ["다 했어"]


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


async def test_empty_audio_and_text_are_not_sent() -> None:
    connection = FakeConnection(SETUP_COMPLETE)
    async with session(connection) as live:
        await live.send_audio(b"")
        await live.send_text("   ")

    assert connection.messages("realtimeInput") == []


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
        # Everything queued behind the interruption is dropped, and the next turn
        # is heard normally.
        assert [item async for item in stream] == [b"next turn"]


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
        assert await drain(live) == [b"early"]


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
    ("code", "permanent"),
    [(1008, True), (1007, True), (1011, False), (1006, False)],
)
async def test_close_codes_are_classified(code: int, permanent: bool) -> None:
    connection = FakeConnection(closed(code, "rejected"))
    with pytest.raises(GeminiLiveError) as caught:
        async with session(connection, max_attempts=1):
            pass  # pragma: no cover

    assert caught.value.permanent is permanent


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
