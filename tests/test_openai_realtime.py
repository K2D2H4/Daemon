"""OpenAI Realtime session tests. No network, no audio hardware, no API key.

Mirrors tests/test_voice.py's fake-socket harness for the sibling provider
(daemon/voice/gemini_live.py). This file covers only what Task 2 builds: connect,
`session.update`, and `send_audio`'s 16k->24k upsample. `receive`/`send_*`/
`interrupt` land in later tasks.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math

import pytest

from daemon.llm.base import ToolCall, ToolSpec
from daemon.voice.base import Interrupted, Transcript, VoiceSession
from daemon.voice.openai_realtime import (
    OpenAIRealtimeError,
    OpenAIRealtimeSession,
    _permanent_close,
    _resample_16k_to_24k,
)

KEY, MODEL = "sk-test", "gpt-realtime"
SESSION_UPDATED = {"type": "session.updated"}
HANG = object()
"""A scripted item meaning "never deliver another message" - for proving a wait
is actually bounded, rather than the fake socket just running out of script."""


def _a_tool_spec_named(name: str) -> ToolSpec:
    return ToolSpec(
        name=name, description="A tool.", parameters={"type": "object", "properties": {}}
    )


class FakeConnection:
    def __init__(self, *scripted):
        self.scripted = list(scripted)
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def close(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.scripted:
            raise StopAsyncIteration
        item = self.scripted.pop(0)
        if item is HANG:
            # Never resolves; the caller's own bounded wait is what ends the test.
            await asyncio.Event().wait()
        if isinstance(item, Exception):
            raise item
        return json.dumps(item)

    def sent_of_type(self, t):
        return [m for m in self.sent if m.get("type") == t]


def connector(*conns):
    q = list(conns)

    async def connect(url, additional_headers=None, ssl=None):
        connect.calls.append((url, additional_headers))
        # Reuse the last item once the queue would otherwise run dry, so a
        # scripted exception can stand for every retry attempt, not just one.
        item = q.pop(0) if len(q) > 1 else q[0]
        if isinstance(item, BaseException):
            raise item
        return item

    connect.calls = []
    return connect


def make(*conns, **kw):
    return OpenAIRealtimeSession(KEY, MODEL, connect=connector(*conns), **kw)


def test_satisfies_voice_session_protocol():
    assert isinstance(OpenAIRealtimeSession(KEY, MODEL), VoiceSession)


def test_missing_credentials_are_rejected_before_any_connection():
    with pytest.raises(ValueError):
        OpenAIRealtimeSession("", MODEL)
    with pytest.raises(ValueError):
        OpenAIRealtimeSession(KEY, "")


def _sine_pcm16(freq_hz: float, n_samples: int, amplitude: int = 3000) -> bytes:
    return b"".join(
        int(amplitude * math.sin(2 * math.pi * freq_hz * i / 16_000)).to_bytes(
            2, "little", signed=True
        )
        for i in range(n_samples)
    )


def test_resample_grows_length_by_3_over_2_across_a_full_stream():
    """Fed through in realistic 20ms-mic-chunk pieces, carrying phase and tail
    across calls the way `send_audio` does, total output length matches the
    3/2 ratio to within one sample - the invariant a chunk-boundary bug could
    break by dropping or duplicating a sample at each boundary, even though it
    would not show up in a single-chunk test."""
    n_total = 1600  # 100ms at 16kHz
    pcm = _sine_pcm16(300, n_total)
    phase, tail = 0.0, b""
    out = bytearray()
    for i in range(0, len(pcm), 640):  # 320 samples (20ms) per chunk, like the mic
        chunk_out, phase, tail = _resample_16k_to_24k(pcm[i : i + 640], phase, tail)
        out += chunk_out
    out_samples = len(out) // 2
    assert abs(out_samples - (n_total * 3) // 2) <= 1


def test_resample_is_continuous_across_chunk_boundaries():
    """The assertion the endpoint-anchored implementation failed: resampling a
    signal in many small chunks (phase/tail carried, as `send_audio` does)
    must match resampling the SAME signal in one call, because the
    interpolation is one continuous fixed-step process, not a fresh stretch
    per chunk. The old implementation anchored each chunk's first/last input
    sample to its own first/last output sample, which reset the phase every
    ~20ms and measurably hurt SNR (300 Hz 32.9 dB vs 58.0 dB; see
    `_resample_16k_to_24k`'s docstring) - this is the test that would have
    caught it."""
    n_total = 1600
    pcm = _sine_pcm16(1000, n_total)

    whole, _, _ = _resample_16k_to_24k(pcm, 0.0, b"")

    phase, tail = 0.0, b""
    chunked = bytearray()
    for i in range(0, len(pcm), 640):
        chunk_out, phase, tail = _resample_16k_to_24k(pcm[i : i + 640], phase, tail)
        chunked += chunk_out

    n = min(len(whole), len(chunked)) // 2
    assert n > 0
    whole_vals = [
        int.from_bytes(whole[j * 2 : j * 2 + 2], "little", signed=True) for j in range(n)
    ]
    chunked_vals = [
        int.from_bytes(chunked[j * 2 : j * 2 + 2], "little", signed=True) for j in range(n)
    ]
    diffs = [abs(a - b) for a, b in zip(whole_vals, chunked_vals, strict=True)]
    assert max(diffs) <= 2, "chunked resampling drifted from the single-call reference"


async def test_session_update_uses_the_ga_shape():
    # The GA gpt-realtime shape, confirmed against the live socket: audio config nested
    # under audio.{input,output}, output_modalities, format objects, voice under output.
    # The old flat beta shape (input_audio_format="pcm16", top-level voice) is rejected
    # 4000 by gpt-realtime.
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn, system_instruction="be brief", voice_name="alloy"):
        pass
    (upd,) = conn.sent_of_type("session.update")
    s = upd["session"]
    assert s["type"] == "realtime"
    assert s["output_modalities"] == ["audio"]
    assert s["instructions"] == "be brief"
    assert s["audio"]["output"]["voice"] == "alloy"
    assert s["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert s["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert s["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert s["audio"]["input"]["transcription"]["model"] == "whisper-1"
    # No beta-shape leftovers at the top level.
    assert "voice" not in s and "input_audio_format" not in s and "modalities" not in s


async def test_no_voice_omits_the_voice_field():
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn):  # no voice_name -> server default
        pass
    (upd,) = conn.sent_of_type("session.update")
    assert "voice" not in upd["session"]["audio"]["output"]


async def test_send_audio_appends_resampled_base64_across_calls():
    """Two mic chunks on the same session: the resample state (phase, tail)
    must carry across the two `send_audio` calls, not reset each time - the
    live bug (Finding 2, PR #79 review). Total output for 4 input samples
    should land within one sample of the 3/2 ratio (6 samples)."""
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn) as live:
        await live.send_audio(b"\x00\x01\x02\x03")  # 2 samples
        await live.send_audio(b"\x04\x05\x06\x07")  # 2 more samples
    appends = conn.sent_of_type("input_audio_buffer.append")
    total_samples = sum(len(base64.b64decode(a["audio"])) for a in appends) // 2
    assert abs(total_samples - 6) <= 1


async def test_send_audio_buffers_a_lone_sample_until_the_next_chunk():
    """A chunk with fewer than two samples cannot be interpolated yet - it must
    be held (in the session's resample tail) rather than dropped, and produce
    no `input_audio_buffer.append` of its own."""
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn) as live:
        await live.send_audio(b"\x00\x01")  # 1 sample: nothing to interpolate to yet
        assert conn.sent_of_type("input_audio_buffer.append") == []
        await live.send_audio(b"\x02\x03")  # completes the pair
    appends = conn.sent_of_type("input_audio_buffer.append")
    assert len(appends) == 1
    assert len(base64.b64decode(appends[0]["audio"])) > 0


async def test_bearer_auth_header_is_sent():
    conn = FakeConnection(SESSION_UPDATED)
    connect = connector(conn)
    async with OpenAIRealtimeSession(KEY, MODEL, connect=connect):
        pass
    _, headers = connect.calls[0]
    assert headers["Authorization"] == f"Bearer {KEY}"
    # GA drops the beta header; re-adding it makes gpt-realtime close 4000.
    assert "OpenAI-Beta" not in headers


async def collect(live):
    return [item async for item in live.receive()]


def audio_delta(b):
    return {"type": "response.output_audio.delta", "delta": base64.b64encode(b).decode()}


async def test_receive_yields_audio_then_transcripts_then_ends_on_response_done():
    conn = FakeConnection(
        SESSION_UPDATED,
        audio_delta(b"\x01\x02"),
        {"type": "response.output_audio_transcript.done", "transcript": "안녕하세요"},
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "뭐 해"},
        {"type": "response.done"},
    )
    async with make(conn) as live:
        items = await collect(live)
    assert b"\x01\x02" in [i for i in items if isinstance(i, bytes)]
    texts = {(t.role, t.text, t.final) for t in items if isinstance(t, Transcript)}
    assert ("assistant", "안녕하세요", True) in texts
    assert ("user", "뭐 해", True) in texts
    # No double-emit: the user transcript arrived before `response.done` here,
    # and it must be released exactly once, not once from the immediate path
    # and again from `_flush`.
    assert sum(1 for t in texts if t[0] == "user") == 1


async def test_user_transcript_arriving_after_response_done_lands_before_the_reply():
    # The reproduced live defect. `input_audio_buffer.committed` tells `_decode`
    # a user transcript is OWED for this turn; whisper-1 then emits only
    # `input_audio_transcription.completed` - a full transcript, no delta stream
    # - and it commonly arrives *after* `response.done` (~200ms, measured).
    # Losing it to the next turn - or misfiling it as belonging to the next
    # turn's own speech - would record the assistant's reply BEFORE the user's
    # question that prompted it: the exact ordering bug gemini_live.py's
    # `_flush` docstring calls out ("user first: that is the order it happened
    # in"). `receive()`'s grace wait (USER_TRANSCRIPT_GRACE_SECONDS) - paid only
    # because a transcript is owed - is what keeps this in the same turn, user
    # transcript yielded before the assistant's.
    conn = FakeConnection(
        SESSION_UPDATED,
        {"type": "input_audio_buffer.committed", "item_id": "item_1"},
        audio_delta(b"\x01\x02"),
        {"type": "response.output_audio_transcript.done", "transcript": "hello"},
        {"type": "response.done"},
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1",
            "transcript": "뭐 해",
        },
    )
    async with make(conn) as live:
        items = await collect(live)
    texts = [(t.role, t.text, t.final) for t in items if isinstance(t, Transcript)]
    assert texts == [("user", "뭐 해", True), ("assistant", "hello", True)]
    # No double-emit: the late arrival is released exactly once, from the grace
    # wait, not again from the assistant `_flush`.
    assert sum(1 for role, _, _ in texts if role == "user") == 1


async def test_user_transcript_arriving_before_response_done_is_not_double_emitted():
    # The early-arrival counterpart of the test above: `…completed` lands
    # before `response.done` this time. It must still be one turn, user before
    # assistant, with no double-emit - the accumulate-then-flush structure
    # should not care which order the two events happened to arrive in.
    conn = FakeConnection(
        SESSION_UPDATED,
        {"type": "input_audio_buffer.committed", "item_id": "item_1"},
        audio_delta(b"\x01\x02"),
        {"type": "response.output_audio_transcript.done", "transcript": "hello"},
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1",
            "transcript": "뭐 해",
        },
        {"type": "response.done"},
    )
    async with make(conn) as live:
        items = await collect(live)
    texts = [(t.role, t.text, t.final) for t in items if isinstance(t, Transcript)]
    assert texts == [("user", "뭐 해", True), ("assistant", "hello", True)]
    assert sum(1 for role, _, _ in texts if role == "user") == 1


async def test_turn_with_no_user_transcription_completes_within_the_grace_window():
    # A turn where the user never spoke (e.g. a send_text-only turn) never gets
    # `input_audio_buffer.committed`, so nothing is OWED and `receive()` must not
    # wait at all. `HANG` makes the fake socket never deliver another message; a
    # bound well under USER_TRANSCRIPT_GRACE_SECONDS (0.5s) is what proves the
    # wait was skipped entirely - a regression that waits unconditionally would
    # blow through this timeout instead of finishing.
    conn = FakeConnection(
        SESSION_UPDATED,
        {"type": "response.output_audio_transcript.done", "transcript": "hi"},
        {"type": "response.done"},
        HANG,
    )
    async with make(conn) as live:
        async with asyncio.timeout(0.2):
            items = await collect(live)
    texts = [t for t in items if isinstance(t, Transcript)]
    assert texts == [Transcript(text="hi", role="assistant", final=True)]


async def test_consecutive_turns_do_not_bleed_user_transcripts():
    # Two full turns on one connection, each with its own `committed` +
    # `completed`. Turn 1's user transcript must not appear in turn 2's result,
    # and turn 2 must not inherit any state - `owed`, `committed_item_id` -
    # left over from turn 1.
    conn = FakeConnection(
        SESSION_UPDATED,
        # Turn 1
        {"type": "input_audio_buffer.committed", "item_id": "item_1"},
        audio_delta(b"\x01\x02"),
        {"type": "response.output_audio_transcript.done", "transcript": "first reply"},
        {"type": "response.done"},
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1",
            "transcript": "첫 turn",
        },
        # Turn 2
        {"type": "input_audio_buffer.committed", "item_id": "item_2"},
        audio_delta(b"\x03\x04"),
        {"type": "response.output_audio_transcript.done", "transcript": "second reply"},
        {"type": "response.done"},
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_2",
            "transcript": "두번째 turn",
        },
    )
    async with make(conn) as live:
        turn1 = [
            (t.role, t.text, t.final) for t in await collect(live) if isinstance(t, Transcript)
        ]
        turn2 = [
            (t.role, t.text, t.final) for t in await collect(live) if isinstance(t, Transcript)
        ]
    assert turn1 == [("user", "첫 turn", True), ("assistant", "first reply", True)]
    assert turn2 == [("user", "두번째 turn", True), ("assistant", "second reply", True)]


async def test_partial_transcripts_yields_one_completed_utterance_per_turn():
    """Finding 3 (PR #79 review): whisper-1 sends no deltas, so what actually
    reaches `partial_transcripts()` is the complete, already-final transcript
    from `…completed`, offered as `final=False`. This drives one turn and
    proves exactly that - one item, carrying the full text, not a growing
    fragment - matching what the docstring now says rather than what it used
    to claim ("one item per delta")."""
    conn = FakeConnection(
        SESSION_UPDATED,
        {"type": "input_audio_buffer.committed", "item_id": "item_1"},
        {"type": "response.output_audio_transcript.done", "transcript": "hello"},
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1",
            "transcript": "안녕하세요 반갑습니다",
        },
        {"type": "response.done"},
    )
    live = make(conn)
    async with live:
        await collect(live)  # drive the turn so `_decode` pushes to the partial queue
        gen = live.partial_transcripts()
        async with asyncio.timeout(0.2):
            first = await anext(gen)
    # The session is now closed (the sentinel was pushed by `close()`), so
    # draining the rest of the generator must end immediately - proving
    # nothing else was queued this turn, i.e. one item per turn, not per delta.
    async with asyncio.timeout(0.2):
        rest = [item async for item in gen]
    assert first == Transcript(text="안녕하세요 반갑습니다", role="user", final=False)
    assert rest == []


async def test_speech_started_is_a_barge_in():
    conn = FakeConnection(
        SESSION_UPDATED,
        audio_delta(b"\x01\x02"),
        {"type": "input_audio_buffer.speech_started"},
        {"type": "response.done"},
    )
    async with make(conn) as live:
        items = await collect(live)
    assert any(isinstance(i, Interrupted) for i in items)


async def test_speech_started_with_nothing_generating_is_not_a_barge_in():
    conn = FakeConnection(
        SESSION_UPDATED,
        {"type": "input_audio_buffer.speech_started"},
        {"type": "response.done"},
    )
    async with make(conn) as live:
        items = await collect(live)
    assert not any(isinstance(i, Interrupted) for i in items)


async def test_function_call_becomes_a_toolcall():
    conn = FakeConnection(
        SESSION_UPDATED,
        {"type": "response.output_item.added",
         "item": {"type": "function_call", "name": "open_path", "call_id": "c1"}},
        {
            "type": "response.function_call_arguments.done",
            "call_id": "c1",
            "arguments": '{"path": "/tmp"}',
        },
        {"type": "response.done"},
    )
    async with make(conn, tools=(_a_tool_spec_named("open_path"),)) as live:
        items = await collect(live)
    calls = [i for i in items if isinstance(i, ToolCall)]
    assert calls and calls[0].name == "open_path" and calls[0].arguments == {"path": "/tmp"}


async def test_send_context_adds_history_without_a_response():
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn) as live:
        await live.send_context("recall: the user likes tea")
    assert conn.sent_of_type("conversation.item.create")
    assert not conn.sent_of_type("response.create")  # context must not make the model answer


async def test_send_text_prompts_a_response():
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn) as live:
        await live.send_text("say hello")
    assert conn.sent_of_type("conversation.item.create")
    assert conn.sent_of_type("response.create")


async def test_send_tool_response_emits_function_output_then_response():
    from daemon.tools.base import ToolResult
    conn = FakeConnection(SESSION_UPDATED)
    result = ToolResult(call_id="c1", name="open_path", ok=True, content="done")
    async with make(conn) as live:
        await live.send_tool_response([result])
    outs = conn.sent_of_type("conversation.item.create")
    assert any(
        m["item"]["type"] == "function_call_output" and m["item"]["call_id"] == "c1"
        for m in outs
    )
    assert conn.sent_of_type("response.create")


async def test_send_frame_is_a_noop():
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn) as live:
        await live.send_frame(b"\xff\xd8jpeg")
    assert conn.sent == [conn.sent[0]] if conn.sent else True  # nothing beyond session.update
    assert not conn.sent_of_type("input_audio_buffer.append")


async def test_setup_error_fails_fast_instead_of_hanging_for_20s():
    # OpenAI reports an invalid session.update via a top-level {"type": "error"}
    # event and leaves the socket open - it never sends session.updated. Before
    # the fix this fell through to the SETUP_TIMEOUT_SECONDS (20s) wait and then
    # raised a misleading "no session.updated" message. The fake socket returns
    # this as the very first message, so a passing test here is proof the error
    # is caught immediately rather than proof of a fast clock.
    conn = FakeConnection({"type": "error", "error": {"message": "unknown parameter"}})
    with pytest.raises(OpenAIRealtimeError) as exc_info:
        async with make(conn):
            pass
    assert "unknown parameter" in str(exc_info.value)


async def test_closing_removes_the_log_filter_it_installed():
    """A filter left on the root handlers would outlive the session and keep a
    reference to the key."""
    root = logging.getLogger()
    before = [len(h.filters) for h in root.handlers]
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn):
        pass
    assert [len(h.filters) for h in root.handlers] == before


async def test_a_session_that_never_opens_still_removes_its_log_filter():
    """__aexit__ does not run when __aenter__ raises, so this is the path where a
    filter holding the key would be left behind - the failed-handshake path that
    historically leaked one log filter per retry attempt."""
    root = logging.getLogger()
    before = [len(h.filters) for h in root.handlers]
    with pytest.raises(OpenAIRealtimeError):
        async with make(OSError("down"), max_attempts=1):
            pass  # pragma: no cover
    assert [len(h.filters) for h in root.handlers] == before


def test_close_code_classification():
    # gemini_live.py's sibling measured 1008 as an idle timeout wearing a
    # policy-violation code; classifying it permanent by code alone defeats
    # reconnect for something that was just weather. OpenAI's 1008 semantics are
    # unconfirmed against a live socket, so only 1007 and 4000 are permanent.
    assert _permanent_close(1007, "") is True
    # 4000 invalid_request (measured: a beta-shaped session.update closes 4000) is a
    # malformed request - retrying dials into the same close.
    assert _permanent_close(4000, "invalid_request_error.beta_api_shape_disabled") is True
    assert _permanent_close(1008, "anything") is False
