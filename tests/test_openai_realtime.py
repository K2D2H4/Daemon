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

import pytest

from daemon.llm.base import ToolCall, ToolSpec
from daemon.voice.base import Interrupted, Transcript, VoiceSession
from daemon.voice.openai_realtime import (
    USER_TRANSCRIPT_GRACE_SECONDS,
    OpenAIRealtimeError,
    OpenAIRealtimeSession,
    _permanent_close,
    _upsample_16k_to_24k,
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


def test_upsample_16k_to_24k_grows_length_by_3_over_2():
    # 4 samples (8 bytes) of 16-bit LE mono -> 6 samples (12 bytes) at 24k.
    pcm = b"".join(int(v).to_bytes(2, "little", signed=True) for v in (0, 100, 200, 300))
    out = _upsample_16k_to_24k(pcm)
    assert len(out) == 12
    assert len(out) % 2 == 0


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


async def test_send_audio_appends_upsampled_base64():
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn) as live:
        await live.send_audio(b"\x00\x01\x02\x03")  # 2 samples 16k -> 3 samples 24k
    (append,) = conn.sent_of_type("input_audio_buffer.append")
    raw = base64.b64decode(append["audio"])
    assert len(raw) == 6  # 3 samples * 2 bytes


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
    # whisper-1 emits only `input_audio_transcription.completed` - a full
    # transcript, no delta stream - and it commonly arrives *after*
    # `response.done` (~200ms, measured). Losing it to the next turn - or
    # misfiling it as belonging to the next turn's own speech - would record the
    # assistant's reply BEFORE the user's question that prompted it: the exact
    # ordering bug gemini_live.py's `_flush` docstring calls out ("user first:
    # that is the order it happened in"). `receive()`'s grace wait
    # (USER_TRANSCRIPT_GRACE_SECONDS) is what keeps this in the same turn, user
    # transcript yielded before the assistant's.
    conn = FakeConnection(
        SESSION_UPDATED,
        audio_delta(b"\x01\x02"),
        {"type": "response.output_audio_transcript.done", "transcript": "hello"},
        {"type": "response.done"},
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "뭐 해"},
    )
    async with make(conn) as live:
        items = await collect(live)
    texts = [(t.role, t.text, t.final) for t in items if isinstance(t, Transcript)]
    assert texts == [("user", "뭐 해", True), ("assistant", "hello", True)]
    # No double-emit: the late arrival is released exactly once, from the grace
    # wait, not again from the assistant `_flush`.
    assert sum(1 for role, _, _ in texts if role == "user") == 1


async def test_turn_with_no_user_transcription_completes_within_the_grace_window():
    # A turn where the user never spoke (e.g. a send_text-only turn, or one
    # where whisper-1 simply never reports anything) has no `…completed` event
    # to wait for. `HANG` makes the fake socket never deliver another message,
    # so a passing test here is proof the internal wait is bounded by
    # USER_TRANSCRIPT_GRACE_SECONDS, not proof that the fake socket ran dry.
    conn = FakeConnection(
        SESSION_UPDATED,
        {"type": "response.output_audio_transcript.done", "transcript": "hi"},
        {"type": "response.done"},
        HANG,
    )
    async with make(conn) as live:
        async with asyncio.timeout(USER_TRANSCRIPT_GRACE_SECONDS + 2.0):
            items = await collect(live)
    texts = [t for t in items if isinstance(t, Transcript)]
    assert texts == [Transcript(text="hi", role="assistant", final=True)]


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
