"""OpenAI Realtime session tests. No network, no audio hardware, no API key.

Mirrors tests/test_voice.py's fake-socket harness for the sibling provider
(daemon/voice/gemini_live.py). This file covers only what Task 2 builds: connect,
`session.update`, and `send_audio`'s 16k->24k upsample. `receive`/`send_*`/
`interrupt` land in later tasks.
"""

from __future__ import annotations

import base64
import json

import pytest

from daemon.voice.openai_realtime import OpenAIRealtimeSession, _upsample_16k_to_24k

KEY, MODEL = "sk-test", "gpt-realtime"
SESSION_UPDATED = {"type": "session.updated"}


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
        if isinstance(item, Exception):
            raise item
        return json.dumps(item)

    def sent_of_type(self, t):
        return [m for m in self.sent if m.get("type") == t]


def connector(*conns):
    q = list(conns)

    async def connect(url, additional_headers=None, ssl=None):
        connect.calls.append((url, additional_headers))
        return q.pop(0)

    connect.calls = []
    return connect


def make(*conns, **kw):
    return OpenAIRealtimeSession(KEY, MODEL, connect=connector(*conns), **kw)


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


async def test_session_update_sets_voice_instructions_and_pcm16():
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn, system_instruction="be brief", voice_name="alloy"):
        pass
    (upd,) = conn.sent_of_type("session.update")
    s = upd["session"]
    assert s["voice"] == "alloy"
    assert s["instructions"] == "be brief"
    assert s["input_audio_format"] == "pcm16" and s["output_audio_format"] == "pcm16"
    assert s["turn_detection"]["type"] == "server_vad"
    assert "input_audio_transcription" in s


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
