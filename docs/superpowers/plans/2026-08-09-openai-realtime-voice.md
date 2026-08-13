# OpenAI Realtime Voice (Phase B1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** With `DAEMON_VOICE_PROVIDER=openai`, `daemon voice` holds a conversation over the OpenAI Realtime WebSocket API instead of Gemini Live, satisfying the same `VoiceSession` protocol so `conversation.py`, memory, tools, and the wake gate are unchanged.

**Architecture:** A new `daemon/voice/openai_realtime.py` implements the `VoiceSession` protocol, structurally mirroring `daemon/voice/gemini_live.py` (connect-with-backoff, an API-key log filter, permanent-vs-transient close classification, transcript accumulation, a turn-boundary `receive()`). `config.py` gains a `voice_provider` dimension so `route_for(CHAT_VOICE)` and `run_voice` pick the provider; `app.run_voice` branches to build the right session. Audio is reconciled by upsampling the 16 kHz mic capture to OpenAI's required 24 kHz inside the session.

**Tech Stack:** Python 3.13, `websockets` (already a dep, injected as `connect=` in tests), `base64`/`json`/`audioop`-free manual PCM resample, pydantic-settings, `fastapi.testclient` not needed here.

## Global Constraints

- **Protocol is frozen and unchanged:** implement exactly the `VoiceSession` methods in `daemon/voice/base.py` — `__aenter__`, `__aexit__`, `send_audio`, `send_frame`, `send_context`, `send_text`, `send_tool_response`, `receive`, `pending_transcripts`, `partial_transcripts`, `interrupt`. No new protocol methods.
- **Layering:** only `daemon/app.py` may import `OpenAIRealtimeSession` (concrete provider). `config.py` must not import `daemon/voice/*`; voice-name/provider allowlists are `config.py` constants.
- **No test may touch the network, a key, a microphone, or a speaker** (tests/CLAUDE.md). Drive the session with an injected fake websocket (`connect=`), exactly like `tests/test_voice.py`. At least one Korean transcript case.
- **Fail early, loud:** a bad `voice_provider`, a missing realtime model/key for the chosen provider, or a bad voice name raises `ConfigError` at `Settings` construction.
- **Audio:** OpenAI Realtime pcm16 is **24 kHz mono, both directions** (input rate fixed). The mic/`AudioIO` capture is 16 kHz — upsample 16→24 kHz inside `send_audio`. Output is 24 kHz, already matching `AudioIO.playback_sample_rate`.
- **Two loop-facing semantics** (the current `conversation.py` depends on them): `send_context` must NOT trigger a response; `Interrupted` fires only on a genuine user barge-in (`input_audio_buffer.speech_started`), never as a side effect of our own sends.
- **`send_frame` is a no-op + one-time warning** (OpenAI Realtime has no realtime video input).
- **Voices (10):** alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar.
- **Event-name drift:** GA (`gpt-realtime`) uses `response.output_audio.delta` / `response.output_audio_transcript.*`; older beta uses `response.audio.delta` / `response.audio_transcript.*`. Accept BOTH names in the decoder. Concrete names are confirmed by the live spike (Task 6).

---

### Task 1: config — the `voice_provider` dimension

**Files:**
- Modify: `daemon/config.py` (constants near line 75–96; fields near line 299–305; `routing` property near 985; `route_for` near 995–1013; `_check` near 752–781)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `VOICE_PROVIDERS: tuple[str,...]`, `OPENAI_REALTIME_VOICES: frozenset[str]`; `Settings.voice_provider: str`, `Settings.openai_realtime_model: str`, `Settings.openai_realtime_voice: str`; `route_for(Task.CHAT_VOICE)` returns `Route(provider=<voice_provider>, model=<that provider's realtime model>)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py` (use the module's `make_settings` helper; `balanced`/`quality` presets need keys, so build voice cases on a preset that routes voice — mirror the existing voice tests' preset choice, e.g. `quality` with the needed keys, or whichever the existing `test_enabling_voice_*` tests use):

```python
def test_voice_provider_defaults_to_gemini() -> None:
    assert make_settings().voice_provider == "gemini"


def test_unknown_voice_provider_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_VOICE_PROVIDER"):
        make_settings(voice_provider="anthropic")


def test_openai_voice_route_uses_the_realtime_model_and_provider() -> None:
    s = make_settings(
        preset="quality", voice_enabled=True, voice_provider="openai",
        openai_realtime_model="gpt-realtime", openai_api_key="sk-o", gemini_api_key="g",
        anthropic_api_key="a",
    )
    route = s.route_for(Task.CHAT_VOICE)
    assert route.provider == "openai"
    assert route.model == "gpt-realtime"


def test_openai_voice_requires_its_own_model_and_key() -> None:
    with pytest.raises(ConfigError, match="DAEMON_OPENAI_REALTIME_MODEL"):
        make_settings(preset="quality", voice_enabled=True, voice_provider="openai",
                      openai_api_key="sk-o", gemini_api_key="g", anthropic_api_key="a")
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        make_settings(preset="quality", voice_enabled=True, voice_provider="openai",
                      openai_realtime_model="gpt-realtime", gemini_api_key="g", anthropic_api_key="a")


def test_unknown_openai_realtime_voice_fails() -> None:
    with pytest.raises(ConfigError, match="DAEMON_OPENAI_REALTIME_VOICE"):
        make_settings(openai_realtime_voice="not-a-voice")
    assert make_settings(openai_realtime_voice="alloy").openai_realtime_voice == "alloy"
    assert make_settings(openai_realtime_voice="").openai_realtime_voice == ""
```

(Adjust `make_settings(...)` kwargs to whatever that helper requires for a valid `quality` preset — read the existing `test_enabling_voice_on_*` tests and copy their setup. The assertions above are the contract.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_config.py -k "voice_provider or openai_realtime or openai_voice" -v`
Expected: FAIL — `Settings` has no `voice_provider` / `openai_realtime_model` / `openai_realtime_voice`.

- [ ] **Step 3: Add constants, fields, routing, route_for, validation**

In `daemon/config.py`, beside `GEMINI_LIVE_VOICES` (near line 75–96):

```python
VOICE_PROVIDERS = ("gemini", "openai")
"""Which hosted native-audio backend a voice session uses. Independent of the text
`hosted_provider`: voice-model availability is a separate axis, and being explicit
turns a mismatch into a startup error, not a first-turn failure."""

OPENAI_REALTIME_VOICES = frozenset({
    "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse",
    "marin", "cedar",
})
"""OpenAI Realtime voices. `marin`/`cedar` are gpt-realtime-only; a voice the chosen
model rejects comes back as a session error, the same class as Gemini's 1007."""
```

Add fields beside `gemini_live_voice` (near line 305):

```python
voice_provider: str = Field(default="gemini", alias="DAEMON_VOICE_PROVIDER")
"""Which native-audio backend voice mode uses: one of VOICE_PROVIDERS. Not derived
from the text hosted_provider."""

openai_realtime_model: str = Field(default="", alias="DAEMON_OPENAI_REALTIME_MODEL")
"""OpenAI Realtime model id (e.g. gpt-realtime), distinct from DAEMON_OPENAI_MODEL:
the realtime endpoint takes its own id. No default - a guessed id fails at the first
voice turn, which is what this module exists to prevent."""

openai_realtime_voice: str = Field(default="", alias="DAEMON_OPENAI_REALTIME_VOICE")
"""Which prebuilt OpenAI voice: one of OPENAI_REALTIME_VOICES, or empty for the
server default. Checked at construction, not on the wire."""
```

In the `routing` property (near line 985), make `CHAT_VOICE` follow `voice_provider`. Add, just before the `return {**resolved, **self.route_overrides}` line:

```python
        # Voice provider is its own axis (DAEMON_VOICE_PROVIDER), not the preset's
        # literal CHAT_VOICE entry. Override it here so route_for, active_tasks and
        # the key/model checks all see the provider that will actually be dialled.
        if Task.CHAT_VOICE in resolved:
            resolved[Task.CHAT_VOICE] = self.voice_provider
```

In `route_for` (line 1009–1012), replace the voice-route model with a provider-aware one:

```python
        if task in VOICE_TASKS:
            # The native-audio endpoint takes its own model id (never DAEMON_*_MODEL).
            model = self.gemini_live_model if provider == "gemini" else self.openai_realtime_model
            return Route(provider=provider, model=model)
```

In `_check` (replace the block at lines 757–761, and extend the voice-name check):

```python
        if self.voice_provider not in VOICE_PROVIDERS:
            problems.append(
                f"DAEMON_VOICE_PROVIDER is {self.voice_provider!r}; expected one of "
                f"{', '.join(VOICE_PROVIDERS)}"
            )
        elif self.voice_enabled:
            # The chosen provider's own realtime model + that provider's key. The
            # text model (DAEMON_*_MODEL) is neither required nor read for voice.
            if self.voice_provider == "gemini" and not self.gemini_live_model:
                problems.append(
                    "DAEMON_VOICE_ENABLED is on with DAEMON_VOICE_PROVIDER=gemini but "
                    "DAEMON_GEMINI_LIVE_MODEL is empty; the native-audio endpoint needs its own id"
                )
            if self.voice_provider == "openai":
                if not self.openai_realtime_model:
                    problems.append(
                        "DAEMON_VOICE_ENABLED is on with DAEMON_VOICE_PROVIDER=openai but "
                        "DAEMON_OPENAI_REALTIME_MODEL is empty; the realtime endpoint needs its own id"
                    )
                if not self.openai_api_key:
                    problems.append(
                        "DAEMON_VOICE_ENABLED is on with DAEMON_VOICE_PROVIDER=openai but "
                        "OPENAI_API_KEY is empty"
                    )
```

And after the existing `gemini_live_voice` check (line 781), add:

```python
        if self.openai_realtime_voice and self.openai_realtime_voice not in OPENAI_REALTIME_VOICES:
            problems.append(
                f"DAEMON_OPENAI_REALTIME_VOICE is {self.openai_realtime_voice!r}; expected one of "
                "the OpenAI Realtime voices, or empty to leave it to the server"
            )
```

Note: the existing `_provider_problems` early-return for `VOICE_TASKS` (near line 953) stays — it already avoids demanding `DAEMON_OPENAI_MODEL` for a voice route.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_config.py -v` then `python3 -m pytest -q`
Expected: PASS (the whole suite; the routing override must not break the existing Gemini voice tests — `voice_provider` defaults to `gemini`).

- [ ] **Step 5: Commit**

```bash
git add daemon/config.py tests/test_config.py
git commit -m "config: DAEMON_VOICE_PROVIDER (gemini|openai) + openai realtime model/voice, provider-aware voice routing"
```

---

### Task 2: `OpenAIRealtimeSession` — connect, configure, take audio

**Files:**
- Create: `daemon/voice/openai_realtime.py`
- Modify: `tests/test_reachable.py` (add `OpenAIRealtimeSession` to `PENDING_CLASSES`)
- Test: `tests/test_openai_realtime.py` (new)

**Interfaces:**
- Consumes: `daemon.voice.base` (`VoiceSession`, `Transcript`, `Interrupted`), `daemon.llm.base.ToolCall`, `daemon.tools.base.ToolResult`, and the tool-arg helpers `gemini_live.py` imports (`decode_tool_arguments`, `synthesise_call_id`, and a schema-narrowing helper for OpenAI tool params — reuse the same JSON-schema the text OpenAI provider uses, or pass `spec.parameters` through unchanged if OpenAI Realtime accepts it; confirm in the spike).
- Produces: `OpenAIRealtimeSession(api_key, model, *, system_instruction=None, voice_name=None, tools=(), connect=None, url=WS_URL, max_attempts=DEFAULT_MAX_ATTEMPTS, ssl_context=None)`; `OpenAIRealtimeError(Exception)` with `.permanent: bool`; module fn `_upsample_16k_to_24k(pcm: bytes) -> bytes`. Later tasks add `receive`/`send_*`/`interrupt`.

**Porting note:** `daemon/voice/gemini_live.py` is the structural template. Port these verbatim, changing only what this task's steps name: the module logging setup, `OpenAIRealtimeError` (from `GeminiLiveError` 257–264), `_KeyFilter` (267–303), `_sleep`/`_backoff_delay`/backoff constants (333–342, 74–85), `_ca_bundle`/`_ssl_context`/`CA_BUNDLE_ENV` (224–254), `_open` (778–825), `_discard` (827–833), `_require_open` (731–734), `_send` (736–745), `__aenter__`/`_enter`/`__aexit__`/`close` (449–492). These are provider-agnostic; the deltas are named below.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_openai_realtime.py`, mirroring `tests/test_voice.py`'s fake-socket harness (`FakeConnection`, `connector`, a `session()` helper) but for `OpenAIRealtimeSession`. The "setup complete" signal is `{"type": "session.updated"}` (OpenAI confirms our `session.update` took):

```python
import base64, json
import pytest
from daemon.voice.openai_realtime import OpenAIRealtimeSession, _upsample_16k_to_24k

KEY, MODEL = "sk-test", "gpt-realtime"
SESSION_UPDATED = {"type": "session.updated"}

class FakeConnection:
    def __init__(self, *scripted): self.scripted=list(scripted); self.sent=[]
    async def send(self, raw): self.sent.append(json.loads(raw))
    async def close(self): pass
    def __aiter__(self): return self
    async def __anext__(self):
        if not self.scripted: raise StopAsyncIteration
        item=self.scripted.pop(0)
        if isinstance(item, Exception): raise item
        return json.dumps(item)
    def sent_of_type(self, t): return [m for m in self.sent if m.get("type")==t]

def connector(*conns):
    q=list(conns)
    async def connect(url, additional_headers=None, ssl=None):
        connect.calls.append((url, additional_headers)); return q.pop(0)
    connect.calls=[]; return connect

def make(*conns, **kw):
    return OpenAIRealtimeSession(KEY, MODEL, connect=connector(*conns), **kw)


def test_missing_credentials_are_rejected_before_any_connection():
    with pytest.raises(ValueError): OpenAIRealtimeSession("", MODEL)
    with pytest.raises(ValueError): OpenAIRealtimeSession(KEY, "")


def test_upsample_16k_to_24k_grows_length_by_3_over_2():
    # 4 samples (8 bytes) of 16-bit LE mono -> 6 samples (12 bytes) at 24k.
    pcm = b"".join(int(v).to_bytes(2, "little", signed=True) for v in (0, 100, 200, 300))
    out = _upsample_16k_to_24k(pcm)
    assert len(out) == 12
    assert len(out) % 2 == 0


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_send_audio_appends_upsampled_base64():
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn) as live:
        await live.send_audio(b"\x00\x01\x02\x03")  # 2 samples 16k -> 3 samples 24k
    (append,) = conn.sent_of_type("input_audio_buffer.append")
    raw = base64.b64decode(append["audio"])
    assert len(raw) == 6  # 3 samples * 2 bytes


@pytest.mark.asyncio
async def test_bearer_auth_header_is_sent():
    conn = FakeConnection(SESSION_UPDATED)
    connect = connector(conn)
    async with OpenAIRealtimeSession(KEY, MODEL, connect=connect):
        pass
    _, headers = connect.calls[0]
    assert headers["Authorization"] == f"Bearer {KEY}"
```

(If the suite uses `anyio`/`asyncio` marks differently, match `tests/test_voice.py`'s async style — copy its decorator/fixture convention.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_openai_realtime.py -v`
Expected: FAIL — `daemon.voice.openai_realtime` does not exist.

- [ ] **Step 3: Create `daemon/voice/openai_realtime.py`**

Write the module. Constants and the OpenAI-specific pieces in full; the ported helpers per the Porting note.

```python
"""OpenAI Realtime - the second hosted native-audio VoiceSession (docs/PLAN.md 6.5).

Structurally a sibling of daemon/voice/gemini_live.py: same connect-with-backoff, the
same API-key log filter, the same permanent-vs-transient close classification, the same
turn-boundary receive(). What differs is the wire - a different socket, JSON events
instead of Gemini's proto-over-JSON, and 24 kHz pcm16 input (so the 16 kHz mic capture is
upsampled here). Event names differ between the GA (gpt-realtime) and older beta surfaces;
the decoder accepts both, and evals/openai_realtime_spike.py pins them against the socket.
"""
from __future__ import annotations
import asyncio, base64, json, logging, ssl
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, InvalidHandshake, InvalidStatus

from daemon.llm.base import ToolCall
from daemon.tools.base import ToolResult
from daemon.voice.base import Interrupted, Transcript, VoiceSession
# The neutral tool-arg helpers live in daemon.llm.base (confirmed: gemini_live.py:47
# imports them from there). Same helpers, same ToolCall — one tool layer serves both.
from daemon.llm.base import decode_tool_arguments, synthesise_call_id

logger = logging.getLogger(__name__)

WS_URL = "wss://api.openai.com/v1/realtime"
INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE = 16_000, 24_000  # mic capture -> OpenAI pcm16
SETUP_TIMEOUT_SECONDS = 20.0
BACKOFF_START_SECONDS, BACKOFF_MAX_SECONDS, _BACKOFF_MAX_SHIFT = 1.0, 30.0, 5
DEFAULT_MAX_ATTEMPTS = 4
PARTIAL_BACKLOG = 32

# Server event types (accept GA and beta spellings).
_AUDIO_DELTA = ("response.output_audio.delta", "response.audio.delta")
_ASSISTANT_TR_DELTA = ("response.output_audio_transcript.delta", "response.audio_transcript.delta")
_ASSISTANT_TR_DONE = ("response.output_audio_transcript.done", "response.audio_transcript.done")
_USER_TR_DELTA = ("conversation.item.input_audio_transcription.delta",)
_USER_TR_DONE = ("conversation.item.input_audio_transcription.completed",)
_SPEECH_STARTED = "input_audio_buffer.speech_started"
_RESPONSE_DONE = "response.done"
_FUNC_ARGS_DONE = "response.function_call_arguments.done"
_OUTPUT_ITEM_ADDED = "response.output_item.added"

_PERMANENT_STATUS = frozenset({400, 401, 403, 404})
_PERMANENT_CLOSE_CODES = frozenset({1007, 1008})  # invalid payload / policy; confirm in spike


def _permanent_close(code: int, reason: str) -> bool:
    return code in _PERMANENT_CLOSE_CODES


def _upsample_16k_to_24k(pcm: bytes) -> bytes:
    """16-bit LE mono, 16 kHz -> 24 kHz by linear interpolation (ratio 3/2).

    OpenAI Realtime pcm16 input is fixed at 24 kHz; the mic captures 16 kHz. Only the
    input needs this - output is already 24 kHz (AudioIO.playback_sample_rate). Speech
    band-limits to ~8 kHz either way, so interpolating adds no artefacts a listener hears.
    """
    n = len(pcm) // 2
    if n == 0:
        return b""
    src = [int.from_bytes(pcm[i * 2:i * 2 + 2], "little", signed=True) for i in range(n)]
    out_n = (n * 3) // 2
    out = bytearray()
    for j in range(out_n):
        pos = j * (n - 1) / max(out_n - 1, 1) if out_n > 1 else 0.0
        lo = int(pos); hi = min(lo + 1, n - 1); frac = pos - lo
        val = int(round(src[lo] + (src[hi] - src[lo]) * frac))
        val = max(-32768, min(32767, val))
        out += val.to_bytes(2, "little", signed=True)
    return bytes(out)


class OpenAIRealtimeError(Exception):
    """A realtime session failed. Never carries the API key (see the key filter)."""
    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message); self.permanent = permanent
```

Then, in the same file:
- **`_KeyFilter`**: port from `gemini_live.py:267–303` unchanged (it is provider-agnostic).
- **`_sleep`, `_backoff_delay`**: port from `gemini_live.py:333–342`.
- **`_ca_bundle`, `_ssl_context`, `CA_BUNDLE_ENV`**: port from `gemini_live.py:224–254`.
- **`class OpenAIRealtimeSession`** with:
  - `name = "openai-realtime"`.
  - `__init__(self, api_key, model, *, system_instruction=None, voice_name=None, tools=(), connect=None, url=WS_URL, max_attempts=DEFAULT_MAX_ATTEMPTS, ssl_context=None)`: raise `ValueError("OPENAI_API_KEY is empty")` if not api_key; `ValueError("DAEMON_OPENAI_REALTIME_MODEL is empty")` if not model. Store fields; `self._model = model` (no `models/` prefix — unlike Gemini). Init `self._said = {"user": [], "assistant": []}`, `self._partials: asyncio.Queue = asyncio.Queue(PARTIAL_BACKLOG)`, `self._turn_over=False`, `self._dropping=False`, `self._generating=False`, `self.ended=None`, `self._funcs: dict[str, dict] = {}` (accumulate function-call items by id). Install the `_KeyFilter` on the `websockets` logger + root handlers exactly as `gemini_live.py:440–447`.
  - `__aenter__`/`_enter`/`__aexit__`/`close`: port from `gemini_live.py:449–492` (rename the class in type hints).
  - `_open`: port from `gemini_live.py:778–825`, changing ONLY the connect call — the URL carries the model as a query param and the auth is a Bearer header:
    ```python
    return await self._connect(
        f"{self._url}?model={self._model}",
        additional_headers={"Authorization": f"Bearer {self._api_key}",
                            "OpenAI-Beta": "realtime=v1"},
        ssl=self._ssl_context or _ssl_context(_ca_bundle()),
    )
    ```
  - `_handshake`: port from `gemini_live.py:747–776`, but send `self._setup_message()` and wait for a message whose `type == "session.updated"` (instead of `"setupComplete" in message`).
  - `_setup_message() -> dict`: return the `session.update` payload:
    ```python
    def _setup_message(self) -> dict[str, Any]:
        session: dict[str, Any] = {
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad"},
            "input_audio_transcription": {"model": "whisper-1"},
        }
        if self._voice_name:
            session["voice"] = self._voice_name
        if self._system_instruction:
            session["instructions"] = self._system_instruction
        if self._tools:
            session["tools"] = self._tool_declarations()
        return {"type": "session.update", "session": session}

    def _tool_declarations(self) -> list[dict[str, Any]]:
        # OpenAI Realtime function tool shape: flat, type "function".
        return [{"type": "function", "name": s.name, "description": s.description,
                 "parameters": s.parameters} for s in self._tools]
    ```
    (If the spike shows the model rejects a raw MCP `inputSchema`, narrow `s.parameters` with the same helper the text OpenAI provider uses — note it, do not guess here.)
  - `_require_open`, `_send`: port from `gemini_live.py:731–745` (rename error type).
  - `send_audio(self, chunk)`: `if not chunk: return`; upsample; append:
    ```python
    async def send_audio(self, chunk: bytes) -> None:
        if not chunk:
            return
        pcm24 = _upsample_16k_to_24k(chunk)
        await self._send({"type": "input_audio_buffer.append",
                          "data": None,  # OpenAI uses "audio" (base64); see below
                          })
    ```
    Correct payload (no `data` key — OpenAI uses `audio`):
    ```python
        await self._send({"type": "input_audio_buffer.append",
                          "audio": base64.b64encode(pcm24).decode("ascii")})
    ```

- [ ] **Step 4: Declare the class in `PENDING_CLASSES`**

`OpenAIRealtimeSession` exists but nothing constructs it yet (app wiring is Task 5), so `tests/test_reachable.py` will fail. Add it to `PENDING_CLASSES` (the `dict[str, str]` at line 46), which is exactly what that dict is for:

```python
PENDING_CLASSES: dict[str, str] = {
    "OpenAIRealtimeSession": (
        "Voice Phase B1 - the OpenAI Realtime VoiceSession, built by app.run_voice "
        "when DAEMON_VOICE_PROVIDER=openai. Wired in the app-branch task; this entry "
        "is removed then."
    ),
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_openai_realtime.py tests/test_reachable.py -v` then `python3 -m pytest -q` and `python3 -m ruff check .`
Expected: PASS. (Task-2 tests green; reachable green because the class is declared PENDING.)

- [ ] **Step 6: Commit**

```bash
git add daemon/voice/openai_realtime.py tests/test_openai_realtime.py tests/test_reachable.py
git commit -m "voice: OpenAIRealtimeSession - connect, session.update, send_audio (16k->24k upsample)"
```

---

### Task 3: `receive()` and the event decoder

**Files:**
- Modify: `daemon/voice/openai_realtime.py` (add `receive`, `_decode`, `_flush`, `_push_partial`, `partial_transcripts`, `pending_transcripts`, `_offer_partial`, `_end_partials`, `_closed_error`)
- Test: `tests/test_openai_realtime.py`

**Interfaces:**
- Consumes: the class from Task 2 and its `self._said`/`self._partials`/`self._turn_over`/`self._funcs` state.
- Produces: `receive() -> AsyncIterator[bytes | Transcript | Interrupted | ToolCall]` (ends at `response.done`); `partial_transcripts()`/`pending_transcripts()` per protocol.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_openai_realtime.py`. A helper to collect one turn (mirror `tests/test_voice.py`'s `turn`):

```python
async def collect(live):
    return [item async for item in live.receive()]

def audio_delta(b): return {"type": "response.output_audio.delta", "delta": base64.b64encode(b).decode()}

@pytest.mark.asyncio
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

@pytest.mark.asyncio
async def test_speech_started_is_a_barge_in():
    conn = FakeConnection(SESSION_UPDATED, audio_delta(b"\x01\x02"),
                          {"type": "input_audio_buffer.speech_started"}, {"type": "response.done"})
    async with make(conn) as live:
        items = await collect(live)
    assert any(isinstance(i, Interrupted) for i in items)

@pytest.mark.asyncio
async def test_function_call_becomes_a_toolcall():
    conn = FakeConnection(
        SESSION_UPDATED,
        {"type": "response.output_item.added",
         "item": {"type": "function_call", "name": "open_path", "call_id": "c1"}},
        {"type": "response.function_call_arguments.done", "call_id": "c1", "arguments": "{\"path\": \"/tmp\"}"},
        {"type": "response.done"},
    )
    async with make(conn, tools=(_a_tool_spec_named("open_path"),)) as live:
        items = await collect(live)
    calls = [i for i in items if isinstance(i, ToolCall)]
    assert calls and calls[0].name == "open_path" and calls[0].arguments == {"path": "/tmp"}
```

(`_a_tool_spec_named` builds a minimal `ToolSpec` — copy how `tests/test_voice.py` constructs tool specs for its tool-call test.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_openai_realtime.py -k "receive or barge or function" -v`
Expected: FAIL — `receive`/decoder not implemented.

- [ ] **Step 3: Implement `receive` + `_decode`**

Port the `receive()` loop shape from `gemini_live.py:614–672` (the `async for raw in ws` → `_decode` → `_turn_over` → flush-on-close structure, and `_closed_error`/`ended`), and write the OpenAI `_decode`:

```python
def _decode(self, raw) -> Iterator[bytes | Transcript | Interrupted | ToolCall]:
    try:
        msg = json.loads(raw)
    except ValueError:
        logger.warning("openai-realtime: dropping a non-JSON message"); return
    if not isinstance(msg, dict):
        return
    t = msg.get("type")
    if t in _AUDIO_DELTA:
        self._generating = True
        if not self._dropping:
            try:
                yield base64.b64decode(msg.get("delta") or "", validate=True)
            except Exception:
                logger.warning("openai-realtime: undecodable audio delta")
        return
    if t == _SPEECH_STARTED:
        if self._generating:
            self._dropping = True
            yield Interrupted()
        return
    if t in _ASSISTANT_TR_DELTA:
        self._said["assistant"].append(msg.get("delta") or ""); return
    if t in _ASSISTANT_TR_DONE:
        # `transcript` carries the full text; prefer it over accumulated deltas.
        text = msg.get("transcript")
        if isinstance(text, str) and text:
            self._said["assistant"] = [text]
        return
    if t in _USER_TR_DELTA:
        self._said["user"].append(msg.get("delta") or ""); self._push_partial(); return
    if t in _USER_TR_DONE:
        text = msg.get("transcript")
        if isinstance(text, str) and text:
            self._said["user"] = [text]; self._push_partial()
        return
    if t == _OUTPUT_ITEM_ADDED:
        item = msg.get("item") or {}
        if item.get("type") == "function_call":
            cid = item.get("call_id") or item.get("id") or ""
            self._funcs[cid] = {"name": item.get("name"), "args": ""}
        return
    if t == _FUNC_ARGS_DONE:
        cid = msg.get("call_id") or ""
        rec = self._funcs.pop(cid, {"name": None})
        name = rec.get("name")
        if isinstance(name, str) and name and self._tools:
            yield ToolCall(id=cid or synthesise_call_id(name, 0), name=name,
                           arguments=decode_tool_arguments(msg.get("arguments")))
        return
    if t == "error":
        logger.warning("openai-realtime: server error %s", msg.get("error"))
        return
    if t == _RESPONSE_DONE:
        self._dropping = False; self._generating = False
        yield from self._flush()
        self._turn_over = True
        return
```

Then `_flush`, `_push_partial`, `_offer_partial`, `_end_partials`, `partial_transcripts`, `pending_transcripts`, `interrupt`-adjacent state: port from `gemini_live.py:696–729, 1055–1093` unchanged (they operate on `self._said`/`self._partials`, which are identical here). Add `_closed_error` (map a `ConnectionClosed` to `OpenAIRealtimeError(..., permanent=_permanent_close(code, reason))`), porting `gemini_live.py`'s version.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_openai_realtime.py -q` then `python3 -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add daemon/voice/openai_realtime.py tests/test_openai_realtime.py
git commit -m "voice: OpenAIRealtimeSession.receive - event decoder (audio, transcripts, barge-in, tool calls, turn end)"
```

---

### Task 4: `send_context` / `send_text` / `send_tool_response` / `interrupt` / `send_frame`

**Files:**
- Modify: `daemon/voice/openai_realtime.py`
- Test: `tests/test_openai_realtime.py`

**Interfaces:**
- Produces the remaining `VoiceSession` methods. Consumes `self._send`, `self._generating`, `self._dropping`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_send_context_adds_history_without_a_response():
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn) as live:
        await live.send_context("recall: the user likes tea")
    assert conn.sent_of_type("conversation.item.create")
    assert not conn.sent_of_type("response.create")  # context must not make the model answer

@pytest.mark.asyncio
async def test_send_text_prompts_a_response():
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn) as live:
        await live.send_text("say hello")
    assert conn.sent_of_type("conversation.item.create")
    assert conn.sent_of_type("response.create")

@pytest.mark.asyncio
async def test_send_tool_response_emits_function_output_then_response():
    from daemon.tools.base import ToolResult
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn) as live:
        await live.send_tool_response([ToolResult(call_id="c1", name="open_path", ok=True, content="done")])
    outs = conn.sent_of_type("conversation.item.create")
    assert any(m["item"]["type"] == "function_call_output" and m["item"]["call_id"] == "c1" for m in outs)
    assert conn.sent_of_type("response.create")

@pytest.mark.asyncio
async def test_send_frame_is_a_noop():
    conn = FakeConnection(SESSION_UPDATED)
    async with make(conn) as live:
        await live.send_frame(b"\xff\xd8jpeg")
    assert conn.sent == [conn.sent[0]] if conn.sent else True  # nothing beyond session.update
    assert not conn.sent_of_type("input_audio_buffer.append")
```

(Match `ToolResult`'s actual constructor from `daemon/tools/base.py` — read it; the fields are `call_id`, `name`, `ok`, `content` per `gemini_live.py:583–612`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_openai_realtime.py -k "send_context or send_text or tool_response or send_frame" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the methods**

```python
async def send_frame(self, jpeg: bytes) -> None:
    # OpenAI Realtime has no realtime video input; screen share is Gemini-only.
    if not self._warned_no_video:
        logger.warning("openai-realtime: screen share is unsupported on OpenAI; frames are dropped")
        self._warned_no_video = True

async def send_context(self, text: str) -> None:
    if not text.strip():
        return
    # An item, and NO response.create: this puts recall in front of the model without
    # asking it to answer (daemon/voice/base.py; the loop relies on this).
    await self._send({"type": "conversation.item.create", "item": {
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": text}]}})

async def send_text(self, text: str) -> None:
    if not text.strip():
        logger.warning("openai-realtime: refusing to send empty text"); return
    await self._send({"type": "conversation.item.create", "item": {
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": text}]}})
    await self._send({"type": "response.create"})

async def send_tool_response(self, results: Sequence[ToolResult]) -> None:
    if not results:
        return
    for r in results:
        await self._send({"type": "conversation.item.create", "item": {
            "type": "function_call_output", "call_id": r.call_id,
            "output": r.content}})
    await self._send({"type": "response.create"})

async def interrupt(self) -> None:
    # Local, like Gemini: stop handing out the abandoned turn's audio. Under server VAD
    # the user's own audio already stopped generation server-side.
    if not self._generating:
        return
    self._dropping = True
```

Add `self._warned_no_video = False` to `__init__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_openai_realtime.py -q` then `python3 -m pytest -q` and `python3 -m ruff check .`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add daemon/voice/openai_realtime.py tests/test_openai_realtime.py
git commit -m "voice: OpenAIRealtimeSession send_context/text/tool_response/interrupt/send_frame"
```

---

### Task 5: wire `run_voice` to branch on the provider

**Files:**
- Modify: `daemon/app.py` (the session factory in `run_voice`, near line 1143–1164; and its import near 1034)
- Modify: `tests/test_reachable.py` (remove `OpenAIRealtimeSession` from `PENDING_CLASSES`; add to `WIRED_CLASSES`)

**Interfaces:**
- Consumes: `route = settings.route_for(Task.CHAT_VOICE)` (already resolved at app.py:1042), `OpenAIRealtimeSession` (Task 2–4), `settings.openai_api_key`, `settings.openai_realtime_voice`.

- [ ] **Step 1: Branch the session factory**

In `daemon/app.py`, add the import beside the Gemini one (near line 1034):

```python
    from daemon.voice.openai_realtime import OpenAIRealtimeError, OpenAIRealtimeSession
```

Replace the `return GeminiLiveSession(...)` factory body (1143–1164) so it branches on `route.provider`:

```python
            if route.provider == "openai":
                return OpenAIRealtimeSession(
                    api_key=settings.openai_api_key,
                    model=route.model,
                    system_instruction=system_instruction,
                    tools=tool_specs,
                    voice_name=settings.openai_realtime_voice,
                )
            return GeminiLiveSession(
                api_key=settings.gemini_api_key,
                model=route.model,
                system_instruction=system_instruction,
                tools=tool_specs,
                start_sensitivity=settings.voice_start_sensitivity,
                end_sensitivity=settings.voice_end_sensitivity,
                prefix_padding_ms=settings.voice_prefix_padding_ms,
                silence_duration_ms=settings.voice_silence_duration_ms,
                voice_name=settings.gemini_live_voice,
            )
```

`run_voice` passes the error type to catch as a positional argument to `_voice_attempts` (app.py:1189–1197 — the `GeminiLiveError,` argument at line 1193). Change that argument to a tuple so an OpenAI failure is retried/reported the same way (an `except` accepts a tuple of types, and both errors carry `.permanent`):

```python
            return await _voice_attempts(
                new_session,
                audio,
                companion,
                (GeminiLiveError, OpenAIRealtimeError),
                opening_audio=opening_audio,
                screen_share=screen_share,
                screen_pump_factory=screen_pump_factory,
            )
```

- [ ] **Step 2: Move the class from PENDING to WIRED**

In `tests/test_reachable.py`: delete the `"OpenAIRealtimeSession": (...)` entry from `PENDING_CLASSES` (back to `{}`), and add `"OpenAIRealtimeSession",` to the `WIRED_CLASSES` tuple beside `"GeminiLiveSession"`.

- [ ] **Step 3: Run the reachability gate + suite**

Run: `python3 -m pytest tests/test_reachable.py -v` then `python3 -m pytest -q` and `python3 -m ruff check .`
Expected: PASS. `test_reachable` now finds `run_voice` constructs `OpenAIRealtimeSession` (a stale PENDING entry would fail — that is why it moved).

- [ ] **Step 4: Commit**

```bash
git add daemon/app.py tests/test_reachable.py
git commit -m "voice: run_voice builds OpenAIRealtimeSession when DAEMON_VOICE_PROVIDER=openai"
```

---

### Task 6: the live spike + full verification

**Files:**
- Create: `evals/openai_realtime_spike.py`

This task has no unit test (evals hit the live API, never in CI — tests/CLAUDE.md). It pins the GA/beta event names the decoder accepts and proves a real round-trip.

- [ ] **Step 1: Write the spike**

Create `evals/openai_realtime_spike.py`, mirroring `evals/m0_voice_spike.py`'s shape: read `OPENAI_API_KEY` and `DAEMON_OPENAI_REALTIME_MODEL` from the env; open an `OpenAIRealtimeSession`; `send_text("Say a one-sentence hello.")`; iterate `receive()` printing each item's kind (bytes length / transcript role+text / Interrupted / ToolCall) and the raw server `type` strings seen; print total audio bytes and first-audio latency. Exit non-zero with guidance if either env var is missing (no network on that path). Read `evals/m0_voice_spike.py` and match its structure/logging.

- [ ] **Step 2: Import + no-key sanity (no network)**

Run: `env -u OPENAI_API_KEY -u DAEMON_OPENAI_REALTIME_MODEL python3 -m evals.openai_realtime_spike`
Expected: prints the missing-env guidance, exits non-zero. Confirms the module imports.

- [ ] **Step 3: Commit the spike**

```bash
git add evals/openai_realtime_spike.py
git commit -m "evals: OpenAI Realtime live spike (pins GA/beta event names)"
```

- [ ] **Step 4: Full gates**

Run:
```bash
python3 -m pytest
python3 -m ruff check .
python3 scripts/check_docs.py
```
Expected: all pass.

- [ ] **Step 5: Live run (deferred to a key-owner)**

With a real key: `OPENAI_API_KEY=… DAEMON_OPENAI_REALTIME_MODEL=gpt-realtime python3 -m evals.openai_realtime_spike`. Confirm audio + transcripts come back, and that the `type` strings printed match the decoder's `_AUDIO_DELTA` / transcript / function-call constants — **adjust those constants to the observed names if they differ**, then re-run the unit suite. Then set `DAEMON_VOICE_PROVIDER=openai`, `DAEMON_OPENAI_REALTIME_MODEL`, `OPENAI_API_KEY`, `DAEMON_VOICE_ENABLED=true` and run `daemon voice` for a real conversation. (No `OPENAI_API_KEY` in CI, so this step is the owner's.)

---

## Self-Review

**Spec coverage:**
- `voice_provider` + `openai_realtime_model` + `openai_realtime_voice` + `OPENAI_REALTIME_VOICES` + provider-aware `route_for`/validation → Task 1. ✓
- New `daemon/voice/openai_realtime.py` `VoiceSession` (connect/session.update/send_audio + upsample) → Task 2. ✓
- `receive()` event decoder (audio, both transcripts, barge-in, tool call, turn end, error) + partials → Task 3. ✓
- `send_context` (no response), `send_text`, `send_tool_response`, `interrupt` (local), `send_frame` (no-op) → Task 4. ✓
- `app.run_voice` branch + reachability (PENDING→WIRED) → Tasks 2 & 5. ✓
- Keyless tests with a fake socket + Korean case; live spike deferred → Tasks 2–4, 6. ✓
- Upsample 16→24 inside the session → Task 2. ✓
- GA/beta event-name tolerance → Task 3 constants + Task 6 pinning. ✓
- Out of scope (B2 admin UX) → not in any task. ✓

**Placeholder scan:** The two spots that say "confirm in the spike" (tool-param narrowing; the exact function-call event shape) are genuine live-API unknowns with a named resolution step (Task 6), not skipped work — the decoder handles the documented shape and both event spellings. The tool-arg helper import path is flagged to verify (`decode_tool_arguments`/`synthesise_call_id`'s real module) rather than guessed. No TBD/TODO in code steps.

**Type/name consistency:** `OpenAIRealtimeSession`, `OpenAIRealtimeError`, `_upsample_16k_to_24k`, `_setup_message`, `_decode`, the `_AUDIO_DELTA`/`_USER_TR_DONE`/… constants, and `self._said`/`self._partials`/`self._funcs`/`self._warned_no_video` are used consistently across Tasks 2–5. `route_for` returns `Route(provider, model)` (Task 1) consumed by `run_voice` (Task 5). `PENDING_CLASSES` add (Task 2) is removed and `WIRED_CLASSES` add done (Task 5).
