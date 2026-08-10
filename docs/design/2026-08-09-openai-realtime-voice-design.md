# OpenAI Realtime voice — Phase B1 (the second native-audio provider)

**Status:** approved design, pre-implementation
**Date:** 2026-08-09
**Base:** off `main` at v0.1.25 (`0ad2d21`). Phase A (the Gemini voice picker) plus the
v0.1.23–25 voice work is merged. Verified against current code: the `VoiceSession`
protocol (`daemon/voice/base.py`) is **unchanged** since Phase A, and `route_for`,
`run_voice`, and the config voice-validation my design edits are structurally the same
spots. The v0.1.23–25 changes live in the consumer loop (`conversation.py`) and
`companion.py`, not the protocol.
**Scope:** B1 only — the OpenAI Realtime `VoiceSession` and the provider-routing
dimension. The admin UX for it (a provider dropdown, an OpenAI voice dropdown, OpenAI
preview clips) is **B2**, a separate spec/plan cycle. In B1 the new settings are plain
editable fields (env / the existing generic admin inputs).

## Problem

`daemon/voice/base.py`, `config.py`, and `conversation.py` all name OpenAI Realtime as
"the second provider", but only Gemini Live is implemented. `route_for(Task.CHAT_VOICE)`
is hardwired to `gemini_live_model` and `app.run_voice` builds a `GeminiLiveSession`
unconditionally. So voice mode is Gemini-only, and the owner cannot choose otherwise.

## Goal

With `DAEMON_VOICE_PROVIDER=openai`, `daemon voice` holds a conversation over the OpenAI
Realtime API instead of Gemini Live, satisfying the exact same `VoiceSession` protocol so
`conversation.py`, memory, tools, and the wake gate are unchanged. Voice-provider choice
is decoupled from the text `hosted_provider` (a dedicated selector), because voice-model
availability is independent of text-model choice and being explicit turns a mismatch into
a startup error rather than a first-turn failure.

## Decisions (from brainstorming)

1. **Dedicated `DAEMON_VOICE_PROVIDER`** (`gemini` | `openai`), default `gemini`
   (back-compat). Not derived from `hosted_provider`.
2. **B1 backend first**, B2 admin UX later.
3. **Upsample 16 kHz → 24 kHz inside the OpenAI session.** The mic/`AudioIO` captures at
   16 kHz (fixed, shared with the wake gate's Silero VAD and Apple ko-KR recognizer);
   OpenAI Realtime requires 24 kHz pcm16 input. Resampling in `send_audio` keeps the
   change inside one file and leaves `AudioIO`, the wake path, and Gemini untouched. Only
   input needs it — OpenAI output is 24 kHz, which already matches
   `AudioIO.playback_sample_rate`.
4. **`send_frame` (screen share, ADR 0009) is a no-op with a one-time warning** on OpenAI —
   the Realtime API has no realtime video-frame input. Screen share stays Gemini-only,
   documented. `conversation.py` only calls `send_frame` during an active screen share
   (`DAEMON_SCREEN_ENABLED` + the screen tool), so this degrades cleanly.
5. **`interrupt()` stays local** (refuse to hand out more of the abandoned turn's audio;
   `AudioIO.stop_playback` drops the queue) — matching the protocol contract. Under
   OpenAI server-VAD the user's own audio already stops generation server-side.
6. **All 10 voices offered**, validated against a fixed set; `marin`/`cedar` are
   documented as `gpt-realtime`-only (a voice the chosen model rejects comes back as a
   session error, the same class as Gemini's 1007 — caught early where we can, surfaced
   honestly where we can't).

## OpenAI Realtime facts (grounded, 2026-08-09)

Sources: OpenAI Realtime guide + Realtime-conversations reference
(`developers.openai.com/api/docs/guides/realtime`). **Event names AND the request shape
drift between the beta (`realtime=v1`) and GA (`gpt-realtime`) surfaces.**

**Confirmed live against gpt-realtime, 2026-08-09** (the repo's "socket wins over docs"
rule): the beta request shape is *rejected* — a beta-header + flat `session.update`
(`input_audio_format:"pcm16"`, top-level `voice`, `modalities`) closes **4000
`invalid_request_error.beta_api_shape_disabled`**. The GA shape (from the socket's own
`session.created`) is: **drop** the `OpenAI-Beta` header; `session` = `{type:"realtime",
output_modalities:["audio"], instructions, audio:{input:{format:{type:"audio/pcm",rate:24000},
turn_detection:{type:"server_vad"}, transcription:{model:"whisper-1"}}, output:{format:{...},
voice}}, tools}`. Server event names (measured): `response.output_audio.delta`,
`response.output_audio_transcript.delta|done`, `conversation.item.input_audio_transcription.delta|completed`,
`input_audio_buffer.speech_started`, `response.done` — all in the decoder's constants. And
the load-bearing timing (measured with fed audio): the user's `...input_audio_transcription.completed`
arrives **~200 ms AFTER `response.done`** — which is exactly why the user transcript is
yielded immediately on `completed`, decoupled from the turn boundary. Names table below.

- **Transport:** `wss://api.openai.com/v1/realtime?model=<model>`; headers
  `Authorization: Bearer <OPENAI_API_KEY>` and (beta models) `OpenAI-Beta: realtime=v1`.
- **Audio:** pcm16, **24 kHz mono, both directions** (input rate fixed, not configurable —
  hence decision 3).
- **Voices (10):** alloy, ash, ballad, coral, echo, sage, shimmer, verse, marin, cedar.

| VoiceSession method / item | OpenAI Realtime wire |
|---|---|
| connect | WS open, then one `session.update`: instructions=persona, voice, input/output pcm16, `turn_detection` server_vad, `input_audio_transcription`, tools, modalities audio+text |
| `send_audio(pcm16 16k)` | upsample→24k, base64, `input_audio_buffer.append` (server VAD auto-commits + responds) |
| `send_frame(jpeg)` | **no-op + warn once** (no video input) |
| `send_context(text)` | `conversation.item.create` (user item), **no** `response.create` — history without an answer |
| `send_text(text)` | `conversation.item.create` + `response.create` — a prompt the model answers |
| `send_tool_response(results)` | one `conversation.item.create` (`function_call_output`) per result, then `response.create` |
| `interrupt()` | local (stop yielding); optional `response.cancel` |
| receive → `bytes` | `response.output_audio.delta` (b64 pcm16 24k) |
| receive → `Transcript(assistant, final)` | accumulate `response.output_audio_transcript.delta`, emit at `…transcript.done` |
| receive → `Transcript(user, final)` | `conversation.item.input_audio_transcription.completed` |
| `partial_transcripts()` | `conversation.item.input_audio_transcription.delta` (OpenAI DOES stream partials — this is the provider `conversation.py` notes makes the recall prefetch pay off) |
| receive → `Interrupted` | `input_audio_buffer.speech_started` (barge-in) |
| receive → `ToolCall` | function-call item / `response.function_call_arguments.done` (accumulate args, decode JSON) |
| turn boundary (end `receive()`) | `response.done` |
| error | `error` event → classify permanent vs transient, set `ended`, raise |

## Components

### 1. `daemon/voice/openai_realtime.py` (new) — `OpenAIRealtimeSession`

Implements the `VoiceSession` protocol (`daemon/voice/base.py`), mirroring
`gemini_live.py`'s structure so the two read alike:

- `__init__(api_key, model, *, system_instruction=None, voice_name=None, tools=(), connect=None, url=WS_URL, max_attempts=…, ssl_context=None)` — `connect`/`url` injectable so tests drive it with a fake websocket and no network (exactly how `GeminiLiveSession` is tested). `voice_name` empty → omit `voice` from `session.update` (server default).
- Connect with retry/backoff and permanent-vs-transient close classification; an API-key log filter installed on the websockets loggers and removed on every exit path (the same key-leak guard `gemini_live.py` carries).
- Transcript accumulation per direction; `receive()` ends at `response.done`; `pending_transcripts()` drains what was accumulated but never yielded; `partial_transcripts()` streams in-progress user transcripts.
- `_upsample_16k_to_24k(pcm: bytes) -> bytes` — a pure function (16-bit LE mono, ratio 3/2, linear interpolation). Unit-tested independently.
- Screen-share `send_frame`: no-op + a single `logger.warning` (guard with a flag so it warns once).

### 2. `daemon/config.py` — the provider dimension

- Constants: `VOICE_PROVIDERS = ("gemini", "openai")`, `OPENAI_REALTIME_VOICES` (the 10).
- Fields:
  - `voice_provider: str = Field(default="gemini", alias="DAEMON_VOICE_PROVIDER")`
  - `openai_realtime_model: str = Field(default="", alias="DAEMON_OPENAI_REALTIME_MODEL")` (no default, like `gemini_live_model`; recommend `gpt-realtime`)
  - `openai_realtime_voice: str = Field(default="", alias="DAEMON_OPENAI_REALTIME_VOICE")`
- `routing`: override `CHAT_VOICE`'s provider with `self.voice_provider` so the effective voice provider follows the selector, not the preset's literal `GEMINI`.
- `route_for(CHAT_VOICE)`: `Route(self.voice_provider, self.gemini_live_model if voice_provider=="gemini" else self.openai_realtime_model)`.
- `_check` (provider-aware, replacing the unconditional `gemini_live_model` check):
  - `voice_provider` must be in `VOICE_PROVIDERS`.
  - if `voice_enabled`: the **chosen** provider's realtime model must be set
    (`gemini_live_model` for gemini, `openai_realtime_model` for openai) **and** that
    provider's API key (`GEMINI_API_KEY` / `OPENAI_API_KEY`).
  - `openai_realtime_voice` must be in `OPENAI_REALTIME_VOICES` or empty (like the Gemini
    voice check). Both voice fields validated unconditionally.
  - Both `"gemini"` and `"openai"` are already in `HOSTED_PROVIDERS` / `PROVIDER_KEY_ENV`,
    so overriding `CHAT_VOICE`'s provider to either is not an "unknown provider". The
    existing early-return for `VOICE_TASKS` in `_provider_problems` **stays**: a voice
    route to `openai` must NOT require `DAEMON_OPENAI_MODEL` (that is the text model, never
    read for voice) — the realtime model is the only model checked, above.

### Matching the current conversation loop (v0.1.23–25)

The consumer loop gained tool-answer coordination since Phase A (`conversation.py`'s
`_answering_tool`: between "the model asked for a tool" and "the model spoke the result",
the loop withholds `send_context` so a recall block does not cancel the tool answer; and a
continuity block is delivered via `send_context` on reconnect). This is loop-side and uses
only the unchanged protocol, so `OpenAIRealtimeSession` needs no special case — but it must
honour the two semantics the loop depends on, both already in this design:
- **`send_context` must not trigger a response** (item added to history, no `response.create`),
  so it can be sent between turns without the model answering it.
- **`Interrupted` fires only on a genuine user barge-in** (`input_audio_buffer.speech_started`),
  never as a side effect of our own `send_*` — the loop reads `Interrupted` as "empty the
  speaker", and a false one throws away a real answer.
The implementer reads the current `conversation.py` and `gemini_live.py` to mirror this,
not a Phase-A snapshot.

### 3. `daemon/app.py` — `run_voice` branches

Resolve the voice route; if `route.provider == "openai"` build `OpenAIRealtimeSession(api_key=settings.openai_api_key, model=route.model, system_instruction=…, voice_name=settings.openai_realtime_voice, tools=tool_specs)`, else the existing `GeminiLiveSession` path (unchanged, incl. the Gemini-only sensitivity/padding args). Only `app.py` imports the concrete session (layering).

## Testing (keyless, no network — tests/CLAUDE.md)

- `_upsample_16k_to_24k`: length ratio and interpolated values for a known PCM buffer.
- Event decoding: feed the session a scripted sequence of fake server JSON events through
  an injected fake websocket; assert `receive()` yields the right `bytes` / `Transcript`
  (user & assistant, `final=True`) / `Interrupted` / `ToolCall`, ends at `response.done`,
  and that `partial_transcripts()` surfaces user deltas. Follows the `GeminiLiveSession`
  fake-socket test pattern; a Korean transcript case included.
- `send_frame` no-op (no send, warns once); `send_context` sends no `response.create`;
  `send_tool_response` emits `function_call_output` + `response.create`.
- config: `voice_provider` validation; `route_for(CHAT_VOICE)` returns the openai
  provider+model when `voice_provider=openai`; `voice_enabled` requires the chosen
  provider's model+key; `openai_realtime_voice` allowlist.
- `tests/test_reachable.py`: `OpenAIRealtimeSession` added to `WIRED_CLASSES` (built by
  `run_voice` when `voice_provider=openai`).

**Live verification (deferred to a key-owner run — no `OPENAI_API_KEY` in CI):**
`evals/openai_realtime_spike.py`, mirroring `evals/m0_voice_spike.py`: open a real session,
send text and audio, print transcripts + first-audio timing, and **pin the concrete GA/beta
event names** the table above marks as best-known. Green units are not proof (tests/CLAUDE.md).

## Out of scope — B2 (next cycle)

- Admin: a `voice_provider` dropdown that drives which voice list + preview clips show;
  `openai_realtime_voice` dropdown; OpenAI preview clips (extend the generator + namespace
  `voice-samples/<provider>/<voice>.mp3` and the `/admin/api/voice-sample` route).
- Mapping any OpenAI `turn_detection` knobs to settings (B1 uses server_vad defaults).
