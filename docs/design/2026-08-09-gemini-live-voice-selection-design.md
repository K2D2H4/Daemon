# Gemini Live voice selection, from the admin web

**Status:** approved design, pre-implementation
**Date:** 2026-08-09
**Scope:** Phase A only. OpenAI Realtime (Phase B) is out of scope and sketched at the end.

## Problem

The Gemini Live session already accepts a `voice_name`
([daemon/voice/gemini_live.py](../../daemon/voice/gemini_live.py) `__init__`,
`_setup_message`): when set it sends
`speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName`, when empty it sends no
`speechConfig` and the server picks its own default. But nothing ever sets it —
`app.run_voice` builds the session without the argument, and there is no config
field or `.env` key for it. So the spoken voice is whatever the server defaults
to, and the owner cannot change it.

## Goal

Let the owner choose the Gemini Live voice from the admin web's Settings tab, save
it to `.env`, and hear a short sample of each voice before committing. A bad value
fails loudly at config construction (startup), not late on the wire — the same
philosophy the sensitivity settings already follow, and for the same reason: an
unknown voice name comes back from the server as a `1007` close, which the session
classifies permanent, so a typo would not be a bad setting, it would be voice mode
gone.

## Decisions

1. **Dedicated field, `gemini_live_voice`** — prefixed so Phase B can add
   `openai_realtime_voice` beside it under a `DAEMON_VOICE_PROVIDER` selector. The
   provider selector is **not** built now (YAGNI: there is one voice provider
   today).
2. **Strict allowlist, validated in `config.py`.** The set of valid voices lives
   in `config.py` as a constant, not imported from the voice layer — importing
   `daemon/voice/*` into `config.py` would invert the layering, the same reason
   `SENSITIVITIES` is duplicated rather than imported (config.py header comment).
3. **Bundled preview samples**, not live-on-click. The admin is documented as
   "static, self-contained, offline" (admin routes docstring, design decision 1).
   A one-time generator hits the live API and commits a short compressed clip per
   voice; the admin serves those files. Preview then costs nothing, needs no key,
   and keeps the admin offline. The generation run doubles as the real-API check.

## The voice set

Native-audio Live models support any voice available to the TTS models — the full
set of 30, confirmed against
`https://ai.google.dev/gemini-api/docs/speech-generation` (2026-08-09):

```
Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, Callirrhoe, Autonoe,
Enceladus, Iapetus, Umbriel, Algieba, Despina, Erinome, Algenib, Rasalgethi,
Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, Achird,
Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat
```

Empty string is also valid and means "leave it to the server."

## Changes, file by file

### 1. `daemon/config.py`

- New constant `GEMINI_LIVE_VOICES: frozenset[str]` holding the 30 names above.
- New field:
  ```python
  gemini_live_voice: str = Field(default="", alias="DAEMON_GEMINI_LIVE_VOICE")
  ```
  with a docstring in the style of the sensitivity fields: empty = server default,
  otherwise one of `GEMINI_LIVE_VOICES`; a wrong name is a `1007` permanent close,
  so it is checked here.
- Validation in `_check`, beside the sensitivity loop (config.py ~line 749):
  ```python
  if self.gemini_live_voice and self.gemini_live_voice not in GEMINI_LIVE_VOICES:
      problems.append(
          f"DAEMON_GEMINI_LIVE_VOICE is {self.gemini_live_voice!r}; expected one of "
          f"the Gemini Live voices, or empty to leave it to the server"
      )
  ```
  Validated unconditionally (independent of `voice_enabled`): a typo is a typo, and
  the field is harmless when unset because the default is empty.

### 2. `daemon/admin/settings_io.py`

- Add to `STR_FIELDS`: `"gemini_live_voice": "DAEMON_GEMINI_LIVE_VOICE"`. This alone
  surfaces the field in `GET /settings` `editable` (the function iterates
  `STR_FIELDS`) and makes `PATCH` accept it (`EDITABLE` includes `STR_FIELDS`), with
  the validate-before-write guarantee inherited unchanged.
- In `current_settings_payload`, add to `options`:
  ```python
  "gemini_live_voices": ["", *sorted(GEMINI_LIVE_VOICES)],
  ```
  imported from `config` (settings_io already imports from `daemon.config`).

### 3. `daemon/app.py`

One argument at the `GeminiLiveSession(...)` construction in `run_voice`
(app.py ~line 938):
```python
voice_name=settings.gemini_live_voice,
```
Empty passes straight through as "leave it to the server", exactly as today.

### 4. `daemon/admin/static/index.html`

- In `loadSettings`, next to the voice controls:
  ```js
  h+=fieldStr('gemini_live_voice',e.gemini_live_voice,o.gemini_live_voices);
  ```
  `fieldStr` with an options list renders a `<select>`; the leading `""` option is
  the "(default)" choice, the same shape `hosted_provider` already uses.
  `collectPatch`'s generic string branch already handles it — no JS change there.
- A preview control beside that select: a `▶` button and one hidden `<audio>`
  element. On click, if a non-empty voice is selected, set
  `audio.src = "/admin/api/voice-sample/" + voice` and `audio.play()`. Kept to a few
  terse lines to match the file. No preview for the empty "(default)" option.

### 5. `daemon/admin/routes.py`

New route, under the existing loopback guard:
```
GET /admin/api/voice-sample/{voice}  ->  audio/mpeg bytes, or 404
```
- `voice` is checked against `GEMINI_LIVE_VOICES` **before** touching the
  filesystem — this closes path traversal (the name is never interpolated into a
  path until it is known to be one of a fixed set).
- Reads `daemon/admin/static/voice-samples/{voice}.mp3` from disk (same
  read-per-request pattern as the shell), returns it as `audio/mpeg`.
- **404 for anything not served**: a name outside the allowlist and a known voice
  whose sample file is absent both return the same 404 with a short message. One
  code path, no allowlist leak, and a missing asset degrades to "no preview" rather
  than a 500. The browser only ever requests allowlisted names, so 404 here is an
  edge case, not the normal path.

### 6. Sample generator (committed assets)

- `evals/gen_voice_samples.py`, run manually with `GEMINI_API_KEY` set. Lives in
  `evals/` because it imports product code (`GeminiLiveSession`) and hits the live
  API — `scripts/` may do neither (scripts import no product code).
- For each voice in `GEMINI_LIVE_VOICES`: open a `GeminiLiveSession` with that
  `voice_name`, prompt one short English greeting (default:
  "Hi, I'm Daemon. This is what I sound like."), collect the audio bytes from
  `receive()` until the turn boundary, and write
  `daemon/admin/static/voice-samples/{voice}.mp3`.
- Encoding: the session returns 24 kHz mono 16-bit PCM. Encode to MP3 (~64 kbps,
  ~2 s) with `ffmpeg` invoked as a subprocess — an offline, one-time dependency on
  the generator's machine, not a runtime dependency of the daemon. Target ≈ 15–20 KB
  per clip, ≈ 0.5 MB for all 30. If `ffmpeg` is absent the generator errors with
  guidance rather than committing raw WAV.
- The generator only needs the network session, not `AudioIO`/PortAudio, so it runs
  without the `voice` extra's hardware dependencies.

## Testing

- `tests/test_admin.py`: a valid `gemini_live_voice` PATCH writes the `.env` key and
  is reflected in the next GET; an invalid voice returns 400 and writes nothing;
  `GET /admin/api/voice-sample/{voice}` returns audio for a present sample, and 404
  (never a filesystem read for the name) both for a known voice with no file and for
  a name outside the allowlist.
- `tests/test_config.py` (or the existing config test module): an unknown
  `DAEMON_GEMINI_LIVE_VOICE` raises `ConfigError`; empty and a known name both
  construct.
- Real-API check (project rule: green units are not proof): run
  `evals/gen_voice_samples.py` for at least one voice and confirm audio comes back;
  run `daemon voice` with a chosen voice and confirm the spoken timbre changes.

## Out of scope — Phase B (next cycle)

Its own spec → plan → live-API verification:

- `daemon/voice/openai_realtime.py`: a full `VoiceSession` implementation over the
  OpenAI Realtime WebSocket API.
- `DAEMON_VOICE_PROVIDER` (`gemini` | `openai`): a dedicated voice-provider selector
  in the admin, independent of the text `hosted_provider`. Voice routing
  (`route_for`, currently hardwired to `gemini_live_model`) grows this dimension.
- `openai_realtime_voice` (alloy/ash/… ) beside `gemini_live_voice`, grouped under
  the provider selector in the admin UI.
- Known gap to design there: OpenAI Realtime has no realtime video-frame input, so
  `send_frame` (screen share, ADR 0009) cannot be supported on that provider —
  decide no-op vs. clear "screen share needs Gemini".
