# OpenAI voice in the admin — Phase B2 (provider/voice UX + preview)

**Status:** approved design, pre-implementation
**Base:** stacks on B1, rebased onto **main v0.1.37**. B1 landed the OpenAI Realtime
`VoiceSession`, `DAEMON_VOICE_PROVIDER` routing, and the provider-aware config; this is the
admin UX on top, in the same PR. **The admin was rewritten in v0.1.37 (#73, "a rail, a
timeline… and the settings it was hiding")** — the plan targets that current structure:
`renderSettings` builds cards (MODEL / VOICE / KEYS …), fields come from helpers
`fieldStr(name,value,opts)` / `fieldVoice(value,opts)` / `fieldSecret` / a `.sw` switch, and
there is dirty-tracking + a revert. It now exposes many previously-hidden fields, including
`gemini_live_model` — so `openai_realtime_model` joins it symmetrically (see below), and the
"gemini_live_model stays .env-only" note from the pre-rewrite plan no longer applies.

## Problem

B1 made `DAEMON_VOICE_PROVIDER=openai` work, but only via `.env`. The admin still shows
only the Gemini voice picker (Phase A). The owner cannot choose the voice provider, pick an
OpenAI voice, set the OpenAI realtime model, or preview OpenAI voices from the web.

## Goal

In the admin Settings tab: a `voice_provider` dropdown (gemini | openai); the voice dropdown
and `▶` preview follow the chosen provider (Gemini's 30 voices + Gemini clips, or OpenAI's 10
+ OpenAI clips); and `openai_realtime_model` is editable — so OpenAI voice is fully
configurable from the web. OpenAI preview clips are bundled like the Gemini ones.

## THE binding constraint: do not regress Gemini Live

The Gemini voice path (Phase A, shipped v0.1.20, working) must behave **exactly as before**
after B2. This governs every change below:
- `voice_provider` defaults to `gemini`, so an untouched install is byte-for-byte the current
  Gemini behavior.
- Replacing the `/{voice}` sample route with `/{provider}/{voice}` and moving the 30 Gemini
  clips to `voice-samples/gemini/` is done **within this one PR** — route, UI fetch, and the
  moved files change together, so Gemini preview keeps working through the new path.
- The Gemini voice dropdown still lists the 30 voices, still PATCHes `gemini_live_voice`, still
  previews — just reached when `voice_provider=gemini`.
- QA (below) drives BOTH providers live in a browser and confirms Gemini still selects, saves,
  and previews.

## Components

### 1. `daemon/admin/settings_io.py` — surface the fields
- `STR_FIELDS +=` `"voice_provider": "DAEMON_VOICE_PROVIDER"`,
  `"openai_realtime_voice": "DAEMON_OPENAI_REALTIME_VOICE"`,
  `"openai_realtime_model": "DAEMON_OPENAI_REALTIME_MODEL"`. (GET/PATCH + validate-before-write
  come for free, as in Phase A.)
- `options +=` `"voice_providers": list(VOICE_PROVIDERS)` and
  `"openai_realtime_voices": ["", *sorted(OPENAI_REALTIME_VOICES)]`. (`gemini_live_voices` is
  already there; `gemini_live_model` is already an admin field as of v0.1.37, so
  `openai_realtime_model` joins it — both realtime models editable from the web.)

### 2. `daemon/admin/routes.py` — provider-namespaced preview route
- Replace `GET /admin/api/voice-sample/{voice}` with `GET /admin/api/voice-sample/{provider}/{voice}`.
- Validate `provider in VOICE_PROVIDERS` AND `voice in {gemini: GEMINI_LIVE_VOICES, openai:
  OPENAI_REALTIME_VOICES}[provider]` **before** building any path (closes traversal, as Phase A
  did for the voice name). Serve `VOICE_SAMPLES / provider / f"{voice}.mp3"`, else 404.
- `VOICE_SAMPLES` stays `static/voice-samples`; clips now live under `gemini/` and `openai/`.

### 3. `daemon/admin/static/index.html` — provider-aware voice UI
- Add a `voice_provider` dropdown (`fieldStr('voice_provider', …, o.voice_providers)`).
- The voice row is rendered for the **currently selected** provider: gemini →
  `gemini_live_voice` select over `o.gemini_live_voices` + preview; openai →
  `openai_realtime_voice` select over `o.openai_realtime_voices` + preview. On a
  `voice_provider` change, re-render the voice row (a `change` listener, delegated on
  `#settings-form` like the ▶ handler). Only the active provider's voice select carries a
  `data-f`, so `collectPatch` PATCHes the right key and leaves the other provider's saved voice
  untouched.
- `▶` preview fetches `/admin/api/voice-sample/<provider>/<selectedVoice>` (empty voice → no
  request, per Phase A). Reuse the existing `.voice-preview` markup + graceful "no preview
  available yet" fallback (v0.1.20).
- Add `openai_realtime_model` as a text field (so OpenAI voice is fully web-configurable).
- Keep the v0.1.20 save-behavior (reflect saved patch locally, "restart to apply" note); the
  provider re-render reads from the local `SETTINGS.editable`, so a just-saved provider stays
  shown.

### 4. Assets + generator
- `git mv daemon/admin/static/voice-samples/*.mp3 daemon/admin/static/voice-samples/gemini/`.
- Extend `evals/gen_voice_samples.py` to run **two passes in one invocation**, one per
  provider: the Gemini pass (unchanged) uses `GeminiLiveSession` for `GEMINI_LIVE_VOICES` →
  `voice-samples/gemini/<Voice>.mp3`; the OpenAI pass uses `OpenAIRealtimeSession` for
  `OPENAI_REALTIME_VOICES` → `voice-samples/openai/<voice>.mp3`. Each pass is **independently
  skipped** when its key/model env is absent (`GEMINI_API_KEY`+`DAEMON_GEMINI_LIVE_MODEL`;
  `OPENAI_API_KEY`+`DAEMON_OPENAI_REALTIME_MODEL`) and logs that it skipped — so a Gemini-only
  owner still regenerates Gemini clips and vice versa. Both passes capture 24 kHz PCM from
  `receive()` and encode to MP3 via ffmpeg (the existing helper).
- Generate and commit the 10 OpenAI clips now (the key is available), same English phrase.

## Testing

Keyless unit (no network/key/mic/speaker):
- `settings_io`: the three new fields are offered + PATCH-accepted; `options` carries
  `voice_providers` and `openai_realtime_voices` (empty-first); an invalid `voice_provider` or
  `openai_realtime_voice` is a 400 that writes nothing.
- `routes`: `/voice-sample/gemini/<known>` and `/voice-sample/openai/<known>` serve `audio/mpeg`
  for a present clip; 404 for a missing clip, an unknown voice, an unknown provider, and a
  traversal name — never a filesystem read for a bad name. **A Gemini-path serving test proves
  no regression** of the moved clips.
- `index.html`: shell still renders offline (no CDN).

Live (real, in a browser + the key):
- Generate the 10 OpenAI clips; spot-listen 2.
- **Gemini no-regression (the load-bearing check):** provider=gemini → the 30-voice dropdown
  still lists/selects, Save keeps the value, ▶ plays a Gemini clip (now from `/gemini/`).
- OpenAI: provider=openai → the 10-voice dropdown, ▶ plays an OpenAI clip (from `/openai/`),
  `openai_realtime_model` is editable and persists.
- Switching provider re-renders the voice row correctly; no console errors.
- Full suite + ruff + check_docs + reachability/acceptance gates green.

## Out of scope
- OpenAI `turn_detection` knobs in the admin (B1 uses server_vad defaults).
- OpenAI `input_audio_transcription` model choice (whisper-1 is fixed in B1).
