# 0020 — Two endpoints serve Gemini Live, and neither is a superset

**Status:** accepted · 2026-09-02 · measured

## Context

Voice mode assumed one endpoint. `GEMINI_API_KEY` in a header, one URI constant in
`daemon/voice/gemini_live.py`, `models/{id}` on the wire, and a model id chosen
from what that endpoint lists. Everything about Gemini Live in this repo — the
model picker in the admin console, `daemon setup`'s probe, the ADRs about
latency — was written inside that assumption.

The owner preferred the timbre of `Despina` on 2.5 native audio over
`gemini-3.1-flash-live-preview`, and the measurement said that preference cost
about four seconds an answer:

| endpoint / model | first audio after speech ends |
|---|---|
| API key `gemini-3.1-flash-live-preview` | 1723 ms |
| API key `gemini-2.5-flash-native-audio-preview-12-2025` | 3137 ms |

Every knob was swept on the slow arm — `silence_duration_ms` at 1500/800/300, a
shorter system instruction, the `-09-2025` and `-latest` builds, the TTS model as
a cascade — and every one came back **worse than the default**. The conclusion
written down at that point was "the voice you want costs four seconds, and nothing
configurable changes it."

That conclusion was wrong, and what disproved it was one of the owner's other
services. ReadyTalk runs `gemini-live-2.5-flash-native-audio` and feels instant.
That id is not a build we had missed; it does not exist on the endpoint this repo
dials. The API-key endpoint closes 1008 `models/gemini-live-2.5-flash-native-audio
is not found` and never lists it. ReadyTalk reaches it through
`genai.Client(vertexai=True, location="us-central1")`.

Re-measured with the same stimulus, the same open-mic behaviour and the same SDK
on both sides, 5 trials per arm, interleaved:

| endpoint / model | median | spread |
|---|---|---|
| **Vertex `gemini-live-2.5-flash-native-audio`** | **1430 ms** | 41 ms |
| API key `gemini-3.1-flash-live-preview` | 1723 ms | 73 ms |
| API key `gemini-2.5-flash-native-audio-preview-12-2025` | 3137 ms | 772 ms |

With everything the resident actually sends (98 tool declarations, affective
dialog, LOW start sensitivity) the Vertex arm still answered in **1739 ms** with no
misses — the same latency as the 3.1 configuration it would replace, in the voice
the owner picked, with an expressive feature 3.1 does not support at all.

Three failures we had attributed to "2.5 native audio" were attributable to the
API-key **preview builds** instead:

| | API key 2.5 | Vertex 2.5 |
|---|---|---|
| `START_SENSITIVITY=low` | 12/12 turns silent, input transcript empty | 1420 ms, 0/3 |
| 98 tool declarations | +1380 ms | +127 ms |
| `enable_affective_dialog` | supported | supported, +147 ms |

## Decision

**The endpoint is a configuration axis, not a migration.** `api_key` stays the
default and `vertex` joins it, chosen by `DAEMON_GEMINI_LIVE_TRANSPORT` and
selectable in the admin console beside the model it constrains.

It is an axis because neither endpoint contains the other:

- The fast, steady native-audio model is **Vertex-only**.
- The newer generation is **API-key-only**. Vertex, checked across all five
  regions that serve any live model, has no conversational live model newer than
  2.5 — no 3.x live, no 4.x. `gemini-3.5-transcribe-live-preview` transcribes and
  does not speak.
- Vertex serves live models in us-central1, us-east1, us-east4, us-west1 and
  europe-west4 only. **asia-northeast3 (Seoul) serves none**, so a Korean
  self-hoster's nearest region is not an option — and from Seoul us-west1 measured
  identical to us-central1 (1441 ms both), so the delay is serving rather than
  distance.

It stays *default off* for a reason that is not about speed: `api_key` needs one
key in `.env`, and `vertex` needs a GCP project, a service account or an ADC login,
and a region that happens to serve the model. That is a different onboarding, and
`daemon setup` must not lead with it.

## Consequences

The protocol body did not change, which is what made this small: same
proto-over-JSON `setup`, same `serverContent`, same tool frames. Three things
differ and they are all `daemon/voice/vertex.py` — the regional URI, an
`Authorization: Bearer` provider, and the model as
`projects/{p}/locations/{l}/publishers/google/models/{id}`.

- **Credentials are now a callable, not a string.** An access token lives about an
  hour while a resident reconnects on its own schedule, so a token fetched at
  startup dies at a later handshake. `GeminiLiveSession` asks its `auth` provider
  per connect attempt.
- **Credential failures are permanent.** Every one this can reach — no
  credentials, an expired ADC login, an unreadable key file — needs a person, and
  the alternative is the failure mode ADR-era voice work kept hitting: a process
  that retries forever while `/health` says running.
- **The log filter had to learn new secrets after construction.** It held exactly
  one, given to the constructor. A bearer token is not known until the provider
  runs, and `websockets` logs handshake headers at DEBUG.
- **One new dependency, lazily imported.** `google-auth`, in the `voice` extra.
  This repo builds its own REST and websocket clients and had no Google SDK; the
  alternative was signing service-account JWTs by hand, which trades one
  dependency for a worse one.
- **A self-hoster can now pick a model their endpoint cannot serve.** The admin
  console offers the API-key catalogue from a live probe; the Vertex catalogue
  cannot be probed with an API key, so it is a named constant
  (`VERTEX_LIVE_MODELS`) that will go stale when Google ships a Vertex live model.
  `evals/vertex_live_spike.py` is how that gets re-checked.

**Not decided here:** which project and service account a Daemon install should
use, and what the endpoint does to the bill. The measurement above used the
owner's ReadyTalk service account, and ReadyTalk's own price table (that
service's pricing module, audio in $3.00 / out $12.00 per 1M) is not evidence
about the API-key path's billing. A resident holds sessions open all day, so that comparison
may matter more than the 1.4 seconds this ADR is about.

## Measured by

`docs/design/vertex-live-transport.md` (the sweep and the region survey),
`evals/vertex_live_spike.py` (this repo's own client against the endpoint, rather
than the SDK's).
