# 0004 — Voice is hosted native audio

**Status:** accepted, argument revised after measurement · 2026-08-03

## Context

Voice is how the two differentiators are actually felt; text carries about half of
them. Two ways to build it: a local cascade (STT → model → TTS) or a hosted
native-audio model.

## Decision

Hosted native audio. Gemini Live first, OpenAI Realtime second behind the same
protocol — supporting two is the only real mitigation for depending on either.

Three things forced it away from local, none of them preference:

1. A cascade **structurally discards paralinguistics**. Tone, hesitation and pace
   are lost at the STT step, and the TTS reads flat text with no knowledge of
   context. Listeners identify a robotic reply within ~400 ms when it fails to
   match sarcasm or urgency.
2. Open-weight native audio exists (Qwen3-Omni, Apache 2.0) but the Apple Silicon
   MLX conversion is **text-only** — no audio generation path on this machine.
3. Korean. Moshi and similar are English/French-centric; there is effectively no
   Korean unified speech-to-speech.

Hosted realtime also returns **transcripts**, which is what keeps memory and
persona evolution working in voice mode. Without them the differentiator switches
itself off.

## Consequences, and the part we got wrong

The latency argument does not hold up. Research put unified speech-to-speech at
200–300 ms and tuned pipelines at 0.7–1.1 s, and we cited that. **Measured against
the live API: 740 ms to first audio** — the low end of the pipeline range, not the
native-audio end.

So the decision stands but the justification is narrower than we wrote: it rests on
the paralinguistic round trip, which a cascade cannot carry at any latency. PLAN
§6.5 now says that and not the other thing.

Costs accepted: audio leaves the machine in voice mode, which required rewriting the
privacy section honestly; a preset voice rather than an arbitrary cloned one; and
per-minute billing, which is why a session opens per conversation rather than
staying open.

## What would change our mind

An open-weight native-audio model that generates on Apple Silicon and handles
Korean. Also worth watching: Google's docs now recommend the Interactions API and
Live API is still Preview, with two documented model ids already shut down.
