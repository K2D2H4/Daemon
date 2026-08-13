# 0012 — Voice is its own axis, not a preset tier

**Status:** accepted · 2026-08-11

## Context

ADR 0007 split two axes: a preset answers *where work runs*, `DAEMON_HOSTED_PROVIDER`
answers *whose model*. Voice was later given a third, `DAEMON_VOICE_PROVIDER`, and
`Settings.routing` overrode `CHAT_VOICE` with it — the config comment says "voice
provider is its own axis" in as many words.

One thing contradicted that. `PRESETS["offline"]` has no `CHAT_VOICE` row, and the
validator turned the absence into a rule: `DAEMON_VOICE_ENABLED=true` under `offline`
failed at startup with "preset routes no voice task". So **local text with hosted
voice was unconfigurable** — a combination with an obvious owner (keep conversation,
reflection and the proactive judge on this machine; use Gemini or OpenAI for spoken
turns) and no way to ask for it.

The absence was defended as what made the privacy promise true. It is not what makes
it true. `docs/PLAN.md` §7 states the promise as "**text mode** + local models —
nothing leaves the device", and one line above it: "turn voice on and audio goes to
the provider you chose (BYOK); leave it off and it does not". The promise was already
conditioned on the switch. Only the table disagreed.

The admin redesign is what surfaced it: a page whose whole shape is "provider, then
that provider's model" had to grey out the voice toggle under a local provider and
explain a limitation that no requirement asked for.

## Decision

`Settings.routing` adds the `CHAT_VOICE` row whenever `voice_enabled` is on, under
every preset, mapped to `voice_provider`. The row is still kept when a preset carries
one and voice is off, so that `routing` stays a faithful rendering of the preset
table regardless of the switch. `route_for` itself already answers "voice is off
(DAEMON_VOICE_ENABLED)" before it ever consults the table, independent of whether
this row is present.

The validator problem that refused voice under `offline` is deleted as unreachable.
`providers_for` — which decides the keys onboarding asks for — contributes
`voice_provider` while voice is on, under every preset, instead of reading the preset
table's literal `CHAT_VOICE` entry. That second half also fixes a standing bug: a user
who chose OpenAI voice was asked for a Gemini key.

`PRESETS` is unchanged. No new preset, no new setting, no migration: every existing
`.env` keeps its meaning, and the configurations that used to fail at startup now
start.

## Consequences

- "Local text, hosted voice" works. `offline` + `DAEMON_VOICE_ENABLED=true` needs the
  voice provider's key and its own realtime model id, which the existing voice-model
  checks already demand.
- `offline` is no longer a promise by construction. Turning voice on there sends audio
  out, and the wizard's preset menu now says so instead of saying voice is
  unavailable. The promise moved from "this preset cannot" to "this switch is off",
  which is where PLAN.md always had it.
- One fewer message for one situation: voice-off now reports the switch under every
  preset rather than the preset under one of them.
- The admin's provider picker no longer needs to disable voice for a local text
  provider — the reason that rule existed is gone.

## What would change our mind

A voice path that could run locally. Every reason above assumes native audio means a
hosted model (ADR 0004). If a local native-audio session became real, "voice implies
something leaves the machine" stops holding and this record needs revisiting rather
than extending.
