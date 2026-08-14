# 0014 — The provider is the axis; presets are gone

**Status:** accepted · 2026-08-13

## Context

ADR 0007 split configuration into two axes: a **preset** (`offline` / `balanced` /
`quality`) answered *where work runs*, and `DAEMON_HOSTED_PROVIDER` answered *whose
model* answers wherever a preset said "hosted". Three presets, not nine, because the
two axes were not multiplied together.

After the admin redesign and the openai_compatible merge, that table stopped paying
for itself. `balanced` and `quality` differ in exactly one routed task -
`PROACTIVE_JUDGE` - so "preset" named a three-row table over a single boolean.
`offline` was not a tier at all; it was "everything local", which is what
`DAEMON_PROVIDER=ollama` already says once ollama is a nameable provider. And once
both the admin and the wizard ask "provider first, then that provider's model(s)"
(the M5 admin work), nobody types a preset name again - `DAEMON_PRESET` became a
table with no caller but the code itself, the exact shape ADR 0007 was written to
avoid on the provider side.

The admin card made the cost visible first: eight controls, one or two ever in
force, `preset` and `hosted_provider` both aiming at the same decision from two
directions.

## Decision

**Collapse the preset table into two axes a person actually sets.** `DAEMON_PROVIDER`
(the chat provider - `ollama` included, alongside `anthropic` / `openai` / `gemini` /
`openai_compatible`) and `DAEMON_PROACTIVE_JUDGE_LOCAL` (a bool). `Settings.routing`
is computed from these two rather than looked up in a `PRESETS` table:

| Task | role |
|---|---|
| `CHAT_TEXT`, `RECALL_ESCALATION`, `REFLECTION`, `PERSONA_RULE` | `DAEMON_PROVIDER` |
| `PROACTIVE_JUDGE` | `ollama` if `DAEMON_PROACTIVE_JUDGE_LOCAL` else `DAEMON_PROVIDER` |
| `EMBED` | always `ollama` |
| `CHAT_VOICE` | its own axis, unchanged (ADR 0012) |

`DAEMON_PROVIDER=ollama` reconstructs the old `offline` preset exactly - every
hosted role resolves to `ollama`, and the judge toggle stops mattering because there
is no hosted provider to send it to. The toggle **controls `PROACTIVE_JUDGE` only**,
not reflection: reflection was hosted in both `balanced` and `quality`, so folding it
into a "background work" toggle would have been false advertising. `route_overrides`
remains the hand-edit escape hatch for anyone who wants finer control than two axes
give; per-task exposure in the UI stays out, the same call ADR 0007 made.

**No migration (single-owner install).** `DAEMON_HOSTED_PROVIDER` is **renamed** to
`DAEMON_PROVIDER` - the old key is not read, not aliased, not silently accepted.
Encountering a stale `DAEMON_PRESET` in the environment **raises loudly at
construction**, naming both new keys, rather than being ignored: a
`DAEMON_PRESET=offline` install silently starting to dial a hosted provider is the
privacy-facing version of the footgun ADR 0007 refused to build on the model side. A
read-and-rewrite migration was rejected too - unneeded machinery for one install
that a human runs `daemon setup` on once.

This **amends ADR 0007**: its preset axis is replaced by the provider axis above;
its other half - no default provider, a hosted task routed with none fails loudly
naming `daemon setup` - stands unchanged, now keyed off `DAEMON_PROVIDER` instead of
`DAEMON_HOSTED_PROVIDER`. It **composes with ADR 0012**: voice stays its own axis,
independent of both `DAEMON_PROVIDER` and `DAEMON_PROACTIVE_JUDGE_LOCAL` - a local
chat provider does not disable voice, and a hosted one does not require it.

## Consequences

- One fewer concept to teach. `daemon setup` and the admin both ask "provider, then
  that provider's model(s), then that provider's extras" - the same order, the same
  two questions, on both surfaces.
- The admin's MODEL card shows only controls currently in force: a provider picker,
  that provider's model field(s), and the judge toggle when a hosted provider is
  picked. No more silent boxes belonging to a provider that is not selected.
- Every existing `.env` needs a one-time hand edit (rename `DAEMON_HOSTED_PROVIDER`
  to `DAEMON_PROVIDER`, drop `DAEMON_PRESET`) rather than loading unchanged - the
  cost of no migration, paid once, made loud rather than silent.
- `route_overrides` is now the only way to route a single task off the two axes,
  which was already true in practice (nobody exposed per-task routing in the UI
  before this either).

## What would change our mind

A need for per-task provider control common enough that `route_overrides`
(hand-edit-only) is too coarse for it - that would argue for a UI surface this ADR
deliberately keeps out, not for bringing the preset table back.
