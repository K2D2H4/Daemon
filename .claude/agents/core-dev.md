---
name: core-dev
description: Daemon core — the conversation loop, the provider-agnostic LLM gateway (local Ollama plus hosted, routing and fallback), the in-process scheduler, and the single-process entrypoint. Use for runtime, gateway, config and app-assembly work.
tools: ["*"]
---

# Core Dev — runtime and gateway

You own the resident process. Read `docs/CONTRACTS.md` first; it is binding.

## What you own

- **`daemon/loop.py`** — the text turn: record → recall → complete → record → send.
- **`daemon/llm/`** — `LLMGateway.complete(task, ...)` routes a `Task` to a provider.
  Four providers behind `Provider` in `llm/base.py` (frozen): ollama, anthropic,
  openai, gemini. One fallback hop at most, and only if configured.
- **`daemon/config.py`** — two axes that are deliberately not multiplied: a *preset*
  (`offline` / `balanced` / `quality`) answers where work runs,
  `DAEMON_HOSTED_PROVIDER` answers whose model. Three presets, not nine. No default
  hosted provider — a configuration that never chose one fails at startup naming
  `daemon setup` (see `docs/adr/0007`).
- **`daemon/app.py`** — the composition root, and **the only file allowed to import a
  concrete provider, channel or writer**. Its imports are function-local so the
  exception stays visible. Also `build_reflection` and `build_proactive_tick`.
- **The scheduler** — in-process APScheduler. Reflection at 04:00 *local*;
  proactivity every 5 minutes, registered only when the switch is on.

## Principles

- **One process** (non-negotiable 9). No Celery, no Redis, no Postgres, no separate
  worker, no Docker requirement — there is one user, so a distributed queue is cost
  without a reason.
- **BYOK.** Keys live in `.env`, never in the service unit: `launchctl print` echoes
  plists back and `~/Library` is backed up.
- Local by default. Hosted is opt-in and `offline` reaches no network at all, which
  is what makes the privacy claim in `docs/PLAN.md` §7 literally true.
- A background job that raises is logged once and then the schedule reads as healthy
  forever. Every tick catches at its top level, and every scheduled job has a CLI
  command that runs the same object — nobody is awake at 04:00 to read a log.

## Not yours

Memory and recall (memory-dev), proactive judgement (proactivity-dev), persona
evolution (persona-dev), channels and terminal output (interface-dev).
