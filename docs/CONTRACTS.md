# Contracts — read this before writing any code

This file exists so that work happening in parallel does not produce
incompatible code. **These are not suggestions.** If a contract is wrong, say so
and stop — do not quietly work around it.

Design rationale for everything here: [docs/PLAN.md](PLAN.md).

## Layout

```
daemon/
  tasks.py            Task enum — the LLM routing key. FROZEN.
  config.py           settings + the 3 presets
  app.py              single-process entrypoint (FastAPI + APScheduler)
  llm/
    base.py           Provider protocol, Message, Completion. FROZEN.
    gateway.py        LLMGateway: routes Task -> Provider
    providers/        ollama.py, anthropic.py, openai.py, gemini.py
  channels/
    base.py           Channel protocol, InboundMessage, OutboundMessage. FROZEN.
    telegram.py
  memory/
    schema.sql        storage contract. FROZEN.
    store.py          sqlite access
    log.py            markdown log writer (the source of truth)
    recall.py         Lane 1 — must make ZERO model calls
  persona/
    loader.py         assembles seed.md + learned.md
  proactivity/        (M3)
tests/
  conftest.py         shared fixtures. Use them; do not invent parallel ones.
```

FROZEN means: do not edit without flagging it first.

## Non-negotiables

1. **Markdown is the source of truth. SQLite is a rebuildable index.**
   Deleting the sqlite file must never lose user data. Every write path writes
   the markdown first, then mirrors into sqlite.

2. **Recall Lane 1 makes zero LLM calls.** It is on the voice latency path. If
   you find yourself wanting a model call there, stop and flag it.

3. **Provenance is columns, never prose.** Never encode origin/importance/dates
   as markdown comments that a model could write or mangle. Columns only
   (`origin`, `session_kind`, `modality`, `created_at`, `importance`,
   `supersession_key`).

4. **Layering.** Nothing outside `daemon/llm/providers/` imports a provider.
   Nothing outside `daemon/channels/` imports a channel implementation.
   Callers use `LLMGateway.complete(task, ...)` and the `Channel` protocol.

5. **`persona/seed.md` is human-owned. Code must never write to it.**
   That asymmetry is the anchor that prevents personality collapse.
   `persona/learned.md` is AI-owned; humans only read it or request deletion.

6. **`observations` is append-only.** No UPDATE, no DELETE. Only `consumed_by`
   may be set later.

7. **Proactivity: silence is the default.** Candidate generation and the gate
   are deterministic (no model). Exactly one LLM call, and only for candidates
   that already passed the gate.

8. **Timestamps** are ISO-8601 UTC with `Z`, stored as TEXT. Use one helper,
   do not scatter `datetime.now()` calls.

9. **Single process.** No Celery, no Redis, no Postgres, no separate worker.
   Background work runs on the in-process APScheduler.

## Testing (required, not optional)

- `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`, so no decorator needed).
- **Every module you add ships with tests in the same PR-sized unit of work.**
- **No test may hit the network or a real LLM.** Use the `fake_provider`
  fixture. A test that needs an API key is a broken test.
- Database tests use the `db` fixture (fresh schema in `tmp_path`). Never touch
  a developer's real data dir.
- Assert behaviour, not implementation. Prefer one clear failing assertion over
  five weak ones.
- Cover the failure paths that matter: provider raising `ProviderError`,
  malformed inbound message, non-allowlisted sender, corrupt/absent markdown,
  concurrent write to the same file.
- Run `python3 -m pytest` and `python3 -m ruff check .` before declaring done.
  Report actual output; do not claim green without running it.

## Style

- Python 3.13. `from __future__ import annotations`, modern generics (`list[str]`).
- Async everywhere on the I/O path. No blocking calls inside async functions.
- Comments explain *why*, not *what*. Match the density of the existing files.
- No speculative abstraction. If it has one caller, it does not need an interface.
- Code, comments, commit messages, and identifiers in **English**. Design docs
  in `docs/` stay Korean.

## Milestone scope

M1a is done: a Telegram message gets an answer and the exchange lands in
`memory/log/YYYY-MM-DD.md`.

Now building **M1b**:

> It quotes yesterday accurately, it survives a reboot, voice works, and the
> golden set gives recall a number.

Four pieces, one owner each:

1. **Recall** — Lane 1, `daemon/memory/recall.py`. FTS5 **and** vectors, scored
   `similarity × recency decay (30d half-life) × importance`. **Zero LLM calls**
   (an embedder call is fine — local, milliseconds). Vectors are float32 BLOBs in
   the `embeddings` table, searched brute-force with numpy: no sqlite extension,
   because this Python build cannot load one and neither can many others.
2. **Voice** — `daemon/voice/`. Gemini Live first, behind `VoiceSession`.
   Audio hardware behind `AudioIO` so tests need none. Voice deps live in the
   `voice` extra.
3. **Residency** — LaunchAgent / systemd install, so proactivity (M3) has a
   process to run in after the terminal closes and the machine reboots.
4. **Golden set** — `evals/`. Recall gets a pass rate that moves when recall
   changes, so M1b's gate is a number rather than an impression.

Still out of scope: reflection, entity notes, observations, proactivity,
persona rules. Their tables exist so those milestones need no migration.
