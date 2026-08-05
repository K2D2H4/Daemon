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
  tui.py              terminal presentation: colours, boxes, CJK-aware widths
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

5. **`data/persona/seed.md` is human-owned. Code must never write to it.**
   That asymmetry is the anchor that prevents personality collapse.
   `data/persona/learned.md` is AI-owned; humans only read it or request deletion.

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

**Unit tests are not enough, and we learned that the expensive way.** A milestone
shipped with 470 passing tests and failed on first contact three separate ways:
`daemon run` refused to start, the bot never answered, and voice was reported
complete while nothing could reach it. Every one of those was invisible to unit
tests and obvious after thirty seconds of actually using the thing. So two more
kinds of test are required, and a change is not done without them:

- **`tests/test_reachable.py` — is it reachable?** Every `Task` needs a caller,
  every nameable provider needs to be buildable, every protocol implementation
  needs something that constructs it. Anything genuinely not built yet must be
  declared PENDING with the milestone that owns it. The check runs both ways: a
  stale PENDING fails too, so the file cannot quietly stop working. The recurring
  defect it exists for is *contract satisfied, unit-tested, unreachable*.
- **`tests/test_acceptance.py` — does the journey work?** Assemble the app the way
  the entrypoint does, drive a real conversation through it, and assert the whole
  chain the product promises: the reply, the markdown, the mirror, the vector, and
  the recall on the next turn. Fakes stop at the network edge, because the defects
  live between.

And when you have run the suite, **run the product**. `pytest` passing is not the
same as it working, and the difference is where every defect above lived.


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

M1a, M1b, and M2 are done. M3's pipeline is done too — generators, gate,
judge, delivery, and the label loop are all wired — but its own gate stays
open: precision needs weeks of labels, not more code (docs/PLAN.md 8.2.2).

Now building **M4**:

> Two weeks of real observations make the personality shift felt (by which
> point the log is already three months old).

M4's code is done as well — `daemon/persona/loader.py` (assembles
`seed.md` + `learned.md`, read every turn), `daemon/persona/rules.py` (the
only write path for `data/persona/learned.md` and its `persona_rules` mirror),
and `daemon/persona/evolve.py` (the weekly pass: observations → rule
proposals, at most one model call) — wired into `daemon/app.py` (a Monday
05:00 job, one hour after reflection) and `daemon/cli.py` (`daemon persona`,
`daemon persona evolve --force`, `daemon persona forget <id> --why`).

Its gate has no input to measure yet, the same shape as M3's: the live
database before M4 held 0 observations, 0 persona rules, and no resident
process installed (no LaunchAgent). Blocked on wall-clock, not on code
(docs/PLAN.md 8.2.3).

Still out of scope: the type-E associative candidate generator (PLAN.md
§6.1), the `osascript`-under-LaunchAgent question (PLAN.md §6.3.1), and
pointing `daemon/proactivity/judge.py` at learned rules — it deliberately
stays seed-only, a separate decision from this milestone
(docs/design/2026-08-05-m4-persona-design.md). The `recalled = 1` hygiene
rule starving the observation table (PLAN.md §9) is a related but separate
open item, not fixed here.
