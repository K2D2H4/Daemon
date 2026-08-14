# Contracts — read this before writing any code

This file exists so that work happening in parallel does not produce
incompatible code. **These are not suggestions.** If a contract is wrong, say so
and stop — do not quietly work around it.

Design rationale for everything here: [docs/PLAN.md](PLAN.md).

## Layout

```
daemon/
  tasks.py            Task enum — the LLM routing key. FROZEN.
  config.py           settings + the two routing axes (provider, proactive-judge-local)
  app.py              single-process entrypoint (FastAPI + APScheduler)
  companion.py        what the daemon can do, for both endpoints. Read this before
                      adding a capability to loop.py or voice/conversation.py.
  llm/
    base.py           Provider protocol, Message, Completion. FROZEN.
    gateway.py        LLMGateway: routes Task -> Provider
    providers/        ollama.py, anthropic.py, openai.py, gemini.py, openai_compatible.py
  channels/
    base.py           Channel protocol, InboundMessage, OutboundMessage. FROZEN.
    telegram.py
  tools/
    base.py           Tool + Executable protocols, Registry, ToolResult
    policy.py         ToolPolicy: the origin gate, modes, approvals. Read this first.
    builtin.py        the seven built-in tools
    browser.py        fetch_page, list_tabs, read_page. Read the module docstring.
                      `system_state` delegates to proactivity/presence.py - one
                      implementation of the machine probes, not two.
    mcp.py            MCP client; the only file that imports the `mcp` package
    runner.py         ToolRunner: decide -> execute -> audit, in that order
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

`daemon/llm/base.py`'s `Message` now carries `images` and `daemon/voice/base.py`'s
`VoiceSession` now has `send_frame` - both additive, both for screen sharing,
justified in [docs/adr/0009-images-in-the-message-contract.md](adr/0009-images-in-the-message-contract.md).

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
   make zero *LLM* calls - same distinction non-negotiable 2 draws, and the same
   allowance: type E's `association_candidates` awaits the embedder every tick,
   which is not a call that thinks. Exactly one LLM call, and only for candidates
   that already passed the gate.

8. **Timestamps** are ISO-8601 UTC with `Z`, stored as TEXT. Use one helper,
   do not scatter `datetime.now()` calls.

9. **Single process.** No Celery, no Redis, no Postgres, no separate worker.
   Background work runs on the in-process APScheduler.

10. **No tool runs on a turn whose origin is not `owner`.** Not in any mode, not
    with any setting, `full` included. `InboundMessage.authored_by_sender` is False
    for a forward or an inline-bot result, and recall replays arbitrary old text
    into every prompt - so without this gate, "look at this message" is a way to
    hand a stranger a shell. Enforced twice on purpose: `Companion.specs`
    (`daemon/companion.py`) offers such a turn no tools at all, and
    `tools/policy.py:decide` refuses every call regardless of mode, allowlist or
    standing grant. The offering side is a convenience; `decide` is the guarantee.
    The offering side lives with the capability rather than in an endpoint so that a
    second endpoint getting tools cannot get them ungated. If you find yourself
    needing an exception, stop and flag it.

11. **The tool policy makes no model calls.** Same rule as recall Lane 1 and for a
    different reason: a gate that asks a model whether to open the gate is not a
    gate. OpenClaw's `auto` mode (an LLM reviewer for edge cases) was deliberately
    not ported.

12. **Every executed tool call leaves a `tool_calls` audit row.** That row is the
    owner's ground-truth record of what touched the machine, readable with `daemon
    tools log`; a tool that ran without one is a defect, which is why `ToolRunner`
    owns decide, execute and audit together instead of exposing them separately. The
    reply the owner reads carries **only the model's answer** - the raw
    `run`/`write`/`rm` lines are not folded into it, in text or spoken aloud in
    voice, because narrating every call reads as clutter and the audit is the record
    that cannot be omitted or misrepresented by the model's prose. Tools are **on**
    by default, and the mode is **`full`** - a guarded
    tool runs without asking. The safety is not the mode and not the switch: it is
    rule 10, the origin gate, which no mode can turn off. `full` is the default
    because a companion that stops to ask before every action on the owner's own
    machine is the "chat with extra steps" this product exists not to be; `ask` and
    `allowlist` remain for anyone who wants a prompt before the machine changes. So
    `daemon doctor` has to say which way it is set - a capability nobody was asked
    about and which is reported nowhere is the silent state this project keeps being
    bitten by. The browser group - the one that reads an authenticated session the
    owner never named - stays off behind its own switch; MCP keeps its own switch
    too but defaults on, since it launches nothing until a server is configured.

13. **Code is never built from data.** Three places would be natural to get this
    wrong and all three are done the same way — the script is a constant and the
    values travel as `argv`:
    - `notify`'s AppleScript (a title with a quote in it would be AppleScript),
    - `read_page`'s JavaScript (it runs in a logged-in session),
    - `run_command` (there is no shell, so an argv vector, never a string).

    **There must be no tool that runs model-supplied code in the browser.** Not
    `execute_javascript`, not an `eval` parameter, not a `script` argument. The
    owner's browser holds their sessions; a page's author must never reach it
    through us. `tests/test_browser.py` asserts this and is meant to.

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
- **`tests/test_e2e.py` — does the *assembled daemon* work?** One layer further out:
  boot the real `create_app` and its lifespan, so the real `_build_io`, the real
  `_build_tools`, the real loop task and the real Telegram channel all run. Messages
  go in through `getUpdates` and replies are read off `sendMessage`, so every
  assertion is about what a person would see on their phone. Exactly three things
  are faked and all three are network edges: the model, the embedder, Telegram's
  HTTP. It also covers what the other two structurally cannot - a **restart** with
  the same data dir, and shutdown leaving nothing running.

And when you have run the suite, **run the product**. `pytest` passing is not the
same as it working, and the difference is where every defect above lived.


- `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`, so no decorator needed).
- **Every module you add ships with tests in the same PR-sized unit of work.**
- **No test may hit the network or a real LLM.** Use the `fake_provider`
  fixture. A test that needs an API key is a broken test. Verifying against a live
  API is not a test and lives in `evals/` — run by hand, never in CI — which is
  where a real-key check like the Gemini tool round-trip
  (`evals/m1c_text_tools_spike.py`) belongs. It is what catches a contract a mock
  cannot see, the way `thoughtSignature` slipped a green 2.5-pinned suite.
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

And **M1c — PC control**, pulled forward from post-M4 (docs/PLAN.md 8.2, §10):

> It can read a file, run a command it has been allowed to run, and ask before
> anything else - and a forwarded message can do none of it.

One piece: `daemon/tools/`, plus tool calling in the provider contract and four
tables in `daemon/memory/schema.sql`. Both frozen files were extended additively; nothing that
compiled before stopped compiling. No new `Task` - tool-using chat is still
`chat_text`, so routing is untouched.

The fourth table, `tool_grants`, was added later and is a *second* standing axis
rather than a widening of the first. `tool_allowlist` holds argv prefixes, and a
tool with no argv can never match one - so `mode=allowlist` was a permanent refusal
for `write_file` and for **every** MCP tool (`McpTool` is deliberately not
`Executable`), with no setting that reached it. `tool_grants` allows a whole tool;
`tools/policy.py:decide` reads it only for tools that are *not* `Executable`,
because a tool-level grant on `run_command` would be `mode=full` wearing one table
row. Rule 10 above is unchanged and covers it: a grant does not reach past the
origin gate, the switch, or `mode=off`.

Then **the browser group** (`daemon/tools/browser.py`), behind its own
`DAEMON_BROWSER_ENABLED`:

> It can read the page the owner is looking at, so they can say "what does this
> say?" instead of pasting it - and a forwarded message still cannot.

Still out of scope: the `osascript`-under-LaunchAgent question (PLAN.md §6.3.1).
The type-E associative candidate generator (PLAN.md §6.1) that used to be listed
here is built — `daemon/proactivity/candidates.py`'s `association_candidates`,
wired into `daemon/proactivity/tick.py` — and its one exception to "no user text
in a reason" is docs/adr/0013.

Pointing `daemon/proactivity/judge.py` at learned rules used to be out of scope
too — it deliberately stayed seed-only, a separate decision from M4
(docs/design/2026-08-05-m4-persona-design.md) — and no longer is: `daemon/proactivity/judge.py` now
calls `load_persona`, so a proactive utterance carries the same learned rules the
text loop and voice already do, one persona regardless of which path reached the
user. Decided 2026-08-11, in the same milestone that built type E.

The `recalled = 1` hygiene rule that was starving the observation table **was**
fixed, after being scoped out first: PLAN.md §4.2's rule 2 is retired, because it
cost 29 of 38 messages on a real day and blocked no loop — recall's hits are
injected as a system block and never become rows. Read
`Store.messages_for_day`'s docstring before reinstating any filter on that flag.
