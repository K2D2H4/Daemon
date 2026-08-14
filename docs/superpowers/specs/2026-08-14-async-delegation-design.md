# Async delegation for the voice path — design

- Date: 2026-08-14
- Status: approved for planning
- Branch of origin: `claude/daemon-notion-write-hang-95c345`

## Problem

Over voice, the daemon reliably **reads** (search, fetch, list) and performs the
side-effecting actions the tool contract *names* — `open_path`, `run_command` — but
it **cannot perform writes it is not told about**: asked to create a Notion subpage
it says "만들었어요" and calls no tool at all, or invents a fake error ("만들려다
오류가 났어"). Nothing is created; the assistant confabulates.

## Evidence (measured, not argued)

`evals/voice_write_nudge_spike.py` drives the live `gemini-3.1-flash-live-preview`
socket with Korean TTS audio (no microphone), one crowded 80-tool declaration list,
N sessions per cell:

| cell (same 80-tool set) | create/act call rate |
|---|---|
| `open_path` — flat schema `{target: string}` | **4/4** |
| `notion-create-pages` — nested schema `{parent{…}, pages[…]}` | **0/4** (confabulated) |
| `notion-create-pages` + a "name your writes" contract nudge | **0/4** (nudge inert) |
| `create_note` — flat schema `{title, content}` | **4/4** |

**The wall is argument-schema complexity, not tool crowding, not the contract
wording, not read-vs-write.** The native-audio model emits a flat-argument tool call
every time and a nested-argument one almost never, faking the result instead. This
also matches upstream reports: [python-genai #813] (native-audio "claims a call, the
call does not go through, and it persists for the session") and [#843].

[python-genai #813]: https://github.com/googleapis/python-genai/issues/813
[#843]: https://github.com/googleapis/python-genai/issues/843

The audit agrees: over the whole voice history, `open_path` ran 23×, `run_command`
ran, every read ran — and no `notion-create-pages` / `write_file` / any nested-schema
write ever fired once. The one successful Notion create in the record was on
**telegram** (text), which handles nested schemas and the Gemini-3 `thoughtSignature`
contract fine (`evals/m1c_text_tools_spike.py`).

## Idea (owner's)

Give the voice model **one flat tool it can always call** — `delegate_task(request)`,
a single free-text string argument — that hands the natural-language task to the
**text** agent (which the evidence shows is reliable), runs it **asynchronously**, and
lets the voice agent say only "started, I'll report back", then reports the result
when it lands.

## Decisions (approved)

1. **Voice gets flat tools + `delegate_task` only.** Nested-schema tools are not
   offered to a voice session at all — the model cannot call them, so offering them
   only invites confabulation. Split is by **automatic schema classification**, not a
   hand-maintained whitelist.
2. **Presence-routed reporting**, reusing `ProactiveDelivery` — speaker + telegram
   when the owner is at the machine, telegram when away — **not** counted against the
   proactive budget or the silence clock (this is a result the owner asked for, not
   daemon-initiated speech).
3. **Durable.** The task is persisted in SQLite before the ack is spoken, so a restart
   (this session's own failure mode) cannot silently swallow a promised task. v1 is
   minimal: persist request + status, a single worker, and on boot **report** any
   unfinished task; full automatic resume is later.

## Non-goals (v1, YAGNI)

Parallel workers (execution is serial), automatic mid-task resume after restart (boot
only *reports* the unfinished task), cancellation, progress/"still working" queries,
and recursive delegation (the delegated text agent is not offered `delegate_task`).

## Architecture

Flow:

1. Voice: "UJET JD 밑에 예상질문 하위페이지 만들어줘."
2. The voice model calls `delegate_task(request="…")` — flat schema, so it *can*
   (measured 4/4).
3. The tool **commits a `queued` row** to SQLite and returns a flat, truthful ack
   string ("접수했어. 백그라운드로 처리하고 끝나면 알려줄게."). The voice speaks that.
   It does **not** run the task — the per-minute socket is never held open on it.
4. The worker claims the row, runs the request through the **text tool loop** with the
   full tool set (search → `notion-create-pages` with nested args, which text handles),
   captures the final reply text, marks the row `done`.
5. The worker delivers the reply through presence routing.
6. On restart mid-task: boot finds the non-terminal row and reports "아까 그거 못
   끝냈어" (v1) rather than losing it.

### Components

| # | Component | Location | Responsibility |
|---|---|---|---|
| 1 | **Voice tool filter** | `daemon/companion.py` `specs()` | New `surface: "text"\|"voice"` param. `voice` → flat-schema tools + `delegate_task`; `text` → all tools except `delegate_task`. `run_voice` passes `voice`; `loop.py` passes `text`. |
| 2 | **Schema classifier** | `daemon/companion.py` (or `tools/`) | `is_flat_schema(params)`: true iff every top-level property is a primitive (`string`/`number`/`integer`/`boolean`) or enum thereof, with no `object`/`array` property and no `$ref`/`anyOf`/`oneOf`/`allOf`. Unit-tested against real specs. |
| 3 | **`delegate_task` tool** | `daemon/tools/` (builtin-style) | Flat spec `{request: string}`. On call: `enqueue` a durable row, return the ack. Never executes the task. Registered in the registry so the runner can execute it; offered only to `voice` by (1). |
| 4 | **Task store** | `daemon/memory/schema.sql` (**FROZEN — change declared here**), `daemon/memory/store.py` | Table `delegated_tasks(id, request, status, result, error, created_ts, finished_ts, origin, channel, sender_id)`; `status ∈ {queued, running, done, failed}`. Methods: `enqueue_task`, `claim_next_queued`, `mark_running/done/failed`, `pending_tasks`. |
| 5 | **Worker** | `daemon/` new module (e.g. `delegation.py`), assembled in `daemon/app.py` lifespan | Serial loop: claim → run via **capture-channel + `ConversationLoop.handle`** (the proven text path) → mark → deliver. Boot: `pending_tasks` → report unfinished. |
| 6 | **Delivery variant** | `daemon/proactivity/delivery.py` | A path/param that routes a delegated result by presence **without** consuming the proactive budget or logging it as a `proactive` utterance for the silence clock. |

### Reuse seams (verified in this session)

- `ConversationLoop.handle(InboundMessage)` is the full proven text turn (recall,
  tools, memory writes) but ends by `self._channel.send(...)`. The worker supplies a
  **capture channel** — a `Channel` whose `send()` records the reply text and whose
  `listen()` is unused — so the whole real loop runs headless and the worker takes the
  captured text to deliver. No duplication of the tool-round logic.
- The worker builds a **text-mode `Companion` + `LLMGateway` (`Task.CHAT_TEXT`)** with
  the full tool registry, mirroring how `daemon/app.py` assembles the text channel
  loop. Origin is `owner` (a microphone has no relay path; the request came from the
  owner), so the tool policy's origin gate still holds.

## Truthfulness invariants (the point of the whole thing)

- The ack is returned **only after** the row is committed — so "접수했어" is never a lie.
- The result is delivered **only after** the worker actually completes the run — never
  from the voice model's imagination.
- A failed task delivers the **failure**, honestly ("그거 하려다 실패: <이유>").
- Because voice is only ever offered a flat `delegate_task`, the confabulation-inducing
  nested call is never on the table.

## Error handling

- Task raises / provider errors: `mark_failed(error)`, deliver the failure.
- Worker/process dies mid-run: the row stays `running`; boot recovery finds it and
  reports it (v1) — not silently dropped.
- Delivery reaches nobody: same contract as `ProactiveDelivery` today (the result is
  still recorded; it is not re-sent forever).

## Testing

- **Unit**: `is_flat_schema` against `open_path` (flat), `run_command` (flat),
  `notion-create-pages` (nested), a bare `{}`; `specs(surface="voice")` excludes
  nested tools and includes `delegate_task`, `specs(surface="text")` is unchanged and
  excludes `delegate_task`; `delegate_task` commits a `queued` row and returns the ack
  **without** running anything; the worker takes a `queued` row through a fake
  capture-channel/loop to `done` and calls delivery; a failing run → `failed` +
  failure delivered; boot recovery reports a `running`/`queued` row left by a restart.
  Korean cases where text is involved.
- **Reachability** (`tests/test_reachable.py`): register the new tool, worker class,
  and store methods, or declare them PENDING with the owning milestone.
- **Premise already proven**: `evals/voice_write_nudge_spike.py` establishes the
  schema wall live; it is the evidence this whole design rests on and stays in `evals/`.

## Frozen-contract note

`daemon/memory/schema.sql` is frozen (CLAUDE.md / CONTRACTS.md). This design adds one
table, `delegated_tasks`; the change is declared here and must be called out in the
implementing PR. No existing table or column changes.

## Open questions for planning

- Exact worker cadence: an APScheduler job vs a lifespan `asyncio` task polling the
  queue vs an `asyncio.Event` the tool sets on enqueue. (Prefer event-driven with a
  poll fallback.)
- Where `is_flat_schema` lives (companion vs a small `tools/` helper) — a plan detail.
- Whether the ack text is fixed or model-composed (fixed is safer for truthfulness).
