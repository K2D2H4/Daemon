# daemon/ — the process

## Owns

One process: a FastAPI control plane, an in-process APScheduler, a channel loop — assembled
in `app.py`, which owns every concrete-implementation import here.

## Layout

| | |
|---|---|
| `app.py` | composition root and lifespan. **The only file allowed to import concrete providers, channels and writers** — its imports are function-local to keep that exception visible. Also `run_voice`, `build_reflection` and `build_proactive_tick` |
| `cli.py` | `run` · `setup` · `install` · `uninstall` · `status` · `doctor` · `reindex` · `reflect` · `voice` · `proactive` · `pairing` · `wake calibrate` · `wake test` |
| `setup.py` | the onboarding wizard: preset, hosted provider, keys, persona seed, pairing |
| `wake_cli.py` | `daemon wake`: measure what the recognizer returns for the owner's phrase, save it as `DAEMON_WAKE_ALIASES`, then run the gate and print what fires. Writes `.env` through `setup.py`'s writer |
| `config.py` | settings and the three presets. `HOSTED` resolves to the chosen provider |
| `loop.py` | the text conversation: record → recall → complete → record → send |
| `reflection.py` | the 04:00 pass: one local day → curated facts, entity notes, observations |
| `tasks.py` | the `Task` enum. **Frozen** — it is the LLM routing key |
| `tui.py` · `service.py` | terminal presentation, CJK-aware widths, plain text when not a tty · the LaunchAgent / systemd user unit, which holds no secrets |
| `fs.py` · `clock.py` | 0700 dirs, 0600 files, and the two durable writes — append, and atomic replace · the one timestamp helper, so nobody scatters `datetime.now()` |
| `channels/` | `channels/base.py` (frozen) · `channels/telegram.py` · `channels/pairing.py` |
| `llm/` | `llm/base.py` (frozen) · `llm/gateway.py` · `llm/providers/` (4) · `llm/embedders/` |
| `memory/` | `memory/schema.sql` (frozen) · `store` · `log` · `writer` · `recall` · `curated` · `entities` · `reindex` |
| `voice/` | `voice/base.py` (frozen) · `voice/gemini_live.py` · `voice/audio.py` · `voice/conversation.py` |
| `proactivity/` | `proactivity/base.py` (frozen) · `candidates` · `gate` · `presence` · `judge` · `delivery` · `speaker` · `tick` |
| `persona/` | `persona/loader.py` (assembles `seed.md` + `learned.md` into one system message, read every turn) · `persona/rules.py` (the only write path for `data/persona/learned.md` and its `persona_rules` mirror) · `persona/evolve.py` (the weekly pass: observations → rule proposals, at most one model call) |

## Layering

Callers use `LLMGateway.complete(task, ...)` and the protocols in the frozen files above;
importing a concrete provider or channel outside `app.py` is a layering break. Changing a
frozen file is allowed, doing it silently is not.

```bash
daemon doctor                  # config, reachability, and what reflection has built
daemon run                     # the loop, in this terminal
daemon reflect                 # the 04:00 pass, now, by hand (`--date`, `--force`)
daemon proactive               # one proactivity round, verdicts only (`--speak` to let it)
daemon persona                 # active learned rules, last evolution, last diary
daemon persona evolve          # the Monday 05:00 pass, now, by hand (`--force`)
daemon persona forget <id>     # retire a learned rule - a human's deletion request (`--why`)
daemon reindex                 # rebuild the mirror from markdown - log, curated, entities, persona rules
daemon wake calibrate          # what does the recognizer actually hear you say? (`--takes`)
daemon wake test               # run the wake gate here and print what fires (`--seconds`)
python3 -m pytest tests/test_reachable.py   # is everything you built reachable?
```

## Common changes

**A new LLM provider.** Implement `Provider` in `daemon/llm/base.py` under `daemon/llm/providers/`, name
it in `HOSTED_PROVIDERS`, build it in `daemon/app.py`, offer it in `daemon/setup.py`. Missing the last two shipped once.

**A new channel.** Implement `Channel` in `daemon/channels/base.py`; the `Cursor` it needs is already
`daemon.memory.store.Store`. **Don't** invent a second allowlist — `daemon/channels/pairing.py` owns who the owner is.

**A new background job.** Register it on the scheduler in `daemon/app.py`'s lifespan
and copy the reflection job or the proactivity tick: `timezone=None` to fire on *local*
time, `max_instances=1` + `coalesce=True` so a slow run cannot stack, a top-level
`except` in the tick because a raising job is logged once and the schedule then reads
healthy forever, and **a CLI command running the same object** — nobody is awake at
04:00 to read a log. Register it only when its switch is on — an absent job is a clearer
"off" than a disabled one — and give polling a floor; see the last bullet.

**Anything a model writes into a path, a score or a date.** Reflection's output names files, scales
the recall score and can retire a fact the user stated, so `daemon/reflection.py` clamps every number,
narrows keys to a fixed charset, and checks names with `entities.safe_name` **and** a boundary on the
resolved path — a blocklist is not exhaustive. Date records by the day they are *about*.

**A new `Task`.** `daemon/tasks.py` is the routing key: add it to all three presets in
`daemon/config.py` *and* give it a caller, or `tests/test_reachable.py` fails unless it is declared
PENDING with the milestone that owns it.

## Depends on

Downwards only: `daemon/loop.py`, `daemon/voice/conversation.py`, `daemon/reflection.py`
and `daemon/proactivity/` depend on the protocols in `daemon/llm/base.py`,
`daemon/channels/base.py`, `daemon/memory/base.py` and `daemon/proactivity/base.py`,
never on what implements them; `daemon/memory/` depends on `daemon/fs.py` and
`daemon/clock.py` and on nothing above it. Depended on by [tests/](../tests/CLAUDE.md)
and nothing else; [scripts/](../scripts/CLAUDE.md) deliberately does not import this
package. Rules: [CONTRACTS.md](../docs/CONTRACTS.md). Data flow:
[ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## Things measured, not assumed

- **Running reflection for real found two defects unit tests did not.** Entity notes were
  stamped with the day of the *run*, so a months-long catch-up reads as all-today; and two
  facts sharing a supersession key retired the wrong half (`data/memory/core.md` kept the 3).
- **A 4B local model is unstable at reflection.** Same 6-message Korean fixture, two
  runs: 1 fact / 1 entity / 0 observations, then 2 / 2 / 1. Point it hosted if anything.
- **Two of the plan's three macOS presence probes were wrong as documented.** `HIDIdleTime`
  is **nanoseconds**; every `ioreg` audio class is absent on Apple Silicon, so CoreAudio via
  ctypes is the only one that answers (input too); `name of ... application process` gives
  the *executable* — `stable`, not `Warp`.
- **Narrowing the question is not enough to make a model decline.** Merely permitting it:
  **0 declines of 15**. Stating silence as the default and speaking as the exception: 3.
- **Do not use a reasoning model for local chat.** gemma3:4b answers in 1.7 s, qwen3:4b
  in 11.8 s, and the gap is chain-of-thought nobody sees; it cannot be turned off
  (`think: false` stops Ollama *separating* it, so it becomes the reply).
- **The vector lane is the cheap half of recall** (0.22 ms vs FTS5's 1.9 ms at
  10k). What dominates is the embedder round trip, ~117 ms, mostly fixed overhead.
- **FTS5 with `unicode61` cannot carry Korean recall alone** — whole-token matching,
  so an inflected word is a different token: 50% keyword-only vs 93% hybrid.
- **Two Telegram traps.** The inbound poll needs a floor — left to the long poll alone, a
  transport that returns immediately spins at ~16,000 requests/second. And `allowed_updates`
  is **server-side**: at `["message"]` a 👍 press is never delivered at all.
- **M4's gate had no input to measure.** The live database held 0 observations, 0
  `persona_rules`, no LaunchAgent installed — reflection had never run on real data. The
  cause: `Store.messages_for_day` excludes `recalled = 1` rows permanently (PLAN.md 4.2),
  which dropped 29 of 38 real messages in one day, and the excluded lines were exactly the
  persona-relevant ones.
- **`daemon doctor` and `daemon reflect` disagreed about the same day.** Doctor's backlog
  count includes today; `Reflection.catch_up` deliberately excludes it. Running the command
  doctor recommended answered "nothing to reflect on."
- **A real weekly evolution pass, run against Gemini with 30 seeded observations:** 30 read
  -> 7 proposed -> 3 added, 10.8 s, with the 4 the per-cycle cap dropped reported rather than
  discarded. A same-week rerun skipped in 0.64 s and made no model call.
- **Printing a real turn's assembled prompt found a defect.** `load_persona` was injecting
  all of `learned.md`, including its human-facing header (`daemon persona forget <id>`, a
  repeat of the sentence the loader already prefixes). Only the rule bullets go in now;
  `seed.md` still goes in verbatim.
- **`learned.md` was being rewritten from the mirror, so deleting the rebuildable sqlite file
  cost 5 rules out of 5 on the next ordinary write** — non-negotiable 1, measured, not argued.
  `add`/`retire` now refuse on divergence (`LearnedFileDiverged`), `daemon reindex` restores
  rows from the file additively, and `doctor` reports the divergence as a blocker. A crash
  between the two writes has the same shape: the instant after is allowed, the *next* write
  was what destroyed.
- **`write_private_replace` used one fixed temp filename**, so two writers on the same path
  (the Monday job and a hand-run `evolve`) raced: one `O_TRUNC`ed the other's bytes, then the
  loser's `os.replace` raised. Random suffix now. Two concurrent writers still do not merge —
  the later `replace` wins outright, and that is not fixed.
