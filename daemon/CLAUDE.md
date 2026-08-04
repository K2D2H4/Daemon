# daemon/ — the process

## Owns

One process: a FastAPI control plane, an in-process APScheduler, a channel loop.
Assembled in `app.py`, which owns every concrete-implementation import here.

## Layout

| | |
|---|---|
| `app.py` | composition root and lifespan. **The only file allowed to import concrete providers, channels and writers** — its imports are function-local to keep that exception visible. Also `run_voice` and `build_reflection` |
| `cli.py` | `run` · `setup` · `install` · `uninstall` · `status` · `doctor` · `reindex` · `reflect` · `voice` · `pairing` |
| `setup.py` | the onboarding wizard: preset, hosted provider, keys, persona seed, pairing |
| `config.py` | settings and the three presets. `HOSTED` resolves to the chosen provider |
| `loop.py` | the text conversation: record → recall → complete → record → send |
| `reflection.py` | the 04:00 pass: one local day → curated facts, entity notes, observations |
| `tasks.py` | the `Task` enum. **Frozen** — it is the LLM routing key |
| `tui.py` | terminal presentation. CJK-aware widths; plain text when not a tty |
| `service.py` | LaunchAgent / systemd user unit. Holds no secrets |
| `fs.py` | 0700 dirs, 0600 files, and the two durable writes — append, and atomic replace |
| `clock.py` | the one timestamp helper. Do not scatter `datetime.now()` |
| `channels/` | `channels/base.py` (frozen) · `channels/telegram.py` · `channels/pairing.py` |
| `llm/` | `llm/base.py` (frozen) · `llm/gateway.py` · `llm/providers/` (4) · `llm/embedders/` |
| `memory/` | `memory/schema.sql` (frozen) · `store` · `log` · `writer` · `recall` · `curated` · `entities` · `reindex` |
| `voice/` | `voice/base.py` (frozen) · `voice/gemini_live.py` · `voice/audio.py` · `voice/conversation.py` |
| `persona/` · `proactivity/` | empty — M4 and M3 |

## Layering

Callers use `LLMGateway.complete(task, ...)` and the protocols in the frozen files
above; importing a concrete provider or channel outside `app.py` is a layering
break. Changing a frozen file is allowed, doing it silently is not.

```bash
daemon doctor                  # config, reachability, and what reflection has built
daemon run                     # the loop, in this terminal
daemon reflect                 # the 04:00 pass, now, by hand (`--date`, `--force`)
daemon reindex                 # rebuild all three markdown tiers into the mirror
python3 -m pytest tests/test_reachable.py   # is everything you built reachable?
```

## Common changes

**A new LLM provider.** Implement `Provider` in `daemon/llm/base.py` under
`daemon/llm/providers/`, name it in `HOSTED_PROVIDERS` (`daemon/config.py`), build it
in `daemon/app.py`, offer it in `daemon/setup.py`. Missing the last two shipped once.

**A new channel.** Implement `Channel` in `daemon/channels/base.py`; the `Cursor`
it needs is already `daemon.memory.store.Store`. **Don't** invent a second
allowlist — pairing in `daemon/channels/pairing.py` owns who the owner is.

**A new background job.** Register it on the scheduler in `daemon/app.py`'s
lifespan and copy the reflection job: `timezone=None` to fire on *local* time,
`max_instances=1` + `coalesce=True` so a slow run cannot stack, a top-level `except`
in the tick because a raising job is logged once and the schedule then reads as
healthy forever, and **a CLI command running the same object** — nobody is awake at
04:00 to read a log. Polling needs a floor; see the last bullet.

**Anything a model writes into a path, a score or a date.** Reflection's output
names files, scales the recall score and can retire a fact the user stated, so
`daemon/reflection.py` clamps every number, narrows keys to a fixed charset, and
checks names with `entities.safe_name` **and** a boundary on the resolved path — a
blocklist is not exhaustive. Date records by the day they are *about*.

**A new `Task`.** `daemon/tasks.py` is the routing key: add it to all three presets
in `daemon/config.py` *and* give it a caller. A task with no caller fails
`tests/test_reachable.py` unless declared PENDING with the milestone that owns it.

## Depends on

Downwards only: `daemon/loop.py`, `daemon/voice/conversation.py` and
`daemon/reflection.py` depend on the protocols in `daemon/llm/base.py`,
`daemon/channels/base.py` and `daemon/memory/base.py`, never on what implements
them; `daemon/memory/` depends on `daemon/fs.py` and `daemon/clock.py` and on
nothing above it. Depended on by [tests/](../tests/CLAUDE.md) and nothing else;
[scripts/](../scripts/CLAUDE.md) deliberately does not import this package. Rules:
[CONTRACTS.md](../docs/CONTRACTS.md). Data flow:
[ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## Things measured, not assumed

- **Running reflection for real found two defects unit tests did not.** Entity notes
  were stamped with the day of the *run*, so a catch-up over months of history would
  read as all-today. And two facts sharing one supersession key inverted the result:
  gemma3:4b keyed both halves of a move `location`, the second retired the first, so
  `data/memory/core.md` kept the importance-3 half and the artifact claimed the 8.
- **A 4B local model is unstable at reflection.** Same 6-message Korean fixture,
  two runs: 1 fact / 1 entity / 0 observations, then 2 / 2 / 1. It is the task most
  worth pointing at a hosted model — the whole graph is built from its output.
- **Do not use a reasoning model for local chat.** gemma3:4b answers in 1.7 s,
  qwen3:4b in 11.8 s, and the gap is chain-of-thought nobody sees, cannot be turned
  off (`think: false` stops Ollama *separating* it, so it becomes the reply).
- **The vector lane is the cheap half of recall** (0.22 ms vs FTS5's 1.9 ms at
  10k). What dominates is the embedder round trip, ~117 ms, mostly fixed overhead.
- **FTS5 with `unicode61` cannot carry Korean recall alone** — whole-token matching,
  so an inflected word is a different token: 50% keyword-only vs 93% hybrid.
- **The inbound poll needs a floor.** Left to Telegram's long poll alone, a
  transport that returns immediately spins at ~16,000 requests/second.
