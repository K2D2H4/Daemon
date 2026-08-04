# daemon/ — the process

## Owns

One process: FastAPI for a local control plane, an in-process APScheduler for
background work, a channel loop for conversation. Assembled in `app.py`, which
also owns every concrete-implementation import in the package.

## Layout

| | |
|---|---|
| `app.py` | composition root and lifespan. **The only file allowed to import concrete providers, channels and writers** — its imports are function-local to keep that exception visible. Also `run_voice`. |
| `cli.py` | `run` · `setup` · `install` · `uninstall` · `status` · `doctor` · `reindex` · `voice` · `pairing` |
| `setup.py` | the onboarding wizard: preset, hosted provider, keys, persona seed, pairing |
| `config.py` | settings and the three presets. `HOSTED` resolves to the chosen provider |
| `loop.py` | the text conversation: record → recall → complete → record → send |
| `tasks.py` | the `Task` enum. **Frozen** — it is the LLM routing key |
| `tui.py` | terminal presentation. CJK-aware widths; plain text when not a tty |
| `service.py` | LaunchAgent / systemd user unit. Holds no secrets |
| `fs.py` | 0700 dirs, 0600 files. Everything written here is private conversation |
| `clock.py` | the one timestamp helper. Do not scatter `datetime.now()` |
| `channels/` | `channels/base.py` (frozen) · `channels/telegram.py` · `channels/pairing.py` |
| `llm/` | `llm/base.py` (frozen) · `llm/gateway.py` · `llm/providers/` (4) · `llm/embedders/` |
| `memory/` | `memory/schema.sql` (frozen) · `store` · `log` · `writer` · `recall` · `reindex` |
| `voice/` | `voice/base.py` (frozen) · `voice/gemini_live.py` · `voice/audio.py` · `voice/conversation.py` |
| `persona/` | empty — M4 |
| `proactivity/` | empty — M3 |

## Layering

Callers use `LLMGateway.complete(task, ...)` and the protocols declared in the
frozen files above. A file that imports a concrete provider or channel outside
`app.py` is a layering break, not a shortcut.

The files marked frozen above are contracts. Changing one is allowed; doing
it silently is not — say so, because implementations elsewhere depend on the exact
shape and several have been corrected by measurement rather than by reading docs.

```bash
daemon doctor                  # what is configured, reachable and indexed
daemon run                     # the loop, in this terminal
daemon reindex                 # rebuild the sqlite mirror from the markdown
python3 -m pytest tests/test_reachable.py   # is everything you built reachable?
```

## Common changes

**A new LLM provider.** Implement `Provider` in `daemon/llm/base.py` under
`daemon/llm/providers/`, add the name to `HOSTED_PROVIDERS` in `daemon/config.py`,
build it in `daemon/app.py`, and offer it in `daemon/setup.py`. Miss the last two
and it is nameable in a route and unbuildable in the app — which shipped once.
`tests/test_reachable.py` is what catches it now.

**A new channel.** Implement `Channel` in `daemon/channels/base.py`; the `Cursor`
it needs is already `daemon.memory.store.Store`. **Don't** invent a second
allowlist — pairing in `daemon/channels/pairing.py` owns who the owner is.

**A new background job.** Register it on the scheduler in `daemon/app.py`'s
lifespan. Anything that polls needs a floor; see the last bullet below.

**A new `Task`.** `daemon/tasks.py` is the routing key, so adding one means adding
it to all three presets in `daemon/config.py` *and* giving it a caller. A task
with no caller fails `tests/test_reachable.py` unless it is declared PENDING with
the milestone that owns it.

## Depends on

Downwards only: `daemon/loop.py` and `daemon/voice/conversation.py` depend on the
protocols in `daemon/llm/base.py`, `daemon/channels/base.py` and
`daemon/memory/base.py` — never on what implements them. `daemon/memory/` depends
on `daemon/fs.py` and `daemon/clock.py` and on nothing above it.

Depended on by [tests/](../tests/CLAUDE.md) and by nothing else in the repo;
[scripts/](../scripts/CLAUDE.md) deliberately does not import this package. The
rules these seams exist to protect are in [CONTRACTS.md](../docs/CONTRACTS.md);
the runtime data flow is in [ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## Things measured, not assumed

- **Do not use a reasoning model for local chat.** gemma3:4b answers in 1.7 s,
  qwen3:4b in 11.8 s, and the gap is chain-of-thought nobody sees. It cannot be
  switched off: `think: false` stops Ollama *separating* the reasoning, so it
  becomes the reply.
- **The vector lane is the cheap half of recall** (0.22 ms vs FTS5's 1.9 ms at
  10k). What dominates is the embedder round trip, ~117 ms, mostly fixed overhead.
- **FTS5 with `unicode61` cannot carry Korean recall alone** — whole-token
  matching only, so an inflected word is a different token. Keyword-only tops out
  at 50% on the golden set against 93% hybrid.
- **The inbound poll needs a floor.** Left to Telegram's long poll alone, a
  transport that returns immediately spins at ~16,000 requests/second.
