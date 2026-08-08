# daemon/ — the process

## Owns

One process: a FastAPI control plane, an in-process APScheduler, a channel loop — assembled
in `app.py`, which owns every concrete-implementation import here.

## Layout

| | |
|---|---|
| `app.py` | composition root and lifespan. **The only file allowed to import concrete providers, channels and writers** — its imports are function-local to keep that exception visible. Also `run_voice`, `build_reflection` and `build_proactive_tick` |
| `cli.py` | `run` · `setup` · `install` · `uninstall` · `status` · `doctor` · `reindex` · `update` · `reflect` · `voice` · `proactive` · `persona` · `tools` · `pairing` · `wake calibrate` · `wake test` |
| `setup.py` | the onboarding wizard: PC control, preset, hosted provider, keys, persona seed, pairing, then the residency finish — offer `daemon install` and confirm the resident woke up via `service.status()` + `/health` |
| `wake_cli.py` | `daemon wake`: measure what the recognizer returns for the owner's phrase, save it as `DAEMON_WAKE_ALIASES`, then run the gate and print what fires. Writes `.env` through `setup.py`'s writer |
| `config.py` | settings and the three presets. `HOSTED` resolves to the chosen provider |
| `companion.py` | **what both endpoints can do, in one place**: `context` (persona + tool rules + the recall block), `record` + `index_recorded`, `specs` / `run_tools`. Add a capability here, not twice |
| `loop.py` | the text transport: record → context → complete → record → send. Owns the tool loop's shape, `/approve`, and the wire |
| `reflection.py` | the 04:00 pass: one local day → curated facts, entity notes, observations |
| `tasks.py` | the `Task` enum. **Frozen** — it is the LLM routing key |
| `tui.py` · `service.py` | terminal presentation, CJK-aware widths, plain text when not a tty · the LaunchAgent / systemd user unit, which holds no secrets |
| `fs.py` · `clock.py` | 0700 dirs, 0600 files, and the two durable writes — append, and atomic replace · the one timestamp helper, so nobody scatters `datetime.now()` |
| `channels/` | `channels/base.py` (frozen) · `channels/telegram.py` · `channels/pairing.py` |
| `llm/` | `llm/base.py` (frozen) · `llm/gateway.py` · `llm/providers/` (4) · `llm/embedders/` |
| `memory/` | `memory/schema.sql` (frozen) · `store` · `log` · `writer` · `recall` · `curated` · `entities` · `reindex` |
| `voice/` | `voice/base.py` (frozen) · `voice/gemini_live.py` · `voice/audio.py` (PortAudio) · `voice/apple_audio.py` (macOS echo cancellation) · `voice/conversation.py` · `voice/vad.py` · `voice/apple_speech.py` · `voice/wake.py` |
| `proactivity/` | `proactivity/base.py` (frozen) · `candidates` · `gate` · `presence` · `judge` · `delivery` · `speaker` · `tick` |
| `persona/` | `persona/loader.py` (assembles `seed.md` + `learned.md` into one system message, read every turn — both conversation endpoints reach it through `companion.py`) · `persona/rules.py` (the only write path for `data/persona/learned.md` and its `persona_rules` mirror) · `persona/evolve.py` (the weekly pass: observations → rule proposals, at most one model call) |
| `tools/` | `tools/base.py` (the `Tool` protocol, `Registry`, `ToolResult`) · `tools/policy.py` — **read this one first**: the origin gate, the modes, the standing approvals, and **zero model calls** · `tools/builtin.py` (the seven: `list_dir` · `read_file` · `write_file` · `run_command` · `open_path` · `notify` · `system_state`) · `tools/extract.py` (`read_file`'s document text: PDF via `pypdf`, `.docx`/`.xlsx`/`.pptx` via the stdlib) · `tools/browser.py` (`fetch_page` · `list_tabs` · `read_page`, behind their own switch) · `tools/mcp.py` (the only file that imports the `mcp` package) · `tools/runner.py` (decide → execute → audit, in that order, in one object so none of the three can be skipped) |

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
daemon tools list              # the tools that loaded, and the policy actually in force
daemon tools log               # recent calls, including the refused ones
daemon tools pending           # approvals waiting on an answer (`daemon tools forget <pattern>`)
daemon reindex                 # rebuild the mirror from markdown - log, curated, entities, persona rules
daemon wake calibrate          # what does the recognizer actually hear you say? (`--takes`)
daemon wake test               # run the wake gate here and print what fires (`--seconds`)
python3 -m pytest tests/test_reachable.py   # is everything you built reachable?
```

## Read when you need it

Two files, kept out of here so this one stays orientation rather than reference —
and linked rather than imported, so they cost nothing until something asks for them.

| | |
|---|---|
| [RECIPES.md](RECIPES.md) | how to add a provider, a channel, a tool, a background job, a `Task` — each naming the step that was skipped the time it shipped broken |
| [MEASURED.md](MEASURED.md) | what this project believed until it ran it. Read before optimising, before trusting a documented probe, and before re-measuring something already here |

## Depends on

Downwards only: `daemon/loop.py`, `daemon/voice/conversation.py`, `daemon/companion.py`,
`daemon/reflection.py`
and `daemon/proactivity/` depend on the protocols in `daemon/llm/base.py`,
`daemon/channels/base.py`, `daemon/memory/base.py` and `daemon/proactivity/base.py`,
never on what implements them; `daemon/memory/` depends on `daemon/fs.py` and
`daemon/clock.py` and on nothing above it. Depended on by [tests/](../tests/CLAUDE.md)
and nothing else; [scripts/](../scripts/CLAUDE.md) deliberately does not import this
package. Rules: [CONTRACTS.md](../docs/CONTRACTS.md). Data flow:
[ARCHITECTURE.md](../docs/ARCHITECTURE.md).
