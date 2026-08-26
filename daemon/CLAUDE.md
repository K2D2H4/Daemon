# daemon/ — the process

## Owns

One process: a FastAPI control plane, an in-process APScheduler, a channel loop — assembled
in `app.py`, which owns every concrete-implementation import here.

## Layout

| | |
|---|---|
| `app.py` | composition root and lifespan. **The only file allowed to import concrete providers, channels and writers** — its imports are function-local to keep that exception visible. Also `run_voice`, `build_reflection` and `build_proactive_tick` |
| `cli.py` | `run` · `setup` · `install` · `uninstall` · `status` · `log` · `help` · `doctor` · `reindex` · `update` · `reflect` · `voice` · `proactive` · `persona` · `tools` · `pairing` · `wake calibrate` · `wake test` · `request-mic` · `face` · `face-transitions`. Commands are registered through one `add()` that requires a help group, so the grouped listing in `daemon help` cannot drift from what the parser accepts |
| `setup.py` | the onboarding wizard: PC control, provider, its model(s) and background toggle, keys, persona seed, pairing, then the residency finish — offer `daemon install` and confirm the resident woke up via `service.status()` + `/health` |
| `wake_cli.py` | `daemon wake`: measure what the recognizer returns for the owner's phrase, save it as `DAEMON_WAKE_ALIASES`, then run the gate and print what fires. Writes `.env` through `setup.py`'s writer |
| `config.py` | settings and the two routing axes, `DAEMON_PROVIDER` + `DAEMON_PROACTIVE_JUDGE_LOCAL` — `Settings.routing` is computed from them, not looked up in a table |
| `companion.py` | **what both endpoints can do, in one place**: `context` (the time block, persona, tool rules, commitments, recall, then the verbal-tic list — time leads because it is a fact about the world rather than an instruction, commitments trail the tool rules they don't qualify, and the tics go last because that block is about the sentence being written now), `record` + `index_recorded`, `specs` / `run_tools`. Add a capability here, not twice |
| `loop.py` | the text transport: record → context → complete → record → send. Owns the tool loop's shape, `/approve`, and the wire |
| `face.py` | what the daemon is doing, as two kinds of event on one `FaceBus`: `activity` (idle/listening/thinking/speaking/working, a state) and a mood or flourish (a one-shot clip). No subscriber, no cost - `create_app` builds the one bus and hands it by keyword to the loop, the tool runner and the voice conversation |
| `face_routes.py` | the face's HTTP surface: `/face` (`daemon/static/face.html`), `/face/stream` (SSE off the bus), `/face/manifest`, `/face/transitions` (the pose-match table, 404 until it is built), `/face/clips/{name}` reading `<data_dir>/face/*.mp4` - an allowlisted name, not a path. Read-only and carries no conversation, same loopback reasoning as `admin/routes.py` |
| `face_match.py` | offline, run by hand from `daemon face-transitions`: where to enter a clip and when it is a good moment to leave one, written to `<data_dir>/face/transitions.json` for the page to look up. Never on the request path. The order of its two mechanisms is the decision - [ADR 0017](../docs/adr/0017-the-neutral-moment-not-the-matched-pose.md) |
| `reflection.py` | the 04:00 pass: one local day → curated facts, entity notes, observations. Two model calls — the conversation plus a **usage summary** of the day's tool calls (columns only, no output text) produces all three; a second call reads the day's tool *output* and may produce facts and nothing else, stamped `origin='untrusted'`. That asymmetry is the design, not an omission: docs/design/2026-08-18-tool-results-into-memory-design.md |
| `tasks.py` | the `Task` enum. **Frozen** — it is the LLM routing key |
| `tui.py` · `service.py` | terminal presentation, CJK-aware widths, plain text when not a tty · the LaunchAgent / systemd user unit, which holds no secrets |
| `fs.py` · `clock.py` · `mic_hold.py` · `mic_floor.py` | 0700 dirs, 0600 files, and the two durable writes — append, and atomic replace · the one timestamp helper, so nobody scatters `datetime.now()` · whether *this process* holds the microphone, a reentrant counter the wake listener and a voice session both increment. Top-level rather than inside `voice/` so `proactivity/presence.py` can subtract our own hold from the CoreAudio probe without importing the voice layer — a text-only install has no PortAudio and still has to answer presence (docs/adr/0013) · the mailbox a proactive line uses to ask the wake loop for the microphone, since the speaker refuses while `mic_hold` says this process holds it - the wake side takes the request and does the speaking, because releasing the capture stream is not something that can be done to the gate from outside |
| `timesense.py` | how the daemon *speaks* about time — the current instant, relative phrasing, where a conversation broke, which commitments in view are past. `clock.py` reads the clock; this renders it. Holds the extraction primitives `proactivity/candidates.py` imports back, so both paths read one judgement instead of two |
| `channels/` | `channels/base.py` (frozen) · `channels/telegram.py` · `channels/pairing.py` |
| `llm/` | `llm/base.py` (frozen) · `llm/gateway.py` · `llm/providers/` (5) · `llm/embedders/` |
| `memory/` | `memory/schema.sql` (frozen) · `store` · `log` · `writer` · `recall` · `curated` · `entities` · `reindex` |
| `voice/` | `voice/base.py` (frozen) · `voice/gemini_live.py` · `voice/audio.py` (PortAudio) · `voice/apple_audio.py` (macOS echo cancellation) · `voice/conversation.py` · `voice/vad.py` · `voice/apple_speech.py` · `voice/wake.py` · `voice/mic_access.py` (mic TCC request + status, Apple-guarded) |
| `macapp/` | the thin native-launcher `Daemon.app` (macOS): `build_bundle` assembles and ad-hoc-signs the bundle whose code identity is the microphone grant; `launcher.c`/`launcher` is the committed universal2 Mach-O it copies in |
| `proactivity/` | `proactivity/base.py` (frozen) · `candidates` · `gate` · `presence` · `judge` · `delivery` · `speaker` · `tick` |
| `persona/` | `persona/loader.py` (assembles `seed.md` + `learned.md` into one system message, read every turn — both conversation endpoints reach it through `companion.py`) · `persona/rules.py` (the only write path for `data/persona/learned.md` and its `persona_rules` mirror) · `persona/evolve.py` (the weekly pass: observations → rule proposals, at most one model call) · `persona/tics.py` (phrases the daemon repeats and the owner never uses, named into the prompt because the abstract "do not repeat yourself" rule loses to the window full of it) |
| `tools/` | `tools/base.py` (the `Tool` protocol, `Registry`, `ToolResult`) · `tools/policy.py` — **read this one first**: the origin gate, the modes, the standing approvals, and **zero model calls** · `tools/builtin.py` (the seven: `list_dir` · `read_file` · `write_file` · `run_command` · `open_path` · `notify` · `system_state`) · `tools/extract.py` (`read_file`'s document text: PDF via `pypdf`, `.docx`/`.xlsx`/`.pptx` via the stdlib) · `tools/browser.py` (`fetch_page` · `list_tabs` · `read_page`, behind their own switch) · `tools/screen.py` (`see_screen` · `start_screen_share` · `stop_screen_share`, behind `DAEMON_SCREEN_ENABLED` — the live pump is `voice/screen_share.py`, Pillow, voice extra) · `tools/mcp.py` (the only file that imports the `mcp` package) · `tools/runner.py` (decide → execute → audit, in that order, in one object so none of the three can be skipped) |
| `admin/` | the loopback control plane (M5): `admin/routes.py` (the JSON API and the server-rendered shell — health, chat-test, settings, restart, activity, memory, persona, MCP Phase 2) · `admin/mind.py` (read-only Memory and Persona tab payloads over the store and its markdown) · `admin/activity.py` (read-only Activity, Proactive and Tools tab payloads) · `admin/settings_io.py` (validated `.env` patch — a candidate `Settings` must construct before a byte is written) · `admin/seed_io.py` (the seed editor's read and its one write — the only path besides the wizard that writes `data/persona/seed.md`, and it writes nothing but what the owner typed: [ADR 0019](../docs/adr/0019-the-seed-is-authored-not-unreachable.md)) · `admin/mcp_oauth.py` (the inline SDK OAuth flow bridged across two HTTP requests) · `admin/restart.py` (exit only when a supervisor will revive the process) · `admin/static/index.html` (the whole front end, one self-contained offline file) |

## Layering

Callers use `LLMGateway.complete(task, ...)` and the protocols in the frozen files above;
importing a concrete provider or channel outside `app.py` is a layering break. Changing a
frozen file is allowed, doing it silently is not.

```bash
daemon help                    # the commands, grouped; `daemon help <command>` for one
daemon doctor                  # config, reachability, and what reflection has built
daemon log                     # the resident's own stderr (`-f` follows, `--raw` keeps the polling)
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
