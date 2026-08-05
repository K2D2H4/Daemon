# daemon/ — the process

## Owns

One process: a FastAPI control plane, an in-process APScheduler, a channel loop — assembled
in `app.py`, which owns every concrete-implementation import here.

## Layout

| | |
|---|---|
| `app.py` | composition root and lifespan. **The only file allowed to import concrete providers, channels and writers** — its imports are function-local to keep that exception visible. Also `run_voice`, `build_reflection` and `build_proactive_tick` |
| `cli.py` | `run` · `setup` · `install` · `uninstall` · `status` · `doctor` · `reindex` · `reflect` · `voice` · `proactive` · `pairing` · `wake calibrate` · `wake test` |
| `setup.py` | the onboarding wizard: PC control, preset, hosted provider, keys, persona seed, pairing |
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
| `persona/` | `persona/loader.py` — the only place the persona is assembled. `seed.md` today; M4 adds `learned.md` to this one file |

## Layering

Callers use `LLMGateway.complete(task, ...)` and the protocols in the frozen files above;
importing a concrete provider or channel outside `app.py` is a layering break. Changing a
frozen file is allowed, doing it silently is not.

```bash
daemon doctor                  # config, reachability, and what reflection has built
daemon run                     # the loop, in this terminal
daemon reflect                 # the 04:00 pass, now, by hand (`--date`, `--force`)
daemon proactive               # one proactivity round, verdicts only (`--speak` to let it)
daemon reindex                 # rebuild all three markdown tiers into the mirror
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

**Anything the daemon can *do*** — something new in front of the model, something new
written down. It goes in `daemon/companion.py`, once, and both endpoints get it.
Adding it to `daemon/loop.py` is how voice ended up recording every spoken turn and
embedding none of them: two implementations of one thing, and only one of them
complete. What genuinely belongs to an endpoint is *transport* — the wire, and when
it is safe to write to it. That includes voice's injection timing, which is measured
and must stay in `daemon/voice/conversation.py`: `clientContent` sent mid-generation
kills the answer.

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
- **Voice mode interrupted itself on every single turn, and echo was only half of
  it.** Two independent causes, both measured. (1) The speaker leaks into the open
  microphone: 80.1% of mic frames read as speech through PortAudio while the speaker
  played, 0.0% through macOS voice processing, and 81.6% for speech the canceller
  does not know about - so the echo goes and the user does not
  (`daemon/voice/apple_audio.py`). (2) Gemini delivers `inputTranscription` **at the
  turn boundary, in the same server event as the answer's first audio chunk**, and
  `daemon/voice/conversation.py` inferred a barge-in from the transcript growing while audio
  played. So the question's own transcript condemned its own answer: a complete,
  fluent reply generated and **0.0s of it played**. The barge-in is now the server's
  `interrupted`, which is what `daemon/voice/base.py` always said the authority was. Same
  measurement retires a documented claim: the recall prefetch cannot be free against
  this provider, because the partial it fires on arrives with the answer.
- **Recall was killing the answer it was fetched for, and this was the biggest of the
  three.** `send_context` is `clientContent`, and the Live API says plainly that "a
  message here will interrupt any current model generation". The prefetch landed
  mid-answer, so seeding a memory cut the reply off at "아..." - the owner's log paired
  every barge-in with the embed call immediately before it, 1:1. Measured, one
  conversation, same room and microphone: **2.2s of audio with recall on, 46.7s with
  it off, 38.8s with it deferred to the turn boundary.** Deferring costs nothing that
  was not already lost, because the prefetch fires on a partial that arrives with the
  answer anyway. And note what `serverContent.interrupted` actually means - "a client
  message has interrupted current model generation", *not* "the user spoke" - so it is
  two failures wearing one flag, and reading it as pure user-VAD is what let the daemon
  mistake its own memory for the owner talking over it.
- **An interruption arriving after `generationComplete` is not an interruption.**
  Measured four times: it lands ~0.25 s later on a turn nobody touched, and acted on
  it empties the speaker of an answer that was fully delivered.
- **Two Telegram traps.** The inbound poll needs a floor — left to the long poll alone, a
  transport that returns immediately spins at ~16,000 requests/second. And `allowed_updates`
  is **server-side**: at `["message"]` a 👍 press is never delivered at all.
