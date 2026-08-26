# Ollama comes up with the daemon — design

- Date: 2026-08-26
- Status: approved for planning
- Branch of origin: `claude/ollama-embedder-unreachable-162244`

## Problem

The owner rebooted, and recall lost its vector lane for fifteen hours without
anything failing.

```
2026-08-25 23:47:38   reboot
2026-08-25 23:53:45   first  "ollama embedder unreachable at http://127.0.0.1:11434/api/embed"
2026-08-26 14:47:43   last   same warning — 237 occurrences
```

Nothing was broken. `pgrep ollama` found no process, nothing listened on 11434,
`curl` returned exit 7. The binary was installed (`/opt/homebrew/bin/ollama`) and
every model was pulled, `bge-m3` included. Only the server was down, and nothing
on the machine was responsible for starting it.

The daemon itself survives a reboot: `~/Library/LaunchAgents/ai.daemon.default.plist`
carries `RunAtLoad` and `KeepAlive`. Its embedder gets neither. The plist does not
mention Ollama, no code starts it, and `brew services list` reports `ollama none`.
`docs/adr/` has seventeen entries and none of them decides this, which reads less
like a deliberate omission than a gap nobody stood in front of.

The owner named it exactly:

> 어쨋든 daemon을 설치한 사람들은 daemon이 실행될때 ollama도 같이 실행되야할거아냐

### Why it was invisible

By design, and the design is right. `recall.py:394` drops to keyword-only and
`recall.py:566` swallows a failed index so a turn still gets its reply. Chat was
routed to `gemini`, so only `embed->ollama` was severed. `daemon doctor` reported
it correctly the whole time — but nothing makes anybody run `daemon doctor`.

### The two failures it caused

**Ongoing indexing.** 49 messages took their turn with no vector. Recoverable:
`Recall.index` retries every turn, so it healed the moment Ollama came back.

**The startup backfill.** Not recoverable without a restart:

```
14:42:16 WARNING recall: backfill stopped after 0 message(s) (unreachable)
14:50:23 INFO    recall backfill embedded 49 message(s)
```

`_backfill` (`app.py:769`) runs once at startup and never again. Its own docstring
says "no retry, no periodic job" about the bug it fixed, and one layer of that bug
survived: when the embedder is unreachable it gives up at zero and nothing calls it
again. The 14:50 line is luck — the daemon happened to restart after Ollama was up.

### The setup step that should have caught it

`setup.py:2387` returns early for anyone not on a local chat model:

```python
if provider != OLLAMA:
    self.prompt.say("Nothing here needs a local chat model.")
    self.prompt.say("Recall embeddings are always local, so `ollama pull")
    self.prompt.say(f"{embed_model}` is still worth doing - `daemon doctor` checks it.")
    return
```

The owner is on `gemini`. So onboarding never probed reachability, never checked
whether `bge-m3` existed, and offered one sentence of advice. Embeddings are always
local whatever `DAEMON_PROVIDER` says (`Task.EMBED` is pinned to `ollama`), so this
branch skips the check for precisely the users who most need it.

## Measured constraint: the service has no useful PATH

The resident is a launchd job, and launchd hands it the bare default:

```
resident PATH   = /usr/bin:/bin:/usr/sbin:/sbin
ollama binary   = /opt/homebrew/bin/ollama          ← not on that PATH
```

`_render_plist` omits `EnvironmentVariables` on purpose (`service.py:226`:
"WorkingDirectory is how the process finds .env, and that is the only place
credentials live"). That stays. The consequence is that `shutil.which("ollama")`
inside `daemon run` finds nothing, while the same call from a terminal succeeds —
a failure that only appears in the real service. Binary discovery must not rely on
`which` alone.

## Decisions

Two, taken with the owner before this document existed.

**The daemon owns the process (a child), not an OS unit.** It answers the question
as asked, the ordering problem disappears by construction, and it is less code than
a second unit file per platform. The alternative — `daemon install` registering an
Ollama service — cannot order two LaunchAgents against each other, which would
leave the backfill-at-zero failure in a quieter form.

**Heavy work happens in setup, where a person is watching; boot only starts what is
ready.** `bge-m3` is a 1.2GB download. `daemon setup` asks and pulls; `daemon run`
spawns and nothing else.

### A deliberate decision this reverses

`setup.py:2418` says:

> Deliberately not run for them: these are gigabytes, and a wizard should not
> start a download the user did not ask for.

Setup will now offer to run it. Asking first keeps the intent — the download is
still one the user asked for — but the comment is no longer true as written and
gets rewritten with the change. Flagged rather than done quietly, per
`CLAUDE.md`.

## Design

### 1. Setup — `daemon/setup.py`

Delete the `provider != OLLAMA` early return. Regardless of provider, check:

- **reachability** — `check_ollama(base_url)`, which already exists
- **`bge-m3` presence** — embeddings are always local

Check the **chat** model (`qwen3`/`gemma3`) only when `provider == ollama`. That
much the early return had right, and it is the only part worth keeping.

When `bge-m3` is missing, ask, then pull:

```
[missing] bge-m3: not pulled yet — recall drops to keyword-only for Korean
          Pull it now? (1.2GB) [y/N]
```

Declining is not an error. Setup continues and says `daemon doctor` re-checks it,
as it does today.

### 2. Boot — `daemon/ollama_process.py` (new)

A small module owning one child process. `daemon/app.py` is the only importer
(CONTRACTS 4).

Four gates, in order. Every one must pass before anything is spawned:

| gate | on failure |
|---|---|
| `base_url` names this machine (loopback) | remote Ollama — never spawn |
| not already reachable (`probe_ollama`) | already up — **leave it alone** |
| binary found | log once, carry on |
| it answers within a bounded wait | log once, carry on |

**Binary discovery** tries `shutil.which()` first, then known locations —
`/opt/homebrew/bin`, `/usr/local/bin`, `Ollama.app/Contents/Resources`. Required by
the PATH measurement above, not defensive habit.

**Spawning** uses `asyncio.create_subprocess_exec`, the pattern already in
`tools/screen.py:96` and `tools/builtin.py:436`.

**Shutdown** stops only what this daemon started. An Ollama that was already
running outlives the daemon. The teardown sits with the stdio-MCP cleanup in the
`lifespan` `finally` for the reason stated there (`app.py:568`): otherwise every
restart leaves one more orphan.

Soft dependency is preserved. Any gate can fail and the daemon still boots, still
serves, still answers — keyword-only, exactly as it did for fifteen hours.

### 3. Backfill ordering — `daemon/app.py`

Without this, gate-passing does not actually fix anything: Ollama takes seconds to
answer, the backfill fires at t=0, and `backfill stopped after 0 message(s)`
reappears.

The backfill task waits for the embedder **inside itself**. The wait must not move
into `lifespan`, because `app.py:490` requires that "a cold embedder must not delay
the log clock" (`docs/PLAN.md` 8.1) — startup stays unblocked and `/health` stays
honest.

### 4. Observability

No `doctor` change: it already reports this correctly, in both states. One log line
when the daemon spawns Ollama, so "who started this process" is answerable from the
log during an incident.

## Testing

TDD — a failing test per gate before its implementation.

- a remote `base_url` never spawns
- an already-reachable Ollama is never spawned and never stopped
- a missing binary leaves the daemon booting and serving
- shutdown stops a daemon-spawned process, and leaves a pre-existing one alone
- the backfill waits rather than reporting zero against a cold embedder
- setup checks reachability and `bge-m3` under a hosted provider
- setup pulls only after consent, and a decline is not an error

Use the fixtures in `tests/conftest.py`; CONTRACTS forbids inventing parallel ones.
Check whether `tests/test_reachable.py` needs an entry — `CLAUDE.md` requires
closing `PENDING_TASKS`/`PENDING_CLASSES` as pieces get wired up, and a module
nothing constructs is the defect that file exists to catch.

## Out of scope (YAGNI)

- **A `DAEMON_MANAGE_OLLAMA` toggle.** "Never touch a running one" and "never spawn
  for a remote URL" already are the opt-out. Anyone running Ollama under
  `brew services` fails gate 2 on every boot, which is the correct outcome.
- **Periodic Ollama supervision after boot.** A failed spawn is in the log and in
  `doctor`; ongoing indexing retries per turn and heals itself.
- **Installing Ollama for the user.** Setup points at ollama.com, as it does now.
  Pulling a model into an existing install is a different act from installing a
  package manager's worth of software.
