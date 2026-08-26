# Ollama comes up with the daemon — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The daemon starts the local Ollama it needs for embeddings, so a reboot
no longer silently costs recall its vector lane.

**Architecture:** A new `daemon/ollama_process.py` owns one child process behind
four gates (local `base_url`, not already reachable, binary found, answers in
time) and stops only what it started. `daemon/app.py` constructs it in `lifespan`,
hands it to the backfill task — which waits for the embedder inside itself so
startup stays unblocked — and closes it in the `finally`. `daemon/setup.py` stops
skipping the Ollama check for hosted providers and offers to pull `bge-m3`.

**Tech Stack:** Python 3.12+, asyncio, httpx, pytest (`asyncio_mode = "auto"` —
async tests need no decorator), ruff.

**Spec:** [docs/superpowers/specs/2026-08-26-ollama-with-the-daemon-design.md](../specs/2026-08-26-ollama-with-the-daemon-design.md)

## Global Constraints

- **Layering (CONTRACTS 4).** Only `daemon/app.py` imports `ollama_process`. Do
  not import it from `cli.py`, `setup.py`, `loop.py`, or `companion.py`.
- **Do not import `daemon.cli` from `daemon.app`.** `probe_ollama` lives in
  `cli.py`; `ollama_process` gets its own probe rather than importing upward.
- **Soft dependency.** Any gate may fail and the daemon must still boot, serve
  and answer. No gate failure raises out of `lifespan`.
- **A cold embedder must not delay the log clock** (`daemon/app.py:490`,
  `docs/PLAN.md` 8.1). Nothing added to `lifespan` may `await` readiness before
  `yield`.
- **Never stop an Ollama this process did not start.**
- **No `EnvironmentVariables` in the plist.** `daemon/service.py:226` omits it
  deliberately; this work does not add it. Binary discovery compensates instead.
- **Measured PATH constraint.** The resident's PATH is
  `/usr/bin:/bin:/usr/sbin:/sbin` and `ollama` lives at `/opt/homebrew/bin/ollama`.
  `shutil.which()` alone finds nothing in the real service.
- **Embed model default:** `bge-m3` (`DEFAULT_EMBED_MODEL`, `daemon/setup.py:178`),
  1.16 GB measured.
- **Fixtures:** use `tests/conftest.py`; CONTRACTS forbids parallel fixtures.
- **Baseline to preserve:** `tests/test_setup.py` is 232 passing tests today.

---

### Task 1: The child-process module

**Files:**
- Create: `daemon/ollama_process.py`
- Test: `tests/test_ollama_process.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, for Task 2:
  - `LocalOllama(base_url: str, *, find: Callable[[], str | None] = find_binary, probe: Callable[[str], Awaitable[bool]] | None = None, spawn: Callable[[str], Awaitable[Process]] | None = None)`
  - `async def ensure_running(self) -> bool` — True when Ollama answers at the
    end, whether or not this object started it.
  - `async def aclose(self) -> None`
  - `property started_by_us: bool`
  - `def is_local(base_url: str) -> bool`
  - `def find_binary() -> str | None`
  - `READY_TIMEOUT_SECONDS: float = 30.0`

- [ ] **Step 1: Write the failing tests for the two gates that must not spawn**

Create `tests/test_ollama_process.py`:

```python
"""The four gates in front of spawning Ollama.

Two of them exist to *not* act: a remote Ollama is somebody else's process, and a
reachable one is already doing its job. Getting either wrong means this daemon
kills or duplicates a server it does not own.
"""

from __future__ import annotations

from typing import Any

from daemon.ollama_process import LocalOllama, find_binary, is_local


def _never_spawn(binary: str) -> Any:
    raise AssertionError(f"spawned {binary} when no gate allowed it")


async def test_a_remote_base_url_is_never_spawned_locally() -> None:
    """Starting a local server for a URL pointing at another machine would run a
    second, empty Ollama and leave the real one untouched."""
    local = LocalOllama(
        "http://192.168.1.50:11434",
        find=lambda: "/opt/homebrew/bin/ollama",
        probe=_unreachable,
        spawn=_never_spawn,
    )

    assert await local.ensure_running() is False
    assert local.started_by_us is False


async def test_an_already_reachable_ollama_is_left_alone() -> None:
    local = LocalOllama(
        "http://127.0.0.1:11434",
        find=lambda: "/opt/homebrew/bin/ollama",
        probe=_reachable,
        spawn=_never_spawn,
    )

    assert await local.ensure_running() is True
    assert local.started_by_us is False


async def test_a_missing_binary_is_not_fatal() -> None:
    """The daemon serves without Ollama - keyword-only recall. `ensure_running`
    reports the failure; it never raises it."""
    local = LocalOllama(
        "http://127.0.0.1:11434",
        find=lambda: None,
        probe=_unreachable,
        spawn=_never_spawn,
    )

    assert await local.ensure_running() is False
    assert local.started_by_us is False


async def _reachable(base_url: str) -> bool:
    return True


async def _unreachable(base_url: str) -> bool:
    return False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_ollama_process.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'daemon.ollama_process'`

- [ ] **Step 3: Write the module**

Create `daemon/ollama_process.py`:

```python
"""The local Ollama the daemon starts when nothing else has.

`Task.EMBED` is always `ollama` whatever `DAEMON_PROVIDER` says, so recall's
vector lane depends on a server that - measured 2026-08-26 - nothing on a fresh
macOS install is responsible for starting. The daemon survives a reboot on the
LaunchAgent's `RunAtLoad`; its embedder had no equivalent, and the result was
fifteen hours of keyword-only Korean recall with nothing failing.

Four gates, and two of them exist in order *not* to act:

  * The URL must name this machine. `DAEMON_OLLAMA_BASE_URL` may point at another
    host, and spawning a local server for it would start an empty second Ollama
    while the real one carried on unused.
  * It must not already answer. Somebody running Ollama under `brew services` or
    Ollama.app owns that process; this one neither duplicates nor stops it.
  * The binary must be findable - see `find_binary` for why `which` is not enough.
  * It must answer within `READY_TIMEOUT_SECONDS`.

Failing any gate is not an error. The daemon boots, serves, and answers with
keyword-only recall, exactly as it did for those fifteen hours.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Awaitable, Callable
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

READY_TIMEOUT_SECONDS = 30.0
"""How long to wait for a spawned Ollama to answer. Generous on purpose: a cold
start loads no model, but a machine still finishing login is slow at everything."""

POLL_SECONDS = 0.5

BINARY_FALLBACKS = (
    "/opt/homebrew/bin/ollama",
    "/usr/local/bin/ollama",
    "/Applications/Ollama.app/Contents/Resources/ollama",
)
"""Where to look when PATH does not say.

Measured 2026-08-26: the resident launchd job runs with
`PATH=/usr/bin:/bin:/usr/sbin:/sbin` - `_render_plist` omits
`EnvironmentVariables` deliberately (daemon/service.py:226) and
`launchctl getenv PATH` is unset. Homebrew's bin is on neither, so
`shutil.which("ollama")` succeeds in a terminal and finds nothing in the service
this code actually runs in."""

LOCAL_HOSTNAMES = frozenset({"localhost", ""})


def is_local(base_url: str) -> bool:
    """Does this URL name the machine we are running on?"""
    host = urlparse(base_url).hostname or ""
    if host in LOCAL_HOSTNAMES:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def find_binary() -> str | None:
    """`ollama` on PATH, or in a known install location. See BINARY_FALLBACKS."""
    found = shutil.which("ollama")
    if found:
        return found
    for candidate in BINARY_FALLBACKS:
        if Path(candidate).is_file():
            return candidate
    return None


async def _probe(base_url: str) -> bool:
    """Does Ollama answer? Its own client, not a provider's: this runs before the
    gateway exists and must not depend on it (CONTRACTS 4)."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/tags")
    except httpx.HTTPError:
        return False
    return response.status_code == 200


async def _spawn(binary: str) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        binary,
        "serve",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


class LocalOllama:
    """One child Ollama. Cheap to construct; nothing runs until `ensure_running`."""

    def __init__(
        self,
        base_url: str,
        *,
        find: Callable[[], str | None] = find_binary,
        probe: Callable[[str], Awaitable[bool]] | None = None,
        spawn: Callable[[str], Awaitable[asyncio.subprocess.Process]] | None = None,
    ) -> None:
        self._base_url = base_url
        self._find = find
        self._probe = probe or _probe
        self._spawn = spawn or _spawn
        self._process: asyncio.subprocess.Process | None = None

    @property
    def started_by_us(self) -> bool:
        return self._process is not None

    async def ensure_running(self) -> bool:
        """True when Ollama answers at the end of this call.

        Never raises. A gate that closes is a log line and a keyword-only recall,
        not a dead daemon."""
        if not is_local(self._base_url):
            logger.info(
                "ollama at %s is not on this machine; leaving it alone", self._base_url
            )
            return await self._probe(self._base_url)
        if await self._probe(self._base_url):
            logger.info("ollama already running at %s", self._base_url)
            return True
        binary = self._find()
        if binary is None:
            logger.warning(
                "ollama is not installed where this process can see it; recall stays "
                "keyword-only. Install it from https://ollama.com, or start it yourself"
            )
            return False
        try:
            self._process = await self._spawn(binary)
        except OSError as exc:
            logger.warning("could not start %s serve (%s); recall stays keyword-only", binary, exc)
            return False
        logger.info("started %s serve (pid %s)", binary, self._process.pid)
        if await self._wait_until_ready():
            return True
        logger.warning(
            "ollama did not answer within %.0fs of being started; recall stays "
            "keyword-only until it does",
            READY_TIMEOUT_SECONDS,
        )
        return False

    async def _wait_until_ready(self) -> bool:
        deadline = asyncio.get_running_loop().time() + READY_TIMEOUT_SECONDS
        while asyncio.get_running_loop().time() < deadline:
            if await self._probe(self._base_url):
                return True
            await asyncio.sleep(POLL_SECONDS)
        return False

    async def aclose(self) -> None:
        """Stop only what this object started.

        An Ollama that was already up belongs to whoever started it and outlives
        this daemon. Same orphan reasoning as the stdio MCP servers in
        `app.py`'s lifespan `finally`: without this, every restart leaves one more."""
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            async with asyncio.timeout(5.0):
                await process.wait()
        except TimeoutError:
            process.kill()
            await process.wait()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_ollama_process.py -v`
Expected: PASS — 3 tests

- [ ] **Step 5: Write the failing tests for spawning and for shutdown ownership**

Append to `tests/test_ollama_process.py`:

```python
class FakeProcess:
    """Enough of `asyncio.subprocess.Process` for the ownership tests."""

    def __init__(self) -> None:
        self.pid = 4242
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


async def test_a_cold_local_ollama_is_started_and_waited_for() -> None:
    answers = iter([False, False, True])
    spawned: list[str] = []

    async def probe(base_url: str) -> bool:
        return next(answers, True)

    async def spawn(binary: str) -> FakeProcess:
        spawned.append(binary)
        return FakeProcess()

    local = LocalOllama(
        "http://127.0.0.1:11434",
        find=lambda: "/opt/homebrew/bin/ollama",
        probe=probe,
        spawn=spawn,
    )

    assert await local.ensure_running() is True
    assert spawned == ["/opt/homebrew/bin/ollama"]
    assert local.started_by_us is True


async def test_aclose_stops_a_process_this_daemon_started() -> None:
    process = FakeProcess()

    async def spawn(binary: str) -> FakeProcess:
        return process

    answers = iter([False, True])

    async def probe(base_url: str) -> bool:
        return next(answers, True)

    local = LocalOllama(
        "http://127.0.0.1:11434",
        find=lambda: "/opt/homebrew/bin/ollama",
        probe=probe,
        spawn=spawn,
    )
    await local.ensure_running()
    await local.aclose()

    assert process.terminated is True


async def test_aclose_leaves_an_ollama_it_did_not_start_running() -> None:
    """The regression that would be invisible: a daemon restart killing the
    owner's own `brew services` Ollama, which then stays dead."""
    local = LocalOllama(
        "http://127.0.0.1:11434",
        find=lambda: "/opt/homebrew/bin/ollama",
        probe=_reachable,
        spawn=_never_spawn,
    )
    await local.ensure_running()
    await local.aclose()

    assert local.started_by_us is False


def test_find_binary_looks_past_a_path_that_lacks_homebrew(monkeypatch) -> None:
    """The measured service condition: PATH is /usr/bin:/bin:/usr/sbin:/sbin, so
    `which` finds nothing and the fallbacks are the only way through."""
    monkeypatch.setattr("daemon.ollama_process.shutil.which", lambda name: None)
    monkeypatch.setattr(
        "daemon.ollama_process.BINARY_FALLBACKS", ("/opt/homebrew/bin/ollama",)
    )
    monkeypatch.setattr(
        "daemon.ollama_process.Path", _PathSaying("/opt/homebrew/bin/ollama")
    )

    assert find_binary() == "/opt/homebrew/bin/ollama"


class _PathSaying:
    def __init__(self, existing: str) -> None:
        self._existing = existing

    def __call__(self, value: str) -> Any:
        existing = self._existing

        class _P:
            def is_file(self) -> bool:
                return value == existing

        return _P()


def test_is_local_accepts_loopback_and_rejects_another_host() -> None:
    assert is_local("http://127.0.0.1:11434") is True
    assert is_local("http://localhost:11434") is True
    assert is_local("http://[::1]:11434") is True
    assert is_local("http://192.168.1.50:11434") is False
    assert is_local("https://ollama.internal.example:11434") is False
```

- [ ] **Step 6: Run the tests to verify they fail, then pass**

Run: `python3 -m pytest tests/test_ollama_process.py -v`
Expected: the five new tests FAIL first if the module is wrong about ownership or
discovery; with Step 3's module as written they PASS. If
`test_find_binary_looks_past_a_path_that_lacks_homebrew` fails on the `Path`
monkeypatch, replace `_PathSaying` with `monkeypatch.setattr` on
`daemon.ollama_process.Path.is_file` — do not weaken the assertion.

- [ ] **Step 7: Lint**

Run: `python3 -m ruff check daemon/ollama_process.py tests/test_ollama_process.py`
Expected: no findings.

- [ ] **Step 8: Commit**

```bash
git add daemon/ollama_process.py tests/test_ollama_process.py
git commit -m "ollama: a child process behind four gates

The daemon survives a reboot on the LaunchAgent's RunAtLoad; its embedder
had no equivalent, and nothing on the machine started Ollama. This module
does, and refuses to in the three cases where acting would be wrong: a
remote base_url, an Ollama already answering, and no binary in sight.

Binary discovery does not trust PATH. Measured: the resident launchd job
runs with PATH=/usr/bin:/bin:/usr/sbin:/sbin and ollama lives in
/opt/homebrew/bin, so which() succeeds in a terminal and fails in the
service."
```

---

### Task 2: Wire it into the daemon, and make the backfill wait

**Files:**
- Modify: `daemon/app.py` — `_backfill` (line 769), the `recall is not None` block
  (line 487-493), the `finally` block (line 546-565)
- Test: `tests/test_ollama_process_wiring.py`

**Interfaces:**
- Consumes: `LocalOllama`, `ensure_running`, `aclose`, `started_by_us` from Task 1.
- Produces: `_backfill(recall: Recall, local_ollama: LocalOllama | None) -> None`
  — the second parameter is new and positional.

- [ ] **Step 1: Write the failing test that the backfill waits**

Create `tests/test_ollama_process_wiring.py`:

```python
"""The ordering that makes Task 1 worth anything.

Measured on the owner's machine, 2026-08-26:

    14:42:16 WARNING recall: backfill stopped after 0 message(s) (unreachable)
    14:50:23 INFO    recall backfill embedded 49 message(s)

The first line is the backfill firing before the embedder could answer; the
second is a restart that happened to land after Ollama was up. Starting Ollama
without fixing the order just reproduces the first line with a shorter gap.
"""

from __future__ import annotations

from daemon.app import _backfill


class RecordingRecall:
    def __init__(self) -> None:
        self.calls = 0

    async def backfill(self, limit: int = 500) -> int:
        self.calls += 1
        return 0


class Waiter:
    """A `LocalOllama` that reports when it was asked, and answers slowly."""

    def __init__(self, *, ready: bool) -> None:
        self._ready = ready
        self.asked = False

    async def ensure_running(self) -> bool:
        self.asked = True
        return self._ready


async def test_the_backfill_asks_for_ollama_before_embedding_anything() -> None:
    recall = RecordingRecall()
    waiter = Waiter(ready=True)

    await _backfill(recall, waiter)

    assert waiter.asked is True
    assert recall.calls == 1


async def test_the_backfill_does_not_run_against_an_embedder_that_never_came_up() -> None:
    """Not a silent skip - `backfill` would only log `stopped after 0 message(s)`
    and never run again, which is the bug this whole change exists to remove."""
    recall = RecordingRecall()
    waiter = Waiter(ready=False)

    await _backfill(recall, waiter)

    assert waiter.asked is True
    assert recall.calls == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 -m pytest tests/test_ollama_process_wiring.py -v`
Expected: FAIL — `TypeError: _backfill() takes 1 positional argument but 2 were given`

- [ ] **Step 3: Give `_backfill` the wait**

In `daemon/app.py`, change `_backfill`'s signature and add the wait at the top of
the body, leaving the exhaustion loop and its comments untouched:

```python
async def _backfill(recall: Recall, local_ollama: LocalOllama | None = None) -> None:
    """Embed history the vector lane is missing, to exhaustion.

    One call was not enough. It stopped at its default 500 rows and never ran
    again - no retry, no periodic job - so a rebuilt sqlite file over a year of
    history left the great majority of messages with no vector while /health
    still reported recall ready. That is the same invisible Korean ceiling the
    protocol change was meant to prevent, just further along.

    The wait for Ollama lives here rather than in `lifespan` on purpose: a cold
    embedder must not delay the log clock (docs/PLAN.md 8.1), and awaiting
    readiness before the `yield` would block uvicorn's "startup complete" and
    /health for as long as a cold start takes. Measured 2026-08-26, skipping the
    wait entirely is what logged `backfill stopped after 0 message(s)` and left 49
    messages unembedded until an unrelated restart.

    Never fatal: recall degrades to keyword-only, which is worse than the full
    answer and far better than a dead conversation loop.
    """
    if local_ollama is not None and not await local_ollama.ensure_running():
        logger.info(
            "recall backfill skipped: no embedder answered. Recall stays keyword-only "
            "and the next restart tries again"
        )
        return
    total = 0
    ...
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_ollama_process_wiring.py -v`
Expected: PASS — 2 tests

- [ ] **Step 5: Construct and close it in `lifespan`**

In `daemon/app.py`, add the import at the top with the other `daemon` imports:

```python
from daemon.ollama_process import LocalOllama
```

Replace the `recall is not None` block (currently lines 487-493) so the object is
built and passed in. Construction is cheap and touches nothing, so it stays
outside the `if`— the `finally` needs it either way:

```python
    local_ollama = LocalOllama(settings.ollama_base_url)
    app.state.local_ollama = local_ollama

    if recall is not None:
        # Backfill after the loop is already serving, and in the background: a
        # rebuilt sqlite file gives every message a new id and drops `embeddings`
        # by cascade, so without this the vector lane stays empty for all history
        # while /health still says recall is ready. Measured on the golden set,
        # that silent state is a 50% ceiling for Korean rather than the hybrid
        # number - a regression where nothing fails. Not awaited, because a cold
        # embedder must not delay the log clock (docs/PLAN.md 8.1) - and that is
        # also why `_backfill` waits for Ollama inside itself rather than here.
        app.state.backfill_task = asyncio.create_task(
            _backfill(recall, local_ollama), name="recall-backfill"
        )
```

In the `finally` block, close it next to the MCP cleanup — after the task
cancellations, so a backfill mid-`ensure_running` is already cancelled:

```python
        local = getattr(app.state, "local_ollama", None)
        if local is not None:
            # A child process, like the stdio MCP servers below: an Ollama this
            # daemon started and did not stop is one more orphan per restart. One
            # it did *not* start is somebody else's and stays running.
            with suppress(Exception):
                await local.aclose()
```

- [ ] **Step 6: Write the failing test that the lifespan closes it**

Append to `tests/test_ollama_process_wiring.py`:

```python
from pathlib import Path

from daemon.app import create_app


async def test_the_lifespan_builds_and_closes_the_ollama_it_owns(tmp_path: Path) -> None:
    """`app.state.local_ollama` has to exist, and the `finally` has to reach it -
    a child process nothing closes is one orphan per restart."""
    from httpx import ASGITransport, AsyncClient

    app = create_app(_settings_for(tmp_path))
    closed: list[bool] = []

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test"):
        local = app.state.local_ollama
        assert local is not None
        original = local.aclose

        async def record() -> None:
            closed.append(True)
            await original()

        local.aclose = record  # type: ignore[method-assign]

    assert closed == [True]
```

Reuse the settings helper the acceptance tests already use — see
`tests/test_acceptance.py:157` `test_the_lifespan_actually_starts_the_conversation_loop`
and its `_settings(tmp_path)`. Import that rather than writing `_settings_for`; if
it is private to that module, copy its body into this test file's own
`_settings_for` and say so in a comment.

- [ ] **Step 7: Run it, then the neighbouring suites**

Run: `python3 -m pytest tests/test_ollama_process_wiring.py tests/test_acceptance.py tests/test_reliability.py tests/test_recall.py tests/test_m1b_audit.py -q`
Expected: PASS. `_backfill`'s new parameter defaults to `None`, so existing
single-argument callers keep working — but **measure it, do not assume**: if any
test constructs `_backfill` or asserts on startup log lines, fix the test to the
new behaviour rather than widening the default.

- [ ] **Step 8: Lint and commit**

```bash
python3 -m ruff check daemon/app.py tests/test_ollama_process_wiring.py
git add daemon/app.py tests/test_ollama_process_wiring.py
git commit -m "ollama: start it at boot, and make the backfill wait for it

The backfill fired at t=0 against an embedder that answers seconds later,
logged 'stopped after 0 message(s)', and never ran again - 49 messages
stayed unembedded until an unrelated restart. The wait lives inside the
backfill task, not in lifespan, because a cold embedder must not delay the
log clock (PLAN 8.1).

The lifespan closes only an Ollama it started."
```

---

### Task 3: Setup stops skipping the check that mattered

**Files:**
- Modify: `daemon/setup.py` — `_check_ollama` (line 2386-2423), `Checks` (line 889)
- Modify: `tests/test_setup.py` — `working_checks` (line 71), `Recorder.checks`
  (line 111), `test_missing_ollama_models_are_printed_as_commands_not_run`
  (line 419)

**Interfaces:**
- Consumes: nothing from Tasks 1-2. This task is independent of them.
- Produces: `Checks.pull: Callable[[str], bool]` — pulls one model, returns
  whether it succeeded. Default `pull_model`.

**Measured blast radius — read before starting.** `tests/test_setup.py` has 232
passing tests and **131 `drive()` calls whose answers are positional lists**. A new
question consumes one answer and shifts every list on a path that reaches it.
`working_checks()` (line 71) returns models `("gemma3:4b",)` — **`bge-m3` is
missing** — and today's early return is the only reason hosted-provider tests never
see the embed-model step. Removing it exposes ~22 `working_checks` tests to a new
prompt at once. Step 1 defuses this before touching `setup.py`.

- [ ] **Step 1: Make the shared fixture describe a working install**

In `tests/test_setup.py`, line 71, add the embed model — a fixture named
`working_checks` should mean everything works:

```python
        ollama=lambda url: OllamaState(
            True, f"reachable at {url} (v0.5.0)", ("gemma3:4b", "bge-m3")
        ),
```

`Recorder.checks`'s `ollama` (line 111-114) already returns
`("gemma3:4b", "bge-m3")`; leave it alone.

- [ ] **Step 2: Run the suite to confirm the fixture change alone is green**

Run: `python3 -m pytest tests/test_setup.py -q`
Expected: 232 passed. A failure here is a test that *asserted on the missing
model* — fix that test to say what it means, then continue.

- [ ] **Step 3: Commit the fixture change on its own**

```bash
git add tests/test_setup.py
git commit -m "tests: working_checks means the embed model is installed too

bge-m3 was missing from a fixture named for everything working, which is
only invisible because setup skips the embed check for hosted providers."
```

- [ ] **Step 4: Write the failing tests for the new behaviour**

Add to `tests/test_setup.py`, next to the existing Ollama tests (~line 419):

```python
def test_a_hosted_provider_still_gets_the_embed_model_checked(tmp_path: Path) -> None:
    """The gap that cost the owner fifteen hours. `Task.EMBED` is always ollama,
    so 'nothing here needs a local chat model' was true and irrelevant: the embed
    model is needed under every provider, and setup never looked."""
    checks = Checks(
        anthropic=lambda key, model: Verdict(True, "key works"),
        gemini=lambda key: Verdict(True, "key works"),
        telegram=lambda token: Verdict(True, "connected to @test_bot"),
        ollama=lambda url: OllamaState(True, f"reachable at {url} (v0.5.0)", ()),
        pull=lambda model: False,
    )
    result = drive(tmp_path, GEMINI_ANSWERS + ["n"], checks=checks)

    assert result.code == 0
    assert "bge-m3" in result.out


def test_the_chat_model_is_not_checked_under_a_hosted_provider(tmp_path: Path) -> None:
    """The one thing the early return had right: a gemini user needs no qwen3."""
    checks = Checks(
        gemini=lambda key: Verdict(True, "key works"),
        telegram=lambda token: Verdict(True, "connected to @test_bot"),
        ollama=lambda url: OllamaState(True, f"reachable at {url} (v0.5.0)", ()),
        pull=lambda model: False,
    )
    result = drive(tmp_path, GEMINI_ANSWERS + ["n"], checks=checks)

    assert result.code == 0
    assert "gemma3:4b" not in result.out
    assert "qwen3" not in result.out


def test_declining_the_embed_model_pull_is_not_an_error(tmp_path: Path) -> None:
    pulled: list[str] = []

    def pull(model: str) -> bool:
        pulled.append(model)
        return True

    checks = Checks(
        gemini=lambda key: Verdict(True, "key works"),
        telegram=lambda token: Verdict(True, "connected to @test_bot"),
        ollama=lambda url: OllamaState(True, f"reachable at {url} (v0.5.0)", ()),
        pull=pull,
    )
    result = drive(tmp_path, GEMINI_ANSWERS + ["n"], checks=checks)

    assert result.code == 0
    assert pulled == []
    assert result.env_path.exists()


def test_accepting_the_embed_model_pull_runs_it(tmp_path: Path) -> None:
    pulled: list[str] = []

    def pull(model: str) -> bool:
        pulled.append(model)
        return True

    checks = Checks(
        gemini=lambda key: Verdict(True, "key works"),
        telegram=lambda token: Verdict(True, "connected to @test_bot"),
        ollama=lambda url: OllamaState(True, f"reachable at {url} (v0.5.0)", ()),
        pull=pull,
    )
    result = drive(tmp_path, GEMINI_ANSWERS + ["y"], checks=checks)

    assert result.code == 0
    assert pulled == ["bge-m3"]


def test_an_unreachable_ollama_does_not_offer_to_pull(tmp_path: Path) -> None:
    """Nothing to pull *into*. Offering a 1.2GB download to a dead server is a
    question with no right answer."""
    pulled: list[str] = []

    def pull(model: str) -> bool:
        pulled.append(model)
        return True

    checks = Checks(
        gemini=lambda key: Verdict(True, "key works"),
        telegram=lambda token: Verdict(True, "connected to @test_bot"),
        ollama=lambda url: OllamaState(False, f"not reachable at {url}"),
        pull=pull,
    )
    result = drive(tmp_path, GEMINI_ANSWERS, checks=checks)

    assert result.code == 0
    assert pulled == []
    assert "https://ollama.com" in result.out
```

`GEMINI_ANSWERS` may not exist. Find the answer list an existing gemini test
passes to `drive` — grep `def test_` around `gemini` in `tests/test_setup.py` — and
either reuse its module-level constant or define `GEMINI_ANSWERS` beside the
existing `TOOLS_YES` / `GOOD_TOKEN` constants with the same values that test uses.
Do not guess the order; copy it from a passing test.

- [ ] **Step 5: Run them to verify they fail**

Run: `python3 -m pytest tests/test_setup.py -k "embed_model or pull" -v`
Expected: FAIL — `TypeError: Checks.__init__() got an unexpected keyword argument 'pull'`

- [ ] **Step 6: Add the pull probe**

In `daemon/setup.py`, beside `check_ollama` (line 834):

```python
def pull_model(model: str) -> bool:
    """`ollama pull <model>`, straight through to the terminal.

    Not an HTTP call to `/api/pull`: the CLI already prints a progress bar, and a
    1.2GB download with no visible progress is a wizard that looks hung."""
    return subprocess.run(("ollama", "pull", model), check=False).returncode == 0
```

Add it to `Checks` (line 889), keeping the one-bundle-of-probes shape:

```python
    pull: Callable[[str], bool] = pull_model
```

- [ ] **Step 7: Rewrite `_check_ollama`**

Replace `daemon/setup.py:2386-2423` entirely:

```python
    def _check_ollama(self, provider: str, env: Mapping[str, str]) -> None:
        """Reachability and the embed model, under every provider.

        This used to return early for anyone not on a local chat model, saying
        "Nothing here needs a local chat model" - true, and beside the point.
        `Task.EMBED` is always ollama whatever `DAEMON_PROVIDER` says, so the embed
        model is needed under every provider and the branch skipped the check for
        exactly the users who would never think to run it. Measured 2026-08-26:
        that gap cost a gemini user fifteen hours of keyword-only Korean recall,
        with `daemon doctor` reporting it correctly the whole time and nothing
        prompting anybody to run `daemon doctor`.
        """
        theme = self.prompt.theme
        say = self.prompt.say
        base_url = env.get("DAEMON_OLLAMA_BASE_URL") or _config_default("ollama_base_url")
        embed_model = env.get("DAEMON_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        state = self.checks.ollama(base_url)
        if not state.reachable:
            say(status(theme, "warn", f"Ollama {state.detail}"))
            say("  Install it from https://ollama.com, then run `ollama serve`.")
            say("  Recall stays keyword-only until it answers - `daemon doctor` re-checks.")
            say()
            return

        say(status(theme, "ok", f"Ollama {state.detail}"))
        wanted = [embed_model]
        if provider == OLLAMA:
            # The one thing the old early return had right.
            wanted.append(env.get("DAEMON_OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL)
        missing = [model for model in wanted if not _installed(state.models, model)]
        for model in wanted:
            if model not in missing:
                say(status(theme, "ok", f"{model}: installed"))
        for model in missing:
            say(status(theme, "missing", f"{model}: not pulled yet"))
        if not missing:
            say()
            return

        # It used to print the commands and stop, on the grounds that "a wizard
        # should not start a download the user did not ask for". Asking is how the
        # user asks for it, and the alternative measured worse: `bge-m3` missing is
        # a Korean recall ceiling nobody sees, and a printed command is one a person
        # reads past. Chat models stay printed-only - they are several GB and only a
        # local-provider user needs them.
        if embed_model in missing:
            say("  Without it, recall drops to keyword-only - Korean worst of all.")
            if self.prompt.ask_yes_no(f"  Pull {embed_model} now? (1.2GB)", default=False):
                if self.checks.pull(embed_model):
                    say(status(theme, "ok", f"{embed_model}: pulled"))
                    missing = [model for model in missing if model != embed_model]
                else:
                    say(status(theme, "warn", f"{embed_model}: pull failed"))
        if missing:
            say("  Run these yourself (they are large):")
            for model in missing:
                say(f"    ollama pull {model}")
        say()
```

- [ ] **Step 8: Run the new tests, then the whole setup suite**

Run: `python3 -m pytest tests/test_setup.py -q`
Expected: the 5 new tests pass and the other 232 stay green.

**When answer lists have shifted,** the failure is a `drive()` call whose
positional answers desynchronized because its path now reaches the pull question.
Fix the answer list — append the `"n"` the new question consumes. Do **not** make
the question conditional to keep tests quiet.

`test_missing_ollama_models_are_printed_as_commands_not_run` (line 419) asserts the
behaviour this task deliberately changes. Its name is now wrong. Rename it to
`test_missing_chat_models_are_printed_as_commands_not_run`, keep the `gemma3:4b`
and "they are large" assertions, drop the `bge-m3` one, and add `"y"`/`"n"` for the
pull question.

- [ ] **Step 9: Lint and commit**

```bash
python3 -m ruff check daemon/setup.py tests/test_setup.py
git add daemon/setup.py tests/test_setup.py
git commit -m "setup: check the embed model under every provider, and offer to pull it

_check_ollama returned early for anyone not on a local chat model, saying
'Nothing here needs a local chat model'. Task.EMBED is always ollama, so
the embed model is needed under every provider and the branch skipped the
check for exactly the users who would never run it themselves. A gemini
user lost fifteen hours of vector recall to that gap.

Reverses the 'a wizard should not start a download the user did not ask
for' comment: it now asks. Chat models stay printed-only."
```

---

### Task 4: The whole-repo gates

**Files:**
- Modify: `docs/ARCHITECTURE.md` — add `ollama_process.py` to the layout
- Modify: `docs/CONTRACTS.md` — add it to the `daemon/` tree
- Modify: `daemon/MEASURED.md` — the PATH measurement
- Verify: `tests/test_reachable.py`

- [ ] **Step 1: Check whether reachability needs an entry**

Run: `python3 -m pytest tests/test_reachable.py -v`
Expected: PASS with no new entry. `LocalOllama` is constructed by
`daemon/app.py`'s `lifespan`, so `_constructed` finds it — unlike a grant or a
constructor argument, which is why `PENDING_CLASSES` is empty and its comment
explains the blind spot. **If it fails,** read the comment at
`tests/test_reachable.py:46` before adding anything: the fix is usually to
construct the thing, not to declare it pending.

- [ ] **Step 2: Add the module to both layout docs**

In `docs/CONTRACTS.md`, in the `daemon/` tree after the `companion.py` entry:

```
  ollama_process.py   the local Ollama the daemon starts for embeddings. Only
                      app.py imports it (rule 4).
```

`docs/ARCHITECTURE.md` is **not** a module list — `## The shape` is a mermaid
`flowchart LR` of subgraphs. Add one node to the `memory["memory/"]` subgraph
(line 49, after the `RECALL` node on line 52), because this module exists to serve
the embedder recall depends on:

```
    OLL["ollama_process.py<br/>starts the local embedder · app.py only"]
```

Then add the edge `APP --> OLL` alongside the other `APP -->` edges. Read the
existing edge block first and match its arrow style; do not invent a new one.

- [ ] **Step 3: Record the measurement**

`daemon/MEASURED.md` has exactly one heading and is otherwise a flat bulleted
list of `- **bold claim.** evidence` entries. Add one bullet in that style — no
new heading:

```markdown
- **The resident cannot see homebrew, so `shutil.which` is not binary discovery.**
  Measured 2026-08-26 on the live `ai.daemon.default` job: `PATH` is
  `/usr/bin:/bin:/usr/sbin:/sbin` while `ollama` is at `/opt/homebrew/bin/ollama`.
  `_render_plist` omits `EnvironmentVariables` deliberately (`service.py:226` - the
  working directory is how the process finds `.env`) and `launchctl getenv PATH` is
  unset, so nothing puts homebrew's bin back. `which("ollama")` therefore resolves
  in every terminal test and returns `None` in the service the code actually runs
  in - a soft dependency that then degrades silently instead of failing.
  `ollama_process.py:BINARY_FALLBACKS` is the consequence.
```

- [ ] **Step 4: Run every repo gate**

```bash
python3 -m pytest -q
python3 -m ruff check .
python3 scripts/check_docs.py
```

Expected: suite green, no lint findings, `check_docs.py` reporting every
documented path exists (it counted 321 before this change).

- [ ] **Step 5: Drive the real thing**

Green unit tests are not proof this works. Run the actual command against the
actual binary:

```bash
pkill -f "ollama serve"
cd /Users/gimdaehyeon/Daemon && PYTHONPATH=/Users/gimdaehyeon/Daemon/.claude/worktrees/ollama-embedder-unreachable-162244 python3 -c "
import asyncio, logging
logging.basicConfig(level=logging.INFO)
from daemon.ollama_process import LocalOllama, find_binary
print('binary:', find_binary())
async def main():
    local = LocalOllama('http://127.0.0.1:11434')
    print('ensure_running ->', await local.ensure_running())
    print('started_by_us ->', local.started_by_us)
    await local.aclose()
asyncio.run(main())
"
curl -s -m 3 -o /dev/null -w 'after aclose: http=%{http_code}\n' http://127.0.0.1:11434/api/tags
```

Expected: `binary:` prints a real path; `ensure_running -> True`;
`started_by_us -> True`; and after `aclose` the curl fails — proof the child was
stopped. Then run it a second time **while** an Ollama is already up and confirm
`started_by_us -> False` and that curl still succeeds afterwards.

Note the `PYTHONPATH` and `cd`: `.env` is cwd-relative and cwd precedes
`PYTHONPATH` on `sys.path`, so running from the main checkout would import the
main checkout's `daemon/` and prove nothing about this code.

- [ ] **Step 6: Commit**

```bash
git add docs/ARCHITECTURE.md docs/CONTRACTS.md daemon/MEASURED.md
git commit -m "docs: record ollama_process and the PATH it cannot trust"
```

---

## Self-review

**Spec coverage.** Spec §1 (setup) → Task 3. §2 (boot, four gates, discovery,
shutdown ownership) → Task 1 + Task 2 Step 5. §3 (backfill ordering) → Task 2
Steps 1-4. §4 (observability: one log line, no doctor change) → Task 1 Step 3's
`logger.info("started %s serve (pid %s)")`; no doctor task exists, deliberately.
Spec "Testing" list → Task 1 Steps 1/5, Task 3 Step 4. Spec "Out of scope" → no
task, correct.

**The reversed decision** the spec flags is carried into the code as a rewritten
comment in Task 3 Step 7, not dropped.

**Type consistency.** `ensure_running` / `aclose` / `started_by_us` /
`find_binary` / `is_local` / `READY_TIMEOUT_SECONDS` / `BINARY_FALLBACKS` are
spelled identically in Tasks 1, 2 and 4. `Checks.pull` is
`Callable[[str], bool]` in both its definition (Task 3 Step 6) and every test
that injects it (Step 4). `_backfill`'s second parameter is positional with a
`None` default in both the test (Step 1) and the implementation (Step 3).

**Known unknowns, flagged rather than guessed.** Three places tell the
implementer to read the repo instead of trusting this plan: `GEMINI_ANSWERS`
(Task 3 Step 4), the `_settings` helper (Task 2 Step 6), and the `Path`
monkeypatch shape (Task 1 Step 6). Each names where to look and what not to
weaken.
