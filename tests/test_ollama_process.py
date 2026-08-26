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


async def test_a_malformed_port_closes_the_gate_instead_of_raising() -> None:
    """`urlparse(...).hostname` never touches the port, so `is_local` passes a
    base_url like this straight through - it is the real `_probe` (unmocked
    here on purpose) that hands httpx a URL it cannot parse.
    `httpx.InvalidURL.__mro__` is `(InvalidURL, Exception, ...)`, not
    `HTTPError`, so a one-character `.env` typo used to escape `ensure_running`'s
    "never raises" promise entirely."""
    local = LocalOllama(
        "http://127.0.0.1:notaport",
        find=lambda: None,
        spawn=_never_spawn,
    )

    assert await local.ensure_running() is False
