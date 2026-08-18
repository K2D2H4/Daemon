"""Shared fixtures. Use these; do not invent parallel ones.

Hard rule: no test in this suite may hit the network or a real LLM.
A test that needs an API key is a broken test.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from daemon.llm.base import Completion, Message, ProviderError, ToolCall, ToolSpec

SCHEMA = Path(__file__).resolve().parents[1] / "daemon" / "memory" / "schema.sql"

CONFIG_PREFIXES = ("DAEMON_", "TELEGRAM_", "GEMINI_", "ANTHROPIC_", "OPENAI_")
"""Every prefix `daemon/config.py` reads from the environment."""

CANARY = "DAEMON_CONFTEST_CANARY"
"""Planted in the environment at collection so the isolation below is falsifiable.

Without it, `test_the_environment_is_isolated` only fails on a machine that happens
to have `DAEMON_*` set - so CI, which is clean, would pass with the isolation
deleted. Removing the fixture now fails everywhere instead of nowhere."""


def pytest_configure(config: pytest.Config) -> None:
    os.environ[CANARY] = "1"


def pytest_unconfigure(config: pytest.Config) -> None:
    os.environ.pop(CANARY, None)


@pytest.fixture(autouse=True)
def no_real_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide the developer's own configuration from every test - both halves of it.

    **The process environment.** `Settings(_env_file=None, ...)` stops
    pydantic-settings reading `.env` from disk but *not* from the process
    environment, so a shell that had sourced `.env` - which is how you run the
    product - leaked real values into tests that had explicitly chosen their own.

    Found by sourcing `.env` and running the suite: `DAEMON_VOICE_ENABLED=true`
    made `test_the_allowlist_accepts_what_a_person_would_type` raise ConfigError,
    because voice is on and the preset that test picks is `offline`. Three
    allowlist cases failed for a reason that had nothing to do with allowlists.

    **The file on disk.** The mirror image, and it survived the fix above: a test
    that does *not* pass `_env_file=None` - `daemon.cli.main`, building `Settings()`
    the way the product does - reads `ENV_FILE = ".env"` relative to the *current
    directory*, and pytest runs in the repo root. So the developer's real `.env`
    decided the result: the same `DAEMON_VOICE_ENABLED=true` plus a monkeypatched
    `DAEMON_PRESET=offline` made `test_the_audit_trail_is_readable_from_the_cli`
    exit 2 instead of 0. It passed in CI only because CI has no `.env`. Chdir'ing
    into `tmp_path` puts every test somewhere with no `.env` in it; a test that
    wants one writes it there itself, which is what test_cli.py and test_setup.py
    were already doing per-file.

    Autouse and suite-wide rather than a fix to those tests: a green suite that
    depends on a clean shell - or on the developer not having run `daemon setup` -
    is not a gate, and both leaks were available to every test that builds
    `Settings`. It also makes CONTRACTS' "a test that needs an API key is a broken
    test" enforceable rather than aspirational - a test cannot now accidentally
    *see* a real key either.
    """
    for name in [k for k in os.environ if k.startswith(CONFIG_PREFIXES)]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _no_ambient_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own exported settings must not decide what the suite asserts.

    `Settings(_env_file=None)` still reads the process environment, so a shell with
    `DAEMON_BROWSER_ENABLED=true` in it turned three tests red - and the people who
    export that are precisely the people using the feature those tests cover.
    test_config.py has carried its own copy of this for the same reason; this one
    covers the rest of the suite.
    """
    for name in list(os.environ):
        if name.startswith(("DAEMON_", "TELEGRAM_")) or name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """Fresh database with the real schema applied, in a temp dir."""
    conn = sqlite3.connect(tmp_path / "daemon.sqlite3")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA.read_text())
    yield conn
    conn.close()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Isolated data dir. Never point a test at a developer's real one."""
    for sub in ("memory/log", "memory/entities", "persona"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def seoul(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the machine's local zone to Asia/Seoul.

    The single definition. `test_timesense.py` and `test_candidates.py` each wrap
    this in their own autouse `_seoul` fixture that just requests it, for any test
    whose assertion is a Korean phrase built from a local-time conversion
    (`daemon/timesense.py`). CI (`ubuntu-latest`) sets no timezone, so a string like
    "오후 4시" that reads right on a KST machine would name the wrong hour there
    without this.
    """
    monkeypatch.setenv("TZ", "Asia/Seoul")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


class FakeProvider:
    """Deterministic stand-in for a real LLM.

    Records every call so tests can assert on routing, and can be told to fail
    so fallback paths get exercised.

    `scripted_calls` drives the tool loop: each entry is the tool calls to return
    from the corresponding completion, and once the script runs out it answers with
    `reply` - which is what ends the loop. A list rather than one value because the
    interesting cases are multi-round.
    """

    name = "fake"

    def __init__(
        self,
        reply: str = "ok",
        *,
        fail: bool = False,
        scripted_calls: Sequence[Sequence[ToolCall]] | None = None,
    ) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[list[Message]] = []
        self.models: list[str] = []
        self.offered_tools: list[tuple[ToolSpec, ...]] = []
        self._script = [tuple(round_) for round_ in (scripted_calls or ())]

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> Completion:
        self.calls.append(list(messages))
        self.models.append(model)
        self.offered_tools.append(tuple(tools or ()))
        if self.fail:
            raise ProviderError("fake provider was told to fail")
        # Only when tools were actually offered, because that is the one thing a
        # real provider cannot do otherwise. Without this the fake could answer a
        # deliberately tool-free call - the loop's round-limit escape hatch - with a
        # tool call, and the escape hatch would look broken when it was not.
        asked = self._script.pop(0) if (self._script and tools) else ()
        return Completion(
            # Empty text alongside tool calls, the way a real provider answers a
            # tool-use turn.
            text="" if asked else self.reply,
            model=model,
            input_tokens=1,
            output_tokens=1,
            tool_calls=asked,
        )

    async def health(self) -> bool:
        return not self.fail


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
