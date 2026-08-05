"""Shared fixtures. Use these; do not invent parallel ones.

Hard rule: no test in this suite may hit the network or a real LLM.
A test that needs an API key is a broken test.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from daemon.llm.base import Completion, Message, ProviderError

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
def no_real_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the developer's own environment from every test.

    `Settings(_env_file=None, ...)` stops pydantic-settings reading `.env` from
    disk but *not* from the process environment, so a shell that had sourced
    `.env` - which is how you run the product - leaked real values into tests that
    had explicitly chosen their own.

    Found by sourcing `.env` and running the suite: `DAEMON_VOICE_ENABLED=true`
    made `test_the_allowlist_accepts_what_a_person_would_type` raise ConfigError,
    because voice is on and the preset that test picks is `offline`. Three
    allowlist cases failed for a reason that had nothing to do with allowlists.

    Autouse and suite-wide rather than a fix to that one test: a green suite that
    depends on a clean shell is not a gate, and the same leak was available to
    every test that builds `Settings`. It also makes CONTRACTS' "a test that needs
    an API key is a broken test" enforceable rather than aspirational - a test
    cannot now accidentally *see* a real key either.
    """
    for name in [k for k in os.environ if k.startswith(CONFIG_PREFIXES)]:
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


class FakeProvider:
    """Deterministic stand-in for a real LLM.

    Records every call so tests can assert on routing, and can be told to fail
    so fallback paths get exercised.
    """

    name = "fake"

    def __init__(self, reply: str = "ok", *, fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[list[Message]] = []
        self.models: list[str] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Completion:
        self.calls.append(list(messages))
        self.models.append(model)
        if self.fail:
            raise ProviderError("fake provider was told to fail")
        return Completion(text=self.reply, model=model, input_tokens=1, output_tokens=1)

    async def health(self) -> bool:
        return not self.fail


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
