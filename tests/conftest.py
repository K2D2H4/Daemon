"""Shared fixtures. Use these; do not invent parallel ones.

Hard rule: no test in this suite may hit the network or a real LLM.
A test that needs an API key is a broken test.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from daemon.llm.base import Completion, Message, ProviderError

SCHEMA = Path(__file__).resolve().parents[1] / "daemon" / "memory" / "schema.sql"


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
