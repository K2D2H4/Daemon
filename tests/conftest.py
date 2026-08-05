"""Shared fixtures. Use these; do not invent parallel ones.

Hard rule: no test in this suite may hit the network or a real LLM.
A test that needs an API key is a broken test.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from daemon.llm.base import Completion, Message, ProviderError, ToolCall, ToolSpec

SCHEMA = Path(__file__).resolve().parents[1] / "daemon" / "memory" / "schema.sql"


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
