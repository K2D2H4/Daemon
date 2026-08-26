"""The ordering that makes Task 1 worth anything.

Measured on the owner's machine, 2026-08-26:

    14:42:16 WARNING recall: backfill stopped after 0 message(s) (unreachable)
    14:50:23 INFO    recall backfill embedded 49 message(s)

The first line is the backfill firing before the embedder could answer; the
second is a restart that happened to land after Ollama was up. Starting Ollama
without fixing the order just reproduces the first line with a shorter gap.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from starlette.testclient import TestClient

from daemon.app import _backfill, create_app

# `_settings`, `_Idle` and `_Mem` are private to test_acceptance.py by convention
# only (no `tests/__init__.py` gating the import) - the same pattern
# tests/test_face.py's `RecordingBus` and tests/test_mcp.py's fakes are already
# reused under (see test_voice_conversation.py, test_mcp_engine.py). `_Idle`/`_Mem`
# keep the lifespan off the network, exactly as test_acceptance.py's own
# `test_the_lifespan_actually_starts_the_conversation_loop` needs them for.
from tests.test_acceptance import _Idle, _Mem
from tests.test_acceptance import _settings as _settings_for


class RecordingRecall:
    def __init__(self) -> None:
        self.calls = 0

    async def backfill(self, limit: int = 500) -> int:
        self.calls += 1
        return 0


async def test_the_backfill_waits_rather_than_embedding_against_a_cold_ollama() -> None:
    """The ordering assertion, not just the outcome: while Ollama is still coming
    up, `backfill` must not have been called even once. Asserting only the final
    count would pass against code that never waited at all."""
    recall = RecordingRecall()
    gate = asyncio.Event()

    async def ollama_ready() -> bool:
        await gate.wait()
        return True

    running = asyncio.create_task(_backfill(recall, asyncio.create_task(ollama_ready())))
    await asyncio.sleep(0)
    assert recall.calls == 0  # the whole point: still waiting

    gate.set()
    await running

    assert recall.calls == 1


async def test_the_backfill_does_not_run_against_an_embedder_that_never_came_up() -> None:
    """Not a silent skip - `backfill` would only log `stopped after 0 message(s)`
    and never run again, which is the bug this whole change exists to remove."""
    recall = RecordingRecall()

    async def never() -> bool:
        return False

    await _backfill(recall, asyncio.create_task(never()))

    assert recall.calls == 0


class FakeOllama:
    """An injected stand-in. A real `LocalOllama` here would probe localhost and
    start a server, which is what `local_ollama=None` exists to prevent."""

    def __init__(self) -> None:
        self.asked = False
        self.closed = False

    async def ensure_running(self) -> bool:
        self.asked = True
        return True

    async def aclose(self) -> None:
        self.closed = True


def test_the_lifespan_starts_and_closes_the_ollama_it_was_given(tmp_path: Path) -> None:
    """A child process nothing closes is one orphan per restart - the reason the
    stdio MCP servers are closed in the same `finally`."""
    fake = FakeOllama()

    app = create_app(
        _settings_for(tmp_path), channel=_Idle(), memory=_Mem(), local_ollama=fake
    )
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"

    assert fake.asked is True
    assert fake.closed is True


def test_no_ollama_injected_means_no_start_task(tmp_path: Path) -> None:
    """The default every existing test takes. If this ever regresses, the suite
    starts spawning real servers on whatever machine runs it."""
    app = create_app(_settings_for(tmp_path), channel=_Idle(), memory=_Mem())
    with TestClient(app):
        assert app.state.ollama_task is None
