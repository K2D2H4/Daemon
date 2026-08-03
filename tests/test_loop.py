"""The conversation loop, and the app assembly around it.

The channel and the memory writer are local fakes: both are protocols, and the
real implementations are owned elsewhere. The loop must not be able to tell.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeProvider

from daemon.app import create_app
from daemon.channels.base import Channel, InboundMessage, OutboundMessage
from daemon.config import Route, Settings
from daemon.llm.base import Completion, Message, ProviderError
from daemon.llm.gateway import LLMGateway
from daemon.loop import FAILURE_NOTICE, ConversationLoop
from daemon.memory.base import LoggedMessage, MemoryWriter
from daemon.tasks import Task


class FakeMemory:
    """Mirrors instantly, which is the harder case for context assembly: the
    user turn is already in `recent()` when the prompt is built."""

    def __init__(self) -> None:
        self.records: list[LoggedMessage] = []

    async def record(self, message: LoggedMessage) -> None:
        self.records.append(message)

    async def seen(self, channel: str, external_id: str) -> bool:
        return any(
            r.channel == channel and r.external_id == external_id for r in self.records
        )

    async def recent(self, limit: int = 20) -> list[LoggedMessage]:
        return self.records[-limit:]


class FakeChannel:
    name = "fake"

    def __init__(self, inbound: list[InboundMessage]) -> None:
        self._inbound = inbound
        self.sent: list[OutboundMessage] = []
        self.closed = False

    async def send(self, message: OutboundMessage) -> None:
        self.sent.append(message)

    def listen(self) -> AsyncIterator[InboundMessage]:
        async def stream() -> AsyncIterator[InboundMessage]:
            for message in self._inbound:
                yield message

        return stream()

    async def close(self) -> None:
        self.closed = True


class FlakyProvider:
    """Fails the first call, then behaves."""

    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Completion:
        self.calls += 1
        if self.calls == 1:
            raise ProviderError("first turn is cursed")
        return Completion(text="recovered", model=model)

    async def health(self) -> bool:
        return True


def inbound(text: str) -> InboundMessage:
    return InboundMessage(
        text=text,
        sender_id="42",
        received_at=datetime(2026, 8, 3, 7, 14, tzinfo=UTC),
        channel="fake",
    )


def gateway_for(provider: Any) -> LLMGateway:
    return LLMGateway({provider.name: provider}, {Task.CHAT_TEXT: Route(provider.name, "m")})


def test_fakes_satisfy_the_protocols() -> None:
    assert isinstance(FakeChannel([]), Channel)
    assert isinstance(FakeMemory(), MemoryWriter)


# --- one turn ---------------------------------------------------------------


async def test_records_the_user_turn_before_the_reply(
    data_dir: Path, fake_provider: FakeProvider
) -> None:
    memory = FakeMemory()
    channel = FakeChannel([inbound("hello")])

    await ConversationLoop(
        channel, gateway_for(fake_provider), memory, data_dir=data_dir
    ).run()

    assert [(m.role, m.content) for m in memory.records] == [
        ("user", "hello"),
        ("assistant", "ok"),
    ]
    assert [m.origin for m in memory.records] == ["owner", "agent"]
    assert [m.session_kind for m in memory.records] == ["interactive", "interactive"]
    assert memory.records[0].sender_id == "42"
    assert [m.text for m in channel.sent] == ["ok"]


async def test_the_user_turn_is_not_duplicated_in_the_prompt(
    data_dir: Path, fake_provider: FakeProvider
) -> None:
    memory = FakeMemory()

    await ConversationLoop(
        FakeChannel([inbound("hello")]), gateway_for(fake_provider), memory, data_dir=data_dir
    ).run()

    prompt = fake_provider.calls[0]
    assert prompt == [Message(role="user", content="hello")]


async def test_history_is_carried_into_the_next_turn(
    data_dir: Path, fake_provider: FakeProvider
) -> None:
    memory = FakeMemory()

    await ConversationLoop(
        FakeChannel([inbound("first"), inbound("second")]),
        gateway_for(fake_provider),
        memory,
        data_dir=data_dir,
    ).run()

    assert fake_provider.calls[1] == [
        Message(role="user", content="first"),
        Message(role="assistant", content="ok"),
        Message(role="user", content="second"),
    ]


async def test_persona_seed_becomes_the_system_turn(
    data_dir: Path, fake_provider: FakeProvider
) -> None:
    (data_dir / "persona" / "seed.md").write_text("You disagree when you disagree.\n")

    await ConversationLoop(
        FakeChannel([inbound("hello")]), gateway_for(fake_provider), FakeMemory(), data_dir=data_dir
    ).run()

    assert fake_provider.calls[0][0] == Message(
        role="system", content="You disagree when you disagree."
    )


async def test_missing_seed_means_no_system_turn(
    data_dir: Path, fake_provider: FakeProvider
) -> None:
    await ConversationLoop(
        FakeChannel([inbound("hello")]), gateway_for(fake_provider), FakeMemory(), data_dir=data_dir
    ).run()

    assert all(m.role != "system" for m in fake_provider.calls[0])


async def test_voice_inbound_is_recorded_as_a_voice_session(data_dir: Path) -> None:
    memory = FakeMemory()
    spoken = InboundMessage(
        text="hello",
        sender_id="42",
        received_at=datetime(2026, 8, 3, 7, 14, tzinfo=UTC),
        channel="fake",
        modality="voice",
    )

    await ConversationLoop(
        FakeChannel([spoken]), gateway_for(FakeProvider()), memory, data_dir=data_dir
    ).run()

    assert [m.session_kind for m in memory.records] == ["voice", "voice"]
    assert [m.modality for m in memory.records] == ["voice", "voice"]


# --- a bad turn must not end the loop ---------------------------------------


async def test_a_failed_turn_is_reported_and_the_loop_continues(data_dir: Path) -> None:
    memory = FakeMemory()
    channel = FakeChannel([inbound("first"), inbound("second")])

    await ConversationLoop(
        channel, gateway_for(FlakyProvider()), memory, data_dir=data_dir
    ).run()

    assert [m.text for m in channel.sent] == [FAILURE_NOTICE, "recovered"]
    # The failed turn still recorded what the user said - the log clock does not
    # depend on the model answering.
    assert [(m.role, m.content) for m in memory.records] == [
        ("user", "first"),
        ("user", "second"),
        ("assistant", "recovered"),
    ]


async def test_a_broken_channel_does_not_end_the_loop(data_dir: Path) -> None:
    class MuteChannel(FakeChannel):
        async def send(self, message: OutboundMessage) -> None:
            raise RuntimeError("telegram is down")

    channel = MuteChannel([inbound("first"), inbound("second")])

    await ConversationLoop(
        channel, gateway_for(FlakyProvider()), FakeMemory(), data_dir=data_dir
    ).run()  # must return rather than raise


# --- the M1a gate -----------------------------------------------------------


async def test_the_exchange_lands_in_the_markdown_log(
    data_dir: Path, db: Any, fake_provider: FakeProvider
) -> None:
    """docs/PLAN.md 8.2, M1a gate: message it, it answers, and the exchange is in
    memory/log/YYYY-MM-DD.md. Everything but the channel is real here."""
    from daemon.memory.store import Store
    from daemon.memory.writer import FileMemoryWriter

    memory = FileMemoryWriter(data_dir, Store(db))
    channel = FakeChannel([inbound("what did I say about the talk?")])

    await ConversationLoop(
        channel, gateway_for(fake_provider), memory, data_dir=data_dir
    ).run()

    logs = sorted((data_dir / "memory" / "log").glob("*.md"))
    assert len(logs) == 1
    written = logs[0].read_text(encoding="utf-8")
    assert "what did I say about the talk?" in written
    assert "ok" in written
    assert [m.text for m in channel.sent] == ["ok"]


# --- app assembly -----------------------------------------------------------


@pytest.fixture
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("DAEMON_", "TELEGRAM_")) or name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)


def test_health_reports_routing_and_a_stopped_loop(_isolated_env: None) -> None:
    # No telegram credentials, so the loop cannot start. That must be visible in
    # /health rather than swallowed.
    from fastapi.testclient import TestClient

    app = create_app(Settings(_env_file=None, preset="offline"))

    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["preset"] == "offline"
    assert body["routing"][Task.CHAT_TEXT.value] == "ollama"
    assert body["conversation_loop"] == "stopped"


def test_health_reports_a_running_loop_when_io_is_injected(_isolated_env: None) -> None:
    from fastapi.testclient import TestClient

    class IdleChannel(FakeChannel):
        """Listens forever without yielding, like a real channel between messages."""

        def listen(self) -> AsyncIterator[InboundMessage]:
            async def stream() -> AsyncIterator[InboundMessage]:
                await asyncio.Event().wait()
                yield inbound("unreachable")

            return stream()

    app = create_app(
        Settings(_env_file=None, preset="offline"),
        channel=IdleChannel([]),
        memory=FakeMemory(),
    )

    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["conversation_loop"] == "running"
