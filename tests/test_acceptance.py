"""The user's journey, end to end, with fakes only at the network edge.

Written after shipping a milestone that passed 470 unit tests and then failed on
first contact, three separate ways:

  * `daemon run` logged "TELEGRAM_ALLOWED_USER_IDS is empty; refusing to start"
    and served nothing, because pairing was implemented but no config selected it
    and no assembly passed it.
  * The bot never answered, because nothing was actually running.
  * Voice was reported complete while no code path could reach it.

Not one of those is visible from a unit test, and every one of them is visible
from starting the thing and talking to it. So that is what this file does: it
assembles the app the way `_build_io` does, drives a conversation through it, and
checks the whole chain the product actually promises - the reply, the markdown, the
mirror, the vector, and the recall on the next turn.

Fakes stop at the edge: the LLM, the embedder and Telegram's HTTP. Everything
between is the real thing, because the defects live between.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from daemon.channels.base import InboundMessage
from daemon.channels.pairing import Pairing
from daemon.channels.telegram import TelegramChannel
from daemon.config import Route, Settings
from daemon.llm.base import Completion, Message
from daemon.llm.gateway import LLMGateway
from daemon.loop import ConversationLoop
from daemon.memory.recall import MemoryRecall
from daemon.memory.store import Store
from daemon.memory.writer import FileMemoryWriter
from daemon.tasks import Task

OWNER = 5502877373
TOKEN = "123456:AAHfake-token-value"


class Provider:
    """Answers with something a person could plausibly have said, so recall has
    real text to find rather than a sentinel."""

    name = "fake"

    def __init__(self, reply: str = "좋네. 몇 시에 만나?") -> None:
        self.prompts: list[list[Message]] = []
        self.reply = reply

    @property
    def calls(self) -> list[list[Message]]:
        """Alias so a test can say "no model call happened" in those words."""
        return self.prompts

    async def complete(self, messages: list[Message], *, model: str, **kw: Any) -> Completion:
        self.prompts.append(list(messages))
        return Completion(text=self.reply, model=model)

    async def health(self) -> bool:
        return True


class Embedder:
    """Deterministic and offline. Token overlap is enough for the one recall
    assertion here; the golden set is where recall quality is measured."""

    name = "fake"
    model = "fake-embed"
    dimensions = 32

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in text.split():
                vector[hash(token) % self.dimensions] += 1.0
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            out.append([v / norm for v in vector])
        return out


def _settings(tmp_path: Path, **extra: Any) -> Settings:
    return Settings(
        _env_file=None,
        DAEMON_PRESET="offline",
        DAEMON_OLLAMA_MODEL="gemma3:4b",
        DAEMON_DATA_DIR=str(tmp_path),
        TELEGRAM_BOT_TOKEN=TOKEN,
        **extra,
    )


# --- the defect that reached the user ---------------------------------------


def test_a_first_run_with_no_allowlist_can_actually_start(tmp_path: Path) -> None:
    """This is the failure a person hit: a fresh install has no allowlist, and
    `allowlist` mode refuses to start on one - correct as a policy, and the exact
    state a first run is in. Assembly has to produce a channel that can run."""
    from daemon.app import _build_io

    settings = _settings(tmp_path)
    assert settings.telegram_allowed_user_ids == ()
    assert settings.telegram_dm_policy == "pairing"

    io = _build_io(settings)  # must not raise
    try:
        assert io.channel is not None
        assert io.memory is not None
    finally:
        io.close()


def test_the_lifespan_actually_starts_the_conversation_loop(tmp_path: Path) -> None:
    """/health reporting `conversation_loop: running` is the only external sign
    that the daemon is not a healthy-looking process with no inbound path.

    The channel and writer are injected, not because assembly is uninteresting -
    the test above covers that - but because the real one long-polls
    api.telegram.org, and a test that reaches the network is a broken test.
    """
    from starlette.testclient import TestClient

    from daemon.app import create_app
    from daemon.memory.base import LoggedMessage

    class Idle:
        """Yields nothing and stays open, like a quiet channel."""

        name = "telegram"

        async def send(self, message: Any) -> None: ...

        async def listen(self) -> Any:
            import asyncio

            await asyncio.Event().wait()  # never fires; cancelled at shutdown
            yield  # pragma: no cover

        async def close(self) -> None: ...

    class Mem:
        async def record(self, message: LoggedMessage) -> None: ...

        async def seen(self, channel: str, external_id: str) -> bool:
            return False

        async def recent(self, limit: int = 20) -> list[LoggedMessage]:
            return []

    app = create_app(_settings(tmp_path), channel=Idle(), memory=Mem())
    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["conversation_loop"] == "running", (
        "the app booted without a conversation loop - the shape of the defect "
        "where a person messages the bot forever and nothing answers"
    )


# --- pairing, the way a person does it --------------------------------------


class FakeTelegram:
    """Telegram's HTTP surface. Serves each batch once, then nothing."""

    def __init__(self, batches: list[list[dict]]) -> None:
        self.batches = batches
        self.sent: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/getUpdates"):
            result = self.batches.pop(0) if self.batches else []
            return httpx.Response(200, json={"ok": True, "result": result})
        self.sent.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True, "result": {}})


def _update(update_id: int, text: str, sender: int = OWNER) -> dict:
    return {
        "update_id": update_id,
        "message": {"from": {"id": sender}, "date": 1785744000, "text": text},
    }


async def test_a_stranger_gets_a_code_and_the_owner_gets_answers(tmp_path: Path) -> None:
    """The whole onboarding path: the first message is refused with a code, the
    code is approved from the machine, and only then does the daemon talk."""
    store = Store.open(tmp_path / "daemon.sqlite3")
    api = FakeTelegram([[_update(1, "안녕")]])
    try:
        pairing = Pairing(store, TelegramChannel.name)
        channel = TelegramChannel(
            TOKEN,
            (),
            client=httpx.AsyncClient(transport=httpx.MockTransport(api.handler)),
            dm_policy="pairing",
            pairing=pairing,
            cursor=store,
        )
        # listen() yields nothing for an unpaired sender and keeps polling, so it
        # is driven as a task and stopped - waiting for it to end would wait
        # forever, which is exactly the behaviour being asserted.
        heard: list[InboundMessage] = []

        async def drain() -> None:
            async for inbound in channel.listen():
                heard.append(inbound)

        task = asyncio.create_task(drain())
        for _ in range(200):  # up to ~2s of real time, then give up and assert
            await asyncio.sleep(0.01)
            if api.sent:
                break
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await channel.close()

        assert heard == [], "an unpaired sender must not be heard"
        assert len(api.sent) == 1, "the stranger was not answered with a code"
        code = next(r.code for r in pairing.pending())
        assert code in api.sent[0]["text"]

        approval = pairing.approve(code)
        assert approval.is_owner and approval.sender_id == str(OWNER)
        assert store.is_allowed("telegram", str(OWNER))
    finally:
        store.close()


# --- the chain the product promises ------------------------------------------


async def test_a_turn_lands_in_markdown_the_mirror_and_the_vector_index(
    tmp_path: Path,
) -> None:
    """Reply, then markdown, then sqlite, then a vector - in that order, because
    markdown is the source of truth and the rest is rebuildable from it."""
    store = Store.open(tmp_path / "daemon.sqlite3")
    try:
        writer = FileMemoryWriter(tmp_path, store)
        recall = MemoryRecall(store, Embedder())
        provider = Provider()
        gateway = LLMGateway({"fake": provider}, {Task.CHAT_TEXT: Route("fake", "m")})

        sent: list[str] = []

        class Channel:
            name = "telegram"

            async def send(self, message: Any) -> None:
                sent.append(message.text)

            def listen(self) -> Any:  # pragma: no cover - driven directly
                raise NotImplementedError

            async def close(self) -> None: ...

        loop = ConversationLoop(
            Channel(),
            gateway,
            writer,
            data_dir=tmp_path,
            recall=recall,
            resolve_id=lambda _text: writer.last_inserted_id,
        )
        await loop.handle(
            InboundMessage(
                text="내일 저녁에 연희동에서 만나자",
                sender_id=str(OWNER),
                received_at=datetime.now(UTC),
                channel="telegram",
                external_id="1",
            )
        )

        assert sent == ["좋네. 몇 시에 만나?"], "the user got no reply"

        day = next((tmp_path / "memory" / "log").glob("*.md")).read_text()
        assert "연희동" in day, "the source of truth does not have the user's words"
        assert "좋네. 몇 시에 만나?" in day

        rows = store.recent(10)
        assert [r["role"] for r in rows] == ["user", "assistant"]
        assert store.count_embeddings("fake-embed") == 2, "nothing was indexed"
    finally:
        store.close()


async def test_yesterday_is_quoted_back_tomorrow(tmp_path: Path) -> None:
    """The M1b gate, as a person would check it: say something, ask about it
    later, and see it in the prompt the model was given."""
    store = Store.open(tmp_path / "daemon.sqlite3")
    try:
        writer = FileMemoryWriter(tmp_path, store)
        recall = MemoryRecall(store, Embedder())
        provider = Provider()
        gateway = LLMGateway({"fake": provider}, {Task.CHAT_TEXT: Route("fake", "m")})

        class Channel:
            name = "telegram"

            async def send(self, message: Any) -> None: ...

            def listen(self) -> Any:  # pragma: no cover
                raise NotImplementedError

            async def close(self) -> None: ...

        loop = ConversationLoop(
            Channel(),
            gateway,
            writer,
            data_dir=tmp_path,
            recall=recall,
            resolve_id=lambda _text: writer.last_inserted_id,
            context_turns=1,  # so recall, not the recent window, has to do the work
        )

        def inbound(text: str, external_id: str) -> InboundMessage:
            return InboundMessage(
                text=text,
                sender_id=str(OWNER),
                received_at=datetime.now(UTC),
                channel="telegram",
                external_id=external_id,
            )

        await loop.handle(inbound("치과 예약은 8월 5일 오후 3시야", "1"))
        await loop.handle(inbound("아무 얘기나 해봐", "2"))
        await loop.handle(inbound("치과 예약 언제였지?", "3"))

        recalled = [
            m.content
            for m in provider.prompts[-1]
            if "recalled-memory:" in m.content
        ]
        assert recalled, "recall put nothing in the prompt"
        assert "8월 5일 오후 3시" in recalled[0], (
            "the earlier answer was not recalled - the M1b gate, failing"
        )
    finally:
        store.close()


# --- what the product says about itself --------------------------------------


def test_doctor_reports_a_fresh_install_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Doctor is the one command that has to survive a configuration it cannot
    load, because explaining the breakage is its whole job."""
    from daemon.cli import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DAEMON_PRESET", "nonsense-preset")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert main(["doctor"]) != 0, "doctor passed a configuration that cannot load"


def test_uninstall_does_not_claim_to_have_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It printed "ai.daemon.default installed at ..." after a successful
    bootout. The action was right and the sentence was wrong, which is worse than
    a crash: nothing looks broken."""
    from daemon.service import RunResult, Service

    calls: list[tuple[str, ...]] = []

    def runner(command: Any) -> RunResult:
        calls.append(tuple(command))
        return RunResult(returncode=0)

    service = Service(
        working_dir=tmp_path,
        program=("/opt/venv/bin/daemon", "run"),
        log_dir=tmp_path / "logs",
        home=tmp_path,
        platform="darwin",
        runner=runner,
    )
    service.install()
    action = service.uninstall()
    assert action.applied

    printed: list[str] = []
    monkeypatch_print = printed.append
    import builtins

    real_print = builtins.print
    builtins.print = lambda *a, **k: monkeypatch_print(" ".join(str(x) for x in a))
    try:
        from daemon.cli import _print_action

        _print_action(action, verb="removed")
    finally:
        builtins.print = real_print

    joined = "\n".join(printed)
    assert "installed" not in joined, f"uninstall printed: {joined!r}"
    assert "removed" in joined


def _sqlite_version_is_recent() -> bool:
    return sqlite3.sqlite_version_info >= (3, 37)


def test_the_schema_features_the_storage_layer_relies_on_are_present() -> None:
    """STRICT, FTS5 and json1 are load-bearing in schema.sql. A build without them
    fails at first write rather than at import, which is the worst moment."""
    assert _sqlite_version_is_recent()
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
    conn.execute("CREATE TABLE s(a TEXT NOT NULL) STRICT")
    conn.execute("CREATE TABLE j(a TEXT, CHECK (json_valid(a)))")


# --- the M3 gate, as a person would check it ---------------------------------


async def test_it_speaks_first_and_the_utterance_can_be_labelled(tmp_path: Path) -> None:
    """The M3 gate end to end: something the user said becomes a reason, the reason
    survives the gate, one model call turns it into a sentence, the sentence reaches
    the channel with a label button, and pressing that button lands a verdict.

    Fakes stop at the network edge - the store, the generators, the gate, the tick
    and the delivery are all real, because every defect this milestone found lived
    between them.
    """
    from daemon.channels.base import OutboundMessage
    from daemon.config import Settings
    from daemon.memory.base import LoggedMessage
    from daemon.proactivity.base import Reading
    from daemon.proactivity.delivery import ProactiveDelivery
    from daemon.proactivity.judge import Judge
    from daemon.proactivity.tick import ProactiveTick

    store = Store.open(tmp_path / "daemon.sqlite3")
    try:
        (tmp_path / "persona").mkdir(exist_ok=True)
        (tmp_path / "persona" / "seed.md").write_text("짧고 담백하게 말한다.\n", encoding="utf-8")
        writer = FileMemoryWriter(tmp_path, store)

        # Yesterday, the user mentioned something with a time attached.
        await writer.record(
            LoggedMessage(
                ts=datetime.now(UTC) - timedelta(hours=40),
                role="user",
                content="나 내일 오후에 팀 발표 있어. 좀 걱정된다.",
                origin="owner",
                session_kind="interactive",
                modality="text",
                channel="telegram",
                sender_id=str(OWNER),
            )
        )

        outbound: list[OutboundMessage] = []

        class Channel:
            name = "telegram"

            async def send(self, message: OutboundMessage) -> None:
                outbound.append(message)

            def listen(self) -> Any:  # pragma: no cover - driven directly
                raise NotImplementedError

            async def close(self) -> None: ...

        class Present:
            async def read(self) -> Reading:
                return Reading(
                    at=datetime.now(UTC),
                    idle_seconds=5.0,
                    foreground_app="Terminal",
                    audio_busy=False,
                )

        provider = Provider(reply='{"say": "발표 어떻게 됐어?"}')
        settings = Settings(
            _env_file=None,
            preset="offline",
            data_dir=tmp_path,
            proactive_enabled=True,
            proactive_quiet_hours="",
        )
        tick = ProactiveTick(
            store,
            settings,
            Present(),
            judge=Judge(
                LLMGateway({"fake": provider}, {Task.PROACTIVE_JUDGE: Route("fake", "m")}),
                data_dir=tmp_path,
            ),
            delivery=ProactiveDelivery(store, writer, channel=Channel()),
        )

        result = await tick.run()

        assert result.generated, "nothing became a candidate from a real conversation"
        assert result.spoke == 1, f"it stayed silent: {result.blocked_by}"

        said = outbound[0]
        assert said.text == "발표 어떻게 됐어?"
        assert said.labelable and said.utterance_id, "the label clock cannot start"
        # Unsolicited: no request to answer, so the channel picks its own owner.
        assert said.recipient_id is None

        # The tap a person makes.
        assert store.label_utterance(said.utterance_id, "good", now=datetime.now(UTC))
        assert store.label_counts()["good"] == 1

        # And the daemon's own voice is not evidence for the next round.
        assert store.last_conversation_at() is not None
        latest = store.recent(1)[0]
        assert latest["session_kind"] == "proactive"
    finally:
        store.close()


async def test_it_stays_silent_when_the_gate_says_so(tmp_path: Path) -> None:
    """The other half of the same gate, and the one that matters more: quiet hours
    must cost no model call at all (docs/CONTRACTS.md 7)."""
    from daemon.config import Settings
    from daemon.memory.base import LoggedMessage
    from daemon.proactivity.base import Reading
    from daemon.proactivity.delivery import ProactiveDelivery
    from daemon.proactivity.judge import Judge
    from daemon.proactivity.tick import ProactiveTick

    store = Store.open(tmp_path / "daemon.sqlite3")
    try:
        (tmp_path / "persona").mkdir(exist_ok=True)
        (tmp_path / "persona" / "seed.md").write_text("짧게.\n", encoding="utf-8")
        writer = FileMemoryWriter(tmp_path, store)
        await writer.record(
            LoggedMessage(
                ts=datetime.now(UTC) - timedelta(hours=40),
                role="user",
                content="요즘 진짜 힘들다.",
                origin="owner",
                session_kind="interactive",
                modality="text",
                channel="telegram",
                sender_id=str(OWNER),
            )
        )

        class Away:
            async def read(self) -> Reading:
                return Reading(at=datetime.now(UTC), idle_seconds=9000.0, audio_busy=False)

        provider = Provider()
        tick = ProactiveTick(
            store,
            Settings(
                _env_file=None,
                preset="offline",
                data_dir=tmp_path,
                proactive_enabled=True,
                proactive_quiet_hours="00:00-23:59",
            ),
            Away(),
            judge=Judge(
                LLMGateway({"fake": provider}, {Task.PROACTIVE_JUDGE: Route("fake", "m")}),
                data_dir=tmp_path,
            ),
            delivery=ProactiveDelivery(store, writer),
        )

        result = await tick.run()

        assert result.spoke == 0
        assert result.blocked_by, "it was allowed during quiet hours"
        assert provider.calls == [], "a blocked candidate still cost a model call"
        assert store.utterances_since(since=datetime.now(UTC) - timedelta(hours=1)) == []
    finally:
        store.close()


async def test_a_label_press_lands_through_the_channel_the_app_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole label path, through `app._build_channel` rather than a channel this
    test constructed.

    Written because a mutation exposed a hole: deleting `labels=store` from
    `_build_channel` broke nothing in 1,120 tests. Every piece was covered - the
    button is attached, the press is authorised, the verdict is stored - and the
    assembled app dropped the press with an error, so the label clock would have
    read as "the owner never labels anything". That is this project's signature
    defect (`docs/CONTRACTS.md`, Testing), and a constructor argument is exactly the
    kind of wiring `tests/test_reachable.py` cannot see.
    """
    from daemon import app as daemon_app
    from daemon.config import Settings

    store = Store.open(tmp_path / "daemon.sqlite3")
    try:
        store.insert_utterance(
            utterance_id="u-1",
            candidate_id=None,
            kind="open_loop",
            text="발표 어떻게 됐어?",
            route="telegram",
            gate_snapshot="{}",
            now=datetime.now(UTC),
        )
        api = FakeTelegram(
            [
                [
                    {
                        "update_id": 1,
                        "callback_query": {
                            "id": "q-1",
                            "from": {"id": OWNER},
                            "data": "label:up:u-1",
                        },
                    }
                ]
            ]
        )
        # The real class captured first: a lambda that calls `httpx.AsyncClient`
        # after patching that same name recurses into itself.
        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            httpx,
            "AsyncClient",
            lambda *a, **k: real_client(transport=httpx.MockTransport(api.handler)),
        )
        settings = Settings(
            _env_file=None,
            preset="offline",
            data_dir=tmp_path,
            telegram_bot_token=TOKEN,
            telegram_allowed_user_ids=(str(OWNER),),
            telegram_dm_policy="allowlist",
        )
        channel = daemon_app._build_channel(settings, store)

        async def drain() -> None:
            async for _ in channel.listen():
                pass

        task = asyncio.create_task(drain())
        for _ in range(200):
            await asyncio.sleep(0.01)
            if store.label_counts().get("good"):
                break
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await channel.close()

        assert store.label_counts()["good"] == 1, (
            "the press never reached storage - `labels=` is probably not wired"
        )
        answered = [c for c in api.sent if "callback_query_id" in c]
        assert answered, "Telegram was left with a spinner on the button"
    finally:
        store.close()
