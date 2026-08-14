import asyncio

from conftest import FakeProvider
from test_loop import FakeMemory, gateway_for

from daemon.channels.base import OutboundMessage
from daemon.companion import Companion
from daemon.delegation import (
    EMPTY_REPLY_MESSAGE,
    FAILURE_PREFIX,
    CaptureChannel,
    DelegationWorker,
    build_run_request,
    deliver_result,
)
from daemon.llm.base import ToolCall, ToolSpec
from daemon.memory.store import Store
from daemon.tools.base import Registry
from daemon.tools.policy import ToolPolicy
from daemon.tools.runner import ToolRunner


class _FakeReading:
    def __init__(self, at_keyboard):
        self.at_keyboard = at_keyboard


class _FakePresence:
    def __init__(self, at_keyboard):
        self._r = _FakeReading(at_keyboard)

    async def read(self):
        return self._r


class _FakeSpeaker:
    def __init__(self, ok=True):
        self.said = []
        self._ok = ok

    async def say(self, text):
        self.said.append(text)
        return self._ok


class _FakeChannel:
    name = "telegram"

    def __init__(self, ok=True):
        self.sent = []
        self._ok = ok

    async def send(self, message: OutboundMessage):
        if not self._ok:
            raise RuntimeError("channel down")
        self.sent.append(message)


async def test_at_keyboard_speaks_and_sends():
    speaker, channel = _FakeSpeaker(), _FakeChannel()
    route = await deliver_result("만들었어요", presence=_FakePresence(True),
                                 speaker=speaker, channel=channel, recipient_id="42")
    assert route == "both"
    assert speaker.said == ["만들었어요"]
    assert channel.sent[0].text == "만들었어요"
    assert channel.sent[0].recipient_id == "42"


async def test_away_sends_to_channel_only():
    speaker, channel = _FakeSpeaker(), _FakeChannel()
    route = await deliver_result("만들었어요", presence=_FakePresence(False),
                                 speaker=speaker, channel=channel, recipient_id="42")
    assert route == "telegram"
    assert speaker.said == []
    assert len(channel.sent) == 1


async def test_channel_failure_degrades_route_not_raises():
    speaker, channel = _FakeSpeaker(), _FakeChannel(ok=False)
    route = await deliver_result("x", presence=_FakePresence(True),
                                 speaker=speaker, channel=channel, recipient_id="42")
    assert route == "local_speaker"  # spoke, send failed


async def test_capture_channel_records_the_reply():
    ch = CaptureChannel()
    await ch.send(OutboundMessage(text="만들었어요"))
    assert ch.reply == "만들었어요"


async def test_worker_runs_a_queued_task_marks_done_and_delivers(db):
    store = Store(db)
    tid = store.enqueue_task(request="노션에 페이지 만들어줘", origin="owner",
                             channel="voice", sender_id=None)
    delivered = []

    async def fake_run(request):
        assert request == "노션에 페이지 만들어줘"
        return "만들었어요"

    async def fake_deliver(text, task_row):
        delivered.append((text, task_row["id"]))

    worker = DelegationWorker(store, fake_run, fake_deliver, wake=asyncio.Event())
    ran = await worker.drain_once()
    assert ran is True
    assert delivered == [("만들었어요", tid)]
    row = db.execute("SELECT status, result FROM delegated_tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "done" and row["result"] == "만들었어요"


async def test_worker_marks_failed_and_delivers_the_failure(db):
    store = Store(db)
    tid = store.enqueue_task(request="r", origin="owner", channel="voice", sender_id=None)
    delivered = []

    async def boom(request):
        raise RuntimeError("notion 400")

    async def fake_deliver(text, task_row):
        delivered.append(text)

    worker = DelegationWorker(store, boom, fake_deliver, wake=asyncio.Event())
    ran = await worker.drain_once()
    assert ran is True
    assert delivered and "notion 400" in delivered[0]
    row = db.execute("SELECT status, error FROM delegated_tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "failed" and "notion 400" in row["error"]


async def test_drain_once_returns_false_when_the_queue_is_empty(db):
    store = Store(db)
    worker = DelegationWorker(store, None, None, wake=asyncio.Event())
    assert await worker.drain_once() is False


class _RaisingDoneStore:
    """Wraps a real Store but blows up on mark_task_done - a mid-write crash."""

    def __init__(self, store):
        self._store = store

    def claim_next_queued(self, *args, **kwargs):
        return self._store.claim_next_queued(*args, **kwargs)

    def mark_task_done(self, *args, **kwargs):
        raise RuntimeError("disk full")

    def mark_task_failed(self, *args, **kwargs):
        return self._store.mark_task_failed(*args, **kwargs)


async def test_mark_done_failure_does_not_flip_a_real_success_into_a_failure(db):
    real_store = Store(db)
    tid = real_store.enqueue_task(request="노션에 페이지 만들어줘", origin="owner",
                                   channel="voice", sender_id=None)
    store = _RaisingDoneStore(real_store)
    delivered = []

    async def fake_run(request):
        return "만들었어요"

    async def fake_deliver(text, task_row):
        delivered.append(text)

    worker = DelegationWorker(store, fake_run, fake_deliver, wake=asyncio.Event())
    ran = await worker.drain_once()
    assert ran is True
    # The work really happened - the owner must hear the real reply, not a
    # manufactured failure, even though recording it as done blew up.
    assert delivered == ["만들었어요"]
    assert FAILURE_PREFIX not in delivered[0]
    row = db.execute("SELECT status FROM delegated_tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "running"  # the mark-done write never landed


async def test_empty_reply_is_delivered_as_an_honest_message_not_silence(db):
    store = Store(db)
    tid = store.enqueue_task(request="r", origin="owner", channel="voice", sender_id=None)
    delivered = []

    async def fake_run(request):
        return ""

    async def fake_deliver(text, task_row):
        delivered.append(text)

    worker = DelegationWorker(store, fake_run, fake_deliver, wake=asyncio.Event())
    ran = await worker.drain_once()
    assert ran is True
    assert delivered == [EMPTY_REPLY_MESSAGE]
    assert delivered[0] != ""
    row = db.execute("SELECT status, result FROM delegated_tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "done" and row["result"] == EMPTY_REPLY_MESSAGE


async def test_build_run_request_runs_the_text_loop_and_returns_the_reply(monkeypatch):
    captured = {}

    class _FakeLoop:
        def __init__(self, channel, gateway, companion, **kw):
            captured["channel"] = channel
            self._channel = channel

        async def handle(self, inbound):
            captured["inbound"] = inbound
            await self._channel.send(type("M", (), {"text": "만들었어요"})())

    monkeypatch.setattr("daemon.delegation.ConversationLoop", _FakeLoop)
    run = build_run_request(gateway=object(), companion_factory=lambda: object())
    reply = await run("노션에 페이지 만들어줘")
    assert reply == "만들었어요"
    assert captured["inbound"].text == "노션에 페이지 만들어줘"
    assert captured["inbound"].authored_by_sender is True
    assert captured["channel"].name == "delegate"


# --- end-to-end: the assembled backend chain -------------------------------------
#
# The wired path the unit tests above fake seam-by-seam, run for real once: enqueue ->
# worker claims -> build_run_request -> CaptureChannel + the real ConversationLoop ->
# Companion offers the owner turn its full tool set -> the model asks for a nested
# write tool (the kind voice cannot call) -> ToolRunner runs it -> the reply is
# captured -> the task is marked done -> it is delivered. Fakes stop at the network
# edge only: the LLM decision (FakeProvider) and the write tool's leaf effect. Revert
# the worker's mark/deliver or the build_run_request wiring and this fails - not a
# tautology.


class _RecordingCreateTool:
    """Stands in for a real nested-schema write tool (like notion-create-pages): the
    text path can call it where the voice model could not. Records that it ran."""

    risk = "guarded"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.spec = ToolSpec(
            name="create_page",
            description="Create a page under a parent (nested schema).",
            parameters={
                "type": "object",
                "properties": {"parent": {"type": "object"}, "pages": {"type": "array"}},
            },
        )

    def preview(self, arguments):
        return "create_page(...)"

    async def run(self, arguments):
        self.calls.append(dict(arguments))
        return "created 1 page"


async def test_end_to_end_a_delegated_task_runs_the_real_text_loop_and_reports(db, data_dir):
    store = Store(db)
    create_tool = _RecordingCreateTool()
    registry = Registry()
    registry.register(create_tool)
    tools = ToolRunner(registry, ToolPolicy(store, mode="full"), store)

    reply = "노션에 '내일 준비물' 페이지 만들어서 저장했어."
    provider = FakeProvider(
        reply=reply,
        scripted_calls=[
            [
                ToolCall(
                    id="1",
                    name="create_page",
                    arguments={"parent": {"page_id": "x"}, "pages": [{"title": "내일 준비물"}]},
                )
            ]
        ],
    )

    def companion_factory():
        return Companion(FakeMemory(), data_dir=data_dir, tools=tools)

    run_request = build_run_request(
        gateway=gateway_for(provider), companion_factory=companion_factory
    )
    delivered: list[tuple[str, int]] = []

    async def deliver(text, row):
        delivered.append((text, row["id"]))

    worker = DelegationWorker(store, run_request, deliver, wake=asyncio.Event())

    tid = store.enqueue_task(
        request="노션에 '내일 준비물' 하위 페이지 만들어줘",
        origin="owner",
        channel="voice",
        sender_id=None,
    )
    ran = await worker.drain_once()

    assert ran is True
    # the assembled text loop actually ran the nested write tool the voice model cannot call
    assert len(create_tool.calls) == 1
    # the delegated turn was offered the full tool set (origin=owner, surface="text")
    assert any(
        spec.name == "create_page" for offered in provider.offered_tools for spec in offered
    )
    # the row went queued -> done with the real reply, and it was delivered
    row = store.conn.execute(
        "SELECT status, result FROM delegated_tasks WHERE id=?", (tid,)
    ).fetchone()
    assert row["status"] == "done"
    assert row["result"] == reply
    assert delivered == [(reply, tid)]
