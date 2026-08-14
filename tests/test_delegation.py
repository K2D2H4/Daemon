import asyncio

from daemon.channels.base import OutboundMessage
from daemon.delegation import (
    CaptureChannel,
    DelegationWorker,
    build_run_request,
    deliver_result,
)
from daemon.memory.store import Store


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
