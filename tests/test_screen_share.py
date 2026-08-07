"""dhash/hamming (pure image logic) and ScreenSharePump's send-decision + loop.

No macOS, no socket: frames are synthesised with Pillow in-test and `capture` /
`session` are fakes injected into the pump, per daemon/voice/screen_share.py's
module docstring on why Pillow lives there rather than in tools/screen.py.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from PIL import Image, ImageDraw

from daemon.voice.screen_share import ScreenSharePump, _should_send, dhash, hamming


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


# --- dhash / hamming ----------------------------------------------------------


def test_identical_frames_have_zero_distance():
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (30, 30, 30)).save(buf, "JPEG")
    b = buf.getvalue()
    assert hamming(dhash(b), dhash(b)) == 0


def test_small_local_change_stays_under_threshold():
    # A 1-pixel tweak on a large frame washes out at the 9x8 resize.
    base = Image.new("RGB", (400, 300), (30, 30, 30))
    a = _jpeg(base)
    tweaked = base.copy()
    tweaked.putpixel((0, 0), (255, 255, 255))
    b = _jpeg(tweaked)
    assert hamming(dhash(a), dhash(b)) < 6


def test_page_change_exceeds_threshold():
    dark = Image.new("RGB", (400, 300), (10, 10, 10))
    a = _jpeg(dark)

    split = Image.new("RGB", (400, 300), (10, 10, 10))
    draw = ImageDraw.Draw(split)
    draw.rectangle([(200, 0), (400, 300)], fill=(255, 255, 255))
    b = _jpeg(split)

    assert hamming(dhash(a), dhash(b)) > 6


# --- _should_send (pure) -------------------------------------------------------


def test_should_send_first_frame_always_sends():
    assert _should_send(
        123, None, now=100.0, last_sent_at=90.0, threshold=6, keepalive_secs=8.0
    )


def test_should_send_big_change_sends():
    # 0 vs 0xFF...F: every bit differs, far above any reasonable threshold.
    assert _should_send(
        0,
        (1 << 64) - 1,
        now=100.0,
        last_sent_at=99.0,
        threshold=6,
        keepalive_secs=8.0,
    )


def test_should_send_small_change_before_keepalive_holds_back():
    # hamming(0b1, 0b0) == 1, under threshold=6; keepalive (8s) not elapsed.
    assert not _should_send(
        0b1, 0b0, now=100.0, last_sent_at=95.0, threshold=6, keepalive_secs=8.0
    )


def test_should_send_small_change_after_keepalive_sends():
    assert _should_send(
        0b1, 0b0, now=104.0, last_sent_at=95.0, threshold=6, keepalive_secs=8.0
    )


# --- ScreenSharePump loop -------------------------------------------------------


class _FakeSession:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    async def send_frame(self, jpeg: bytes) -> None:
        self.sent.append(jpeg)


def _make_frames() -> tuple[bytes, bytes, bytes]:
    dark = Image.new("RGB", (400, 300), (10, 10, 10))
    split = Image.new("RGB", (400, 300), (10, 10, 10))
    draw = ImageDraw.Draw(split)
    draw.rectangle([(200, 0), (400, 300)], fill=(255, 255, 255))
    return _jpeg(dark), _jpeg(dark), _jpeg(split)


async def test_pump_dedups_identical_frames_and_sends_changed_one(monkeypatch):
    frame_a, frame_a_again, frame_b = _make_frames()
    script = [frame_a, frame_a_again, frame_b]

    async def fake_sleep(_secs: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    fake_time = {"t": 0.0}

    def fake_monotonic() -> float:
        fake_time["t"] += 0.1
        return fake_time["t"]

    monkeypatch.setattr("daemon.voice.screen_share.time.monotonic", fake_monotonic)

    calls = {"n": 0}

    async def capture() -> bytes:
        i = calls["n"]
        calls["n"] += 1
        if i >= len(script):
            raise asyncio.CancelledError
        return script[i]

    session = _FakeSession()
    pump = ScreenSharePump(
        session=session, capture=capture, fps=1.0, dedup_threshold=6, keepalive_secs=8.0
    )
    pump.start()
    task = pump._task
    assert task is not None
    with pytest.raises(asyncio.CancelledError):
        await task

    # first frame always sent; identical 2nd frame deduped; changed 3rd frame sent.
    assert session.sent == [frame_a, frame_b]


async def test_stop_cancels_the_running_task():
    async def capture() -> bytes:
        await asyncio.sleep(3600)
        raise AssertionError("should have been cancelled first")

    session = _FakeSession()
    pump = ScreenSharePump(session=session, capture=capture, fps=1.0)
    pump.start()
    await pump.stop()
    assert pump._task is None or pump._task.done()


async def test_start_is_idempotent():
    async def capture() -> bytes:
        await asyncio.sleep(3600)
        raise AssertionError("should have been cancelled first")

    session = _FakeSession()
    pump = ScreenSharePump(session=session, capture=capture, fps=1.0)
    pump.start()
    task = pump._task
    pump.start()
    assert pump._task is task
    await pump.stop()


async def test_send_frame_failure_is_logged_and_the_loop_continues(monkeypatch, caplog):
    """2.2's deferred minor: a broken sink must not kill the share silently."""
    import logging

    async def fake_sleep(_secs: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    frame_a, _, frame_b = _make_frames()
    script = [frame_a, frame_b]

    calls = {"n": 0}

    async def capture() -> bytes:
        i = calls["n"]
        calls["n"] += 1
        if i >= len(script):
            raise asyncio.CancelledError
        return script[i]

    class _BrokenSession:
        def __init__(self) -> None:
            self.attempts = 0

        async def send_frame(self, jpeg: bytes) -> None:
            self.attempts += 1
            raise RuntimeError("socket is gone")

    session = _BrokenSession()
    pump = ScreenSharePump(session=session, capture=capture, fps=1.0)
    with caplog.at_level(logging.WARNING):
        pump.start()
        task = pump._task
        assert task is not None
        with pytest.raises(asyncio.CancelledError):
            await task

    # Both frames were changed enough to send, both attempts failed, and the loop
    # kept running instead of dying on the first one.
    assert session.attempts == 2
    assert any("send" in record.message.lower() for record in caplog.records)


# --- ScreenShareController -----------------------------------------------------


class _FakePump:
    """Records start/stop rather than driving a real capture loop."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True


def test_controller_start_with_no_pump_bound_apologises_and_does_not_crash():
    from daemon.voice.screen_share import ScreenShareController

    controller = ScreenShareController()
    message = controller.start()
    assert "voice conversation" in message
    assert controller.active is False


async def test_controller_stop_with_no_pump_bound_says_not_sharing():
    from daemon.voice.screen_share import ScreenShareController

    controller = ScreenShareController()
    message = await controller.stop()
    assert "not sharing" in message.lower()
    assert controller.active is False


def test_controller_bind_then_start_flips_the_pump_and_acknowledges():
    from daemon.voice.screen_share import ScreenShareController

    controller = ScreenShareController()
    pump = _FakePump()
    controller.bind(pump)
    message = controller.start()
    assert pump.started is True
    assert controller.active is True
    assert "watching your screen" in message.lower()


def test_controller_start_twice_says_already_watching():
    from daemon.voice.screen_share import ScreenShareController

    controller = ScreenShareController()
    pump = _FakePump()
    controller.bind(pump)
    controller.start()
    message = controller.start()
    assert "already" in message.lower()


async def test_controller_stop_stops_the_pump_and_deactivates():
    from daemon.voice.screen_share import ScreenShareController

    controller = ScreenShareController()
    pump = _FakePump()
    controller.bind(pump)
    controller.start()
    message = await controller.stop()
    assert pump.stopped is True
    assert controller.active is False
    assert "stopped" in message.lower()


async def test_stop_and_unbind_stops_an_active_pump_and_clears_it():
    from daemon.voice.screen_share import ScreenShareController

    controller = ScreenShareController()
    pump = _FakePump()
    controller.bind(pump)
    controller.start()
    await controller.stop_and_unbind()
    assert pump.stopped is True
    assert controller.active is False
    # A second start with nothing bound must not crash and must not report success.
    assert "voice conversation" in controller.start()


async def test_stop_and_unbind_never_raises_even_if_the_pump_stop_fails():
    from daemon.voice.screen_share import ScreenShareController

    class _BrokenPump:
        def start(self) -> None:
            pass

        async def stop(self) -> None:
            raise RuntimeError("boom")

    controller = ScreenShareController()
    controller.bind(_BrokenPump())
    controller.start()
    await controller.stop_and_unbind()  # must not raise
    assert controller.active is False


async def test_stop_and_unbind_with_nothing_bound_is_a_no_op():
    from daemon.voice.screen_share import ScreenShareController

    controller = ScreenShareController()
    await controller.stop_and_unbind()  # must not raise
    assert controller.active is False
