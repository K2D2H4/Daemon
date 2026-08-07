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
