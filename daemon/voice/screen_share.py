"""The live screen-share frame pump: capture, dedup, keepalive, send.

Drives a `capture` callable at a fixed cadence (`screen_fps`), downscales and
compares each frame to the last one *sent* using a perceptual hash so an
unchanging screen does not resend the same pixels every tick, and forwards a
frame to a `VoiceSession` only when it changed enough or the keepalive window
elapsed. This is the one place in the screen-sharing stack that imports
Pillow: `daemon/tools/screen.py` is the dependency-free capture core (shell
out to `screencapture`/`sips`), and the perceptual hash needs real image
decoding and resampling that shelling out cannot give cheaply - so that
dependency lives here, in the voice extra, rather than leaking into core.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from PIL import Image

logger = logging.getLogger(__name__)


class _FrameSink(Protocol):
    async def send_frame(self, jpeg: bytes) -> None: ...


def dhash(jpeg: bytes) -> int:
    """Pillow difference hash: grayscale, resize to 9x8, compare adjacent
    pixels along each row. Returns a 64-bit int (8 rows x 8 comparisons)."""
    image = Image.open(io.BytesIO(jpeg)).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(image.getdata())  # row-major, 9 wide x 8 tall
    bits = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            bits = (bits << 1) | (1 if pixels[offset + col] > pixels[offset + col + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _should_send(
    new_hash: int,
    last_hash: int | None,
    *,
    now: float,
    last_sent_at: float,
    threshold: int,
    keepalive_secs: float,
) -> bool:
    """Pure send decision: first frame, big enough change, or keepalive elapsed."""
    if last_hash is None:
        return True
    if hamming(new_hash, last_hash) > threshold:
        return True
    return now - last_sent_at >= keepalive_secs


class ScreenSharePump:
    """Captures at `fps`, sends only frames that changed enough or are due for
    a keepalive. `capture` and `session` are injected so this can be driven
    without a real screen or a real socket."""

    def __init__(
        self,
        *,
        session: _FrameSink,
        capture: Callable[[], Awaitable[bytes]],
        fps: float = 1.0,
        dedup_threshold: int = 6,
        keepalive_secs: float = 8.0,
    ) -> None:
        self._session = session
        self._capture = capture
        self._fps = fps
        self._dedup_threshold = dedup_threshold
        self._keepalive_secs = keepalive_secs
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Idempotent: does nothing if already running."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Idempotent: does nothing if not running."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        last_hash: int | None = None
        last_sent_at = time.monotonic()
        while True:
            try:
                jpeg = await self._capture()
            except Exception:
                # A transient TCC/capture failure must not kill the share.
                logger.warning("screen-share capture failed, skipping this frame", exc_info=True)
            else:
                new_hash = dhash(jpeg)
                now = time.monotonic()
                if _should_send(
                    new_hash,
                    last_hash,
                    now=now,
                    last_sent_at=last_sent_at,
                    threshold=self._dedup_threshold,
                    keepalive_secs=self._keepalive_secs,
                ):
                    try:
                        await self._session.send_frame(jpeg)
                    except Exception:
                        # A dead sink (dropped socket, closed session) must not
                        # kill the share silently - the same principle as the
                        # capture-failure branch above. Logged and skipped rather
                        # than raised, so a transient send failure costs one frame
                        # instead of the whole share (2.2 review, deferred Minor).
                        logger.warning(
                            "screen-share send_frame failed, skipping this frame",
                            exc_info=True,
                        )
                    else:
                        last_hash = new_hash
                        last_sent_at = now
            await asyncio.sleep(1 / self._fps)


class ScreenShareController:
    """Owns the live screen-share pump for one voice conversation.

    The pump needs a live `VoiceSession`, so the conversation binds it once the
    session is open; the `start_screen_share`/`stop_screen_share` tools toggle it.
    Off until the owner asks - and never silent (start logs and returns a spoken
    acknowledgement; a live share must always be announced, per the plan's
    explicit-signal rule).
    """

    def __init__(self) -> None:
        self._pump: Any = None
        self._active = False

    def bind(self, pump: Any) -> None:
        """The conversation hands over the pump for its now-live session."""
        self._pump = pump

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> str:
        """Turn the share on. `pump.start()` is sync; this returns the spoken
        acknowledgement the model relays - the explicit signal is carried in the
        tool result, not left to the model to invent."""
        if self._pump is None:
            return (
                "I can only share your screen during a voice conversation. Ask me "
                "to look at your screen for a one-off view instead."
            )
        if self._active:
            return "I'm already watching your screen."
        self._pump.start()
        self._active = True
        logger.info("screen share on")
        return "I'm watching your screen now — tell me to stop sharing when you're done."

    async def stop(self) -> str:
        if self._pump is None or not self._active:
            return "I'm not sharing your screen right now."
        await self._pump.stop()
        self._active = False
        logger.info("screen share off")
        return "I've stopped watching your screen."

    async def stop_and_unbind(self) -> None:
        """Conversation teardown. Never raises: a share must never outlive its
        session, but a failure to stop cleanly must not take the shutdown with it.
        """
        try:
            if self._pump is not None and self._active:
                await self._pump.stop()
        except Exception:
            logger.exception("stopping the screen-share pump failed")
        finally:
            self._pump = None
            self._active = False
