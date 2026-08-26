"""Turn one audio window plus one driving frame into one JPEG."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from daemon.face_lipsync import Cache, LipsyncEngine, composite
from daemon.face_lipsync.audio import CONTEXT_MS, latest_audio_ms
from daemon.face_lipsync.ring import PcmRing, Slot

logger = logging.getLogger(__name__)

JPEG_QUALITY = 85

DETAIL_SIGMA = 1.1
"""Gaussian sigma separating "texture" from "structure" in the driving frame.

Small on purpose. The mouth this borrows from is the driving clip's own, usually
closed; at a larger sigma the transfer carries that closed lip *edge* across and
it ghosts through an open mouth. 1.1 was checked on the six most-open generated
frames of `idle2` - single set of teeth, single lip contour, no doubling - and a
global average cannot see that failure, so it has to be checked there.
"""


def restore_detail(
    mouth: np.ndarray,
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    sigma: float = DETAIL_SIGMA,
) -> np.ndarray:
    """Add the driving frame's fine texture back onto a generated mouth.

    The generated mouth is soft: measured on a landmark-derived lip box it keeps
    71% of the driving clip's detail at product size. That softness is what the
    owner saw and called trembling - a blurred patch whose texture shifts frame to
    frame reads as vibration, which is why nine temporal fixes all missed.

    This is not a face enhancer and deliberately hallucinates nothing: the avatar
    is a fixed clip, so the original texture at this exact box is already aligned,
    and only its high-frequency residual is added. Measured 71% -> 97% of the
    driving clip's lip detail for 0.56ms on a 320x394 crop.

    Do not reach for temporal smoothing instead. It lowers frame-to-frame RMSE in
    the lip region, but that number ran *opposite* to the owner's own ranking of
    three renders - it buys the metric by destroying the detail.

    `mouth` must already be `box`-sized, as `composite` requires.
    """
    x1, y1, x2, y2 = box
    orig = frame[y1:y2, x1:x2].astype(np.float32)
    high = orig - cv2.GaussianBlur(orig, (0, 0), sigma)
    return np.clip(mouth.astype(np.float32) + high, 0, 255).astype(np.uint8)


BATCH = 2
"""Frames computed per model step, and this is arithmetic rather than taste.

Measured on the assembled engine: at N=1 a frame costs 49.3ms against a 41.67ms
budget - UNet 41.55, TAESD 4.72, convert 0.93, features 2.12 - so 24fps is not
reachable one frame at a time on any of it. At N=2 the UNet drops to 29.29ms/frame
and the whole two-frame step is 72.21ms, i.e. 36.10ms/frame with 13% headroom.

N=3 is not the next step up: the batch cannot start until its last frame's audio has
arrived, so widening costs `(N-1) x 41.67ms` of latency and N=3 misses the 250ms
ceiling by 7.6ms.
"""


class FrameClock:
    """Which frame to render on this tick, or `None` because its audio is not here yet.

    The render loop in `daemon/app.py` ticks at `fps` and asks this each time. Three
    things it exists to get right, none of which are visible from the loop:

    **The batch-fill wait.** A step covers `BATCH` frames, so it cannot start until
    the LAST of them has its audio - `(BATCH - 1) x 41.67ms` past what the first frame
    alone needs. `PcmRing.window` zero-fills what has not arrived instead of erroring,
    so a loop that just renders "the frame for now" gets a mouth conditioned on
    silence and nothing anywhere complains.

    **Re-anchoring.** `PcmRing.origin` moves - a new turn, a long silence, a barge-in
    that rebuilds the clock, and continuously once the ring is full and dropping
    samples. Frame indices are relative to it, so the count restarts whenever it
    jumps. Small forward creep from sample dropping is not a new turn, hence the
    tolerance.

    **One frame per tick, never a skip.** Returning "the newest ready frame" would
    jump the mouth forward after any hitch. Falling behind is corrected by the
    renderer's own held-frame ticks, which cost nothing.

    `now` and `origin` must come from the SAME clock, and in production that clock is
    `loop.time()` - `daemon/voice/conversation.py` stamps audio with it deliberately
    (see its `_playback_until` note). Passing `daemon.clock.now()` here type-checks,
    runs, and puts the mouth an arbitrary offset away from the sound.
    """

    __slots__ = ("_fps", "_frame", "_origin")

    RE_ANCHOR_TOLERANCE = 0.2
    """Seconds of forward movement in `origin` treated as the ring dropping old
    samples rather than as a new turn. A full ring creeps every feed; a turn boundary
    jumps."""

    def __init__(self, *, fps: float) -> None:
        self._fps = fps
        self._origin: float | None = None
        self._frame = 0

    def due(self, *, now: float, origin: float) -> int | None:
        if self._origin is None or abs(origin - self._origin) > self.RE_ANCHOR_TOLERANCE:
            self._origin = origin
            self._frame = 0
        needed = latest_audio_ms(self._frame + BATCH - 1, self._fps) / 1000.0
        if now - origin < needed:
            return None
        frame = self._frame
        self._frame += 1
        return frame


class Renderer:
    """Two frames per model step, released one frame apart.

    The loop that calls this lives in `daemon/app.py` and still calls `render` once
    per displayed frame. Alternate calls do no model work at all: a step computes
    `frame_index` and `frame_index + 1`, publishes the first, and holds the second
    for the next call.

    That holding is not a convenience. `Slot` is latest-wins and never queues, so
    putting both frames of a batch in it at once would overwrite the first before
    anything read it - the mouth would advance two frames at a time and read as
    12fps however fast the model ran.

    The cost is the batch-fill wait the spec names: a step needs audio through
    `frame_index + 1`'s window, which is 41.67ms past what `frame_index` alone
    needs. `PcmRing.window` zero-fills what has not arrived, so a loop that calls
    this too eagerly gets a mouth conditioned on silence rather than an error.
    """

    def __init__(
        self,
        *,
        engine: LipsyncEngine,
        cache: Cache,
        ring: PcmRing,
        slot: Slot,
    ) -> None:
        self._engine = engine
        self._cache = cache
        self._ring = ring
        self._slot = slot
        self._buffer = np.empty_like(cache.frames[0])
        self._pending: bytes | None = None
        """The batch's second frame, waiting for the next call. See the class docstring."""
        self.failed = False
        """Latched on the first engine failure. The caller drops back to v1 clips
        and logs once; retrying per frame would fill the log at 24Hz."""

    def render(self, *, frame_index: int, origin: float, fps: float) -> None:
        """Render `frame_index` and publish it. Never raises."""
        if self.failed:
            return
        try:
            self._render(frame_index=frame_index, origin=origin, fps=fps)
        except Exception:
            logger.exception("face: lip-sync engine failed, falling back to clips")
            self.failed = True

    def _render(self, *, frame_index: int, origin: float, fps: float) -> None:
        if self._pending is not None:
            self._slot.put(self._pending)
            self._pending = None
            return
        n = len(self._cache.boxes)
        indices = [frame_index + step for step in range(BATCH)]
        windows = [
            self._ring.window(
                frame_index=index, fps=fps, origin=origin, context_ms=CONTEXT_MS
            )
            for index in indices
        ]
        mouths = self._engine.mouths(windows, [index % n for index in indices])
        encoded: list[bytes] = []
        for index, mouth in zip(indices, mouths, strict=True):
            i = index % n
            box = self._cache.boxes[i]
            x1, y1, x2, y2 = box
            sized = cv2.resize(mouth, (x2 - x1, y2 - y1))
            sized = restore_detail(sized, self._cache.frames[i], box)
            out = composite(
                self._cache.frames[i],
                sized,
                box,
                self._cache.crop_boxes[i],
                self._cache.masks[i],
                out=self._buffer,
            )
            # Encode inside the loop: `out` is one reusable buffer, so the second
            # composite overwrites the first frame's pixels.
            ok, buf = cv2.imencode(
                ".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok:
                encoded.append(buf.tobytes())
        if encoded:
            self._slot.put(encoded[0])
            self._pending = encoded[1] if len(encoded) > 1 else None
