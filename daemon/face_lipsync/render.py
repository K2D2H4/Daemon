"""Turn one audio window plus one driving frame into one JPEG."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from daemon.face_lipsync import Cache, LipsyncEngine, composite
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


class Renderer:
    """One frame at a time. The loop that calls this lives in `daemon/app.py`.

    N=1 is the shipped shape - `_render` always calls `mouths(audio, [i])` - even
    though the spec calls N=2 the only viable batch. Widening to N=2 is not a
    protocol change here: it would need `mouths` to take two windows rather than
    one audio array (two frames are two whisper indices apart, so a single array
    cannot express both), and it would mean revisiting `Slot`'s latest-wins
    semantics too.
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
        n = len(self._cache.boxes)
        i = frame_index % n
        audio = self._ring.window(frame_index=frame_index, fps=fps, origin=origin)
        mouth = self._engine.mouths(audio, [i])[0]
        x1, y1, x2, y2 = self._cache.boxes[i]
        sized = cv2.resize(mouth, (x2 - x1, y2 - y1))
        sized = restore_detail(sized, self._cache.frames[i], self._cache.boxes[i])
        out = composite(
            self._cache.frames[i],
            sized,
            self._cache.boxes[i],
            self._cache.crop_boxes[i],
            self._cache.masks[i],
            out=self._buffer,
        )
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            self._slot.put(buf.tobytes())
