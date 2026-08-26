"""Turn one audio window plus one driving frame into one JPEG."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from daemon.face_lipsync import Cache, LipsyncEngine, composite
from daemon.face_lipsync.ring import PcmRing, Slot

logger = logging.getLogger(__name__)

JPEG_QUALITY = 85


class Renderer:
    """One frame at a time. The loop that calls this lives in `daemon/app.py`."""

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
