"""The lip-sync render path: the model boundary, the clip cache, and compositing.

Nothing here imports anything else in `daemon/` (CONTRACTS 4). `daemon/app.py`
builds a renderer and injects it; no other module knows this package exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class LipsyncEngine(Protocol):
    """The only place a model lives.

    Everything else - which driving frame, which audio, how the mouth is blended -
    sits outside this, so the renderer is testable without weights.
    """

    def mouths(
        self, audio: np.ndarray, frame_indices: Sequence[int]
    ) -> list[np.ndarray]:
        """256x256 BGR mouths, one per index, in the same order."""
        ...


@dataclass(frozen=True, slots=True)
class Cache:
    """One driving clip, prepared offline. `frames` is memory-mapped in production."""

    frames: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    crop_boxes: list[tuple[int, int, int, int]]
    masks: np.ndarray


def composite(
    frame: np.ndarray,
    mouth: np.ndarray,
    box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    mask: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Blend one mouth into one driving frame, touching only the crop box.

    MuseTalk's own `get_image_blending` round-trips the whole frame through PIL
    twice for the same pixels; this was measured bit-identical at a fraction of the
    cost, which matters because the per-frame budget is 41.67ms.
    """
    x1, y1, x2, y2 = box
    xs, ys, xe, ye = crop_box
    if out is None:
        out = frame.copy()
    elif out is not frame:
        np.copyto(out, frame)
    orig = frame[ys:ye, xs:xe].astype(np.float32)
    pasted = frame[ys:ye, xs:xe].copy()
    pasted[y1 - ys : y2 - ys, x1 - xs : x2 - xs] = mouth
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    out[ys:ye, xs:xe] = (
        pasted.astype(np.float32) * alpha + orig * (1.0 - alpha)
    ).round().astype(np.uint8)
    return out
