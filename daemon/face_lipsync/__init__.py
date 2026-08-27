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
        self, windows: Sequence[np.ndarray], frame_indices: Sequence[int]
    ) -> list[np.ndarray]:
        """256x256 BGR mouths, one per index, in the same order.

        **One window per frame, and this took two windows rather than one array on
        purpose.** Consecutive video frames sit two whisper indices apart, so a
        single array cannot express both frames' conditioning; and deriving the
        second frame's offset inside the engine would put this module's index
        arithmetic in two places. Handing over both windows costs one extra
        encoder pass - measured 4.27ms for the pair against 2.12ms for one, which
        is inside the frame budget either way.

        Each window is `PcmRing.window()`'s own output and nothing else: float32,
        `-1..1`, at `PcmRing.sample_rate` (24kHz in production -
        `daemon/voice/audio.py`'s OUTPUT_SAMPLE_RATE) - not the 16kHz whisper's
        own mel front-end assumes. An implementation that feeds this straight to
        whisper without resampling first gets a 1.5x-stretched, late mouth.

        A window is longer than the 200ms the model conditions on, and **the
        model's window is the tail**. The lead-in is `audio.CONTEXT_MS` of
        preceding audio, which whisper needs and an earlier version of this
        docstring got wrong by promising exactly 200ms: log-mel clamps at
        `log_spec.max() - 8` and rescales, so a bare 200ms slice normalises
        against its own peak and came out anti-correlated (cosine -0.29) with the
        same audio inside a stream. An implementation must therefore locate its ten
        whisper indices at the END of each window, not the start.
        """
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
    strength: float = 1.0,
) -> np.ndarray:
    """Blend one mouth into one driving frame, touching only the crop box.

    MuseTalk's own `get_image_blending` round-trips the whole frame through PIL
    twice for the same pixels; this was measured bit-identical at a fraction of the
    cost, which matters because the per-frame budget is 41.67ms.

    `strength` scales the whole mask, so 0.0 returns the driving frame untouched and
    1.0 is the paste as MuseTalk does it. It exists for the end of an utterance:
    speech stops and the generated mouth is replaced by the artist's own in one
    frame, which the owner saw as the mouth "갑자기 확 닫히는" snap. That step is
    structural rather than a timing fault - `evals/face_lipsync_idle_spike.py`
    measured all 88 conditioning windows, digital zero included, and **not one
    renders this avatar's resting mouth closed**; every one leaves the lips parted
    with a sliver of teeth, where the driving clip's own mouth is cleanly sealed. So
    the last generated frame is always at least slightly open and the frame after it
    is shut.

    Ramping this to 0 dissolves one into the other, and the dissolve *is* the
    closing motion, because what shows through underneath is a closed mouth. No
    model step pays for it.
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
    alpha = (mask.astype(np.float32) / 255.0)[..., None] * strength
    out[ys:ye, xs:xe] = (
        pasted.astype(np.float32) * alpha + orig * (1.0 - alpha)
    ).round().astype(np.uint8)
    return out
