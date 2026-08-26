"""The renderer, with no model in sight.

Everything that decides how the face looks - which driving frame, which audio, how
the mouth is blended in - lives outside the engine protocol, so a fake that returns
a flat colour exercises all of it. That is the whole reason the protocol exists:
CI must never touch a gigabyte of weights.
"""

import cv2
import numpy as np

from daemon.face_lipsync import Cache, composite
from daemon.face_lipsync.render import Renderer
from daemon.face_lipsync.ring import PcmRing, Slot

# Deliberately non-square, with a different margin on every side - not the
# symmetric 60x60 box / 80x80 crop this fixture used before fix round 1. A
# symmetric fixture cannot tell a correct box/crop_box axis from a transposed
# one, nor a correct cv2.resize(width, height) from a swapped one: both mutations
# passed every test here. The box must sit inside the crop box - MuseTalk expands
# the face box by 1.5x to get the blend region, and the mask is the size of that
# larger box.
BOX = (40, 20, 100, 140)         # x1, y1, x2, y2 - 60 wide x 120 tall
CROP_BOX = (30, 15, 120, 155)    # margins: left 10, right 20, top 5, bottom 15
CROP_H = CROP_BOX[3] - CROP_BOX[1]
CROP_W = CROP_BOX[2] - CROP_BOX[0]


class FakeEngine:
    """Returns a solid colour per requested frame, and records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def mouths(self, audio, frame_indices):
        self.calls.append(list(frame_indices))
        return [np.full((256, 256, 3), 200, np.uint8) for _ in frame_indices]


def _cache(n=4):
    frames = np.zeros((n, 200, 160, 3), np.uint8)
    masks = np.full((n, CROP_H, CROP_W), 255, np.uint8)
    return Cache(
        frames=frames,
        boxes=[BOX] * n,
        crop_boxes=[CROP_BOX] * n,
        masks=masks,
    )


def _cache_with_distinct_frames(n=4):
    """Like `_cache`, but each index has its own background and the frame buffer
    is read-only, simulating the memory-mapped clip in production. `_cache`'s
    indices are byte-identical, which would make an index mix-up or a stale
    shared-buffer leak invisible.
    """
    frames = np.zeros((n, 200, 160, 3), np.uint8)
    for i in range(n):
        frames[i] = i * 50
    frames.flags.writeable = False
    masks = np.full((n, CROP_H, CROP_W), 255, np.uint8)
    return Cache(
        frames=frames,
        boxes=[BOX] * n,
        crop_boxes=[CROP_BOX] * n,
        masks=masks,
    )


def test_composite_only_touches_the_crop_box():
    cache = _cache()
    frame = cache.frames[0].copy()
    mouth = np.full((BOX[3] - BOX[1], BOX[2] - BOX[0], 3), 200, np.uint8)
    out = composite(frame, mouth, cache.boxes[0], cache.crop_boxes[0], cache.masks[0])
    assert out[0, 0].tolist() == [0, 0, 0]           # outside the crop box
    assert out[80, 70].tolist() == [200, 200, 200]   # inside the box, full alpha


def test_composite_blends_a_partial_mask_value_as_a_weighted_mix():
    """`composite` exists instead of a plain paste because of this arithmetic -
    alpha = mask / 255, so a mid-range mask value must weight-average the mouth
    against the original pixel, not fully paste it (alpha=1) or fully skip it
    (alpha=0)."""
    frame = np.full((10, 10, 3), 10, np.uint8)
    mouth = np.full((4, 4, 3), 220, np.uint8)
    mask = np.full((10, 10), 255, np.uint8)
    mask[3, 3] = 128
    out = composite(frame, mouth, (2, 2, 6, 6), (0, 0, 10, 10), mask)
    # alpha = 128/255 -> round(220*alpha + 10*(1-alpha)) == 115
    assert out[3, 3].tolist() == [115, 115, 115]
    assert out[2, 2].tolist() == [220, 220, 220]  # full-alpha neighbour, for contrast


def test_the_published_frame_contains_the_engines_mouth_pixels():
    """Not just that a frame was published - that the engine's own pixels are the
    ones in it, and not some unrelated content composited in its place."""
    slot = Slot()
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    Renderer(engine=engine, cache=_cache(), ring=ring, slot=slot).render(
        frame_index=0, origin=0.0, fps=24.0
    )
    published = slot.get()
    assert published is not None
    frame = cv2.imdecode(np.frombuffer(published, np.uint8), cv2.IMREAD_COLOR)
    # Centre of the box: the engine's mouth colour (200), not the background (0).
    # JPEG is lossy, so allow a margin nowhere near the 200-value gap.
    assert abs(int(frame[80, 70, 0]) - 200) < 20


def test_a_failing_engine_does_not_take_the_renderer_down():
    """The design says a mid-stream failure falls back to v1 clips and logs once -
    which it can only do if the frame that failed does not propagate."""

    class Broken:
        def mouths(self, audio, frame_indices):
            raise RuntimeError("weights went away")

    slot = Slot()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    r = Renderer(engine=Broken(), cache=_cache(), ring=ring, slot=slot)
    r.render(frame_index=0, origin=0.0, fps=24.0)   # must not raise
    assert slot.get() is None
    assert r.failed is True


def test_a_latched_failure_stops_calling_the_engine_rather_than_retrying():
    """Retrying per frame would fill the log at 24Hz - once latched, `render()`
    must return immediately without touching the engine again."""

    class CountingBroken:
        def __init__(self) -> None:
            self.calls = 0

        def mouths(self, audio, frame_indices):
            self.calls += 1
            raise RuntimeError("weights went away")

    slot = Slot()
    engine = CountingBroken()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    r = Renderer(engine=engine, cache=_cache(), ring=ring, slot=slot)
    r.render(frame_index=0, origin=0.0, fps=24.0)
    assert engine.calls == 1
    r.render(frame_index=1, origin=0.0, fps=24.0)
    assert engine.calls == 1


def test_the_driving_clip_cycles_rather_than_running_out():
    slot = Slot()
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    r = Renderer(engine=engine, cache=_cache(n=4), ring=ring, slot=slot)
    r.render(frame_index=9, origin=0.0, fps=24.0)
    assert engine.calls[-1] == [1]   # 9 % 4 == 1 - a clamp would give 3 instead


def test_rendering_never_writes_into_a_read_only_cache_or_leaks_a_stale_frame():
    """`Cache.frames` is memory-mapped in production, i.e. read-only. Rendering
    two different indices back-to-back must not raise from a write into that
    array, and the shared internal buffer must not let one index's background
    bleed into the next index's published frame."""
    slot = Slot()
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    cache = _cache_with_distinct_frames()
    r = Renderer(engine=engine, cache=cache, ring=ring, slot=slot)

    r.render(frame_index=0, origin=0.0, fps=24.0)
    assert r.failed is False
    published = slot.get()
    assert published is not None
    first = cv2.imdecode(np.frombuffer(published, np.uint8), cv2.IMREAD_COLOR)
    assert abs(int(first[0, 0, 0]) - 0) < 10       # index 0's own background

    r.render(frame_index=2, origin=0.0, fps=24.0)
    assert r.failed is False
    published = slot.get()
    assert published is not None
    second = cv2.imdecode(np.frombuffer(published, np.uint8), cv2.IMREAD_COLOR)
    assert abs(int(second[0, 0, 0]) - 100) < 10    # index 2's own, not index 0's
