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
    """Returns a solid colour per requested frame, and records what it was asked.

    `audio_calls` keeps a copy of every window handed to `mouths` - added because
    the renderer<->ring seam had zero coverage without it: every test in this file
    fed the ring pure silence and this class discarded `audio` outright, so
    `self._ring.window(...)` could be passed the wrong frame_index, a hardcoded
    origin, or replaced with a hardcoded array of zeros and all 27 tests here would
    still pass (see the three seam tests below, which is what these fields exist
    to make possible).
    """

    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.audio_calls: list[np.ndarray] = []

    def mouths(self, audio, frame_indices):
        self.calls.append(list(frame_indices))
        self.audio_calls.append(audio.copy())
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


def _tone(ms: int, value: int, rate: int = 24_000) -> bytes:
    """Distinctive constant-value waveform - `tests/test_face_lipsync_ring.py`'s
    helper, duplicated rather than imported because this file otherwise has no
    dependency on that one."""
    samples = np.full(int(rate * ms / 1000), value, dtype=np.int16)
    return samples.tobytes()


def _distinct_tone_ring(seconds=2.0):
    """A ring fed a new tone value every 100ms, so two different windows read
    different content - unlike `b"\\x00\\x00" * 24_000` (every other fixture in
    this file), where any array of the right shape is indistinguishable from the
    real one."""
    ring = PcmRing(sample_rate=24_000, width=2, seconds=seconds)
    for step in range(20):
        ring.feed(_tone(100, value=1000 * (step + 1)), audible_at=step * 0.1)
    return ring


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


# --- the renderer<->ring audio seam -------------------------------------------
#
# Every test above feeds the ring silence and none of them look at `audio_calls`,
# so none of them can tell a correct `self._ring.window(frame_index=frame_index,
# fps=fps, origin=origin)` call in `render.py` from one that passes the cycled
# clip index instead of `frame_index`, hardcodes `origin=0.0`, or skips the ring
# entirely and hands the engine a hardcoded silent array. The three tests below
# each target exactly one of those.


def test_the_engine_receives_real_audio_not_silence():
    """Catches `self._ring.window(...)` being replaced by a hardcoded
    `np.zeros(4800)` - indistinguishable from correct in every other test here,
    since they all feed the ring silence too."""
    engine = FakeEngine()
    ring = _distinct_tone_ring()
    r = Renderer(engine=engine, cache=_cache(n=6), ring=ring, slot=Slot())
    r.render(frame_index=8, origin=0.3, fps=24.0)
    assert r.failed is False
    assert np.any(engine.audio_calls[0] != 0.0), "the engine should see real audio"


def test_the_engine_receives_a_window_addressed_by_frame_index_not_the_cycled_clip_index():
    """Catches `window(frame_index=i, ...)` where `i = frame_index % n` - the
    realistic mutation, since `i` is already in scope on the surrounding lines and
    "tidying" them to share one variable looks like cleanup.

    Frame indices 8 and 14 are a cache cycle apart (`n=6`, so both give clip index
    2) but ask for different real audio; a correct call tells them apart, `i` does
    not, so a fixed `i` makes the two calls receive an identical window instead.
    """
    engine = FakeEngine()
    ring = _distinct_tone_ring()
    cache = _cache(n=6)
    r = Renderer(engine=engine, cache=cache, ring=ring, slot=Slot())
    r.render(frame_index=8, origin=0.0, fps=24.0)
    r.render(frame_index=14, origin=0.0, fps=24.0)   # 14 % 6 == 8 % 6 == 2
    assert r.failed is False
    assert not np.array_equal(engine.audio_calls[0], engine.audio_calls[1]), (
        "frame_index=8 and frame_index=14 both cycle to clip index 2, but ask for "
        "different audio - using the clip index in the window call would make "
        "these two identical"
    )


def test_the_engine_receives_a_window_addressed_by_the_real_origin():
    """Catches `window(..., origin=0.0)` hardcoded regardless of the caller's own
    origin - the same shape of bug as the frame_index case above, one argument
    over."""
    engine = FakeEngine()
    ring = _distinct_tone_ring()
    r = Renderer(engine=engine, cache=_cache(n=6), ring=ring, slot=Slot())
    r.render(frame_index=8, origin=0.0, fps=24.0)
    r.render(frame_index=8, origin=0.5, fps=24.0)
    assert r.failed is False
    assert not np.array_equal(engine.audio_calls[0], engine.audio_calls[1]), (
        "the same frame_index at two different origins must read different audio"
    )
