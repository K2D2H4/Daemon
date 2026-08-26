"""The renderer, with no model in sight.

Everything that decides how the face looks - which driving frame, which audio, how
the mouth is blended in - lives outside the engine protocol, so a fake that returns
a flat colour exercises all of it. That is the whole reason the protocol exists:
CI must never touch a gigabyte of weights.
"""

import numpy as np

from daemon.face_lipsync import Cache, composite
from daemon.face_lipsync.render import Renderer
from daemon.face_lipsync.ring import PcmRing, Slot


class FakeEngine:
    """Returns a solid colour per requested frame, and records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def mouths(self, audio, frame_indices):
        self.calls.append(list(frame_indices))
        return [np.full((256, 256, 3), 200, np.uint8) for _ in frame_indices]


def _cache(n=4):
    # The box must sit inside the crop box - MuseTalk expands the face box by 1.5x
    # to get the blend region, and the mask is the size of that larger box.
    frames = np.zeros((n, 200, 160, 3), np.uint8)
    masks = np.full((n, 80, 80), 255, np.uint8)
    return Cache(
        frames=frames,
        boxes=[(40, 40, 100, 100)] * n,        # 60x60
        crop_boxes=[(30, 30, 110, 110)] * n,   # 80x80
        masks=masks,
    )


def test_composite_only_touches_the_crop_box():
    cache = _cache()
    frame = cache.frames[0].copy()
    mouth = np.full((60, 60, 3), 200, np.uint8)
    out = composite(frame, mouth, cache.boxes[0], cache.crop_boxes[0], cache.masks[0])
    assert out[0, 0].tolist() == [0, 0, 0]          # outside the crop box
    assert out[60, 60].tolist() == [200, 200, 200]  # inside it


def test_rendering_publishes_one_frame_to_the_slot():
    slot = Slot()
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    Renderer(engine=engine, cache=_cache(), ring=ring, slot=slot).render(
        frame_index=0, origin=0.0, fps=24.0
    )
    assert slot.get() is not None


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


def test_the_driving_clip_cycles_rather_than_running_out():
    slot = Slot()
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    r = Renderer(engine=engine, cache=_cache(n=4), ring=ring, slot=slot)
    r.render(frame_index=9, origin=0.0, fps=24.0)
    assert engine.calls[-1][0] < 4
