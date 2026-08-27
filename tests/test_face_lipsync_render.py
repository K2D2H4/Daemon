"""The renderer, with no model in sight.

Everything that decides how the face looks - which driving frame, which audio, how
the mouth is blended in - lives outside the engine protocol, so a fake that returns
a flat colour exercises all of it. That is the whole reason the protocol exists:
CI must never touch a gigabyte of weights.
"""

import cv2
import numpy as np

from daemon.face_lipsync import Cache, composite
from daemon.face_lipsync.audio import latest_audio_ms
from daemon.face_lipsync.render import (
    BATCH,
    DISPLAY_LEAD,
    MOTION_BLEND,
    FrameClock,
    Renderer,
    restore_detail,
)
from daemon.face_lipsync.ring import PcmRing

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

    `window_calls` keeps a copy of every window handed to `mouths` - added because
    the renderer<->ring seam had zero coverage without it: every test in this file
    fed the ring pure silence and this class discarded the audio outright, so
    `self._ring.window(...)` could be passed the wrong frame_index, a hardcoded
    origin, or replaced with a hardcoded array of zeros and every test here would
    still pass (see the seam tests below, which is what these fields exist to make
    possible).

    One entry per `mouths` call, holding that call's whole batch - the renderer
    hands over `BATCH` windows at a time, and a batch whose windows are all the
    same frame's audio is its own failure mode.
    """

    def __init__(self) -> None:
        self.calls: list[list[int]] = []
        self.window_calls: list[list[np.ndarray]] = []

    def mouths(self, windows, frame_indices):
        self.calls.append(list(frame_indices))
        self.window_calls.append([w.copy() for w in windows])
        return [np.full((256, 256, 3), 200, np.uint8) for _ in frame_indices]

    @property
    def first_windows(self):
        """The first window of each batch - what the old `audio_calls` meant."""
        return [batch[0] for batch in self.window_calls]


def _cache(n=4):
    frames = np.zeros((n, 200, 160, 3), np.uint8)
    masks = np.full((n, CROP_H, CROP_W), 255, np.uint8)
    return Cache(
        frames=frames,
        boxes=[BOX] * n,
        crop_boxes=[CROP_BOX] * n,
        masks=masks,
    )


def _jpegs(renderer, *, frame_index, origin=0.0, fps=24.0):
    """The pair's JPEGs, through both halves the way the render loop drives them.

    `Renderer` publishes nothing now: `step` is the model half and `encode` the CPU
    one, and they run on different threads so that a pair's compositing overlaps the
    next pair's model step (`render.py:Renderer`, and `daemon/app.py:_lipsync_loop`
    for the pacing that used to live in here). Every test below that used to assert
    on `Slot.get()` asserts on these bytes instead.
    """
    step = renderer.step(frame_index=frame_index, origin=origin, fps=fps)
    return [] if step is None else renderer.encode(step)


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


def _cache_with_distinct_frames(n=4, level=50):
    """Like `_cache`, but each index has its own background and the frame buffer
    is read-only, simulating the memory-mapped clip in production. `_cache`'s
    indices are byte-identical, which would make an index mix-up or a stale
    shared-buffer leak invisible.
    """
    frames = np.zeros((n, 200, 160, 3), np.uint8)
    for i in range(n):
        frames[i] = i * level
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


def test_the_encoded_frame_contains_the_engines_mouth_pixels():
    """Not just that a frame came back - that the engine's own pixels are the ones
    in it, and not some unrelated content composited in its place."""
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    encoded = _jpegs(
        Renderer(engine=engine, cache=_cache(), ring=ring), frame_index=0
    )
    assert encoded
    frame = cv2.imdecode(np.frombuffer(encoded[0], np.uint8), cv2.IMREAD_COLOR)
    # Centre of the box: the engine's mouth colour (200), not the background (0).
    # JPEG is lossy, so allow a margin nowhere near the 200-value gap.
    assert abs(int(frame[80, 70, 0]) - 200) < 20


def test_the_payload_is_the_whole_composited_frame():
    """Two earlier versions of this test pinned a crop instead, and both were wrong.

    The crop is genuinely 3.4x cheaper to send, but this page is served over loopback,
    where that saving buys nothing - and compositing a JPEG over the page's own
    `<video>` cannot be seamless, because a JPEG has no alpha and the video is not on
    the same frame. The spike's comparison is seamless because the SERVER composited
    into the frame. So the payload is the frame.
    """
    cache = _cache()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    encoded = _jpegs(
        Renderer(engine=FakeEngine(), cache=cache, ring=ring), frame_index=0
    )
    got = cv2.imdecode(np.frombuffer(encoded[0], np.uint8), cv2.IMREAD_COLOR)
    assert got.shape[:2] == cache.frames[0].shape[:2]


def test_a_failing_engine_does_not_take_the_renderer_down():
    """The design says a mid-stream failure falls back to v1 clips and logs once -
    which it can only do if the frame that failed does not propagate."""

    class Broken:
        def mouths(self, audio, frame_indices):
            raise RuntimeError("weights went away")

    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    r = Renderer(engine=Broken(), cache=_cache(), ring=ring)
    assert r.step(frame_index=0, origin=0.0, fps=24.0) is None   # must not raise
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

    engine = CountingBroken()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    r = Renderer(engine=engine, cache=_cache(), ring=ring)
    r.step(frame_index=0, origin=0.0, fps=24.0)
    assert engine.calls == 1
    r.step(frame_index=1, origin=0.0, fps=24.0)
    assert engine.calls == 1


def test_the_driving_clip_cycles_rather_than_running_out():
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    r = Renderer(engine=engine, cache=_cache(n=4), ring=ring)
    r.step(frame_index=9, origin=0.0, fps=24.0)
    # Audio frames 9 and 10 land on driving frames 15 and 16 (DISPLAY_LEAD=6), which
    # on a 4-frame clip is 3 then 0. A clamp would give [3, 3] and a missing modulo
    # would give [15, 16].
    assert engine.calls[-1] == [3, 0]


def test_encoding_never_writes_into_a_read_only_cache_or_leaks_a_stale_frame():
    """`Cache.frames` is memory-mapped in production, i.e. read-only. Encoding a pair
    must not raise from a write into that array, and the one reusable buffer both
    frames are composited through must not let the first index's background bleed
    into the second's bytes.

    Both halves of that are now inside a single `encode` call - the pair is encoded
    together rather than one now and one on the next tick - which is why the JPEG is
    taken inside the loop over the pair instead of after it.
    """
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    cache = _cache_with_distinct_frames(n=16, level=15)
    r = Renderer(engine=engine, cache=cache, ring=ring)

    pair = _jpegs(r, frame_index=0)
    assert r.failed is False
    assert len(pair) == BATCH
    first = cv2.imdecode(np.frombuffer(pair[0], np.uint8), cv2.IMREAD_COLOR)
    second = cv2.imdecode(np.frombuffer(pair[1], np.uint8), cv2.IMREAD_COLOR)
    # Audio frames 0 and 1 are drawn on driving frames DISPLAY_LEAD and +1.
    assert abs(int(first[0, 0, 0]) - DISPLAY_LEAD * 15) < 10
    assert abs(int(second[0, 0, 0]) - (DISPLAY_LEAD + 1) * 15) < 10

    later = _jpegs(r, frame_index=2)
    assert r.failed is False
    got = cv2.imdecode(np.frombuffer(later[0], np.uint8), cv2.IMREAD_COLOR)
    assert abs(int(got[0, 0, 0]) - (DISPLAY_LEAD + 2) * 15) < 10


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
    r = Renderer(engine=engine, cache=_cache(n=6), ring=ring)
    r.step(frame_index=8, origin=0.3, fps=24.0)
    assert r.failed is False
    assert np.any(engine.first_windows[0] != 0.0), "the engine should see real audio"


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
    for index in (8, 14):                            # 14 % 6 == 8 % 6 == 2
        r = Renderer(engine=engine, cache=cache, ring=ring)
        r.step(frame_index=index, origin=0.0, fps=24.0)
        assert r.failed is False
    assert not np.array_equal(engine.first_windows[0], engine.first_windows[1]), (
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
    cache = _cache(n=6)
    for origin in (0.0, 0.5):
        r = Renderer(engine=engine, cache=cache, ring=ring)
        r.step(frame_index=8, origin=origin, fps=24.0)
        assert r.failed is False
    assert not np.array_equal(engine.first_windows[0], engine.first_windows[1]), (
        "the same frame_index at two different origins must read different audio"
    )


# --- detail restoration ---------------------------------------------------------
#
# Every test above uses an all-black driving frame, which has no high-frequency
# residual at all - so restore_detail is a no-op there and those ten tests pass
# whether it is wired into _render or deleted from it. These use a textured frame,
# which is the only way to see it.


def _textured(h, w):
    """Fine checkerboard: energy at exactly the scale DETAIL_SIGMA separates."""
    y, x = np.mgrid[0:h, 0:w]
    return np.repeat((((x + y) % 2) * 90 + 70).astype(np.uint8)[..., None], 3, axis=2)


def _lap(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def test_restore_detail_is_a_noop_on_a_flat_frame():
    """No texture to borrow means nothing is borrowed - not a scaled or shifted
    version of the mouth, exactly the mouth."""
    frame = np.full((200, 160, 3), 90, np.uint8)
    mouth = np.full((120, 60, 3), 200, np.uint8)
    assert np.array_equal(restore_detail(mouth, frame, BOX), mouth)


def test_restore_detail_adds_the_frames_texture():
    frame = np.zeros((200, 160, 3), np.uint8)
    frame[BOX[1] : BOX[3], BOX[0] : BOX[2]] = _textured(120, 60)
    mouth = np.full((120, 60, 3), 128, np.uint8)
    out = restore_detail(mouth, frame, BOX)
    assert _lap(out) > _lap(mouth) * 10


def test_restore_detail_does_not_wrap_around_uint8():
    """The whole point of going via float32 and clipping. Naive uint8 addition
    turns an over-bright pixel black and an under-dark one white, which would show
    up as salt-and-pepper speckle on the mouth rather than as a soft error."""
    frame = np.zeros((200, 160, 3), np.uint8)
    frame[BOX[1] : BOX[3], BOX[0] : BOX[2]] = _textured(120, 60)
    for level in (0, 255):
        out = restore_detail(np.full((120, 60, 3), level, np.uint8), frame, BOX)
        assert out.min() >= 0 and out.max() <= 255
        if level == 255:
            assert out.max() == 255          # clipped, not wrapped to near-zero
        else:
            assert out.min() == 0


def test_restore_detail_preserves_shape_and_dtype():
    frame = np.zeros((200, 160, 3), np.uint8)
    frame[BOX[1] : BOX[3], BOX[0] : BOX[2]] = _textured(120, 60)
    mouth = np.full((120, 60, 3), 128, np.uint8)
    out = restore_detail(mouth, frame, BOX)
    assert out.shape == mouth.shape and out.dtype == np.uint8


def _frame_with_dot(at, value=255):
    """A single bright pixel at `at` (row, col) *within the box*, on flat mid-grey.

    A lone dot is the only fixture that pins position and sign at once, which the
    earlier "did the Laplacian go up" assertions could not: reading a shifted
    region still raises high-frequency energy, and so does adding the residual
    with its sign flipped.
    """
    frame = np.full((200, 160, 3), 128, np.uint8)
    frame[BOX[1] + at[0], BOX[0] + at[1]] = value
    return frame


def test_restore_detail_puts_the_texture_where_it_came_from():
    """Reads `box`, aligned. A region shifted by even a few pixels moves the dot."""
    at = (30, 15)
    out = restore_detail(np.full((120, 60, 3), 128, np.uint8), _frame_with_dot(at), BOX)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    assert np.unravel_index(int(gray.argmax()), gray.shape) == at


def test_restore_detail_adds_the_residual_and_does_not_subtract_it():
    """Sign. A bright spot in the driving frame must make that spot *brighter*;
    inverting the residual darkens it instead, while leaving every "detail went up"
    measure looking fine."""
    at = (30, 15)
    flat = np.full((120, 60, 3), 128, np.uint8)
    out = restore_detail(flat, _frame_with_dot(at), BOX)
    assert int(out[at][0]) > 128
    dark = restore_detail(flat, _frame_with_dot(at, value=0), BOX)
    assert int(dark[at][0]) < 128


def test_larger_sigma_transfers_more():
    """Sigma is the ghosting knob: it has to move detail monotonically, or the
    value picked by inspecting the most-open frames means nothing."""
    frame = np.zeros((200, 160, 3), np.uint8)
    frame[BOX[1] : BOX[3], BOX[0] : BOX[2]] = _textured(120, 60)
    mouth = np.full((120, 60, 3), 128, np.uint8)
    small = np.abs(
        restore_detail(mouth, frame, BOX, sigma=0.6).astype(int) - 128
    ).mean()
    large = np.abs(
        restore_detail(mouth, frame, BOX, sigma=3.0).astype(int) - 128
    ).mean()
    assert large > small


def test_encode_actually_restores_detail():
    """The wiring test. With a textured driving frame the encoded JPEG must match
    the restored composite, not the plain one."""
    cache = _cache()
    # The driving frame audio frame 0 actually lands on - DISPLAY_LEAD ahead of it.
    shown = DISPLAY_LEAD % len(cache.boxes)
    cache.frames[shown, BOX[1] : BOX[3], BOX[0] : BOX[2]] = _textured(120, 60)
    ring = _distinct_tone_ring()
    r = Renderer(engine=FakeEngine(), cache=cache, ring=ring)
    encoded = _jpegs(r, frame_index=0)

    got = cv2.imdecode(np.frombuffer(encoded[0], np.uint8), cv2.IMREAD_COLOR)
    flat = np.full((120, 60, 3), 200, np.uint8)
    plain = composite(cache.frames[shown], flat, BOX, CROP_BOX, cache.masks[shown])
    restored = composite(
        cache.frames[shown],
        restore_detail(flat, cache.frames[shown], BOX),
        BOX,
        CROP_BOX,
        cache.masks[shown],
    )
    d_plain = np.abs(got.astype(int) - plain.astype(int)).mean()
    d_restored = np.abs(got.astype(int) - restored.astype(int)).mean()
    assert d_restored < d_plain / 2, (d_restored, d_plain)


# --- batching -------------------------------------------------------------------
#
# BATCH=2 exists because N=1 costs 49.3ms against a 41.67ms budget. Its own failure
# mode is a batch whose windows are the same frame's audio twice - the shapes match
# and nothing raises. What is NOT here any more is the held second frame: pacing the
# pair out one interval apart is `daemon/app.py:_lipsync_loop`'s queue now, and
# `tests/test_face_lipsync_wiring.py` is where that is asserted. It moved because a
# renderer that published its own pair could not overlap the next model step with
# them, which capped the socket at 20.1fps.


def test_a_batch_asks_for_a_different_window_per_frame():
    """Handing the same window to both frames is the cheap-looking mistake here:
    the shapes match, the step succeeds, and the second frame's mouth is simply
    41.67ms stale."""
    engine = FakeEngine()
    r = Renderer(engine=engine, cache=_cache(n=6), ring=_distinct_tone_ring())
    r.step(frame_index=8, origin=0.0, fps=24.0)
    batch = engine.window_calls[0]
    assert len(batch) == BATCH
    assert not np.array_equal(batch[0], batch[1]), (
        "consecutive frames sit two whisper indices apart and must read different "
        "audio - identical windows mean one window was reused"
    )


def test_one_step_yields_two_different_frames_in_order():
    """One model call, two frames, and they must not be the same bytes.

    The loop hands these to its publisher in this order and one interval apart, so a
    pair that came back identical would read as 12fps however fast the model ran.
    """
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    ring.feed(_tone(2000, value=9000), audible_at=0.0)
    r = Renderer(engine=engine, cache=_cache_with_distinct_frames(), ring=ring)

    pair = _jpegs(r, frame_index=0)

    assert len(engine.calls) == 1, "a pair is one model call, not two"
    assert len(pair) == BATCH
    assert pair[0] != pair[1], "the pair's two frames must not be the same bytes"


def test_consecutive_steps_ask_for_contiguous_pairs():
    """Frame indices advance by `BATCH` per step and never overlap or skip - the
    pairs are contiguous, which is what keeps the jump between them under a frame
    when the loop's catch-up drain has to move the index forward."""
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    ring.feed(_tone(2000, value=9000), audible_at=0.0)
    # n=8, not 6: with a clip exactly DISPLAY_LEAD frames long the lead would cancel
    # out modulo the length and this would pass whether it was applied or not.
    r = Renderer(engine=engine, cache=_cache(n=8), ring=ring)

    r.step(frame_index=0, origin=0.0, fps=24.0)
    assert engine.calls == [[6, 7]]
    r.step(frame_index=2, origin=0.0, fps=24.0)
    assert engine.calls == [[6, 7], [0, 1]]


def test_the_mouth_is_drawn_on_the_frame_the_page_will_be_showing():
    """The audio index and the driving index are two different questions.

    A renderer that answers both with `frame_index` composites the mouth into a pose
    242ms stale - measured at the socket - and the page's 180ms crossfade then has to
    dissolve between two poses a quarter-second apart in the avatar's own motion,
    which shows as a ghost over the whole face rather than only the mouth. So the
    reference frame AND the composite move forward by DISPLAY_LEAD while the window
    stays where the audio is. The sizes are in DISPLAY_LEAD's own docstring.
    """
    engine = FakeEngine()
    cache = _cache_with_distinct_frames(n=16, level=15)
    r = Renderer(engine=engine, cache=cache, ring=_distinct_tone_ring())

    pair = _jpegs(r, frame_index=0)

    assert engine.calls == [[DISPLAY_LEAD, DISPLAY_LEAD + 1]], (
        "the reference latent has to be the frame being drawn on, or the mouth is "
        "inpainted for a pose it is not composited into"
    )
    drawn = cv2.imdecode(np.frombuffer(pair[0], np.uint8), cv2.IMREAD_COLOR)
    assert abs(int(drawn[0, 0, 0]) - DISPLAY_LEAD * 15) < 10, (
        "the composite went onto the audio's own frame rather than the one the page "
        "will be showing when this arrives"
    )
    # And the audio is still addressed by the audio index: frame 0's window, not
    # frame 6's, or the mouth would be a quarter-second ahead of the sound.
    ring = _distinct_tone_ring()
    at_audio = Renderer(engine=(e2 := FakeEngine()), cache=cache, ring=ring)
    at_audio.step(frame_index=0, origin=0.0, fps=24.0)
    direct = ring.window(frame_index=0, fps=24.0, origin=0.0, context_ms=2000.0)
    assert np.array_equal(e2.window_calls[0][0], direct)


def test_a_latched_failure_stops_both_halves_rather_than_repeating_a_frame():
    """Neither half raises, and after a latch both must go quiet - a renderer that
    kept handing back a stale pair would freeze the mouth mid-word instead of letting
    the page fall back to clip playback."""

    class Broken:
        def mouths(self, windows, frame_indices):
            raise RuntimeError("engine gone")

    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    ring.feed(_tone(2000, value=9000), audible_at=0.0)
    r = Renderer(engine=Broken(), cache=_cache(n=6), ring=ring)
    assert r.step(frame_index=0, origin=0.0, fps=24.0) is None
    assert r.failed is True
    assert r.step(frame_index=1, origin=0.0, fps=24.0) is None


def test_a_failing_composite_latches_too_rather_than_raising_into_the_thread():
    """The CPU half's own failure path, which nothing else here covers. It runs on a
    worker thread whose future the loop never awaits, so an exception out of it would
    reach nobody and the mouth would freeze with nothing logged."""
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    cache = _cache()
    r = Renderer(engine=engine, cache=cache, ring=ring)
    step = r.step(frame_index=0, origin=0.0, fps=24.0)
    assert step is not None
    # A degenerate box, the shape a mismatched prepared cache would have: cv2.resize
    # refuses an empty destination size.
    cache.boxes[DISPLAY_LEAD % len(cache.boxes)] = (0, 0, 0, 0)
    assert r.encode(step) == []
    assert r.failed is True


# --- the clock that decides when a batch may start -------------------------------


def test_the_first_frame_waits_for_the_LAST_frame_of_its_batch():
    """The whole reason this class exists. Frame 0's own window ends at 100ms, but
    the batch also contains frame 1, whose window ends at 120ms - and PcmRing.window
    zero-fills what has not arrived instead of erroring, so rendering at 100ms
    conditions frame 1 on silence with nothing anywhere complaining."""
    clock = FrameClock(fps=24.0)
    own = latest_audio_ms(0, 24.0) / 1000.0
    batch = latest_audio_ms(BATCH - 1, 24.0) / 1000.0
    assert batch > own, "the fixture is pointless if the batch needs no more audio"
    assert clock.due(now=own, origin=0.0) is None
    assert clock.due(now=batch, origin=0.0) == 0


def test_frames_advance_one_per_tick_and_never_skip():
    """Returning "the newest ready frame" would jump the mouth forward after any
    hitch. Being behind is free - the renderer's held-frame ticks cost nothing."""
    clock = FrameClock(fps=24.0)
    assert clock.due(now=5.0, origin=0.0) == 0        # far past ready
    assert clock.due(now=9.0, origin=0.0) == 1        # still 1, not 200-odd
    assert clock.due(now=9.0, origin=0.0) == 2


def test_a_new_turn_restarts_the_count():
    clock = FrameClock(fps=24.0)
    assert clock.due(now=1.0, origin=0.0) == 0
    assert clock.due(now=1.0, origin=0.0) == 1
    assert clock.due(now=31.0, origin=30.0) == 0, "a re-anchored ring is a new turn"


def test_a_ring_dropping_old_samples_is_not_a_new_turn():
    """`origin` creeps forward every feed once the ring is full. Treating that as a
    turn boundary would reset the frame count several times a second and the mouth
    would restart the utterance continuously."""
    clock = FrameClock(fps=24.0)
    assert clock.due(now=1.0, origin=0.0) == 0
    assert clock.due(now=1.1, origin=0.04) == 1
    assert clock.due(now=1.2, origin=0.08) == 2


def test_creep_does_not_re_anchor_once_it_ADDS_UP_past_the_tolerance():
    """The case the two steps above miss, and the bug they let through.

    Comparing `origin` against the value captured at the last re-anchor makes steady
    creep accumulate against a fixed reference, so it crosses RE_ANCHOR_TOLERANCE
    however small each step is: measured 7 false resets in 40 ticks at 40ms each, five
    restarts of the utterance per second. The comparison has to be against the
    PREVIOUS TICK. 40 ticks here total 1.6s of creep, eight times the tolerance.
    """
    clock = FrameClock(fps=24.0)
    origin, now, frames = 0.0, 1.0, []
    for _ in range(40):
        frames.append(clock.due(now=now, origin=origin))
        origin += 0.04
        now += 1 / 24.0
    advanced = [f for f in frames if f is not None]
    assert advanced == sorted(advanced), f"the frame count went backwards: {frames}"
    assert advanced.count(0) <= 1, (
        f"frame 0 was handed out {advanced.count(0)} times - the utterance restarted"
    )


def test_a_real_turn_boundary_still_re_anchors_after_creep():
    """The fix must not buy creep-immunity by never re-anchoring at all."""
    clock = FrameClock(fps=24.0)
    origin, now = 0.0, 1.0
    for _ in range(10):
        clock.due(now=now, origin=origin)
        origin += 0.04
        now += 1 / 24.0
    assert clock.due(now=now + 30.0, origin=origin + 30.0) == 0


def test_a_backward_jump_also_restarts():
    """Barge-in rebuilds the speech clock from scratch, so origin can move
    backward - see PcmRing.feed's own note on the same hazard."""
    clock = FrameClock(fps=24.0)
    assert clock.due(now=31.0, origin=30.0) == 0
    assert clock.due(now=31.0, origin=30.0) == 1
    assert clock.due(now=1.0, origin=0.0) == 0


# --- motion blending, and the ordering that makes it usable --------------------


class _Mouths:
    """An engine whose mouth colour is chosen per call, so a blend is observable."""

    def __init__(self, values):
        self.values = list(values)
        self.at = 0

    def mouths(self, windows, frame_indices):
        out = []
        for _ in frame_indices:
            v = self.values[min(self.at, len(self.values) - 1)]
            self.at += 1
            out.append(np.full((256, 256, 3), v, np.uint8))
        return out


def _encoded_mouth(renderer, cache, values, *, first=0):
    """Render two frames and return the mouth value each published frame carries."""
    step = renderer.step(frame_index=first, origin=0.0, fps=24.0)
    assert step is not None
    got = []
    for jpeg in renderer.encode(step):
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        got.append(int(frame[80, 70, 0]))
    return got


def test_a_mouth_is_mixed_with_the_one_before_it():
    """The owner's report was that the mouth moves too fast - each frame free to jump,
    because independent per-frame generation has none of a face's inertia."""
    cache = _cache()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    r = Renderer(engine=_Mouths([0, 200]), cache=cache, ring=ring)
    first, second = _encoded_mouth(r, cache, None)
    assert abs(first - 0) < 20, "the first mouth has nothing to mix with"
    # 0.55*200 + 0.45*0 = 110, not 200. JPEG is lossy, so allow room - but nowhere
    # near enough to confuse a blended 110 with an unblended 200.
    assert abs(second - int(MOTION_BLEND * 200)) < 25, second


def test_the_blend_starts_over_at_a_turn_boundary():
    """A turn restarts frame indices at 0. Carrying the last mouth of the previous
    utterance across would open the new one on a stale pose."""
    cache = _cache()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    r = Renderer(engine=_Mouths([0, 0, 200, 200]), cache=cache, ring=ring)
    _encoded_mouth(r, cache, None, first=0)          # leaves a dark mouth behind
    fresh, _ = _encoded_mouth(r, cache, None, first=90)   # a jump: new turn
    assert abs(fresh - 200) < 20, (
        f"a new turn must not mix in the previous one's pose, got {fresh}"
    )


def test_the_borrowed_texture_is_not_damped_by_the_blend():
    """The ordering, and it is the whole reason this is shippable.

    Smoothing as the LAST thing to touch the pixels was ranked unusable by the owner -
    it took the teeth with the judder. Here it runs first and `restore_detail` puts the
    driving frame's own texture back on top, from a source the mix never touched. If
    the two were swapped, that texture would be mixed away too - so this asserts it
    arrives at full strength on a frame that was definitely blended.
    """
    cache = _cache()
    cache.frames[0, BOX[1] : BOX[3], BOX[0] : BOX[2]] = _textured(120, 60)
    cache.frames[1, BOX[1] : BOX[3], BOX[0] : BOX[2]] = _textured(120, 60)
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    r = Renderer(engine=_Mouths([0, 200]), cache=cache, ring=ring)
    step = r.step(frame_index=0, origin=0.0, fps=24.0)
    jpegs = r.encode(step)
    blended = cv2.imdecode(np.frombuffer(jpegs[1], np.uint8), cv2.IMREAD_COLOR)

    flat = np.full((120, 60, 3), int(MOTION_BLEND * 200), np.uint8)
    i = step.indices[1] % len(cache.boxes)
    want = composite(
        cache.frames[i],
        restore_detail(flat, cache.frames[i], BOX),
        BOX,
        CROP_BOX,
        cache.masks[i],
    )
    x1, y1, x2, y2 = CROP_BOX
    got_lap = _lap(blended[y1:y2, x1:x2])
    want_lap = _lap(want[y1:y2, x1:x2])
    assert got_lap > want_lap * 0.7, (
        f"texture arrived damped: {got_lap:.1f} against {want_lap:.1f} - the blend "
        "is running after restore_detail, not before it"
    )
