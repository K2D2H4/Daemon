"""The audio ring and the frame slot.

Both encode a rule from the design. The ring clamps rather than indexes negatively,
because a turn's first frames legitimately ask for audio from before the turn began.
The slot overwrites rather than queues, for the same reason `level` does on the bus:
a window that stopped consuming should resume at the current mouth, not replay a
backlog of stale ones.
"""

import numpy as np

from daemon.face_lipsync.ring import PcmRing, Slot


def _silence(ms: int, rate: int = 24_000) -> bytes:
    return b"\x00\x00" * int(rate * ms / 1000)


def _tone(ms: int, value: int = 16384, rate: int = 24_000) -> bytes:
    """Distinctive constant-value waveform for testing placement."""
    samples = np.full(int(rate * ms / 1000), value, dtype=np.int16)
    return samples.tobytes()


def test_a_window_before_the_start_is_clamped_not_negative():
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    ring.feed(_tone(500, value=16384), audible_at=0.0)
    got = ring.window(frame_index=0, fps=24.0, origin=0.0)
    assert len(got) > 0
    assert not np.isnan(got).any()
    # Padding before audio start should be zero; real audio should be non-zero.
    assert np.any(got > 0.4), "Real audio should appear in window"
    assert np.any(got == 0.0), "Padding should be zero before audio start"


def test_window_contains_correct_samples_at_correct_offset():
    # Frame 0 at 24fps asks for audio from -120ms to +80ms (indices -6..3).
    # We feed 1 second of audio starting at 0.0, so:
    # - First 120ms of window is clamped to zero (before audio start)
    # - Remaining 80ms contains real audio
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    tone_value = 20000
    ring.feed(_tone(1000, value=tone_value), audible_at=0.0)
    got = ring.window(frame_index=0, fps=24.0, origin=0.0)
    expected_value = tone_value / 32768.0
    # First ~2880 samples (120ms) should be zero.
    assert np.all(got[:2880] == 0.0)
    # Remaining ~1920 samples (80ms) should be the tone value.
    assert np.all(np.abs(got[2880:] - expected_value) < 0.001)


def test_window_placement_with_offset_origin():
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    tone_value = -15000
    # Frame 0 at origin 0.5s asks for time 0.38s to 0.58s (indices -6..3).
    # Audio starts at 0.5s, so first 120ms is silence, rest is audio.
    ring.feed(_tone(1000, value=tone_value), audible_at=0.5)
    got = ring.window(frame_index=0, fps=24.0, origin=0.5)
    expected_value = tone_value / 32768.0
    # First 120ms (2880 samples) should be zero.
    assert np.all(got[:2880] == 0.0)
    # Remaining 80ms (1920 samples) should be the tone value.
    assert np.all(np.abs(got[2880:] - expected_value) < 0.001)


def test_the_window_grows_no_longer_than_ten_indices():
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    ring.feed(_silence(2000), audible_at=0.0)
    got = ring.window(frame_index=48, fps=24.0, origin=0.0)
    assert len(got) == int(24_000 * 0.200)


def test_gap_between_turns_re_anchors_the_buffer():
    # Turn 1: feed 1s at time 0.
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    ring.feed(_tone(1000, value=10000), audible_at=0.0)
    # Gap of 3s (nothing fed), then turn 2 starts at time 5.0.
    ring.feed(_tone(1000, value=25000), audible_at=5.0)
    # Window at frame 0, origin=5.0 should find turn 2's audio, not turn 1.
    got = ring.window(frame_index=0, fps=24.0, origin=5.0)
    turn2_value = 25000 / 32768.0
    turn1_value = 10000 / 32768.0
    # Should see turn 2's audio (0.76...), not turn 1's (0.30...).
    assert np.any(np.abs(got - turn2_value) < 0.001), "Turn 2 audio should be present"
    assert not np.all(np.abs(got - turn1_value) < 0.001), "Turn 1 audio should NOT be present"


def test_barge_in_jumping_backward_also_re_anchors_the_buffer():
    """The mirror image of the test above. A forward gap is a new turn starting
    after a silence; barge-in is the opposite shape - `daemon/voice/conversation.py`'s
    `_barge_in` rebuilds `SpeechClock` from scratch (`_until` back to 0.0), so the
    next chunk's `audible_at` can land BEFORE the ring's current buffer end, not
    after it. A one-sided `gap > 0.001` check treats that negative gap as no
    discontinuity at all: the cancelled turn's samples stay exactly where they
    were, the barge-in reply's samples get appended past them, and `window()`
    keeps reading the interrupted sentence instead of the new one.
    """
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    # Old (about-to-be-cancelled) turn: 1s of audio starting at 0.0, so the
    # buffer end sits at 1.0s.
    ring.feed(_tone(1000, value=10000), audible_at=0.0)
    # Barge-in rebuilds the clock: the new turn's first chunk is audible at
    # 0.2s - *before* the old buffer end (1.0s), a gap of -0.8s.
    ring.feed(_tone(200, value=25000), audible_at=0.2)
    got = ring.window(frame_index=0, fps=24.0, origin=0.2)
    new_value = 25000 / 32768.0
    old_value = 10000 / 32768.0
    assert np.any(np.abs(got - new_value) < 0.001), "Barge-in audio should be present"
    assert not np.any(np.abs(got - old_value) < 0.001), (
        "Cancelled turn's audio should NOT be present once barge-in re-anchors"
    )


def test_the_slot_keeps_only_the_latest_frame():
    slot = Slot()
    slot.put(b"first")
    slot.put(b"second")
    assert slot.get() == b"second"
    assert slot.get() == b"second"


def test_an_empty_slot_reports_nothing_rather_than_blocking():
    assert Slot().get() is None


# --- the context lead-in ---------------------------------------------------------
#
# whisper's log-mel normalises against the peak of whatever it is handed, so a bare
# 200ms slice comes out anti-correlated (measured cosine -0.29) with the same audio
# inside a stream. `context_ms` prepends the audio that settles it, and the model's
# own window has to stay at the TAIL - an engine reading from the front would be
# conditioning on audio ~2s stale, which no shape check would catch.


def test_context_lengthens_the_window_at_the_front_only():
    ring = PcmRing(sample_rate=24_000, width=2, seconds=6.0)
    for step in range(50):
        ring.feed(_tone(100, value=1000 * (step % 30 + 1)), audible_at=step * 0.1)
    plain = ring.window(frame_index=72, fps=24.0, origin=0.0)
    with_ctx = ring.window(frame_index=72, fps=24.0, origin=0.0, context_ms=2000.0)
    assert with_ctx.size == plain.size + int(24_000 * 2.0)
    assert np.array_equal(with_ctx[-plain.size :], plain)


def test_context_defaults_to_off_so_existing_callers_are_unchanged():
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    ring.feed(_tone(2000), audible_at=0.0)
    assert np.array_equal(
        ring.window(frame_index=24, fps=24.0, origin=0.0),
        ring.window(frame_index=24, fps=24.0, origin=0.0, context_ms=0.0),
    )


def test_context_reaching_before_the_turn_began_is_zero_filled_not_wrapped():
    """A turn's first frames have no 2s of history. The lead-in must read as silence,
    not as the tail of the previous turn or a wrapped slice of this one."""
    ring = PcmRing(sample_rate=24_000, width=2, seconds=6.0)
    ring.feed(_tone(400), audible_at=0.0)
    got = ring.window(frame_index=0, fps=24.0, origin=0.0, context_ms=2000.0)
    assert got.size == int(24_000 * 2.2)
    head = got[: int(24_000 * 1.9)]
    assert not head.any(), "context before the turn should be silence"
    assert got[-int(24_000 * 0.1) :].any(), "the window itself should still have audio"
