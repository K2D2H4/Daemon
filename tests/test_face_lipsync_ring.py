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


def test_the_slot_keeps_only_the_latest_frame():
    slot = Slot()
    slot.put(b"first")
    slot.put(b"second")
    assert slot.get() == b"second"
    assert slot.get() == b"second"


def test_an_empty_slot_reports_nothing_rather_than_blocking():
    assert Slot().get() is None
