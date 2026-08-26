"""The audio window, which is where the latency budget actually comes from.

MuseTalk v1.5 reads 10 whisper indices per frame at 20ms each - 120ms of past and
80ms of future. That 80ms is the one number a faster GPU cannot reduce, so it is
pinned here rather than left to the model code to imply.
"""

import pytest

from daemon.face_lipsync.audio import latest_audio_ms, window_for


def test_window_is_ten_indices():
    first, last = window_for(600, 24.0)
    assert last - first + 1 == 10


@pytest.mark.parametrize("frame", [24, 600, 0])
def test_lookahead_peaks_at_80ms_on_whisper_aligned_frames(frame):
    """`frame * 50/24` is an integer exactly when `frame % 12 == 0` - the frame's
    own start time lands precisely on a 20ms whisper tick, `floor()` is a no-op,
    and the lookahead hits its ceiling with no float drift (checked across every
    multiple of 12 from 0 to 996, not just these three)."""
    ahead = latest_audio_ms(frame, 24.0) - frame * 1000.0 / 24.0
    assert ahead == 80.0


@pytest.mark.parametrize(
    "frame,expected_ahead_ms",
    [(1, 78.333333), (7, 68.333333), (11, 61.666667)],
)
def test_lookahead_dips_below_80ms_between_whisper_ticks(frame, expected_ahead_ms):
    """Off the 12-frame grid, `floor()` discards a real fraction and the lookahead
    cycles through 12 distinct values (61.67ms .. 80ms) instead of holding at 80.

    Frame 7 is the case that actually exercises `floor()` against a plausible
    regression: `7 * 50/24 == 14.583`, so `floor` gives 14 but `round` would give
    15 - a round()-instead-of-floor() bug is invisible at frames 0/24/600 (exact)
    and at frame 1 (`2.083` rounds the same either way), and only shows up here,
    20ms off.
    """
    ahead = latest_audio_ms(frame, 24.0) - frame * 1000.0 / 24.0
    assert ahead == pytest.approx(expected_ahead_ms, abs=1e-6)


def test_the_80ms_peak_is_never_exceeded_across_a_full_sweep():
    """80ms is the number the whole latency budget rests on, so the claim that
    quantisation only ever pulls the lookahead down - never past it - gets its own
    assertion over many frames rather than resting on the three peak cases above."""
    ahead = [latest_audio_ms(frame, 24.0) - frame * 1000.0 / 24.0 for frame in range(1000)]
    assert max(ahead) == 80.0


def test_the_first_frames_ask_for_audio_before_zero():
    """At a turn's start there is no past, and the caller must clamp rather than
    index negatively - the model repeats the edge feature, which is why the first
    ~200ms degrades to a neutral mouth instead of crashing."""
    first, _ = window_for(0, 24.0)
    assert first < 0
