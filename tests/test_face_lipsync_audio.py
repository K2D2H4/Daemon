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


@pytest.mark.parametrize(
    "frame,expected_lookahead",
    [(24, 80.0), (600, 80.0), (0, 80.0)],
)
def test_steady_state_lookahead_is_80ms(frame, expected_lookahead):
    """Audio needed beyond the frame's own start time."""
    ahead = latest_audio_ms(frame, 24.0) - frame * 1000.0 / 24.0
    assert ahead == pytest.approx(expected_lookahead, abs=7.0)


def test_the_first_frames_ask_for_audio_before_zero():
    """At a turn's start there is no past, and the caller must clamp rather than
    index negatively - the model repeats the edge feature, which is why the first
    ~200ms degrades to a neutral mouth instead of crashing."""
    first, _ = window_for(0, 24.0)
    assert first < 0
