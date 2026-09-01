"""The audio window, which is where the latency budget actually comes from.

MuseTalk v1.5 reads 10 whisper indices per frame at 20ms each - 200ms total. The
split between past and future is not flat: `floor()` quantisation makes the
lookahead cycle 61.67-80ms and the lookbehind cycle 120-138.33ms, mirror images of
each other twelve frames apart. 80ms is the lookahead's peak - the one number a
faster GPU cannot reduce - so it is pinned here rather than left to the model code
to imply.
"""

import numpy as np
import pytest

from daemon.face_lipsync.audio import (
    CONTEXT_MS,
    MS_PER_INDEX,
    SAMPLES_PER_INDEX,
    WINDOW,
    encoder_positions,
    latest_audio_ms,
    resample_to_whisper,
    window_for,
)


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


# --- resampling and the encoder window ------------------------------------------
#
# PcmRing holds the voice path's 24kHz; whisper's mel assumes 16kHz. Every feature
# measurement behind this package went through `librosa.load(sr=16000)`, and the
# daemon has neither librosa nor scipy, so this is the replacement and it has to land
# in the same place.


def test_resample_hits_the_exact_output_length():
    got = resample_to_whisper(np.zeros(24_000, np.float32), rate=24_000)
    assert got.size == 16_000
    assert got.dtype == np.float32


def test_resample_refuses_a_rate_it_was_not_verified_at():
    """Silently accepting 48kHz would produce a plausible array at the wrong pitch."""
    with pytest.raises(ValueError, match="24kHz"):
        resample_to_whisper(np.zeros(1000, np.float32), rate=48_000)


def test_resample_preserves_a_tone_inside_the_band():
    t = np.arange(24_000) / 24_000
    out = resample_to_whisper(np.sin(2 * np.pi * 1000 * t).astype(np.float32), rate=24_000)
    spec = np.abs(np.fft.rfft(out * np.hanning(out.size)))
    peak = np.fft.rfftfreq(out.size, 1 / 16_000)[spec.argmax()]
    assert abs(peak - 1000) < 20


def test_resample_rejects_content_above_the_new_nyquist():
    """The anti-alias filter is the whole reason this is not `pcm[::3]` interpolated.

    16kHz puts Nyquist at 8kHz, which is exactly the top of whisper's mel range, so a
    9kHz tone that folded down would land inside the band the model reads and be
    indistinguishable from speech.
    """
    t = np.arange(24_000) / 24_000

    def rms(hz):
        out = resample_to_whisper(np.sin(2 * np.pi * hz * t).astype(np.float32), rate=24_000)
        return float(np.sqrt((out**2).mean()))

    # RMS, not peak: `np.convolve(mode="same")` leaves a transient at each edge that
    # dominates the time-domain maximum (0.33 for a tone attenuated to 0.05 RMS), so a
    # peak-based assertion would fail on a filter that works.
    assert rms(9000) < rms(1000) / 5


def test_encoder_positions_is_an_int_and_counts_320_sample_steps():
    """A float here survives every arithmetic test and only fails as a slice bound,
    which is deep inside the engine - `MS_PER_INDEX` is a float, so this is one
    edit away from breaking again."""
    p = encoder_positions(35_200)                     # 2.2s at 16kHz
    assert p == 110
    assert isinstance(p, int)
    assert encoder_positions(SAMPLES_PER_INDEX - 1) == 0


def test_the_context_length_covers_the_window_it_precedes():
    """CONTEXT_MS is a measured floor, not a taste: below ~2s the mel had not
    converged. This only guards the relationship, so a later trim cannot make the
    context shorter than the window it is supposed to stabilise."""
    assert CONTEXT_MS >= WINDOW * MS_PER_INDEX
