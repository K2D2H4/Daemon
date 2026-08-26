"""Whisper feature windowing, copied from MuseTalk v1.5's own arithmetic.

`musetalk/utils/audio_processor.py:get_whisper_chunk` left-pads the feature array
by `ceil(50/fps) * padding_left` and then slices 2*(left+right+1) indices from
`floor(frame * 50/fps)`. Reproducing the padding here rather than the padded array
keeps every index in one coordinate system - the unpadded one, where negative means
"before the audio started" and the caller clamps.
"""

from __future__ import annotations

import math

import numpy as np

AUDIO_FPS = 50
"""Whisper encoder frames per second. One index is 20ms."""

MS_PER_INDEX = 1000.0 / AUDIO_FPS

PADDING_LEFT = 2
PADDING_RIGHT = 2
"""MuseTalk v1.5 defaults (`--audio_padding_length_left/right`)."""

WINDOW = 2 * (PADDING_LEFT + PADDING_RIGHT + 1)
"""10 indices = 200ms."""


def window_for(frame_index: int, fps: float) -> tuple[int, int]:
    """Inclusive `[first, last]` unpadded whisper indices for one video frame.

    Negative `first` is normal at a turn's start and means the caller must clamp.
    """
    multiplier = AUDIO_FPS / fps
    left_pad = math.ceil(multiplier) * PADDING_LEFT
    start = math.floor(frame_index * multiplier) - left_pad
    return start, start + WINDOW - 1


def latest_audio_ms(frame_index: int, fps: float) -> float:
    """The newest audio timestamp this frame needs, in ms from the turn's start."""
    _, last = window_for(frame_index, fps)
    return (last + 1) * MS_PER_INDEX


WHISPER_RATE = 16_000
"""What whisper's mel front-end assumes. `PcmRing` holds the voice path's 24kHz
output (`daemon/voice/audio.py:OUTPUT_SAMPLE_RATE`), so the two do not match and
something has to resample - see `resample_to_whisper`."""

SAMPLES_PER_INDEX = int(WHISPER_RATE * MS_PER_INDEX // 1000)
"""320. whisper's mel hop is 160 samples (10ms) and its second conv has stride 2, so
one encoder index is 320 input samples. Needed to turn a length of audio into a count
of encoder positions.

`int()` on purpose: `MS_PER_INDEX` is a float, so without it this is 320.0 and
`encoder_positions` returns a float that blows up as a slice bound.
"""

CONTEXT_MS = 2000.0
"""How much audio must PRECEDE the 200ms window, and why the window alone is not
enough.

whisper's log-mel clamps at `log_spec.max() - 8` and then rescales, so its output
depends on the loudest frame of whatever it was handed. A bare 200ms slice normalises
against its own peak: measured against the same 200ms inside a full clip, its mel came
out *anti-correlated*, cosine -0.29. Sweeping the preceding context, the mel converges
by 2s and does not move after.

Two consequences worth writing down. A live stream can never reproduce a whole-file
reference, because that reference's peak includes audio that has not arrived yet -
measured feature cosine 0.77 against the offline path, which moved the mouth by 0.37x
one frame of its own natural motion, and did not make it jump (the running peak varied
1.235-1.334 over 150 frames). And a turn's first ~2s normalise against less context
than the rest of it.
"""


def encoder_positions(n_samples: int) -> int:
    """How many real encoder positions `n_samples` of 16kHz audio produces.

    Everything past this in a padded mel is silence, and the model reads only the last
    `WINDOW` positions - so this is what bounds the encoder's attention window. Cutting
    1500 positions down to this was measured bit-identical (cosine 1.0000, and the
    mouth pixel-for-pixel unchanged) at a third of the cost.
    """
    return n_samples // SAMPLES_PER_INDEX


def _resample_taps(half: int = 32, beta: float = 8.6) -> np.ndarray:
    """Windowed-sinc low-pass for the 2x-upsampled (48kHz) domain.

    The cutoff is not a matter of taste: after decimating to 16kHz the Nyquist is 8kHz,
    which is exactly the top of whisper's mel range, so anything above it folds down
    into the band the model reads. Measured rejection: a 9kHz tone comes back at 0.073
    amplitude and 11kHz at 0.000. The cost is some droop just under the corner (7.5kHz
    passes at 0.76), which the features did not notice.
    """
    n = np.arange(-half, half + 1, dtype=np.float64)
    h = np.sinc(n * (2.0 / 3.0) * 0.5) * np.kaiser(2 * half + 1, beta)
    return (h / h.sum() * 2.0).astype(np.float32)      # x2 recovers the zero-stuffing


_TAPS: np.ndarray | None = None


def resample_to_whisper(pcm: np.ndarray, *, rate: int) -> np.ndarray:
    """24kHz float32 -> 16kHz float32, the rate whisper's mel assumes.

    Verified against `librosa.load(sr=16000)`, which every measurement behind this
    package went through: same sample count, waveform correlation 0.999904, and
    features through the whole encoder at cosine 0.999413. 0.48ms for a 2.2s window.
    """
    if rate != 24_000:
        raise ValueError(f"only the voice path's 24kHz is supported, got {rate}")
    global _TAPS
    if _TAPS is None:
        _TAPS = _resample_taps()
    up = np.zeros(pcm.size * 2, dtype=np.float32)
    up[::2] = pcm
    return np.convolve(up, _TAPS, mode="same")[::3]
