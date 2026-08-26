"""Whisper feature windowing, copied from MuseTalk v1.5's own arithmetic.

`musetalk/utils/audio_processor.py:get_whisper_chunk` left-pads the feature array
by `ceil(50/fps) * padding_left` and then slices 2*(left+right+1) indices from
`floor(frame * 50/fps)`. Reproducing the padding here rather than the padded array
keeps every index in one coordinate system - the unpadded one, where negative means
"before the audio started" and the caller clamps.
"""

from __future__ import annotations

import math

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
