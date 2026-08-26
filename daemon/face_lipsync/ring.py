"""A rolling PCM buffer and a latest-frame slot.

`SpeechClock` already stamps every chunk with the moment it becomes audible, so
this stores audio on that timeline rather than on arrival time - which is what lets
a frame index be turned into a sample offset without knowing anything about queues.
The ring re-anchors when it detects a gap, so it always holds the current timeline,
not a stale one.
"""

from __future__ import annotations

import threading

import numpy as np

from daemon.face_lipsync.audio import MS_PER_INDEX, WINDOW, window_for


class PcmRing:
    """Recent PCM, addressed by the time it is heard rather than when it arrived."""

    def __init__(self, *, sample_rate: int, width: int, seconds: float) -> None:
        self._rate = sample_rate
        self._width = width
        self._max = int(sample_rate * seconds)
        self._samples = np.zeros(0, dtype=np.int16)
        self._start = 0.0
        """Audible time of `self._samples[0]`."""

    def feed(self, chunk: bytes, audible_at: float) -> None:
        block = np.frombuffer(chunk, dtype=np.int16)
        if self._samples.size == 0:
            self._start = audible_at
        else:
            # Detect if incoming chunk is discontinuous from buffered audio in
            # EITHER direction. A chunk arriving LATER than the buffer end is a
            # new turn or a long silence. One arriving EARLIER means the clock
            # itself was rebuilt from scratch - `daemon/voice/conversation.py`'s
            # `_barge_in` replaces `SpeechClock` wholesale on barge-in, resetting
            # `_until` to 0.0 - and that is just as much a discontinuity as a gap
            # forward: left unhandled, the cancelled turn's audio stays at the
            # front of the buffer and `window()` keeps reading it instead of the
            # new reply.
            buffer_end = self._start + self._samples.size / self._rate
            gap = audible_at - buffer_end
            if abs(gap) > 0.001:  # 1ms tolerance for float noise within a turn
                # New turn, long silence, or a rebuilt clock (barge-in): discard
                # stale samples and re-anchor.
                self._samples = np.zeros(0, dtype=np.int16)
                self._start = audible_at
        self._samples = np.concatenate([self._samples, block])
        if self._samples.size > self._max:
            drop = self._samples.size - self._max
            self._samples = self._samples[drop:]
            self._start += drop / self._rate

    def window(
        self, *, frame_index: int, fps: float, origin: float
    ) -> np.ndarray:
        """The 200ms the model reads for `frame_index`, as float32 in -1..1.

        Clamped at both ends: a turn's first frames ask for audio from before it
        began, and the newest frames may outrun what has arrived.
        """
        first, _ = window_for(frame_index, fps)
        begin_s = origin + first * MS_PER_INDEX / 1000.0
        need = int(self._rate * WINDOW * MS_PER_INDEX / 1000.0)
        offset = int((begin_s - self._start) * self._rate)
        lo = max(0, offset)
        hi = min(self._samples.size, offset + need)
        got = self._samples[lo:hi] if hi > lo else self._samples[:0]
        out = np.zeros(need, dtype=np.float32)
        if got.size:
            at = max(0, -offset)
            out[at : at + got.size] = got.astype(np.float32) / 32768.0
        return out


class Slot:
    """One frame, latest wins. Never queues - see the module docstring."""

    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: bytes | None = None

    def put(self, frame: bytes) -> None:
        with self._lock:
            self._value = frame

    def get(self) -> bytes | None:
        with self._lock:
            return self._value
