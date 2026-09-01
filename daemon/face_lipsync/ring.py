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
        assert width == 2, "PcmRing is 16-bit mono; width must be 2 (bytes/sample)"
        self._rate = sample_rate
        self._max = int(sample_rate * seconds)
        self._samples = np.zeros(0, dtype=np.int16)
        self._start = 0.0
        """Audible time of `self._samples[0]`."""

    @property
    def origin(self) -> float:
        """Audible time of the oldest sample held - the turn's start, in practice.

        `window()` addresses frames relative to an `origin` the caller supplies, and
        this is where that value comes from. It MOVES: `feed` re-anchors on a gap in
        either direction (a new turn, a long silence, or a barge-in that rebuilds the
        clock), and it also creeps forward as the ring drops samples past `seconds`.
        A render loop therefore has to read it each tick rather than capture it once
        - a stale origin points frame indices at audio that is no longer there, and
        `window` answers with silence rather than an error.
        """
        return self._start

    @property
    def sample_rate(self) -> int:
        """The rate `window()`'s output is sampled at - not whisper's assumed
        16kHz. See `LipsyncEngine.mouths`'s docstring for why that gap matters."""
        return self._rate

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
        self, *, frame_index: int, fps: float, origin: float, context_ms: float = 0.0
    ) -> np.ndarray:
        """`context_ms` of preceding audio, then the 200ms the model reads for
        `frame_index`. float32 in -1..1, and the model's window is the TAIL.

        Clamped at both ends: a turn's first frames ask for audio from before it
        began, and the newest frames may outrun what has arrived.

        The context exists because whisper's front-end is not local. Its log-mel
        clamps at `log_spec.max() - 8` and rescales, so handing it a bare 200ms
        normalises that slice against its own peak - measured *anti-correlated*
        (cosine -0.29) with the same 200ms taken from a longer stream. See
        `audio.CONTEXT_MS` for the sweep that settled the length.

        Whoever builds the ring has to size `seconds` for at least
        `context_ms + 200ms`, or the context is silently truncated to whatever the
        ring still holds and the features degrade with no error anywhere.
        """
        first, _ = window_for(frame_index, fps)
        begin_s = origin + first * MS_PER_INDEX / 1000.0
        # The lead-in is subtracted in SAMPLES, not seconds. Doing it in seconds
        # shifted the window by one sample against the no-context call, because the
        # subtraction is inexact in binary: 2.88 - 2.0 is 0.8799999999999999, and
        # int() then floors to 21119 where the direct 2.88 gives 69120. Harmless as
        # audio, but it means the tail is not the window the arithmetic promised.
        lead = int(self._rate * context_ms / 1000.0)
        need = lead + int(self._rate * WINDOW * MS_PER_INDEX / 1000.0)
        offset = int((begin_s - self._start) * self._rate) - lead
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
