"""Only a person talking reaches the model; everything else is digital silence.

Why this exists, measured 2026-09-02 on the owner's Mac (daemon/MEASURED.md):

  * The echo-cancelled microphone (`apple_audio.py`) carries the room at about
    -57 dBFS the whole time, and leaks the daemon's own voice in short bursts - the
    worst seen was three consecutive VAD frames at -24 dBFS while the speaker was
    still talking.
  * A person at the desk reads as speech in 72% of frames, in sustained runs.
  * `gemini-live-2.5-flash-native-audio` answers the first two as though the owner
    had spoken: four of four "user" turns in one session were the tail of the reply
    just played, five of seven in the next were room sounds transcribed as `Ay.`,
    `da`, `what 2 3 4`. The 3.1 model on the same build, room and speaker ignored
    them - 0 such turns. Through the product loop with a -48 dBFS noise tail, 2.5
    also hesitated (empty turns before the answer, +0.6 s) where digital silence
    after the utterance gave 6/6 answers in 1.7 s.

So the server is not shown the room. Frames the local VAD does not call speech go
out as zeros of the same length - the stream stays continuous, which is what the
server's own end-of-speech detection keys on - and speech opens the gate only after
`open_after_frames` in a row, longer than the longest leak. The head of the
utterance is not lost to that delay: a `pre_roll_frames` ring is flushed the moment
the gate opens, the same trick `wake.py` uses so a wake word arrives whole. The
gate closes again after `hangover_frames` of non-speech, long enough to span the
pause inside a sentence.

Pure arithmetic over the VAD's probabilities; no I/O, no clock. The conversation
owns one per session and calls `reset` whenever its half-duplex gate drops audio,
so room sound from before the daemon spoke cannot flush as the head of what the
owner says after.
"""

from __future__ import annotations

from collections import deque

from daemon.voice.base import VoiceActivityDetector

THRESHOLD = 0.5
"""Speech probability at or above which a frame counts as speech - `wake.py`'s."""
OPEN_AFTER_FRAMES = 12
"""Consecutive speech frames before the gate opens: 384 ms at 32 ms a frame.

Fitted, not guessed. Counted by frame index over the owner's own room and speakers
(2026-09-02): the daemon's voice leaking past echo cancellation reached a longest
run of **8** frames, while a voice in the room read 68 in a row. Four frames - the
first value here - is inside the leak, which is exactly the mistake this constant
exists to avoid."""
DENSITY_WINDOW_FRAMES = 25
DENSITY_OPEN_FRAMES = 18
"""...or 18 speech frames out of the last 25. The same measurement: the leak's
densest 25-frame window held 8, a voice in the room held 25. A run alone is
brittle for a quiet voice the VAD scores unevenly; this opens on 576 ms of mostly
speech even when no 12 frames are consecutive."""
PRE_ROLL_FRAMES = 31
"""Audio kept from before the gate opened and flushed the moment it does - about
1 s. It has to cover the opening rule above *and* however late the VAD is on a
real voice: measured live, the gate opened 2.1 s into a sentence spoken across the
room, and a 300 ms pre-roll left the server the tail of a sentence to transcribe -
which it rendered as `ランチョンマット で` and the model answered as nonsense."""
HANGOVER_FRAMES = 25
"""Non-speech that closes the gate again: 800 ms. Longer than a pause inside a
sentence, shorter than the server's own end-of-speech timer needs to fire on the
silence that follows."""


class SpeechGate:
    """Frame-by-frame: is the room's audio a person talking?

    `feed` takes microphone bytes in any block size and returns the bytes to send
    for every whole VAD frame it completed - the recorded audio while open, zeros
    of the same length while shut, plus the pre-roll on the frame that opens it.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        *,
        threshold: float = THRESHOLD,
        open_after_frames: int = OPEN_AFTER_FRAMES,
        pre_roll_frames: int = PRE_ROLL_FRAMES,
        hangover_frames: int = HANGOVER_FRAMES,
        density_window_frames: int = DENSITY_WINDOW_FRAMES,
        density_open_frames: int = DENSITY_OPEN_FRAMES,
    ) -> None:
        self.vad = vad
        self._frame_bytes = vad.frame_samples * 2
        self._threshold = threshold
        self._open_after = max(1, open_after_frames)
        self._hangover = max(1, hangover_frames)
        # The ring holds the opening run too, so it is at least that long.
        self._pre_roll: deque[bytes] = deque(maxlen=max(pre_roll_frames, self._open_after))
        self._recent: deque[bool] = deque(maxlen=max(1, density_window_frames))
        self._density_open = max(1, density_open_frames)
        self._pending = bytearray()
        self._speech_run = 0
        self._silence_run = 0
        self.open = False

    def feed(self, block: bytes) -> bytes:
        """Audio to send for the frames this block completed. Empty when none did."""
        self._pending.extend(block)
        out = bytearray()
        size = self._frame_bytes
        while len(self._pending) >= size:
            frame = bytes(self._pending[:size])
            del self._pending[:size]
            out.extend(self._frame(frame))
        return bytes(out)

    def _frame(self, frame: bytes) -> bytes:
        speech = float(self.vad.probability(frame)) >= self._threshold
        if self.open:
            if speech:
                self._silence_run = 0
            else:
                self._silence_run += 1
                if self._silence_run >= self._hangover:
                    self.open = False
                    self._speech_run = 0
                    self._pre_roll.clear()
                    self._recent.clear()
            return frame
        self._pre_roll.append(frame)
        self._recent.append(speech)
        self._speech_run = self._speech_run + 1 if speech else 0
        dense = sum(self._recent) >= self._density_open
        if self._speech_run >= self._open_after or dense:
            self.open = True
            self._silence_run = 0
            self._recent.clear()
            head = b"".join(self._pre_roll)
            self._pre_roll.clear()
            return head
        return bytes(len(frame))

    def reset(self) -> None:
        """Forget what came before: the pre-roll, a run in progress, the VAD's
        context. Called whenever the caller drops microphone audio, so the gate does
        not open on the far side of a gap with the audio from before it."""
        if not (self.open or self._pre_roll or self._pending or self._speech_run):
            return
        self.open = False
        self._speech_run = 0
        self._silence_run = 0
        self._pre_roll.clear()
        self._recent.clear()
        self._pending.clear()
        self.vad.reset()
