"""The wake gate: two free local stages in front of a session that bills per minute.

A `VoiceSession` held open on the chance of being spoken to costs about 48x what 30
minutes of conversation a day costs (docs/PLAN.md 6.5), so something free has to
decide when to open one:

    mic ──► frames ──► VAD ──► segment ──► recognizer ──► alias ──► WakeEvent
            re-chunk   0.49%    bounded    free, offline   match     (billed session)
                       of a core           in memory                  opens later

**The VAD is never allowed to open a session on its own.** It calls a 3-note chord
with vibrato speech in 46.8% of frames - music is speech to a VAD - and a real
`daemon voice` run against ambient music recorded `nela o trecho da música` as
though the owner had said it. The recognizer stage exists to throw that away, and a
gate with the second stage removed is not a cheaper gate, it is a random one.

## Matching is against what the recognizer returns, not against the name

An on-device recognizer never emits a coined name. Measured on this project's
target machine, 3 runs each, 100% stable: `헤이 데몬` -> `헤이 대문`, `데몬` ->
`질문`, `루시야` -> `루시`, `헤이 루시` -> `헤이씨`. Ordinary Korean transcribes
exactly. So the aliases are a configured list of *observed* transcriptions
(`DAEMON_WAKE_ALIASES`, calibrated per speaker), compared after normalising away
spacing, punctuation and Unicode form - `헤이 자기` and `헤이자기` are the same
phrase, and NFD Korean off macOS is the same text as NFC Korean out of a model.

The match is **head-anchored**: the alias has to start the transcript. Anywhere in
the sentence would be tempting for recall, but the aliases are ordinary Korean
words - the recognizer's rendering of `데몬` is literally `질문` - so a substring
rule fires on ordinary conversation, and the asymmetry is not close: a false fire
opens a paid session and a miss costs the owner one repeat. A segment also *begins*
where speech began, so the wake word being at the head is a property of the audio
rather than an assumption about the speaker.

## It must not die

A wake gate that dies takes the always-listening promise with it and leaves a
process that is alive, `/health`-green and permanently deaf - a failure this repo
has shipped three times. So a raising VAD, a raising or absent recognizer and a
frame that will not decode are all counted, logged periodically and stepped over.
What is *not* swallowed is the microphone stream itself: that is fixed by plugging
something in or installing something (daemon/voice/audio.py), not by retrying, and
pretending to listen with no device is the same silent degradation one level down.

Which is also why every outcome is counted rather than merely logged: "nobody has
said anything" and "the recognizer has been unavailable since Tuesday" look
identical from outside, and `daemon wake test` prints the tally that tells them
apart.
"""

from __future__ import annotations

import logging
import time
import unicodedata
from collections import deque
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from daemon.voice.base import AudioIO, SpeechRecognizer, VoiceActivityDetector, WakeEvent

logger = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 2
"""16-bit little-endian PCM, the one format `AudioIO` and the VAD agree on."""

# Defaults for a direct construction. `daemon/config.py` carries the same numbers as
# DAEMON_WAKE_*, because config.py is foundation and may not import this module -
# the alternative to two literals is an import pointing the wrong way.

DEFAULT_THRESHOLD = 0.5
"""Speech probability at or above which a frame counts as speech."""

DEFAULT_HANGOVER_MS = 600
"""Non-speech that ends a segment. Longer than the pause inside `헤이 루시` and
shorter than the gap between two sentences, so a wake phrase arrives whole and a
monologue does not arrive as one blob."""

DEFAULT_PRE_ROLL_MS = 300
"""Audio kept from *before* the VAD said speech. A VAD notices speech a frame or
two after it starts; a wake word is one or two syllables, so losing the head loses
the match - `루시야` transcribed from its second syllable is not `루시`."""

DEFAULT_MIN_SPEECH_MS = 200
"""Below this, no two-syllable Korean phrase fits, so the segment is dropped
unheard. A 32 ms blip is not a wake word and must not cost a transcription."""

DEFAULT_MAX_SEGMENT_MS = 3_000
"""Hard cap on one segment. Someone talking for a minute must not accumulate a
minute of audio in memory, and must not arrive at the recognizer as one blob:
Apple on-device ko-KR finalised a 2.2 s clip in 760 ms, and that is the size this
stage is fast at."""

DEFAULT_COOLDOWN_SECONDS = 5.0
"""How long after a fire the gate stays quiet, so one `루시야` cannot open two
sessions. Covers the rest of the utterance that carried the wake word - a session
is up in about 1.3 s (0.56 s handshake + 740 ms to first audio, measured) - without
swallowing a deliberate second call."""

ERROR_LOG_EVERY = 20
"""Failures here arrive per frame when they arrive at all, so they are counted and
logged periodically. The count is what a report needs anyway; the flood is not."""


def _monotonic() -> float:
    # Indirection so a test can pin the cooldown clock, as in
    # daemon/channels/telegram.py. Monotonic, not wall clock: a cooldown must not
    # be reopened or extended by an NTP correction.
    return time.monotonic()


def normalise(text: str) -> str:
    """The form aliases are compared in: NFC, case-folded, letters and digits only.

    Spacing and punctuation go because the recognizer's are not the owner's: the
    same phrase came back as `헤이 대문` and would come back as `헤이 대문,`. NFC
    goes first because macOS hands out NFD Korean while a model produces NFC, and
    the two are different byte strings for the same word.
    """
    folded = unicodedata.normalize("NFC", text.casefold())
    return "".join(
        char for char in folded if unicodedata.category(char)[0] in ("L", "N", "M")
    )


def match_alias(heard: str, aliases: Sequence[str]) -> str | None:
    """The configured alias this transcript starts with, or None.

    Head-anchored on purpose - see the module docstring. The longest alias wins, so
    configuring both `헤이` and `헤이 루시` reports the phrase that was actually
    said rather than whichever happened to be listed first. The alias is returned
    as configured, not normalised, because `WakeEvent.matched` is what tells the
    owner which line of `DAEMON_WAKE_ALIASES` fired.
    """
    text = normalise(heard)
    if not text:
        return None
    candidates = [(alias, normalise(alias)) for alias in aliases]
    candidates.sort(key=lambda pair: len(pair[1]), reverse=True)
    for alias, key in candidates:
        if key and text.startswith(key):
            return alias
    return None


@dataclass(slots=True)
class WakeCounters:
    """What the gate has done, as `daemon wake test` reports it.

    Every field is a different answer to "is it working?", and the dangerous
    failures are the ones with no exception attached: a gate whose recognizer is
    unavailable, or whose every segment is too short, hears nothing forever and
    looks exactly like a quiet house.
    """

    frames_seen: int = 0
    """Frames the VAD was asked about. Zero means no audio is arriving at all."""

    segments_closed: int = 0
    """Runs of speech the VAD delimited, before any length filtering."""

    transcribed: int = 0
    """Segments that actually cost a recognizer call."""

    fired: int = 0
    """WakeEvents yielded."""

    skipped_short: int = 0
    """Segments dropped for being below the minimum, un-transcribed."""

    skipped_cooldown: int = 0
    """Segments dropped because a fire was still recent."""

    skipped_unavailable: int = 0
    """Segments dropped because the recognizer could not answer at all. The one
    counter that separates "nobody spoke" from "we have been deaf since Tuesday"."""

    errors: int = 0
    """Exceptions stepped over: a raising VAD, a raising recognizer, a frame that
    would not decode."""


class WakeGate:
    """Always-on listening, and the decision to spend money.

    Depends on the protocols in `daemon/voice/base.py` and on nothing that
    implements them: the VAD is arithmetic and the recognizer is an OS service that
    can be absent, so they fail differently and are mocked differently.
    `daemon/app.py` is where the real ones are chosen.

    It records from the `AudioIO` it is given and does not close it: releasing the
    device has to survive cancellation and a consumer that breaks out of the loop as
    well as an ordinary end, so it belongs to whoever opened it -
    `app.build_wake_gate` hands back a closer for exactly that.
    """

    def __init__(
        self,
        audio: AudioIO,
        vad: VoiceActivityDetector,
        recognizer: SpeechRecognizer,
        aliases: Sequence[str],
        *,
        threshold: float = DEFAULT_THRESHOLD,
        hangover_ms: int = DEFAULT_HANGOVER_MS,
        pre_roll_ms: int = DEFAULT_PRE_ROLL_MS,
        min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
        max_segment_ms: int = DEFAULT_MAX_SEGMENT_MS,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        if vad.sample_rate != audio.sample_rate:
            # Loud, because the quiet version of this is the worst bug the gate can
            # have: a VAD fed at the wrong rate returns plausible probabilities near
            # zero for real speech, so the gate simply never hears anything and
            # nothing anywhere says why.
            raise ValueError(
                f"the VAD wants {vad.sample_rate} Hz and the microphone captures "
                f"{audio.sample_rate} Hz; resample one of them before wiring them together"
            )
        self._audio = audio
        self._vad = vad
        self._recognizer = recognizer
        self._aliases = tuple(aliases)
        self._threshold = threshold
        self._frame_bytes = vad.frame_samples * BYTES_PER_SAMPLE
        # Durations are configured in milliseconds and used in frames: the frame
        # size belongs to the exported model (512 samples, 32 ms at 16 kHz) and
        # nobody configuring a hangover should have to know it.
        frame_ms = 1000.0 * vad.frame_samples / vad.sample_rate
        self._hangover_frames = max(1, round(hangover_ms / frame_ms))
        self._pre_roll_frames = max(0, round(pre_roll_ms / frame_ms))
        self._min_speech_frames = max(1, round(min_speech_ms / frame_ms))
        self._max_segment_frames = max(2, round(max_segment_ms / frame_ms))
        self._cooldown_seconds = cooldown_seconds
        self._last_fire: float | None = None
        self.counters = WakeCounters()

    async def listen(self) -> AsyncIterator[WakeEvent]:
        """WakeEvents, for as long as the microphone yields audio.

        An async generator, like `Channel.listen`: the consumer decides when to stop
        by stopping the iteration.
        """
        self._reset_vad()
        # Bytes that arrived without completing a frame. The microphone's block
        # size is its own (480 samples, 30 ms) and the VAD's is the model's (512),
        # so they never line up and a gate that assumed they did would hand the
        # VAD a short frame and be told, correctly, that it is the wrong size.
        pending = bytearray()
        # +1 because the frame that *opens* a segment is in here too, and the
        # configured pre-roll is audio from before speech started - not including
        # the frame the VAD finally accepted.
        pre_roll: deque[bytes] = deque(maxlen=self._pre_roll_frames + 1)
        segment: list[bytes] = []
        speech: list[float] = []
        silence_run = 0
        in_segment = False

        try:
            async for block in self._audio.record():
                pending += block
                while len(pending) >= self._frame_bytes:
                    frame = bytes(pending[: self._frame_bytes])
                    del pending[: self._frame_bytes]

                    self.counters.frames_seen += 1
                    probability = self._probability(frame)
                    is_speech = probability >= self._threshold
                    # Fed on every frame, speech or not, so the buffer always holds
                    # the audio immediately before now - which is what a segment
                    # needs as its head, and what makes the overlap below free.
                    pre_roll.append(frame)

                    if not in_segment:
                        if not is_speech:
                            continue
                        in_segment = True
                        # The pre-roll already contains this frame.
                        segment = list(pre_roll)
                        speech = [probability]
                        silence_run = 0
                        continue

                    segment.append(frame)
                    if is_speech:
                        speech.append(probability)
                        silence_run = 0
                    else:
                        silence_run += 1

                    full = len(segment) >= self._max_segment_frames
                    if silence_run < self._hangover_frames and not full:
                        continue

                    self.counters.segments_closed += 1
                    pcm = b"".join(segment)
                    # Mean over the frames the VAD *called* speech. Including the
                    # hangover silence would make a clean two-syllable wake word
                    # look like a coin flip.
                    confidence = sum(speech) / len(speech) if speech else 0.0
                    spoken_frames = len(speech)
                    silence_run = 0
                    if full:
                        # A cap on memory, not the end of an utterance: keep
                        # listening inside the same breath, seeded from the
                        # pre-roll so a phrase the cap split still has a head in
                        # the next segment. The VAD is deliberately *not* reset -
                        # the stream is continuing, and resetting mid-speech is
                        # what its docstring warns about.
                        segment = list(pre_roll)
                        speech = [probability] if is_speech else []
                    else:
                        in_segment = False
                        segment = []
                        speech = []
                        self._reset_vad()

                    if spoken_frames < self._min_speech_frames:
                        self.counters.skipped_short += 1
                        continue
                    event = await self._decide(pcm, confidence)
                    if event is not None:
                        yield event
        except Exception:
            # Not swallowed, unlike everything above: this is the microphone going
            # away, which is fixed by plugging it back in or installing something
            # (daemon/voice/audio.py), not by retrying. Logged with its traceback
            # and re-raised, because a gate that retried forever against a device
            # that is not there would report itself healthy while hearing nothing.
            logger.exception("wake: the gate has stopped listening")
            raise

    async def _decide(self, pcm: bytes, confidence: float) -> WakeEvent | None:
        """Transcribe one segment and decide, or explain itself to a counter."""
        if self._last_fire is not None and _monotonic() - self._last_fire < self._cooldown_seconds:
            # Checked before transcribing rather than after matching: the tail of
            # the utterance that just fired is the common case, and it is not worth
            # a recognizer call.
            self.counters.skipped_cooldown += 1
            return None
        if not self._available():
            self.counters.skipped_unavailable += 1
            return None
        try:
            # Awaited inline, which stops draining the microphone for as long as it
            # takes (760 ms measured). That is affordable because `AudioIO` buffers
            # about 1.9 s and drops its oldest blocks: what is lost is audio from
            # after the wake word, in the one moment we are already awake.
            heard = await self._recognizer.transcribe(pcm)
        except Exception as exc:
            self._note("wake: the recognizer failed on a segment", exc)
            return None
        self.counters.transcribed += 1
        matched = match_alias(heard, self._aliases)
        if matched is None:
            # Debug, not info: most of what a room says is not a wake word, and a
            # log that records every passing sentence is both noise and a diary.
            logger.debug("wake: heard %r, no alias matched", heard)
            return None
        self._last_fire = _monotonic()
        self.counters.fired += 1
        logger.info(
            "wake: heard %r, matched alias %r (speech confidence %.2f)",
            heard,
            matched,
            confidence,
        )
        return WakeEvent(heard=heard, matched=matched, confidence=confidence)

    def _available(self) -> bool:
        try:
            return bool(self._recognizer.available)
        except Exception as exc:
            self._note("wake: could not tell whether the recognizer is available", exc)
            return False

    def _probability(self, frame: bytes) -> float:
        try:
            return float(self._vad.probability(frame))
        except Exception as exc:
            self._note("wake: the VAD failed on a frame, treating it as silence", exc)
            return 0.0

    def _reset_vad(self) -> None:
        try:
            self._vad.reset()
        except Exception as exc:
            self._note("wake: the VAD refused to reset", exc)

    def _note(self, message: str, exc: BaseException) -> None:
        self.counters.errors += 1
        if self.counters.errors % ERROR_LOG_EVERY == 1:
            logger.warning("%s (%d so far): %s", message, self.counters.errors, exc)
