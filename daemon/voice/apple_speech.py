"""On-device speech recognition through macOS `SFSpeechRecognizer` - the
`SpeechRecognizer` half of daemon/voice/base.py.

This exists so the wake gate costs nothing. A hosted `VoiceSession` bills per
minute, so holding one open on the chance of being spoken to costs about 48x what
30 minutes a day costs (docs/PLAN.md 6.5); something free has to decide when to
open it. `requiresOnDeviceRecognition` is what makes this that something: no
network, no key, no per-minute charge.

`Foundation`, `Speech` and `AVFoundation` are imported lazily, never at module
scope, for the same reason `sounddevice` is in daemon/voice/audio.py: they are
pyobjc, they exist only on macOS, and a text-only install must still be able to
import this module. On any other platform the import fails, `available` is False,
and the daemon simply has no Apple recognizer.

Measured on the target machine (Apple Silicon, macOS 26, Python 3.13, pyobjc
12.2.1), because several of these decide the shape of the code rather than
decorate it:

* **Never call `SFSpeechRecognizer.requestAuthorization_`.** It SIGABRTs the
  process (exit 134): a plain Python process has no
  `NSSpeechRecognitionUsageDescription` in a bundle Info.plist, so TCC kills it
  when it cannot show the prompt. Recognition works anyway at
  `authorizationStatus() == 0` (notDetermined), which is why nothing here asks.
  `tests/test_apple_speech.py` asserts the call is never made, because the
  failure is a crash rather than an error.
* **The result handler is delivered to `recognizer.queue`, whose default is the
  *main* queue.** That is the one thing that does not survive being moved off the
  main thread: with the default queue, driving recognition from an
  `asyncio.to_thread` worker produced **0 callbacks** in 8 s while
  `NSRunLoop.currentRunLoop().runUntilDate_` returned instantly 1.38 million
  times - a busy-spun core and no transcript, because a worker thread's run loop
  never drains the main queue. Handing the recognizer its own `NSOperationQueue`
  fixes it and removes the need for a CFRunLoop pump entirely: the callback
  arrives on the queue's own thread and a `threading.Event` is enough.
* **Two concurrent recognitions on one recognizer silently lose one.** Run
  together, one segment transcribed and the other returned `""` in 54 ms; run
  one after the other, the same audio returns `'오늘 날씨가 참 좋네요'`. Hence
  `_serialise`, and hence its living in the worker rather than the event loop -
  see `transcribe`.
* **Latency is warm-up, not steady state.** First recognition in a process 6.6 s,
  then 1.9 s, then 0.4-1.1 s for the same 2.2 s clip. So the timeout is generous:
  a bound tight enough for the warm case would drop precisely the first wake word
  after the daemon starts, and drop it silently.
* **`setContextualStrings_` does not help a coined wake word.** With
  `["데몬", "헤이 데몬", "데몬아"]` the output was byte-identical to baseline in
  every case, so it is not implemented here and must not be presented as a fix.
  What actually works is measuring what this recognizer returns for a given
  speaker and matching on that (`WakeEvent.heard` vs `.matched` in
  daemon/voice/base.py, and tests/fixtures/wake/README.md).
* Transcripts come back as precomposed NFC (`'회의들은...'`, 15 codepoints, not
  30), so nothing here normalises. They arrive as `pyobjc_unicode`, a `str`
  subclass that keeps its Objective-C object alive, so they are copied into a
  plain `str` before leaving.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NO_SPEECH_CODE = "Code=1110"
"""`kAFAssistantErrorDomain` 1110, "No speech detected".

Matched on the code rather than the message because the message is localised. This
is the ordinary answer for a segment the VAD accepted and a person did not speak in,
so it is counted and logged at DEBUG - see `transcribe`."""

SAMPLE_RATE = 16_000
"""What the audio is assumed to be: 16 kHz mono 16-bit little-endian PCM, which is
`AudioIO.sample_rate` in daemon/voice/audio.py, so a segment can be handed over
with no resampling."""

DEFAULT_LOCALE = "ko-KR"
"""The product is Korean-first; the class is not, hence a constructor argument."""

CHUNK_FRAMES = 1600
"""100 ms per buffer. The measured-working shape, and small enough that a long
segment is never one huge allocation."""

TRANSCRIBE_TIMEOUT_SECONDS = 10.0
"""Long, deliberately. The first recognition in a process measured 6.6 s against
0.4-1.1 s warm, so a bound sized for the warm case would silently drop the first
wake word after startup - and a wake gate that loses the first attempt reads as a
dead daemon."""

_FULL_SCALE = 32768.0
"""int16 -> float32 in [-1, 1). AVAudioPCMFormatFloat32 is what the recognizer
takes; the microphone gives int16."""


@dataclass(frozen=True, slots=True)
class Frameworks:
    """The three pyobjc modules, as one injectable thing.

    A seam for the same reason `SoundDeviceAudio(backend=...)` has one: CONTRACTS
    forbids a test that touches hardware or an OS service, and CI is Linux where
    none of these import at all.
    """

    foundation: Any
    speech: Any
    avfoundation: Any


class AppleSpeechRecognizer:
    """Implements the `SpeechRecognizer` protocol in daemon/voice/base.py."""

    def __init__(
        self,
        *,
        locale: str = DEFAULT_LOCALE,
        frameworks: Frameworks | None = None,
        timeout: float = TRANSCRIBE_TIMEOUT_SECONDS,
    ) -> None:
        self._locale = locale
        # None means "import the real frameworks on first use"; tests pass a
        # stand-in so no OS speech service is touched.
        self._frameworks = frameworks
        self._timeout = timeout
        self._recognizer: Any = None
        self.declined = 0
        """Segments the recognizer said contained no speech.

        Reported rather than silent: a gate whose every segment is declined is
        indistinguishable from a quiet room, and the two need different actions -
        one is a misaimed microphone, the other is nothing at all."""
        # A threading lock, not an asyncio one, and held inside the worker rather
        # than around the await. `to_thread` cannot be cancelled (see
        # daemon/voice/audio.py), so a cancelled `transcribe` leaves its thread
        # running: an asyncio lock would be released at the cancel and let the
        # next call overlap the abandoned one, which is measured to make one of
        # the two return "" with nothing logged.
        self._serialise = threading.Lock()

    @property
    def available(self) -> bool:
        """Whether this recognizer can answer at all, asking nothing of the user.

        Cheap after the first call - 160 ms once while the framework loads, then
        0.5-1 ms - because the recognizer is built once and kept.

        Deliberately does not consult `authorizationStatus()`: notDetermined
        recognizes perfectly well, and the only call that would change it aborts
        the process (see the module docstring). Deliberately catches everything:
        absence is normal here (daemon/voice/base.py), and an unavailable
        recognizer must read as False rather than take down the caller's loop.
        """
        try:
            recognizer = self._build()
            return (
                recognizer is not None
                and bool(recognizer.isAvailable())
                and bool(recognizer.supportsOnDeviceRecognition())
            )
        except Exception:
            logger.debug("apple speech: no on-device recognizer here", exc_info=True)
            return False

    async def transcribe(self, pcm: bytes) -> str:
        """Best guess at the words in one segment, or `""` if it cannot say.

        The pyobjc work is blocking and runs to a `threading.Event`, so it goes to
        a thread: this runs inside `daemon run`, where blocking the event loop
        would stall the Telegram poll and the scheduler with it. Only the blocking
        call moves - every Objective-C object is created by the code that uses it,
        which is the mistake daemon/reflection.py made in the other direction with
        a sqlite connection.
        """
        if not pcm:
            return ""
        try:
            return await asyncio.to_thread(self._transcribe, pcm)
        except Exception:
            # The protocol forbids raising for ordinary failure: this is called
            # from a loop that has to outlive an unavailable recognizer. Logged
            # loudly all the same, because a wake gate that hears nothing looks
            # exactly like a quiet room.
            logger.exception("apple speech: transcription failed")
            return ""

    # --- everything below runs on the worker thread ---------------------------

    def _transcribe(self, pcm: bytes) -> str:
        with self._serialise:
            frameworks = self._load()
            recognizer = self._build()
            if recognizer is None:
                return ""

            request = frameworks.speech.SFSpeechAudioBufferRecognitionRequest.alloc().init()
            # The whole reason this class is allowed to exist: no network, no key,
            # no per-minute bill.
            request.setRequiresOnDeviceRecognition_(True)
            # Nothing here consumes a partial - the gate is asked about a segment
            # that has already ended - and off it is one callback instead of ten.
            request.setShouldReportPartialResults_(False)

            done = threading.Event()
            heard: list[str] = []
            failure: list[Any] = []

            def on_result(result: Any, error: Any) -> None:
                # Runs on the recognizer's own operation queue, on its thread.
                if error is not None:
                    failure.append(error)
                    done.set()
                    return
                if result is None:
                    return
                if result.isFinal():
                    # Copied out of `pyobjc_unicode`, which keeps an Objective-C
                    # object alive for as long as the string is referenced.
                    heard.append(str(result.bestTranscription().formattedString()))
                    done.set()

            task = recognizer.recognitionTaskWithRequest_resultHandler_(request, on_result)
            for buffer in self._buffers(frameworks, pcm):
                request.appendAudioPCMBuffer_(buffer)
            request.endAudio()

            if not done.wait(self._timeout):
                # Measured safe, and necessary: without it the abandoned task
                # keeps the engine busy. The recognizer stays usable afterwards.
                if task is not None:
                    task.cancel()
                logger.warning(
                    "apple speech: no result within %.1fs for %.2fs of audio",
                    self._timeout,
                    len(pcm) / 2 / SAMPLE_RATE,
                )
                return ""
            if failure:
                detail = str(failure[0])
                if NO_SPEECH_CODE in detail:
                    # Not a failure. The VAD hands us anything with enough energy -
                    # measured, it calls a chord with vibrato speech in 46.8% of
                    # frames - and this is the second gate declining, which is the
                    # whole reason there is a second gate.
                    #
                    # At WARNING it was 15 lines in 52 seconds in a quiet room, so
                    # about 24,000 a day for a process that is meant to run all day,
                    # and the one line that mattered would be buried in them. Only
                    # found by running the resident gate; no test noticed, because a
                    # test does not sit in a room.
                    self.declined += 1
                    logger.debug("apple speech: nothing to hear in this segment")
                    return ""
                logger.warning("apple speech: the recognizer refused this segment: %s", detail)
                return ""
            return heard[0] if heard else ""

    def _build(self) -> Any:
        """The recognizer, built once. None when this machine cannot provide one.

        Safe to build here and use from a worker thread - measured: a recognizer
        constructed on the main thread transcribed correctly from inside
        `to_thread`. What is *not* portable across threads is the result handler's
        queue, which is why one is set here explicitly.
        """
        if self._recognizer is None:
            frameworks = self._load()
            locale = frameworks.foundation.NSLocale.localeWithLocaleIdentifier_(self._locale)
            recognizer = frameworks.speech.SFSpeechRecognizer.alloc().initWithLocale_(locale)
            if recognizer is None:
                # How an unsupported locale reports itself: nil, plus a line on
                # stderr from the framework. Not an exception.
                logger.debug("apple speech: no recognizer for locale %s", self._locale)
                return None
            # The fix for the 0-callbacks-off-the-main-thread failure in the
            # module docstring. Serial, because the handler appends to a list.
            queue = frameworks.foundation.NSOperationQueue.alloc().init()
            queue.setMaxConcurrentOperationCount_(1)
            recognizer.setQueue_(queue)
            self._recognizer = recognizer
        return self._recognizer

    def _buffers(self, frameworks: Frameworks, pcm: bytes) -> Iterator[Any]:
        """The segment as `AVAudioPCMBuffer`s of float32 samples."""
        av = frameworks.avfoundation
        fmt = av.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
            av.AVAudioPCMFormatFloat32, float(SAMPLE_RATE), 1, False
        )
        # Raises on a byte count that is not a whole number of int16 samples,
        # which `transcribe` turns into "" and a logged traceback. Truncating
        # instead would accept malformed audio silently.
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / _FULL_SCALE
        for start in range(0, len(samples), CHUNK_FRAMES):
            window = samples[start : start + CHUNK_FRAMES]
            frames = len(window)
            buffer = av.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(fmt, frames)
            buffer.setFrameLength_(frames)
            buffer.floatChannelData()[0][0:frames] = window
            yield buffer

    def _load(self) -> Frameworks:
        if self._frameworks is None:
            import AVFoundation
            import Foundation
            import Speech

            self._frameworks = Frameworks(
                foundation=Foundation, speech=Speech, avfoundation=AVFoundation
            )
        return self._frameworks
