"""Apple on-device speech recognition, with no speech service and no macOS.

Everything here runs against a fake `Foundation` / `Speech` / `AVFoundation`, for
two reasons that are not negotiable: CONTRACTS forbids a test that touches
hardware or an OS service, and CI is Linux where none of those three modules
import at all.

The first test is the one that protects the others - importing this module must
work on a machine with no pyobjc, or "voice is off" becomes "the daemon does not
start".

Two of these exist because the real thing was measured failing, not because a
branch was uncovered:

* `test_request_authorization_is_never_called` - `requestAuthorization_` SIGABRTs
  a non-bundled Python process (exit 134). A crash is not something to discover
  from a log.
* `test_concurrent_transcribes_do_not_overlap` - two recognitions at once on one
  recognizer made one of them return `""` in 54 ms with nothing raised, which is
  the silent-degradation shape this repo keeps getting caught by.
"""

from __future__ import annotations

import asyncio
import builtins
import logging
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from daemon.voice.apple_speech import (
    CHUNK_FRAMES,
    DEFAULT_LOCALE,
    AppleSpeechRecognizer,
    Frameworks,
)
from daemon.voice.base import SpeechRecognizer

KOREAN = "회의들은 지금 뭐 하고 있어"
"""What the real recognizer returns for tests/fixtures/wake/wake-and-question.wav.
Not the words that were spoken - an on-device recognizer will not emit a coined
name (tests/fixtures/wake/README.md) - which is exactly why the text has to
survive this path byte for byte."""


def pcm(*samples: int) -> bytes:
    """16-bit little-endian mono PCM, the way `AudioIO` captures it."""
    return struct.pack(f"<{len(samples)}h", *samples)


SPEECH = pcm(*([1000, -1000] * 800))
"""3200 bytes = 1600 samples = one whole 100 ms chunk."""


# --- the fake OS -------------------------------------------------------------


class PyObjCString(str):
    """`formattedString()` returns `pyobjc_unicode`, a `str` subclass that keeps
    its Objective-C object alive. The fake is one too, so that copying it out is
    something a test can actually observe."""


class FakeTask:
    def __init__(self) -> None:
        self.cancelled = 0

    def cancel(self) -> None:
        self.cancelled += 1


class FakeTranscription:
    def __init__(self, text: str) -> None:
        self._text = text

    def formattedString(self) -> PyObjCString:  # noqa: N802 - mirrors the framework
        return PyObjCString(self._text)


class FakeResult:
    def __init__(self, text: str, *, final: bool) -> None:
        self._text = text
        self._final = final

    def bestTranscription(self) -> FakeTranscription:  # noqa: N802
        return FakeTranscription(self._text)

    def isFinal(self) -> bool:  # noqa: N802
        return self._final


class FakeChannel:
    """`buffer.floatChannelData()[0]`, which the module slice-assigns into."""

    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.written: list[float] = []

    def __setitem__(self, where: Any, values: Any) -> None:
        assert isinstance(where, slice), "the module writes a slice, not an element"
        assert where.start == 0 and where.stop == self.frames
        self.written = [float(v) for v in values]


class FakeBuffer:
    def __init__(self, fmt: FakeFormat, capacity: int) -> None:
        self.fmt = fmt
        self.capacity = capacity
        self.frame_length: int | None = None
        self.channel = FakeChannel(capacity)

    def setFrameLength_(self, frames: int) -> None:  # noqa: N802
        self.frame_length = frames

    def floatChannelData(self) -> list[FakeChannel]:  # noqa: N802
        return [self.channel]


class FakeFormat:
    def __init__(self, common_format: Any, rate: float, channels: int, interleaved: bool) -> None:
        self.common_format = common_format
        self.rate = rate
        self.channels = channels
        self.interleaved = interleaved


class FakeRequest:
    def __init__(self, world: World) -> None:
        self._world = world
        self.on_device: bool | None = None
        self.partials: bool | None = None
        self.buffers: list[FakeBuffer] = []
        self.ended = False
        self._handler: Any = None
        self._task: FakeTask | None = None

    def setRequiresOnDeviceRecognition_(self, value: bool) -> None:  # noqa: N802
        self.on_device = value

    def setShouldReportPartialResults_(self, value: bool) -> None:  # noqa: N802
        self.partials = value

    def setContextualStrings_(self, strings: Any) -> None:  # noqa: N802
        # Present so that adding it would be visible here. Measured to change
        # nothing for a coined wake word, so nothing should call it.
        self._world.contextual.append(list(strings))

    def appendAudioPCMBuffer_(self, buffer: FakeBuffer) -> None:  # noqa: N802
        self.buffers.append(buffer)

    def endAudio(self) -> None:  # noqa: N802
        self.ended = True
        # The real callback arrives on the recognizer's operation queue once the
        # audio ends. Firing it here is what makes that observable.
        self._world.deliver(self._handler, self._task)


class FakeRecognizer:
    def __init__(self, world: World, locale: str) -> None:
        self._world = world
        self.locale = locale
        self.queue: FakeQueue | None = None
        self.tasks: list[FakeTask] = []

    def isAvailable(self) -> bool:  # noqa: N802
        return self._world.is_available

    def supportsOnDeviceRecognition(self) -> bool:  # noqa: N802
        return self._world.supports_on_device

    def setQueue_(self, queue: FakeQueue) -> None:  # noqa: N802
        self.queue = queue

    def recognitionTaskWithRequest_resultHandler_(  # noqa: N802
        self, request: FakeRequest, handler: Any
    ) -> FakeTask:
        task = FakeTask()
        request._handler = handler
        request._task = task
        self.tasks.append(task)
        self._world.requests.append(request)
        return task


class FakeQueue:
    def __init__(self) -> None:
        self.max_concurrent: int | None = None

    def setMaxConcurrentOperationCount_(self, count: int) -> None:  # noqa: N802
        self.max_concurrent = count


class _Alloc:
    """pyobjc's two-step `Klass.alloc().initWith...()`."""

    def __init__(self, **initialisers: Any) -> None:
        for name, function in initialisers.items():
            setattr(self, name, function)


class World:
    """One fake macOS, with the knobs the failure paths need."""

    def __init__(self) -> None:
        self.text = KOREAN
        self.is_available = True
        self.supports_on_device = True
        self.locale_supported = True
        self.error: Any = None
        self.silent = False
        """Deliver no callback at all - the timeout path."""
        self.gate: threading.Event | None = None
        """Block inside the blocking section until the test releases it."""
        self.entered = threading.Event()
        self.recognizers: list[FakeRecognizer] = []
        self.requests: list[FakeRequest] = []
        self.forbidden: list[str] = []
        """Calls that abort the real process. Must stay empty."""
        self.contextual: list[list[str]] = []
        self.handler_threads: list[threading.Thread] = []
        self.active = 0
        self.max_active = 0
        self._counter = threading.Lock()

    # -- what the module is allowed to see

    def deliver(self, handler: Any, task: FakeTask | None) -> None:
        with self._counter:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.entered.set()
            if self.gate is not None:
                assert self.gate.wait(5), "the test never released the gate"
            self.handler_threads.append(threading.current_thread())
            if self.silent:
                return
            if self.error is not None:
                handler(None, self.error)
                return
            handler(FakeResult(self.text, final=True), None)
        finally:
            with self._counter:
                self.active -= 1

    def _new_recognizer(self, locale: str) -> FakeRecognizer | None:
        if not self.locale_supported:
            return None  # how the real framework reports an unknown locale
        recognizer = FakeRecognizer(self, locale)
        self.recognizers.append(recognizer)
        return recognizer

    @property
    def frameworks(self) -> Frameworks:
        world = self

        class SFSpeechRecognizer:
            @staticmethod
            def alloc() -> _Alloc:
                return _Alloc(initWithLocale_=world._new_recognizer)

            @staticmethod
            def authorizationStatus() -> int:  # noqa: N802
                return 0

            @staticmethod
            def requestAuthorization_(handler: Any) -> None:  # noqa: N802
                world.forbidden.append("requestAuthorization_")
                raise AssertionError("requestAuthorization_ SIGABRTs the real process")

            @staticmethod
            def supportedLocales() -> list[str]:  # noqa: N802
                return [DEFAULT_LOCALE]

        class SFSpeechAudioBufferRecognitionRequest:
            @staticmethod
            def alloc() -> _Alloc:
                return _Alloc(init=lambda: FakeRequest(world))

        class NSLocale:
            @staticmethod
            def localeWithLocaleIdentifier_(identifier: str) -> str:  # noqa: N802
                return identifier

        class NSOperationQueue:
            @staticmethod
            def alloc() -> _Alloc:
                return _Alloc(init=FakeQueue)

        class AVAudioFormat:
            @staticmethod
            def alloc() -> _Alloc:
                return _Alloc(
                    initWithCommonFormat_sampleRate_channels_interleaved_=(
                        lambda *args: FakeFormat(*args)
                    )
                )

        class AVAudioPCMBuffer:
            @staticmethod
            def alloc() -> _Alloc:
                return _Alloc(initWithPCMFormat_frameCapacity_=FakeBuffer)

        speech = type(
            "Speech",
            (),
            {
                "SFSpeechRecognizer": SFSpeechRecognizer,
                "SFSpeechAudioBufferRecognitionRequest": SFSpeechAudioBufferRecognitionRequest,
            },
        )
        foundation = type(
            "Foundation", (), {"NSLocale": NSLocale, "NSOperationQueue": NSOperationQueue}
        )
        avfoundation = type(
            "AVFoundation",
            (),
            {
                "AVAudioFormat": AVAudioFormat,
                "AVAudioPCMBuffer": AVAudioPCMBuffer,
                "AVAudioPCMFormatFloat32": 1,
            },
        )
        return Frameworks(foundation=foundation, speech=speech, avfoundation=avfoundation)


@pytest.fixture
def world() -> World:
    return World()


def recognizer(world: World, **kwargs: Any) -> AppleSpeechRecognizer:
    return AppleSpeechRecognizer(frameworks=world.frameworks, **kwargs)


# --- the lazy import ---------------------------------------------------------


BLOCK_PYOBJC = """
import sys
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name in ("Foundation", "Speech", "AVFoundation"):
            raise ImportError(f"No module named {name!r}")
        return None
sys.meta_path.insert(0, Blocker())
import asyncio
from daemon.voice import apple_speech
one = apple_speech.AppleSpeechRecognizer()
print(apple_speech.SAMPLE_RATE)
print(one.available)
print(repr(asyncio.run(one.transcribe(b"\\x01\\x00" * 16))))
"""


def test_importing_the_module_does_not_need_pyobjc() -> None:
    """CI is Linux and a text-only install has no pyobjc. If importing this
    needed it, "no Apple recognizer" would become "the daemon does not start".

    A subprocess with the imports blocked at the meta path, because faking them
    in-process would keep passing if someone moved the import to module scope.
    """
    done = subprocess.run(
        [sys.executable, "-c", BLOCK_PYOBJC],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["16000", "False", "''"]


# --- available ---------------------------------------------------------------


def test_satisfies_the_speech_recognizer_protocol(world: World) -> None:
    assert isinstance(recognizer(world), SpeechRecognizer)


def test_available_is_false_when_pyobjc_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("Foundation", "Speech", "AVFoundation"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert AppleSpeechRecognizer().available is False


def test_available_is_false_when_the_service_is_not_available(world: World) -> None:
    world.is_available = False
    assert recognizer(world).available is False


def test_available_is_false_without_on_device_support(world: World) -> None:
    """On-device is the entire justification: it is what makes the wake gate free
    and offline. A recognizer that would go to the network is no use here."""
    world.supports_on_device = False
    assert recognizer(world).available is False


def test_available_is_false_when_the_locale_has_no_recognizer(world: World) -> None:
    """A machine without ko-KR reports it as a nil recognizer, not an exception."""
    world.locale_supported = False
    assert recognizer(world).available is False


def test_available_is_true_and_builds_the_recognizer_once(world: World) -> None:
    """It is a property on the event loop's thread; 160 ms of framework load is
    tolerable once, per call is not."""
    subject = recognizer(world)
    assert subject.available is True
    assert subject.available is True
    assert len(world.recognizers) == 1


def test_the_recognizer_gets_its_own_operation_queue(world: World) -> None:
    """The measured failure this whole design turns on: the default queue is the
    *main* queue, and a worker thread's run loop never drains it - 0 callbacks in
    8 s. A private queue is what makes an answer arrive at all."""
    assert recognizer(world).available is True
    (built,) = world.recognizers
    assert built.queue is not None, "left on the main queue, so nothing will call back"
    assert built.queue.max_concurrent == 1


# --- transcribe --------------------------------------------------------------


async def test_transcribe_returns_the_final_korean_text(world: World) -> None:
    heard = await recognizer(world).transcribe(SPEECH)

    assert heard == KOREAN
    assert type(heard) is str, "a pyobjc_unicode escaped, holding an ObjC object alive"
    (request,) = world.requests
    assert request.on_device is True, "this would bill per minute and need a network"
    assert request.partials is False
    assert request.ended is True
    assert world.contextual == [], "measured to change nothing for a coined wake word"


async def test_the_locale_defaults_to_korean_and_can_be_overridden(world: World) -> None:
    await recognizer(world).transcribe(SPEECH)
    assert world.recognizers[-1].locale == "ko-KR"

    other = World()
    await recognizer(other, locale="en-US").transcribe(SPEECH)
    assert other.recognizers[-1].locale == "en-US"


async def test_transcribe_returns_empty_on_a_recognizer_error(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    """The protocol forbids raising for ordinary failure: this is called from a
    loop that has to outlive an unavailable recognizer."""
    world.error = object()
    heard = await recognizer(world).transcribe(SPEECH)

    assert heard == ""
    assert "refused" in caplog.text, "the segment vanished with nothing said about it"


async def test_a_segment_with_no_speech_in_it_is_not_a_warning(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    """Found by running the resident gate, not by any test here.

    The VAD hands over anything with enough energy - it calls a chord with vibrato
    speech in 46.8% of frames - so "no speech detected" is the second gate declining,
    which is the entire reason there is a second gate. At WARNING it was 15 lines in
    52 seconds in a quiet room: about 24,000 a day for a process meant to run all
    day, with the one line that mattered buried among them.
    """
    world.error = "Error Domain=kAFAssistantErrorDomain Code=1110 \"No speech detected\""
    subject = recognizer(world)

    with caplog.at_level(logging.WARNING):
        heard = await subject.transcribe(SPEECH)

    assert heard == ""
    assert caplog.text == "", f"a quiet room logged a warning: {caplog.text!r}"
    # Counted instead, because "every segment declined" and "a quiet room" look
    # identical from outside and need different actions.
    assert subject.declined == 1


async def test_an_unexpected_refusal_is_still_a_warning(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    """Only 1110 is ordinary. Silencing the rest would hide a real breakage behind
    the fix for the noisy one."""
    world.error = "Error Domain=kAFAssistantErrorDomain Code=203 \"Retry\""
    subject = recognizer(world)

    with caplog.at_level(logging.WARNING):
        assert await subject.transcribe(SPEECH) == ""

    assert "refused" in caplog.text
    assert subject.declined == 0


async def test_transcribe_returns_empty_when_it_times_out(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    world.silent = True
    heard = await recognizer(world, timeout=0.05).transcribe(SPEECH)

    assert heard == ""
    # Without this the abandoned task keeps the recognition engine busy.
    assert [task.cancelled for task in world.recognizers[0].tasks] == [1]
    assert "no result" in caplog.text


async def test_transcribe_returns_empty_when_there_is_no_recognizer(world: World) -> None:
    world.locale_supported = False
    assert await recognizer(world).transcribe(SPEECH) == ""


async def test_empty_audio_asks_the_os_nothing(world: World) -> None:
    assert await recognizer(world).transcribe(b"") == ""
    assert world.requests == []


async def test_malformed_audio_is_reported_rather_than_raised(
    world: World, caplog: pytest.LogCaptureFixture
) -> None:
    """An odd byte count is not a whole number of int16 samples. Truncating it
    would accept broken audio silently; raising would kill the wake loop."""
    assert await recognizer(world).transcribe(b"\x01\x02\x03") == ""
    assert "transcription failed" in caplog.text


# --- the audio conversion ----------------------------------------------------


async def test_the_pcm_becomes_float32_at_the_right_scale_and_byte_order(
    world: World,
) -> None:
    """Where a byte-order or scaling bug would live, and it would present as the
    recognizer simply mishearing everything."""
    await recognizer(world).transcribe(pcm(0, 16384, -16384, 32767, -32768))

    (request,) = world.requests
    (buffer,) = request.buffers
    assert buffer.frame_length == 5
    assert buffer.capacity == 5
    assert buffer.channel.written == pytest.approx(
        [0.0, 0.5, -0.5, 32767 / 32768, -1.0], abs=1e-6
    )
    assert buffer.fmt.rate == 16000.0
    assert buffer.fmt.channels == 1
    assert buffer.fmt.interleaved is False


async def test_long_audio_is_appended_in_100ms_buffers(world: World) -> None:
    frames = CHUNK_FRAMES * 2 + 7
    await recognizer(world).transcribe(pcm(*([256] * frames)))

    (request,) = world.requests
    assert [buffer.frame_length for buffer in request.buffers] == [
        CHUNK_FRAMES,
        CHUNK_FRAMES,
        7,
    ]


# --- the two crashes ---------------------------------------------------------


async def test_request_authorization_is_never_called(world: World) -> None:
    """`requestAuthorization_` SIGABRTs a non-bundled Python process (exit 134),
    because TCC kills a process that cannot show the prompt. Recognition works at
    notDetermined, so nothing may ever ask - and a crash is not a thing to find
    out about from a log."""
    subject = recognizer(world)
    assert subject.available is True
    assert await subject.transcribe(SPEECH) == KOREAN

    assert world.forbidden == []


async def test_concurrent_transcribes_do_not_overlap(world: World) -> None:
    """Measured: two recognitions at once on one recognizer made one of them
    return "" in 54 ms, raising nothing, where the same audio one-at-a-time
    transcribed fine."""
    world.gate = threading.Event()
    subject = recognizer(world)
    jobs = [asyncio.create_task(subject.transcribe(SPEECH)) for _ in range(3)]

    async with asyncio.timeout(5):
        assert await asyncio.to_thread(world.entered.wait, 5), "nothing reached the recognizer"
        world.gate.set()
        assert await asyncio.gather(*jobs) == [KOREAN] * 3

    assert world.max_active == 1, "two recognitions ran at once; one of them loses"


# --- the event loop ----------------------------------------------------------


async def test_the_event_loop_keeps_running_while_a_transcribe_blocks(world: World) -> None:
    """This runs inside `daemon run`. Pumping the blocking pyobjc work on the
    event loop would stall the Telegram poll and the scheduler with it."""
    world.gate = threading.Event()
    subject = recognizer(world)
    job = asyncio.create_task(subject.transcribe(SPEECH))

    ticks = 0
    async with asyncio.timeout(5):
        # Reaching this at all proves the loop still runs: the worker is inside
        # the blocking section, and only the loop can release it.
        while not world.entered.is_set():
            await asyncio.sleep(0)
            ticks += 1
        world.gate.set()
        assert await job == KOREAN

    assert ticks > 0, "the loop never got a turn while the worker was blocked"
    assert world.handler_threads, "the recognizer was never driven"
    assert threading.main_thread() not in world.handler_threads, (
        "the pyobjc work ran on the event loop's own thread"
    )
