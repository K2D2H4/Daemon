"""Audio tests. No hardware, no PortAudio, no sound.

A fake sounddevice module stands in for the real one, and the local speaker runs
against a scripted subprocess rather than /usr/bin/say - so this file passes on a
build machine with no audio device and no `voice` extra installed.

The first test is the one that protects everyone else: importing this module must
work in a text-only install, which is why sounddevice is imported lazily.
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from daemon import mic_hold
from daemon.voice import audio
from daemon.voice.audio import AudioUnavailable, SoundDeviceAudio
from daemon.voice.base import AudioIO


class FakeStream:
    """Stands in for sounddevice's RawInputStream / RawOutputStream.

    Writes are announced on a queue as well as recorded, so tests can wait for
    playback to reach the device instead of sleeping and hoping: each chunk goes
    out through a worker thread, and counting event-loop turns would make these
    tests pass for the wrong reason.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.active = False
        self.written: list[bytes] = []
        self.aborted = 0
        self.closed = False
        self.closed_event = threading.Event()
        """Set by `close`, so a test can wait for record()'s detached release thread
        without polling - the close no longer happens on the event loop."""
        self.callback = kwargs.get("callback")
        self.arrivals: asyncio.Queue[bytes] = asyncio.Queue()
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            # Input streams are opened on a worker thread now (record() offloads the
            # blocking Pa_OpenStream off the loop), so construction can happen with no
            # running loop. Only the output path uses `_loop`, and those are still
            # constructed on the loop, so None here is harmless for a microphone stream.
            self._loop = None  # type: ignore[assignment]

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False

    def write(self, chunk: bytes) -> None:
        # Called on a worker thread, hence call_soon_threadsafe.
        self.written.append(bytes(chunk))
        self._loop.call_soon_threadsafe(self.arrivals.put_nowait, bytes(chunk))

    def abort(self) -> None:
        self.aborted += 1
        self.active = False

    def close(self) -> None:
        self.closed = True
        self.closed_event.set()

    def feed(self, pcm: bytes) -> None:
        """Deliver a block the way PortAudio's own thread would."""
        assert self.callback is not None
        self.callback(memoryview(pcm), len(pcm) // 2, None, None)

    async def heard(self, count: int) -> list[bytes]:
        async with asyncio.timeout(5):
            return [await self.arrivals.get() for _ in range(count)]


class FakeSoundDevice:
    def __init__(self) -> None:
        self.inputs: list[FakeStream] = []
        self.outputs: list[FakeStream] = []
        self.opened: asyncio.Queue[FakeStream] = asyncio.Queue()

    def RawInputStream(self, **kwargs: Any) -> FakeStream:  # noqa: N802 - mirrors sounddevice
        stream = FakeStream(**kwargs)
        self.inputs.append(stream)
        return stream

    def RawOutputStream(self, **kwargs: Any) -> FakeStream:  # noqa: N802 - mirrors sounddevice
        stream = FakeStream(**kwargs)
        self.outputs.append(stream)
        self.opened.put_nowait(stream)
        return stream

    async def speaker(self) -> FakeStream:
        """The output stream, once the writer task has opened it."""
        async with asyncio.timeout(5):
            return await self.opened.get()


@pytest.fixture
def backend() -> FakeSoundDevice:
    return FakeSoundDevice()


async def settle() -> None:
    """Give the writer a chance to do something, for the assertions that it did
    not."""
    await asyncio.sleep(0.05)


async def released(stream: FakeStream) -> None:
    """Wait for `record`'s detached release thread to stop and close the stream.

    The stop runs off the event loop now (it can deadlock inside CoreAudio), so
    `aclose()` returns before the device is actually closed. Waited on a worker
    thread rather than polled, and bounded, because tests/CLAUDE.md forbids a test
    that can hang."""
    got = await asyncio.to_thread(stream.closed_event.wait, 5.0)
    assert got, "record()'s release thread did not close the stream within 5s"


# --- the lazy import ---------------------------------------------------------


BLOCK_SOUNDDEVICE = """
import sys
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "sounddevice":
            raise ImportError("No module named 'sounddevice'")
        return None
sys.meta_path.insert(0, Blocker())
import daemon.voice.audio as audio
print(audio.INPUT_SAMPLE_RATE)
"""


def test_importing_the_module_does_not_need_sounddevice() -> None:
    """A text-only install has no PortAudio. If importing this module needed it,
    "voice is off" would become "the daemon does not start".

    Run in a subprocess with sounddevice blocked at the import hook, because
    faking it in-process would still pass once someone installs the voice extra
    and moved the import to the top of the module.
    """
    done = subprocess.run(
        [sys.executable, "-c", BLOCK_SOUNDDEVICE],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "16000"


def _refuse_sounddevice(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sounddevice":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)


def test_a_missing_backend_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _refuse_sounddevice(monkeypatch, ImportError("No module named 'sounddevice'"))
    with pytest.raises(AudioUnavailable, match="pip install -e") as caught:
        audio._sounddevice()

    assert ".[voice]" in str(caught.value)


def test_a_missing_portaudio_library_is_reported_as_such(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wheel installs fine and then fails at import when the shared library
    is absent - a different fix from `pip install`, so a different message."""
    _refuse_sounddevice(monkeypatch, OSError("PortAudio library not found"))
    with pytest.raises(AudioUnavailable, match="PortAudio"):
        audio._sounddevice()


# --- record and play --------------------------------------------------------


def test_satisfies_audio_io_protocol(backend: FakeSoundDevice) -> None:
    assert isinstance(SoundDeviceAudio(backend=backend), AudioIO)


def test_input_and_output_rates_differ(backend: FakeSoundDevice) -> None:
    """Gemini Live takes 16kHz and returns 24kHz. Playing the model back at the
    microphone's rate is the pitch-shifted-chipmunk bug."""
    io = SoundDeviceAudio(backend=backend)
    assert io.sample_rate == 16_000
    assert io.playback_sample_rate == 24_000


async def test_record_yields_microphone_blocks(backend: FakeSoundDevice) -> None:
    io = SoundDeviceAudio(backend=backend)
    blocks = io.record()

    async def push() -> None:
        await asyncio.sleep(0)
        (stream,) = backend.inputs
        stream.feed(b"\x01\x02")
        stream.feed(b"\x03\x04")

    task = asyncio.create_task(push())
    async with asyncio.timeout(5):
        assert await blocks.__anext__() == b"\x01\x02"
        assert await blocks.__anext__() == b"\x03\x04"
    await task

    (stream,) = backend.inputs
    assert stream.kwargs["samplerate"] == 16_000
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "int16"
    await blocks.aclose()
    await released(stream)  # the stop/close runs on a detached thread now
    assert stream.closed  # a microphone nobody reads is a light left on


async def test_the_microphone_queue_drops_the_oldest_audio_not_the_newest(
    backend: FakeSoundDevice, caplog: pytest.LogCaptureFixture
) -> None:
    """The audit finding: the queue was unbounded, so a consumer that fell behind
    kept sending audio from tens of seconds ago. The session stayed alive and the
    transcripts kept arriving while the conversation answered something the user had
    said half a minute earlier. In real-time audio an old block is worth less than
    nothing, so the oldest go."""
    io = SoundDeviceAudio(backend=backend)
    blocks = io.record()
    pull = asyncio.create_task(anext(blocks))  # starts the generator and the stream
    await settle()

    fed = audio.MIC_QUEUE_BLOCKS + 10
    stream = backend.inputs[0]
    for index in range(fed):
        stream.feed(bytes([index, 0]))

    received = [await pull]
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(0.5):
            while True:
                received.append(await anext(blocks))
    await blocks.aclose()

    assert len(received) <= audio.MIC_QUEUE_BLOCKS + 1, "the bound did not hold"
    assert io.dropped_blocks == fed - len(received)
    assert received[-1] == bytes([fed - 1, 0]), "the newest block was thrown away"
    assert received[0] != bytes([0, 0]), "the oldest block survived instead"
    assert "dropped" in caplog.text, "audio went missing with nothing said about it"


async def test_ending_recording_closes_the_stream(backend: FakeSoundDevice) -> None:
    io = SoundDeviceAudio(backend=backend)
    blocks = io.record()

    async def push() -> None:
        await asyncio.sleep(0)
        backend.inputs[0].feed(b"\x00\x00")

    task = asyncio.create_task(push())
    async with asyncio.timeout(5):
        async for _ in blocks:
            break
    await task
    await blocks.aclose()

    (stream,) = backend.inputs
    await released(stream)  # record()'s finally releases the stream off the loop
    assert not stream.active
    assert stream.closed


async def test_recording_marks_the_microphone_held(backend: FakeSoundDevice) -> None:
    """Without this the presence probe cannot tell the wake listener's own hold
    from somebody else's call, and the local speaker route dies for as long as
    voice is switched on. See daemon/mic_hold.py."""
    assert mic_hold.held() is False
    io = SoundDeviceAudio(backend=backend)
    stream = io.record()

    async def push() -> None:
        await asyncio.sleep(0)
        backend.inputs[0].feed(b"\x00\x00")

    task = asyncio.create_task(push())
    async with asyncio.timeout(5):
        await anext(stream)
    await task
    assert mic_hold.held() is True, "the hold must be visible while recording"

    await stream.aclose()
    assert mic_hold.held() is False


async def test_the_hold_is_released_when_recording_raises(
    backend: FakeSoundDevice, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stream that dies mid-read must not leave the daemon convinced it is on
    a call for the rest of the process's life - that is the failure this whole
    milestone exists to remove, reintroduced by an unbalanced counter."""

    def explode(**kwargs: Any) -> FakeStream:
        # The assertion is the point of this test. Without it the test passes
        # whether or not the hold covers the construction call - and covering it
        # is the whole reason the hold sits where it does: PortAudio can fail
        # while opening the device, and a hold taken after that point would
        # never be entered, never be released, and never be wrong in a way a
        # green suite could see.
        assert mic_hold.held() is True, "the hold must already cover stream construction"
        raise RuntimeError("PortAudio went away")

    monkeypatch.setattr(backend, "RawInputStream", explode)
    io = SoundDeviceAudio(backend=backend)

    with pytest.raises(RuntimeError):
        await anext(io.record())
    assert mic_hold.held() is False


async def test_a_wedged_stream_stop_does_not_freeze_teardown() -> None:
    """The daemon-freeze regression.

    `record`'s finally stopped the input stream synchronously on the event-loop
    thread. One day that stop deadlocked inside CoreAudio - the wake gate's PortAudio
    stream and the session's macOS VoiceProcessing unit contending the same device on
    the wake->voice handover - and because it ran on the loop thread the whole daemon
    froze: no logs, no scheduler, no answer to the wake word, only a `sample` of the
    process showing `__psynch_mutexwait` under `AudioOutputUnitStop`.

    The stop runs on a detached thread now, so a stop that never returns costs one
    parked thread, not the loop. Proof: `aclose()` returns even while the stream's
    stop() is wedged forever - under the old code this await never came back."""
    never_returns = threading.Event()

    class WedgingStream(FakeStream):
        def stop(self) -> None:
            never_returns.wait()  # a deadlocked Pa_StopStream, until the test frees it
            super().stop()

    class WedgingBackend(FakeSoundDevice):
        def RawInputStream(self, **kwargs: Any) -> FakeStream:  # noqa: N802
            stream = WedgingStream(**kwargs)
            self.inputs.append(stream)
            return stream

    backend = WedgingBackend()
    io = SoundDeviceAudio(backend=backend)
    blocks = io.record()

    async def push() -> None:
        await asyncio.sleep(0)
        backend.inputs[0].feed(b"\x00\x00")

    task = asyncio.create_task(push())
    async with asyncio.timeout(5):
        async for _ in blocks:
            break
    await task

    try:
        # The finally runs here and starts the release thread, where stop() wedges.
        # Under the bug this await ran stop() inline and never returned; the timeout
        # is what turns "the daemon froze" into a failing test instead of a hung one.
        async with asyncio.timeout(5):
            await blocks.aclose()
    finally:
        never_returns.set()  # let the parked thread finish so it does not outlive us

    (stream,) = backend.inputs
    await released(stream)  # and once freed, it does stop and close the device
    assert stream.closed


async def test_the_input_release_can_be_waited_for_after_aclose() -> None:
    """The handover the wake gate needs, and the one `aclose()` cannot give it.

    The release runs on a detached thread precisely so a wedged `Pa_StopStream`
    cannot take the loop with it, which means `aclose()` returns while the stop is
    still inside CoreAudio. That is right for a microphone nobody wants back, and
    wrong for the wake->voice handover: the next thing the daemon does there is build
    a macOS VoiceProcessing engine on the same device, and the two clients then take
    CoreAudio's HAL mutex and the AudioUnit's recursive mutex in opposite orders.
    Sampled on the resident (2026-08-26): four threads at `__psynch_mutexwait`, one
    under `Pa_StopStream` and one under `AVAudioIOUnit::IOUnitPropertyListener`,
    wedged until the process was killed - the daemon stayed up and never heard
    another word.

    So the caller needs something to wait on that outlasts the generator. Bounded and
    reported, never raised: a release that will not finish is a device already lost,
    and the caller's job is to know that, not to hang on it."""
    stopping = threading.Event()

    class SlowStream(FakeStream):
        def stop(self) -> None:
            stopping.wait(5.0)  # still inside Pa_StopStream when aclose() returns
            super().stop()

    class SlowBackend(FakeSoundDevice):
        def RawInputStream(self, **kwargs: Any) -> FakeStream:  # noqa: N802
            stream = SlowStream(**kwargs)
            self.inputs.append(stream)
            return stream

    backend = SlowBackend()
    io = SoundDeviceAudio(backend=backend)
    blocks = io.record()

    async def push() -> None:
        await asyncio.sleep(0)
        backend.inputs[0].feed(b"\x00\x00")

    task = asyncio.create_task(push())
    async with asyncio.timeout(5):
        async for _ in blocks:
            break
    await task
    async with asyncio.timeout(5):
        await blocks.aclose()

    (stream,) = backend.inputs
    assert not stream.closed, "the premise: aclose() returns before the device is let go"

    waiting = asyncio.create_task(io.wait_for_input_release())
    await asyncio.sleep(0)
    assert not waiting.done(), "the wait returned while the stop was still running"

    stopping.set()
    async with asyncio.timeout(5):
        assert await waiting is True
    assert stream.closed, "the wait came back before the device was released"


async def test_waiting_for_a_wedged_input_release_gives_up_and_says_so() -> None:
    """A stop that never returns must cost the caller a bounded wait and a `False`,
    not the round. The handover is better off knowing the device is gone than
    hanging on it - and better off than today's alternative, which was to open a
    second CoreAudio client on top of it."""
    never_returns = threading.Event()

    class WedgingStream(FakeStream):
        def stop(self) -> None:
            never_returns.wait()
            super().stop()

    class WedgingBackend(FakeSoundDevice):
        def RawInputStream(self, **kwargs: Any) -> FakeStream:  # noqa: N802
            stream = WedgingStream(**kwargs)
            self.inputs.append(stream)
            return stream

    backend = WedgingBackend()
    io = SoundDeviceAudio(backend=backend)
    blocks = io.record()

    async def push() -> None:
        await asyncio.sleep(0)
        backend.inputs[0].feed(b"\x00\x00")

    task = asyncio.create_task(push())
    async with asyncio.timeout(5):
        async for _ in blocks:
            break
    await task
    await blocks.aclose()

    try:
        async with asyncio.timeout(5):
            assert await io.wait_for_input_release(within=0.05) is False
    finally:
        never_returns.set()  # let the parked thread finish so it does not outlive us


async def test_waiting_with_nothing_to_release_is_free() -> None:
    """The ordinary case - no stream was ever opened, or the last one is long gone -
    must not cost a thread hop on the handover's latency path."""
    io = SoundDeviceAudio(backend=FakeSoundDevice())
    assert await io.wait_for_input_release() is True


async def test_a_wedged_stream_open_does_not_freeze_the_loop() -> None:
    """The other face of the daemon-freeze regression - the one v0.1.45 did not cover.

    `record` opened its PortAudio input stream synchronously on the event-loop thread:
    `sd.RawInputStream(...)` is `Pa_OpenStream`, and start() is `Pa_StartStream`. On the
    wake gate's dead-stream rebuild that open deadlocked inside CoreAudio the same way the
    stop once had - the just-dropped stream's detached release thread holding the device's
    HAL mutex while the fresh open waited on it - and because the open ran on the loop
    thread it froze the whole daemon for eleven hours: no logs, no scheduler, no answer to
    the wake word. A `sample` of the resident showed `__psynch_mutexwait` under
    `Pa_OpenStream` on the uvloop thread while a `voice-mic-release` thread sat under
    `AudioOutputUnitStop`. v0.1.45 moved the stop off the loop; this is the open.

    The open runs on a worker thread now, so an open that never returns costs one parked
    thread, not the loop. Proof: a heartbeat keeps ticking on the loop while the stream's
    open is wedged forever - under the old code the loop never turned again and this test
    hangs instead of passing."""
    never_returns = threading.Event()

    class WedgingStream(FakeStream):
        def start(self) -> None:
            never_returns.wait()  # a deadlocked Pa_StartStream, until the test frees it
            super().start()

    class WedgingBackend(FakeSoundDevice):
        def RawInputStream(self, **kwargs: Any) -> FakeStream:  # noqa: N802
            stream = WedgingStream(**kwargs)
            self.inputs.append(stream)
            return stream

    backend = WedgingBackend()
    io = SoundDeviceAudio(backend=backend)
    blocks = io.record()
    opening = asyncio.create_task(anext(blocks))  # drives the generator into the wedged open

    # The loop must keep turning while the open is parked. Under the bug the open ran
    # inline on the loop thread, so the loop froze here and this timeout could never fire.
    beats = 0
    try:
        async with asyncio.timeout(5):
            while beats < 10:
                await asyncio.sleep(0)
                beats += 1
    finally:
        never_returns.set()  # free the parked thread so it does not outlive the test
    assert beats == 10, "the event loop was frozen while the microphone stream opened"

    # Freed, the open completes and the stream is an ordinary, closable microphone.
    (stream,) = backend.inputs

    async def push() -> None:
        await asyncio.sleep(0)
        stream.feed(b"\x00\x00")

    pusher = asyncio.create_task(push())
    async with asyncio.timeout(5):
        assert await opening == b"\x00\x00"
    await pusher
    await blocks.aclose()
    await released(stream)
    assert stream.closed


async def test_an_open_wedged_when_the_generator_is_closed_still_releases_the_stream() -> None:
    """The mic-leak failure path, not the freeze one.

    Moving the open off the loop introduced a new cancellation point: the wake gate's
    45s dead-stream watchdog returns while the open is still in flight (daemon/voice/
    wake.py's timeout -> return -> the generator is closed), and the open then completes
    on its own thread and hands back a live, started stream that nothing in the closed
    generator will ever read. It must still be stopped and closed, or the microphone
    light stays on with mic_hold already released - the exact leak apple_audio.py met
    when its tap install sat outside the try. `record` releases it from `deliver` when
    the late stream arrives to a cancelled future."""
    never_returns = threading.Event()

    class WedgingStream(FakeStream):
        def start(self) -> None:
            never_returns.wait()  # the open is wedged until the test frees it
            super().start()

    class WedgingBackend(FakeSoundDevice):
        def RawInputStream(self, **kwargs: Any) -> FakeStream:  # noqa: N802
            stream = WedgingStream(**kwargs)
            self.inputs.append(stream)
            return stream

    backend = WedgingBackend()
    io = SoundDeviceAudio(backend=backend)
    blocks = io.record()

    # The watchdog, in miniature: drive the open, give up on it, and close the
    # generator - all while the open is still wedged on its worker thread.
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(0.2):
            await anext(blocks)
    with contextlib.suppress(Exception):
        await blocks.aclose()

    # The open thread constructed the stream before parking in start(), and it has had
    # the whole timeout above to do it - so it is already on the backend.
    (stream,) = backend.inputs
    assert not stream.active, "the open should still be wedged, not yet started"

    # The HAL frees: the open completes and hands back a live stream no one will read.
    never_returns.set()
    await released(stream)  # it must be stopped and closed, not leaked
    assert stream.closed, "the microphone stream leaked when the open outran the close"
    assert not stream.active


async def test_play_queues_at_the_output_rate(backend: FakeSoundDevice) -> None:
    io = SoundDeviceAudio(backend=backend)
    await io.play(b"\x01\x01")
    await io.play(b"\x02\x02")
    stream = await backend.speaker()

    assert await stream.heard(2) == [b"\x01\x01", b"\x02\x02"]
    assert stream.kwargs["samplerate"] == 24_000
    assert stream.kwargs["channels"] == 1
    await io.close()


async def test_play_does_not_wait_for_the_speaker(backend: FakeSoundDevice) -> None:
    """The model generates faster than real time; blocking here would stall the
    same loop that has to notice the user interrupting."""
    io = SoundDeviceAudio(backend=backend)
    await io.play(b"\x01" * 4)

    # play() returned without ever yielding, so the device has not even been
    # opened yet - which is the whole point.
    assert backend.outputs == []
    stream = await backend.speaker()
    assert await stream.heard(1) == [b"\x01" * 4]
    await io.close()


async def test_empty_chunks_are_ignored(backend: FakeSoundDevice) -> None:
    io = SoundDeviceAudio(backend=backend)
    await io.play(b"")
    await settle()

    assert backend.outputs == []


# --- stopping ---------------------------------------------------------------


async def test_stop_playback_drops_the_queue_before_it_reaches_the_speaker(
    backend: FakeSoundDevice,
) -> None:
    """Without this the daemon keeps talking over the user for as long as the
    buffer lasts."""
    io = SoundDeviceAudio(backend=backend)
    await io.play(b"\xaa\xaa")
    stream = await backend.speaker()
    assert await stream.heard(1) == [b"\xaa\xaa"]
    # A whole answer's worth of audio, already generated and waiting.
    for index in range(50):
        await io.play(bytes([index, index]))
    await io.stop_playback()
    await settle()

    # Only what had already reached the device before the interruption.
    assert stream.written == [b"\xaa\xaa"]
    await io.close()


async def test_stop_playback_also_discards_what_the_device_already_holds(
    backend: FakeSoundDevice,
) -> None:
    """Emptying our own queue is not enough: PortAudio has already been handed
    several hundred milliseconds of audio, and only abort() drops that."""
    io = SoundDeviceAudio(backend=backend)
    await io.play(b"\x01\x01")
    stream = await backend.speaker()
    await stream.heard(1)
    await io.stop_playback()

    assert stream.aborted == 1
    await io.close()


async def test_playback_resumes_after_being_stopped(backend: FakeSoundDevice) -> None:
    """abort() leaves the stream stopped, so the next turn must restart it or the
    daemon goes permanently silent after one interruption."""
    io = SoundDeviceAudio(backend=backend)
    await io.play(b"\x01\x01")
    stream = await backend.speaker()
    await stream.heard(1)
    await io.stop_playback()
    assert not stream.active  # abort() left it stopped
    await io.play(b"\x02\x02")

    assert await stream.heard(1) == [b"\x02\x02"]
    assert stream.active
    await io.close()


async def test_stop_playback_with_nothing_playing_is_harmless(backend: FakeSoundDevice) -> None:
    io = SoundDeviceAudio(backend=backend)
    await io.stop_playback()
    await io.close()

    assert backend.outputs == []


async def test_close_stops_the_writer_and_the_device(backend: FakeSoundDevice) -> None:
    io = SoundDeviceAudio(backend=backend)
    await io.play(b"\x01\x01")
    stream = await backend.speaker()
    await stream.heard(1)
    await io.close()

    assert stream.closed
    await io.play(b"\x02\x02")  # after close, playing is a no-op, not a crash
    await settle()
    assert stream.written == [b"\x01\x01"]


async def test_closing_waits_for_the_write_in_flight_before_closing_the_device(
    backend: FakeSoundDevice,
) -> None:
    """The audit finding: `close()` cancelled the writer, but
    `to_thread(stream.write)` cannot be cancelled - cancelling only detaches the
    await. So the device was closed while a thread was still inside PortAudio
    writing to it, which at worst is a native crash. The fix is a cooperative stop,
    and this asserts the ordering that proves it."""
    io = SoundDeviceAudio(backend=backend)
    order: list[str] = []
    release = asyncio.Event()
    loop = asyncio.get_running_loop()
    started = asyncio.Event()

    await io.play(b"\x01\x01")
    stream = await backend.speaker()

    def slow_write(chunk: bytes) -> None:
        order.append("write-start")
        loop.call_soon_threadsafe(started.set)
        # Blocks the worker thread the way a full PortAudio buffer does.
        asyncio.run_coroutine_threadsafe(_wait(release), loop).result(5)
        order.append("write-end")

    stream.write = slow_write  # type: ignore[method-assign]
    stream.close = lambda: order.append("close")  # type: ignore[method-assign]
    await io.play(b"\x02\x02")
    async with asyncio.timeout(5):
        await started.wait()

    closing = asyncio.create_task(io.close())
    await settle()
    assert order == ["write-start"], "close() got ahead of the write in flight"

    release.set()
    async with asyncio.timeout(5):
        await closing

    assert order == ["write-start", "write-end", "close"]


async def _wait(event: asyncio.Event) -> None:
    await event.wait()


async def test_a_wedged_speaker_is_left_open_rather_than_closed_underneath(
    backend: FakeSoundDevice, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the write never comes back, the thread is still inside PortAudio. Leaking
    a handle on the way out of the process beats closing the device under it."""
    monkeypatch.setattr(audio, "CLOSE_TIMEOUT_SECONDS", 0.05)
    io = SoundDeviceAudio(backend=backend)
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    await io.play(b"\x01\x01")
    stream = await backend.speaker()

    def wedged(chunk: bytes) -> None:
        loop.call_soon_threadsafe(started.set)
        asyncio.run_coroutine_threadsafe(_wait(asyncio.Event()), loop).result(10)

    stream.write = wedged  # type: ignore[method-assign]
    await io.play(b"\x02\x02")
    async with asyncio.timeout(5):
        await started.wait()

    async with asyncio.timeout(5):
        await io.close()

    assert not stream.closed
    assert "leaving the device open" in caplog.text


async def test_stopping_playback_does_not_abort_during_a_write(
    backend: FakeSoundDevice,
) -> None:
    """`abort()` from one thread while another is inside `write()` is a race in C.
    Barge-in is the moment it would happen, which is also the moment it matters."""
    io = SoundDeviceAudio(backend=backend)
    order: list[str] = []
    release = asyncio.Event()
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    await io.play(b"\x01\x01")
    stream = await backend.speaker()

    def slow_write(chunk: bytes) -> None:
        order.append("write-start")
        loop.call_soon_threadsafe(started.set)
        asyncio.run_coroutine_threadsafe(_wait(release), loop).result(5)
        order.append("write-end")

    stream.write = slow_write  # type: ignore[method-assign]
    stream.abort = lambda: order.append("abort")  # type: ignore[method-assign]
    await io.play(b"\x02\x02")
    async with asyncio.timeout(5):
        await started.wait()

    stopping = asyncio.create_task(io.stop_playback())
    await settle()
    assert order == ["write-start"]

    release.set()
    async with asyncio.timeout(5):
        await stopping

    assert order == ["write-start", "write-end", "abort"]
    await io.close()


async def test_one_refused_chunk_does_not_end_playback(
    backend: FakeSoundDevice, caplog: pytest.LogCaptureFixture
) -> None:
    """An unhandled exception in the writer task would silently end playback for
    the rest of the conversation."""
    io = SoundDeviceAudio(backend=backend)
    await io.play(b"\x01\x01")
    stream = await backend.speaker()
    await stream.heard(1)
    calls = {"n": 0}
    original = stream.write

    def flaky(chunk: bytes) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("device is unhappy")
        original(chunk)

    stream.write = flaky  # type: ignore[method-assign]
    await io.play(b"\x02\x02")
    await io.play(b"\x03\x03")

    # The refused chunk is gone, the one behind it still plays.
    assert await stream.heard(1) == [b"\x03\x03"]
    assert "device is unhappy" in caplog.text  # dropped loudly, not silently
    await io.close()


class NoSpeaker:
    def RawOutputStream(self, **kwargs: Any) -> Any:  # noqa: N802 - mirrors sounddevice
        raise AudioUnavailable("no output device")


async def test_missing_speaker_is_logged_once_and_not_retried_per_chunk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    io = SoundDeviceAudio(backend=NoSpeaker())
    for _ in range(20):
        await io.play(b"\x01\x01")
    # Wait on the writer itself: no stream is ever opened to wait on.
    await asyncio.gather(io._writer)
    await settle()

    assert caplog.text.count("going mute") == 1
    await io.close()


async def test_a_dead_speaker_stops_the_queue_growing_for_ever() -> None:
    """The audit finding: the writer returned on AudioUnavailable and `_closed`
    stayed False, so `play` kept accepting chunks into a queue nothing would ever
    drain - about 48 KB a second at 24 kHz 16-bit, on a process meant to stay up
    for weeks."""
    io = SoundDeviceAudio(backend=NoSpeaker())
    await io.play(b"\x01\x01")
    await asyncio.gather(io._writer)  # let it discover there is no speaker

    for _ in range(500):
        await io.play(b"\x02" * 960)

    assert io._queue.qsize() == 0, "the queue is still growing behind a dead writer"
    await io.close()


# --- the local speaker ------------------------------------------------------

