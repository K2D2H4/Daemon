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
        self._loop = asyncio.get_running_loop()

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

