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
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from daemon.voice import audio
from daemon.voice.audio import AudioUnavailable, LocalSpeaker, SoundDeviceAudio
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
    assert stream.closed  # a microphone nobody reads is a light left on


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
    assert not stream.active
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


async def test_missing_speaker_is_logged_once_and_not_retried_per_chunk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class NoSpeaker:
        def RawOutputStream(self, **kwargs: Any) -> Any:  # noqa: N802 - mirrors sounddevice
            raise AudioUnavailable("no output device")

    io = SoundDeviceAudio(backend=NoSpeaker())
    for _ in range(20):
        await io.play(b"\x01\x01")
    # Wait on the writer itself: no stream is ever opened to wait on.
    await asyncio.gather(io._writer)
    await settle()

    assert caplog.text.count("going mute") == 1
    await io.close()


# --- the local speaker ------------------------------------------------------


def test_macos_reads_the_text_from_stdin_not_argv() -> None:
    """An utterance starting with '-' would otherwise be read as an option, and
    text in argv is visible to every other process on the machine."""
    assert LocalSpeaker(platform="darwin").command() == ["/usr/bin/say", "-f", "-"]


def test_macos_korean_voice_is_selectable() -> None:
    """docs/PLAN.md 6.3: macOS ships ko_KR voices, so the Korean proactive path
    needs nothing installed."""
    assert LocalSpeaker(voice="Yuna", platform="darwin").command() == [
        "/usr/bin/say",
        "-v",
        "Yuna",
        "-f",
        "-",
    ]


def test_a_platform_with_no_synthesiser_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio.shutil, "which", lambda _name: None)
    speaker = LocalSpeaker(platform="linux")

    assert not speaker.available
    with pytest.raises(AudioUnavailable, match="espeak-ng"):
        speaker.command()


def test_linux_uses_whichever_synthesiser_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audio.shutil, "which", lambda name: "/usr/bin/espeak-ng" if name == "espeak-ng" else None
    )
    assert LocalSpeaker(platform="linux").command() == ["/usr/bin/espeak-ng", "--stdin"]


async def test_speaking_writes_the_text_to_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    spoken: list[tuple[tuple[str, ...], bytes]] = []

    class FakeProcess:
        returncode: int | None = None

        async def communicate(self, data: bytes) -> tuple[None, bytes]:
            self.returncode = 0
            spoken.append((command, data))
            return None, b""

    command: tuple[str, ...] = ()

    async def fake_exec(*argv: str, **kwargs: Any) -> FakeProcess:
        nonlocal command
        command = argv
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await LocalSpeaker(voice="Yuna", platform="darwin").say("자기 전에 물 한 잔 마셔")

    (argv, data) = spoken[0]
    assert argv == ("/usr/bin/say", "-v", "Yuna", "-f", "-")
    assert data.decode("utf-8") == "자기 전에 물 한 잔 마셔"


async def test_nothing_to_say_starts_no_process(monkeypatch: pytest.MonkeyPatch) -> None:
    async def refuse(*argv: str, **kwargs: Any) -> Any:
        raise AssertionError("should not have started a process")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)
    await LocalSpeaker(platform="darwin").say("   ")


async def test_a_failing_synthesiser_reports_its_own_complaint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingProcess:
        returncode: int | None = None

        async def communicate(self, data: bytes) -> tuple[None, bytes]:
            self.returncode = 1
            return None, b"Voice 'Nonexistent' not found"

    async def fake_exec(*argv: str, **kwargs: Any) -> FailingProcess:
        return FailingProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(AudioUnavailable, match="not found"):
        await LocalSpeaker(voice="Nonexistent", platform="darwin").say("안녕")


async def test_stopping_kills_the_utterance_mid_word(monkeypatch: pytest.MonkeyPatch) -> None:
    """docs/PLAN.md 6.4: a voice out of the speaker during a meeting is an
    accident, so cutting one off is not a nicety."""
    started = asyncio.Event()
    terminated: list[bool] = []

    class SlowProcess:
        returncode: int | None = None

        async def communicate(self, data: bytes) -> tuple[None, bytes]:
            started.set()
            await asyncio.Event().wait()  # speaks until stopped
            raise AssertionError("unreachable")  # pragma: no cover

        def terminate(self) -> None:
            terminated.append(True)
            self.returncode = -15

        async def wait(self) -> int:
            return -15

    async def fake_exec(*argv: str, **kwargs: Any) -> SlowProcess:
        return SlowProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    speaker = LocalSpeaker(platform="darwin")
    task = asyncio.create_task(speaker.say("아주 긴 이야기를 시작하려고 하는데"))
    async with asyncio.timeout(5):
        await started.wait()
        await speaker.stop()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert terminated == [True]


def test_the_real_say_binary_exists_on_this_macos_machine() -> None:
    """Not a mock: docs/PLAN.md 6.3 counts on /usr/bin/say being there, so if it
    ever is not, that should fail here rather than at the first utterance."""
    if sys.platform != "darwin":
        pytest.skip("macOS only")
    assert LocalSpeaker().available
    voices = subprocess.run(
        ["/usr/bin/say", "-v", "?"], capture_output=True, text=True, check=True
    ).stdout
    assert "ko_KR" in voices  # the Korean proactive path needs no install
