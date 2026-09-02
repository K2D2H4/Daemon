"""The echo-cancelling AudioIO. No microphone, no speaker, no pyobjc.

A fake AVFoundation stands in for the real framework, so this file passes on Linux
CI where none of it imports - the same seam and the same reason as
`SoundDeviceAudio(backend=...)` and `AppleSpeechRecognizer(frameworks=...)`.

What is asserted here is the set of things that were measured the hard way against
the real framework, because each is silent when it is wrong:

  * the hardware output format is read *before* the main mixer is touched, or the
    voice-processing unit will not initialise
  * the input format is taken from the node, never constructed
  * one channel is selected rather than downmixed
  * the converter's status comes back in the block's return tuple
  * voice processing being refused is reported rather than assumed

The numbers those decisions produced are in the module docstring of
daemon/voice/apple_audio.py; they cannot be re-measured without hardware, so what
is guarded here is the shape that produced them.
"""

from __future__ import annotations

import asyncio
import builtins
import sys
import threading
from typing import Any

import numpy as np
import pytest

from daemon import mic_hold
from daemon.voice import apple_audio
from daemon.voice.apple_audio import AudioFrameworks, VoiceProcessingAudio
from daemon.voice.audio import AudioUnavailable
from daemon.voice.base import AudioIO

HAVE_DATA = 0
INPUT_RAN_DRY = 1

INPUT_RATE = 48_000
INPUT_CHANNELS = 5
"""What the real voice-processing input node reports on the target machine. Five,
which is the number that cannot be constructed as a format - hence the fake serving
it from the node like the real one does."""


# --- the fake framework ------------------------------------------------------


class FakeFormat:
    def __init__(self, rate: float, channels: int, *, common: int = 1, interleaved: bool = False):
        self._rate = rate
        self._channels = channels
        self._common = common
        self._interleaved = interleaved

    def sampleRate(self) -> float:  # noqa: N802
        return self._rate

    def channelCount(self) -> int:  # noqa: N802
        return self._channels

    def commonFormat(self) -> int:  # noqa: N802
        return self._common

    def isInterleaved(self) -> bool:  # noqa: N802
        return self._interleaved

    def __repr__(self) -> str:
        return f"<FakeFormat {self._rate:.0f}/{self._channels}ch>"


class FakeVarList:
    """Stands in for `objc.varlist`, which is neither a ctypes pointer nor a list.

    `as_buffer(n)` returns n *elements* worth of bytes - not n bytes. Getting that
    wrong read past the valid region and returned audio at twice its true length,
    so the fake enforces the real semantics.
    """

    def __init__(self, data: bytearray, itemsize: int) -> None:
        self._data = data
        self._itemsize = itemsize

    def as_buffer(self, count: int) -> memoryview:
        need = count * self._itemsize
        if need > len(self._data):
            raise AssertionError(
                f"as_buffer({count}) wants {need} bytes of a {len(self._data)}-byte buffer"
            )
        return memoryview(self._data)[:need]


class FakeBuffer:
    """An AVAudioPCMBuffer: a format, a capacity, and a frame length."""

    def __init__(self, fmt: FakeFormat, capacity: int) -> None:
        self._format = fmt
        self._capacity = capacity
        self._frames = 0
        itemsize = 2 if fmt.commonFormat() == FakeAVFoundation.AVAudioPCMFormatInt16 else 4
        self._itemsize = itemsize
        self._storage = [bytearray(capacity * itemsize) for _ in range(fmt.channelCount())]

    def format(self) -> FakeFormat:
        return self._format

    def frameCapacity(self) -> int:  # noqa: N802
        return self._capacity

    def frameLength(self) -> int:  # noqa: N802
        return self._frames

    def setFrameLength_(self, frames: int) -> None:  # noqa: N802
        assert frames <= self._capacity, "a buffer cannot be longer than its capacity"
        self._frames = frames

    def floatChannelData(self) -> tuple[FakeVarList, ...]:  # noqa: N802
        assert self._itemsize == 4, "floatChannelData on a non-float buffer"
        return tuple(FakeVarList(chan, 4) for chan in self._storage)

    def int16ChannelData(self) -> tuple[FakeVarList, ...]:  # noqa: N802
        assert self._itemsize == 2, "int16ChannelData on a non-int16 buffer"
        return tuple(FakeVarList(chan, 2) for chan in self._storage)

    def fill_int16(self, samples: np.ndarray) -> None:
        self._frames = samples.size
        self._storage[0][: samples.size * 2] = samples.astype("<i2").tobytes()


class FakeConverter:
    """Resamples by taking every third sample. Not fidelity - shape.

    What matters for the test is that the status comes back in the *return tuple*:
    pyobjc hands the block None for the out-pointer, so a converter driven the way
    the C docs describe reports Error and produces silence.
    """

    def __init__(self, source: FakeFormat, target: FakeFormat) -> None:
        self.source = source
        self.target = target
        self.channel_map: list[int] | None = None
        self.calls = 0
        self.statuses_seen: list[Any] = []
        self.pointer_was_none = True

    def setChannelMap_(self, channels: list[int]) -> None:  # noqa: N802
        self.channel_map = list(channels)

    def channelMap(self) -> list[int]:  # noqa: N802
        return self.channel_map or []

    def convertToBuffer_error_withInputFromBlock_(  # noqa: N802
        self, target: FakeBuffer, error: Any, block: Any
    ) -> tuple[int, Any]:
        self.calls += 1
        # The real framework passes None here, and that is the whole finding: there
        # is no pointer to write a status through.
        result = block(target.frameCapacity(), None)
        self.pointer_was_none = True
        if not isinstance(result, tuple) or len(result) != 2:
            # What the C-style usage produces: no status, so no conversion.
            return (3, "block did not return (buffer, status)")
        source, status = result
        self.statuses_seen.append(status)
        if status != HAVE_DATA or source is None:
            return (INPUT_RAN_DRY, None)
        assert self.channel_map is not None, "a 5->1 downmix is refused by the real converter"
        channel = self.channel_map[0]
        raw = bytes(source.floatChannelData()[channel].as_buffer(source.frameLength()))
        floats = np.frombuffer(raw, dtype="<f4")
        step = int(self.source.sampleRate() // self.target.sampleRate())
        out = np.clip(floats[::step] * 32768.0, -32768, 32767)
        target.fill_int16(out)
        return (INPUT_RAN_DRY, None)


class FakeNode:
    def __init__(self, engine: FakeAVFoundation, name: str) -> None:
        self._engine = engine
        self._name = name
        self.voice_processing = False
        self.taps: dict[int, Any] = {}
        self.removed_taps = 0
        self.playing = False
        self.scheduled: list[FakeBuffer] = []
        self.completions: list[tuple[int, Any]] = []
        """(callback type, handler) per buffer scheduled with a completion, so a test
        can be the engine and report them played back."""
        self.stops = 0

    # input node
    def setVoiceProcessingEnabled_error_(self, enabled: bool, error: Any):  # noqa: N802
        if self._engine.refuse_voice_processing:
            return (False, "fake refusal")
        self.voice_processing = bool(enabled)
        self._engine.voice_processing_enabled = bool(enabled)
        return (True, None)

    def inputFormatForBus_(self, bus: int) -> FakeFormat:  # noqa: N802
        if self._name == "output":
            # The measured trap: once the main mixer has been touched, the output
            # node reports the mixer's 44100 instead of the hardware's 48000.
            if self._engine.mixer_touched:
                return FakeFormat(44_100.0, 2)
            return FakeFormat(48_000.0, 2)
        return self.outputFormatForBus_(bus)

    def outputFormatForBus_(self, bus: int) -> FakeFormat:  # noqa: N802
        if self._name == "input":
            channels = INPUT_CHANNELS if self.voice_processing else 1
            return FakeFormat(float(INPUT_RATE), channels)
        return FakeFormat(48_000.0, 2)

    def installTapOnBus_bufferSize_format_block_(  # noqa: N802
        self, bus: int, size: int, fmt: Any, block: Any
    ) -> None:
        self.taps[bus] = (size, fmt, block)

    def removeTapOnBus_(self, bus: int) -> None:  # noqa: N802
        self.taps.pop(bus, None)
        self.removed_taps += 1

    # player node
    def isPlaying(self) -> bool:  # noqa: N802
        return self.playing

    def play(self) -> None:
        self.playing = True

    def stop(self) -> None:
        self.playing = False
        self.stops += 1
        self.scheduled.clear()

    def scheduleBuffer_completionHandler_(self, buffer: FakeBuffer, handler: Any) -> None:  # noqa: N802
        assert self.playing, "a buffer was scheduled on a player that is not playing"
        self.scheduled.append(buffer)

    def scheduleBuffer_completionCallbackType_completionHandler_(  # noqa: N802
        self, buffer: FakeBuffer, kind: int, handler: Any
    ) -> None:
        """The variant the product uses now: the engine reports when a buffer has
        been *played back*. The handler is kept so a test can play the engine."""
        assert self.playing, "a buffer was scheduled on a player that is not playing"
        self.scheduled.append(buffer)
        self.completions.append((kind, handler))


class FakeEngine:
    def __init__(self, framework: FakeAVFoundation) -> None:
        self._framework = framework
        self.started = False
        self.stopped = 0
        self.prepared = 0
        self.attached: list[FakeNode] = []
        self.connections: list[tuple[str, str, Any]] = []
        self._input = FakeNode(framework, "input")
        self._output = FakeNode(framework, "output")
        self._mixer = FakeNode(framework, "mixer")

    def inputNode(self) -> FakeNode:  # noqa: N802
        return self._input

    def outputNode(self) -> FakeNode:  # noqa: N802
        return self._output

    def mainMixerNode(self) -> FakeNode:  # noqa: N802
        self._framework.mixer_touched = True
        return self._mixer

    def attachNode_(self, node: FakeNode) -> None:  # noqa: N802
        self.attached.append(node)

    def connect_to_format_(self, source: FakeNode, target: FakeNode, fmt: Any) -> None:
        self.connections.append((source._name, target._name, fmt))

    def prepare(self) -> None:
        self.prepared += 1

    def startAndReturnError_(self, error: Any):  # noqa: N802
        hardware = next(
            (f for s, t, f in self.connections if s == "mixer" and t == "output"), None
        )
        if hardware is None:
            return (False, "-10875: nothing connected the mixer to the output")
        if self._framework.voice_processing_enabled and hardware.sampleRate() != INPUT_RATE:
            # The real failure, reproduced: the voice-processing unit runs its input
            # side at 48000 and cannot initialise against a 44100 output.
            return (
                False,
                "-10875: PerformCommand(*outputNode, kAUInitialize) - "
                f"output at {hardware.sampleRate():.0f} against input at {INPUT_RATE}",
            )
        if self._framework.refuse_start:
            return (False, "-10851: fake device failure")
        self.started = True
        return (True, None)

    def stop(self) -> None:
        self.started = False
        self.stopped += 1


class FakeAVFoundation:
    """Just enough AVFoundation, with the measured traps intact."""

    AVAudioPCMFormatFloat32 = 1
    AVAudioPCMFormatInt16 = 2
    AVAudioPlayerNodeCompletionDataConsumed = 0
    AVAudioPlayerNodeCompletionDataRendered = 1
    AVAudioPlayerNodeCompletionDataPlayedBack = 2

    def __init__(self) -> None:
        self.refuse_voice_processing = False
        self.refuse_start = False
        self.refuse_converter = False
        self.voice_processing_enabled = False
        self.mixer_touched = False
        self.engines: list[FakeEngine] = []
        self.converters: list[FakeConverter] = []

        framework = self

        class AVAudioEngine:
            @staticmethod
            def alloc() -> Any:
                class _Alloc:
                    @staticmethod
                    def init() -> FakeEngine:
                        engine = FakeEngine(framework)
                        framework.engines.append(engine)
                        return engine

                return _Alloc()

        class AVAudioFormat:
            @staticmethod
            def alloc() -> Any:
                class _Alloc:
                    @staticmethod
                    def initStandardFormatWithSampleRate_channels_(  # noqa: N802
                        rate: float, channels: int
                    ) -> FakeFormat | None:
                        # The real one returns nil for five channels, and handing
                        # that nil onwards segfaults - so it must never be asked.
                        if channels > 2:
                            return None
                        return FakeFormat(rate, channels, common=1)

                    @staticmethod
                    def initWithCommonFormat_sampleRate_channels_interleaved_(  # noqa: N802
                        common: int, rate: float, channels: int, interleaved: bool
                    ) -> FakeFormat:
                        return FakeFormat(
                            rate, channels, common=common, interleaved=interleaved
                        )

                return _Alloc()

        class AVAudioPlayerNode:
            @staticmethod
            def alloc() -> Any:
                class _Alloc:
                    @staticmethod
                    def init() -> FakeNode:
                        return FakeNode(framework, "player")

                return _Alloc()

        class AVAudioPCMBuffer:
            @staticmethod
            def alloc() -> Any:
                class _Alloc:
                    @staticmethod
                    def initWithPCMFormat_frameCapacity_(  # noqa: N802
                        fmt: FakeFormat, capacity: int
                    ) -> FakeBuffer:
                        assert fmt is not None, (
                            "a nil format reached initWithPCMFormat_ - the real "
                            "framework segfaults here"
                        )
                        return FakeBuffer(fmt, capacity)

                return _Alloc()

        class AVAudioConverter:
            @staticmethod
            def alloc() -> Any:
                class _Alloc:
                    @staticmethod
                    def initFromFormat_toFormat_(  # noqa: N802
                        source: FakeFormat, target: FakeFormat
                    ) -> FakeConverter | None:
                        if framework.refuse_converter:
                            return None
                        converter = FakeConverter(source, target)
                        framework.converters.append(converter)
                        return converter

                return _Alloc()

        self.AVAudioEngine = AVAudioEngine
        self.AVAudioFormat = AVAudioFormat
        self.AVAudioPlayerNode = AVAudioPlayerNode
        self.AVAudioPCMBuffer = AVAudioPCMBuffer
        self.AVAudioConverter = AVAudioConverter


@pytest.fixture
def framework() -> FakeAVFoundation:
    return FakeAVFoundation()


@pytest.fixture
def audio(framework: FakeAVFoundation) -> VoiceProcessingAudio:
    return VoiceProcessingAudio(frameworks=AudioFrameworks(avfoundation=framework))


def tap_of(framework: FakeAVFoundation) -> Any:
    engine = framework.engines[-1]
    return engine.inputNode().taps[0][2]


def input_buffer(framework: FakeAVFoundation, samples: np.ndarray) -> FakeBuffer:
    """A tap buffer in the real input format: 48 kHz, five channels, float32."""
    fmt = FakeFormat(float(INPUT_RATE), INPUT_CHANNELS, common=1)
    buffer = FakeBuffer(fmt, samples.size)
    buffer.setFrameLength_(samples.size)
    raw = samples.astype("<f4").tobytes()
    for channel in range(INPUT_CHANNELS):
        buffer._storage[channel][: len(raw)] = raw
    return buffer


# --- the module must import where none of this exists ------------------------


def test_the_module_imports_without_pyobjc(monkeypatch: pytest.MonkeyPatch) -> None:
    """A text-only install and Linux CI have no AVFoundation, and
    `import daemon.voice.apple_audio` still has to work - it is reachable from
    startup code that runs whether or not voice is on."""
    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "AVFoundation":
            raise ImportError("no pyobjc here")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "AVFoundation", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse)

    audio = VoiceProcessingAudio()
    with pytest.raises(AudioUnavailable, match="AVFoundation"):
        audio._modules()


def test_it_satisfies_the_protocol(audio: VoiceProcessingAudio) -> None:
    assert isinstance(audio, AudioIO)
    assert audio.sample_rate == 16_000
    assert audio.playback_sample_rate == 24_000


# --- the graph, in the order that makes it start ------------------------------


async def test_the_hardware_format_is_read_before_the_mixer_is_touched(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """The measured trap, and it is entirely order-dependent: touch the main mixer
    first and the output node reports the mixer's 44100, the mixer -> output
    connection is made at that, and the voice-processing unit will not initialise
    (-10875). The fake reproduces it, so this test fails if the order is swapped."""
    await audio.play(b"\x01\x02")

    engine = framework.engines[-1]
    mixer_to_output = [f for s, t, f in engine.connections if s == "mixer" and t == "output"]
    assert mixer_to_output, "nothing connected the main mixer to the output node"
    assert mixer_to_output[0].sampleRate() == 48_000, (
        "the mixer was connected to the output at the mixer's own default rate, "
        "which is what stops the voice-processing unit initialising"
    )
    assert engine.started


async def test_the_input_format_is_never_constructed(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """A five-channel format cannot be built - the real initStandardFormat returns
    nil and the nil segfaults whatever is handed it - so it has to come from the
    node. The fake returns None for >2 channels and asserts on a nil format
    reaching a buffer, so a regression is a failure rather than a crash."""
    await audio.play(b"\x01\x02")

    converter = framework.converters[-1]
    assert converter.source.channelCount() == INPUT_CHANNELS
    assert converter.source.sampleRate() == INPUT_RATE
    assert converter.target.sampleRate() == 16_000
    assert converter.target.channelCount() == 1


async def test_one_channel_is_selected_rather_than_downmixed(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """All five channels measured byte-identical, so they are copies of one
    processed signal - and a 5->1 downmix is what the real converter refuses."""
    await audio.play(b"\x01\x02")

    assert framework.converters[-1].channel_map == [0]


async def test_recording_starts_the_engine_and_attaches_the_player(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    stream = audio.record()
    task = asyncio.create_task(anext(stream))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    engine = framework.engines[-1]
    assert engine.started
    assert [node._name for node in engine.attached] == ["player"]
    assert 0 in engine.inputNode().taps
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await stream.aclose()


async def test_an_engine_that_will_not_start_says_so(
    framework: FakeAVFoundation,
) -> None:
    framework.refuse_start = True
    audio = VoiceProcessingAudio(frameworks=AudioFrameworks(avfoundation=framework))

    with pytest.raises(AudioUnavailable, match="refused to start"):
        await anext(audio.record())  # type: ignore[arg-type]


async def test_recording_marks_the_microphone_held(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """The second of the two holders `daemon/mic_hold.py`'s counter exists for -
    see `daemon/voice/audio.py`'s equivalent test for the first. Without this the
    presence probe cannot tell this session's own hold from somebody else's call."""
    assert mic_hold.held() is False
    stream = audio.record()
    task = asyncio.create_task(anext(stream))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    assert mic_hold.held() is True, "the hold must be visible while recording"

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await stream.aclose()
    assert mic_hold.held() is False


async def test_the_hold_is_released_when_installing_the_tap_raises(
    audio: VoiceProcessingAudio, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors daemon/voice/audio.py: the hold must cover the tap install, not
    start after it. The assertion inside `explode` is the point of this test -
    without it, the test would pass whether or not the hold actually covers the
    install, since the failure happens either way."""

    def explode(self: FakeNode, bus: int, size: int, fmt: Any, block: Any) -> None:
        assert mic_hold.held() is True, "the hold must already cover the tap install"
        raise RuntimeError("AVAudioEngine went away")

    monkeypatch.setattr(FakeNode, "installTapOnBus_bufferSize_format_block_", explode)

    assert mic_hold.held() is False
    with pytest.raises(RuntimeError):
        await anext(audio.record())  # type: ignore[arg-type]
    assert mic_hold.held() is False


async def test_the_hold_is_released_when_the_engine_fails_to_start(
    framework: FakeAVFoundation,
) -> None:
    """Same failure, one call earlier: `_start()` can also fail before the tap is
    ever reached, and it happens inside the hold too."""
    framework.refuse_start = True
    audio = VoiceProcessingAudio(frameworks=AudioFrameworks(avfoundation=framework))

    assert mic_hold.held() is False
    with pytest.raises(AudioUnavailable, match="refused to start"):
        await anext(audio.record())  # type: ignore[arg-type]
    assert mic_hold.held() is False


async def test_a_missing_converter_is_reported_not_ignored(
    framework: FakeAVFoundation,
) -> None:
    framework.refuse_converter = True
    audio = VoiceProcessingAudio(frameworks=AudioFrameworks(avfoundation=framework))

    with pytest.raises(AudioUnavailable, match="will not convert"):
        await anext(audio.record())  # type: ignore[arg-type]


# --- the speaker reports when it has actually fallen silent -------------------


def _player(framework: FakeAVFoundation) -> FakeNode:
    """The one attached node that had buffers scheduled with a completion."""
    nodes = [n for n in framework.engines[-1].attached if n.completions]
    assert len(nodes) == 1, "expected exactly one node to have scheduled completions"
    return nodes[0]


async def test_the_last_buffer_played_back_is_what_fires_the_idle_hook(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """`play` returns when a buffer is *handed over*; the only estimate of when the
    speaker falls silent used to be bytes / sample rate. Measured 2026-09-02, that
    estimate outran the real speaker by 1.5-3.0 s a turn and the half-duplex gate
    ate the head of every reply the owner began the moment the daemon stopped
    talking. The engine knows the truth - `DataPlayedBack` - so it is asked, and
    only the *last* buffer of an answer counts as the speaker going quiet."""
    fired: list[int] = []
    audio.on_playback_idle = lambda: fired.append(1)

    await audio.play(b"\x01\x02")
    await audio.play(b"\x03\x04")
    player = _player(framework)
    kinds = {kind for kind, _ in player.completions}
    assert kinds == {framework.AVAudioPlayerNodeCompletionDataPlayedBack}, (
        "scheduled with the wrong callback type - consumed/rendered fire before the "
        "listener has heard the end"
    )

    first, second = (handler for _, handler in player.completions)
    first()  # engine thread says: buffer one played back
    await asyncio.sleep(0)  # the hop to the loop
    assert fired == [], "the hook fired with a buffer still to play"

    second()
    await asyncio.sleep(0)
    assert fired == [1], "the last buffer played back and nobody was told"


async def test_a_stop_discards_pending_completions_rather_than_counting_them(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """A barge-in stops the player and throws its queue away. A completion for one
    of those discarded buffers must not be counted against the answer that comes
    next - it would report *that* speaker silent while it was still talking, which
    is the echo leak the gate exists to prevent."""
    fired: list[int] = []
    audio.on_playback_idle = lambda: fired.append(1)

    await audio.play(b"\x01\x02")
    stale = _player(framework).completions[0][1]
    await audio.stop_playback()

    await audio.play(b"\x05\x06")  # the next answer: one buffer pending
    stale()  # the discarded buffer reports in late
    await asyncio.sleep(0)
    assert fired == [], "a discarded buffer's completion was counted against the next answer"

    fresh = _player(framework).completions[-1][1]
    fresh()
    await asyncio.sleep(0)
    assert fired == [1]


async def test_a_raising_idle_hook_does_not_take_the_engine_thread_with_it(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    def explode() -> None:
        raise RuntimeError("the conversation is gone")

    audio.on_playback_idle = explode
    await audio.play(b"\x01\x02")
    handler = _player(framework).completions[0][1]
    handler()
    await asyncio.sleep(0)  # logged, not raised - the render path may not raise


# --- voice processing being refused must be visible --------------------------


async def test_voice_processing_refused_is_reported_rather_than_assumed(
    framework: FakeAVFoundation, caplog: pytest.LogCaptureFixture
) -> None:
    """Without cancellation this class is PortAudio with extra steps. A silent
    degradation is this project's dangerous failure, so it is said out loud and
    readable from the object."""
    framework.refuse_voice_processing = True
    audio = VoiceProcessingAudio(frameworks=AudioFrameworks(avfoundation=framework))

    with caplog.at_level("WARNING"):
        await audio.play(b"\x01\x02")

    assert audio.echo_cancellation is False
    assert "interrupt itself" in caplog.text


async def test_voice_processing_accepted_is_reported_too(
    audio: VoiceProcessingAudio,
) -> None:
    await audio.play(b"\x01\x02")

    assert audio.echo_cancellation is True


# --- capture ------------------------------------------------------------------


async def test_a_tap_buffer_becomes_16k_mono_pcm(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """48 kHz five-channel float in, 16 kHz mono 16-bit out - a third as many
    frames, and two bytes each."""
    stream = audio.record()
    pending = asyncio.create_task(anext(stream))  # type: ignore[arg-type]
    await asyncio.sleep(0)

    frames = 4800
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(frames) / INPUT_RATE)).astype("<f4")
    tap_of(framework)(input_buffer(framework, tone), None)

    block = await asyncio.wait_for(pending, timeout=1.0)
    assert len(block) == frames // 3 * 2, "16 kHz mono 16-bit is a third of the frames"
    samples = np.frombuffer(block, dtype="<i2")
    assert np.abs(samples).max() > 8000, "the audio arrived at nothing like full scale"

    await stream.aclose()


async def test_the_converter_status_comes_back_in_the_return_tuple(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """pyobjc hands the block None for the out-pointer, so the C-style usage cannot
    set a status at all - every conversion reported Error and produced silence. The
    fake returns 3 for a block that does not return a tuple, so this test is what
    stops that regression."""
    stream = audio.record()
    pending = asyncio.create_task(anext(stream))  # type: ignore[arg-type]
    await asyncio.sleep(0)

    tone = np.full(4800, 0.4, dtype="<f4")
    tap_of(framework)(input_buffer(framework, tone), None)
    block = await asyncio.wait_for(pending, timeout=1.0)

    converter = framework.converters[-1]
    assert converter.pointer_was_none, "the fake stopped reproducing the real binding"
    assert converter.statuses_seen == [HAVE_DATA], (
        "the block did not offer its buffer, so nothing could be converted"
    )
    assert block, "the conversion produced nothing"
    # One pull per tap: the real converter answers inputRanDry once it has taken the
    # buffer, so the drain loop is what stops rather than a second offer.
    assert converter.calls == 1

    await stream.aclose()


async def test_a_tap_that_raises_does_not_kill_the_stream(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation, caplog: pytest.LogCaptureFixture
) -> None:
    """The tap runs on the real-time audio thread. An exception escaping it takes
    the tap with it, and then the microphone is silently dead."""
    stream = audio.record()
    pending = asyncio.create_task(anext(stream))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    tap = tap_of(framework)

    class Exploding:
        def frameLength(self) -> int:  # noqa: N802
            raise RuntimeError("a buffer from hell")

    with caplog.at_level("ERROR"):
        tap(Exploding(), None)
    assert "converter refused" in caplog.text

    tone = np.full(4800, 0.4, dtype="<f4")
    tap(input_buffer(framework, tone), None)
    block = await asyncio.wait_for(pending, timeout=1.0)
    assert block, "the stream did not recover from one bad buffer"

    await stream.aclose()


async def test_the_microphone_is_released_when_the_stream_ends(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    stream = audio.record()
    task = asyncio.create_task(anext(stream))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await stream.aclose()

    assert framework.engines[-1].inputNode().taps == {}, "the tap was left installed"


async def test_the_oldest_block_goes_when_the_consumer_falls_behind(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """Same policy and same reason as daemon/voice/audio.py: dropping the newest
    keeps feeding the model audio from tens of seconds ago, which leaves the session
    alive and the conversation in the past."""
    stream = audio.record()
    task = asyncio.create_task(anext(stream))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    tap = tap_of(framework)
    tone = np.full(4800, 0.3, dtype="<f4")

    for _ in range(apple_audio.MIC_QUEUE_BLOCKS + 8):
        tap(input_buffer(framework, tone), None)
    await asyncio.wait_for(task, timeout=1.0)
    await asyncio.sleep(0)

    assert audio.dropped_blocks > 0

    await stream.aclose()


# --- playback -----------------------------------------------------------------


async def test_play_schedules_24k_mono_on_the_player(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    samples = np.array([0, 16384, -16384, 32767], dtype="<i2")
    await audio.play(samples.tobytes())

    player = next(node for node in framework.engines[-1].attached if node._name == "player")
    assert len(player.scheduled) == 1
    buffer = player.scheduled[0]
    assert buffer.frameLength() == samples.size
    assert buffer.format().sampleRate() == 24_000
    assert buffer.format().channelCount() == 1


async def test_an_empty_chunk_is_not_scheduled(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    await audio.play(b"")

    assert framework.engines == [], "an empty chunk built an engine for nothing"


async def test_stop_playback_drops_what_was_scheduled(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """Half of a barge-in. Without it the daemon keeps talking out of the engine's
    buffer for as long as it lasts (daemon/voice/base.py)."""
    await audio.play(np.full(240, 1000, dtype="<i2").tobytes())
    player = next(node for node in framework.engines[-1].attached if node._name == "player")
    assert player.scheduled

    await audio.stop_playback()

    assert player.stops == 1
    assert player.scheduled == []


async def test_the_next_answer_starts_the_player_again(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """`stop` leaves the player stopped, and scheduling onto a stopped player is
    silence - so the turn after a barge-in has to start it again."""
    chunk = np.full(240, 1000, dtype="<i2").tobytes()
    await audio.play(chunk)
    await audio.stop_playback()
    await audio.play(chunk)

    player = next(node for node in framework.engines[-1].attached if node._name == "player")
    assert player.playing
    assert len(player.scheduled) == 1


async def test_stop_playback_before_anything_played_is_harmless(
    audio: VoiceProcessingAudio,
) -> None:
    await audio.stop_playback()


async def test_close_stops_the_engine_and_releases_the_tap(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    stream = audio.record()
    task = asyncio.create_task(anext(stream))  # type: ignore[arg-type]
    await asyncio.sleep(0)
    engine = framework.engines[-1]

    await audio.close()

    assert engine.stopped == 1
    assert engine.inputNode().taps == {}
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_playing_after_close_is_silently_ignored(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """`close` runs in a `finally` while the conversation may still be handing over
    audio, and an exception there would replace the error the caller needs."""
    await audio.close()
    await audio.play(np.full(240, 1000, dtype="<i2").tobytes())

    assert framework.engines == []


async def test_closing_twice_is_harmless(audio: VoiceProcessingAudio) -> None:
    await audio.close()
    await audio.close()


# --- a wedged device must not take the daemon with it -------------------------


async def test_a_stalled_engine_stop_does_not_block_the_event_loop(
    audio: VoiceProcessingAudio, framework: FakeAVFoundation
) -> None:
    """Measured on the owner's Mac: `AudioOutputUnitStop` parked on `HALB_Mutex` and
    never returned. It was awaited straight on the event loop, so it took the whole
    daemon with it - Telegram stopped polling, /health stopped answering, and the
    resident sat alive and mute until it was killed. A wedged audio device must cost
    the conversation, not the process."""
    await audio.play(b"\x00\x00" * 240)
    stalled = threading.Event()

    def wedge() -> None:
        stalled.wait(timeout=5)  # the CoreAudio mutex that never unlocks

    framework.engines[-1].stop = wedge

    closing = asyncio.create_task(audio.close())
    # The loop is still ours while the engine hangs: this only returns if `close`
    # left the event loop free, which is the whole property under test.
    ticks = 0
    for _ in range(20):
        await asyncio.sleep(0)
        ticks += 1
    assert ticks == 20 and not closing.done(), "the stop is blocking, not stalled off-loop"

    stalled.set()
    await asyncio.wait_for(closing, timeout=5)
