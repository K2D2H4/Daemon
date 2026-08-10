"""Microphone and speaker through one AVAudioEngine, with macOS echo cancellation
on - the `AudioIO` half of daemon/voice/base.py, for Macs.

`daemon/voice/audio.py`'s `SoundDeviceAudio` keeps the microphone open while the
model talks, because that is the only way a barge-in can be noticed at all
(daemon/voice/conversation.py). With no echo canceller that open door lets the
daemon's own speaker back in, and the server's activity detection reads it as the
user interrupting - so the daemon cuts itself off mid-sentence. Measured on this
machine, Silero at 0.5 over a 10 s clip of Korean TTS played out of the speaker:

    capture path                              quiet room   speaker playing
    raw PortAudio (SoundDeviceAudio today)         0.0%           82.7%
    this engine, voice processing OFF               0.0%           83.1%
    this engine, voice processing ON                0.0%            0.0%

The middle row is the control: AVAudioEngine on its own changes nothing, so the
83.1 -> 0.0 is voice processing and not the framework. Playing through this engine
rather than another process, the same switch takes the captured RMS from 293.9 to
4.3.

Apple's Voice Processing I/O is one audio unit doing input *and* output, so this
class owns the speaker as well as the microphone - a canceller needs to know what
was played to subtract it, and PortAudio has no path to turn any of this on.

Five things here are load-bearing, and each was measured rather than reasoned to:

1. **The hardware output format is read before `mainMixerNode` is touched.** Touch
   the mixer first and the implicit mixer -> output connection is made at the
   mixer's own 44100 default, which drags the output node's reported format down
   with it; the voice-processing unit then refuses to initialise, because its input
   side runs at 48000. The symptom is `-10875` from `PerformCommand(*outputNode,
   kAUInitialize)` and it is entirely order-dependent - five orderings measured,
   the two that read the format late both fail.
2. **The input format is never constructed, only taken from the node.** With voice
   processing on it is 48 kHz with *five* channels, and
   `initStandardFormatWithSampleRate:channels:` returns nil for five channels.
   Handing that nil to `initWithPCMFormat:frameCapacity:` segfaults the process.
3. **One channel is selected, not downmixed.** All five measured byte-identical -
   same RMS, same peak, silent room and loud - so they are copies of one processed
   signal rather than a processed/raw/reference set. A 5->1 downmix is also what
   `AVAudioConverter` refuses outright.
4. **The converter's pull block returns its status, it does not write it.** pyobjc
   passes the `AVAudioConverterInputStatus *` out-parameter as `None`, so the
   documented C usage cannot work from Python at all: every conversion reported
   `Error` and produced a buffer full of nothing. pyobjc's own convention is a
   return tuple, and with it a 440 Hz tone resamples to a measured 439.9 Hz.
5. **The drain loop is capped.** An uncapped one trapped the process (SIGTRAP),
   allocating buffers on the real-time audio thread.

And one that did *not* repeat a trap this project has hit before: tap callbacks
arrive on the engine's own render threads, never the main thread - 30 callbacks per
3.00 s of audio, from two non-main thread ids. `daemon/voice/apple_speech.py`
documents the opposite case, where a result handler defaulted to the *main* queue
and a worker thread saw 0 callbacks in 8 s while busy-spinning a core. Different
delivery mechanism, so that trap does not apply here - checked, not assumed.

`AVFoundation` is imported lazily, never at module scope, for the same reason
`sounddevice` is in daemon/voice/audio.py: it is pyobjc, it exists only on macOS,
and `import daemon.voice.apple_audio` must still succeed in a text-only install and
on Linux.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from daemon.voice.audio import (
    CHANNELS,
    INPUT_SAMPLE_RATE,
    MIC_DROP_LOG_EVERY,
    MIC_QUEUE_BLOCKS,
    OUTPUT_SAMPLE_RATE,
    AudioUnavailable,
)

logger = logging.getLogger(__name__)

TAP_BUFFER_FRAMES = 1024
"""What the tap asks for. A request, not a promise - the engine delivered 4800
frames a callback here, ten callbacks a second."""

DRAIN_LIMIT = 8
"""How many times one tap may pull the converter before giving up.

Capped because the loop runs on the real-time audio thread: uncapped, it trapped
the process allocating output buffers. Eight is far more than the two a 3:1
conversion needs, so hitting it means something is wrong rather than busy."""

CONVERTER_HEADROOM_FRAMES = 1024
"""Slack on each output buffer. The converter primes - about 19 ms of it, once -
and holds a backlog, so a buffer sized to the exact ratio fills and silently keeps
the remainder for the next call."""

_FULL_SCALE = 32768.0
"""int16 <-> float32 in [-1, 1). 32768 rather than 32767, matching
daemon/voice/vad.py so the two agree about what a sample is worth."""

_HAVE_DATA = 0
_INPUT_RAN_DRY = 1
"""`AVAudioConverterInputStatus`. Only these two are ever returned by the block
here, and only these two are acceptable coming back."""

_INSTALL_HINT = "install with: pip install -e '.[voice]'"


@dataclass(frozen=True, slots=True)
class AudioFrameworks:
    """The one pyobjc module this needs, as an injectable thing.

    A seam for the same reason `SoundDeviceAudio(backend=...)` and
    `AppleSpeechRecognizer(frameworks=...)` have one: CONTRACTS forbids a test that
    touches hardware, and CI is Linux where this does not import at all.
    """

    avfoundation: Any


def _avfoundation() -> AudioFrameworks:
    """Import AVFoundation on first real use, with an error worth reading."""
    try:
        import AVFoundation
    except ImportError:
        raise AudioUnavailable(
            f"AVFoundation (pyobjc) is not installed, so macOS echo cancellation "
            f"is unavailable; {_INSTALL_HINT}"
        ) from None
    return AudioFrameworks(avfoundation=AVFoundation)


class VoiceProcessingAudio:
    """Implements the `AudioIO` protocol in daemon/voice/base.py, on macOS.

    One engine does both directions because Apple's canceller is one unit doing
    both: it subtracts what was played from what was heard, so it has to be the
    thing that played it.
    """

    def __init__(self, *, frameworks: AudioFrameworks | None = None) -> None:
        self.sample_rate = INPUT_SAMPLE_RATE
        self.playback_sample_rate = OUTPUT_SAMPLE_RATE
        # None means "import the real framework on first use"; tests pass a
        # stand-in so no hardware is touched.
        self._frameworks = frameworks
        self._engine: Any = None
        self._input: Any = None
        self._player: Any = None
        self._converter: Any = None
        self._source_format: Any = None
        self._target_format: Any = None
        self._play_format: Any = None
        self._started = False
        self._closed = False
        self._tapped = False
        # Guards engine construction and teardown. A threading lock rather than an
        # asyncio one because the tap callback runs on a render thread and has to
        # be able to see a consistent object graph while `close` is dismantling it.
        self._guard = threading.Lock()
        self.dropped_blocks = 0
        """Microphone blocks thrown away because the consumer fell behind."""

        self.echo_cancellation = False
        """Whether voice processing is actually on.

        Reported rather than assumed, because the whole point of this class is a
        number that moves and a silent degradation is this project's dangerous
        failure: an engine that quietly ran without cancellation would look
        identical to one with it and behave like `SoundDeviceAudio`.
        """

    # --- the engine ---------------------------------------------------------

    def _modules(self) -> Any:
        if self._frameworks is None:
            self._frameworks = _avfoundation()
        return self._frameworks.avfoundation

    def _build(self) -> Any:
        """Assemble the engine, once. Order matters - see the module docstring."""
        if self._engine is not None:
            return self._engine
        avf = self._modules()
        engine = avf.AVAudioEngine.alloc().init()
        source = engine.inputNode()
        output = engine.outputNode()

        enabled, error = source.setVoiceProcessingEnabled_error_(True, None)
        self.echo_cancellation = bool(enabled)
        if not enabled:
            # Not fatal. Without cancellation this is `SoundDeviceAudio` with extra
            # steps, which is worse than nothing only if nobody is told - so it is
            # said at WARNING and reported on the attribute.
            logger.warning(
                "apple audio: macOS voice processing was refused (%s); the "
                "microphone will hear the speaker and the daemon will interrupt "
                "itself",
                error,
            )

        # BEFORE mainMixerNode is touched. See the module docstring: reading it
        # afterwards pins the graph to the mixer's 44100 default and the
        # voice-processing unit will not initialise.
        hardware = output.inputFormatForBus_(0)
        mixer = engine.mainMixerNode()
        engine.connect_to_format_(mixer, output, hardware)

        # Attached whether or not anything is ever played: voice processing is one
        # unit doing both directions, so the output half has to exist for the input
        # half to work.
        play_format = avf.AVAudioFormat.alloc().initStandardFormatWithSampleRate_channels_(
            float(self.playback_sample_rate), CHANNELS
        )
        player = avf.AVAudioPlayerNode.alloc().init()
        engine.attachNode_(player)
        engine.connect_to_format_(player, mixer, play_format)

        # Taken from the node, never constructed - a five-channel format cannot be
        # built, and the nil that comes back segfaults whatever is handed it.
        source_format = source.outputFormatForBus_(0)
        target_format = (
            avf.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
                avf.AVAudioPCMFormatInt16, float(self.sample_rate), CHANNELS, True
            )
        )
        converter = avf.AVAudioConverter.alloc().initFromFormat_toFormat_(
            source_format, target_format
        )
        if converter is None:
            raise AudioUnavailable(
                f"macOS will not convert {source_format.sampleRate():.0f} Hz / "
                f"{source_format.channelCount()} ch to {self.sample_rate} Hz mono"
            )
        # One channel, not a downmix: the five are copies of the same processed
        # signal, and a downmix is refused outright.
        converter.setChannelMap_([0])

        self._engine = engine
        self._input = source
        self._player = player
        self._converter = converter
        self._source_format = source_format
        self._target_format = target_format
        self._play_format = play_format
        logger.info(
            "apple audio: engine at %.0f Hz / %d ch in, %d Hz out, echo cancellation %s",
            source_format.sampleRate(),
            source_format.channelCount(),
            self.playback_sample_rate,
            "on" if self.echo_cancellation else "OFF",
        )
        return engine

    def _start(self) -> None:
        if self._started:
            return
        engine = self._build()
        engine.prepare()
        started, error = engine.startAndReturnError_(None)
        if not started:
            raise AudioUnavailable(f"the audio engine refused to start: {error}")
        self._started = True

    # --- microphone ---------------------------------------------------------

    async def record(self) -> AsyncIterator[bytes]:
        """Microphone PCM at 16 kHz mono, echo-cancelled, for as long as it is
        iterated."""
        loop = asyncio.get_running_loop()
        blocks: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MIC_QUEUE_BLOCKS)

        def enqueue(block: bytes) -> None:
            # Oldest first, same policy and same reason as daemon/voice/audio.py:
            # dropping the newest keeps feeding the model audio from tens of
            # seconds ago, which leaves the session alive and the conversation in
            # the past.
            if blocks.full():
                blocks.get_nowait()
                self.dropped_blocks += 1
                if self.dropped_blocks % MIC_DROP_LOG_EVERY == 1:
                    logger.warning(
                        "apple audio: microphone queue full, dropped %d block(s) so "
                        "far; the session is not keeping up with the microphone",
                        self.dropped_blocks,
                    )
            blocks.put_nowait(block)

        def on_buffer(buffer: Any, when: Any) -> None:
            # Runs on one of the engine's render threads - measured, never the main
            # thread. Nothing here may block or raise: this is the real-time audio
            # path, and an exception on it takes the tap with it.
            try:
                pcm = self._convert(buffer)
            except Exception:
                logger.exception("apple audio: dropping a block the converter refused")
                return
            if pcm:
                loop.call_soon_threadsafe(enqueue, pcm)

        # Both tap calls block inside CoreAudio - `installTap` waits on the engine
        # starting, `removeTap` on the render threads draining - so neither may run
        # on the event loop. See `close` for what that cost when it wedged.
        try:
            # Inside the `try`, so a cancellation landing mid-install still reaches
            # the `finally` that removes the tap. Outside it, the thread finished
            # installing while the generator never entered its own cleanup, and the
            # microphone stayed live (proved by tests/test_apple_audio.py).
            await asyncio.to_thread(self._install_tap_blocking, on_buffer)
            while True:
                yield await blocks.get()
        finally:
            # Reached on cancellation and on the consumer breaking out. A tap
            # nobody reads is a microphone light left on.
            # Synchronous, unlike every other call here, and not an oversight: this
            # runs in an async generator's `finally`, which is reached through
            # `aclose()` on a cancelled task - and a fresh `await` there is
            # cancelled before it can start, so the tap would leak (proved by
            # tests/test_apple_audio.py). Removing a tap is also not where the
            # measured deadlock was; that was `AudioOutputUnitStop` in `close`,
            # which is on a thread now.
            self._remove_tap_blocking()

    def _install_tap_blocking(self, on_buffer: Any) -> None:
        with self._guard:
            self._start()
            self._input.installTapOnBus_bufferSize_format_block_(
                0, TAP_BUFFER_FRAMES, self._source_format, on_buffer
            )
            self._tapped = True

    def _remove_tap_blocking(self) -> None:
        with self._guard:
            if self._tapped and self._input is not None:
                self._input.removeTapOnBus_(0)
                self._tapped = False

    def _convert(self, buffer: Any) -> bytes:
        """One tap buffer to 16 kHz mono 16-bit PCM.

        A loop rather than one call: the converter primes and holds a backlog, so a
        single pull against a buffer sized to the exact ratio silently keeps the
        rest. Capped, because this runs on the audio thread.
        """
        frames = buffer.frameLength()
        if not frames:
            return b""
        avf = self._frameworks.avfoundation  # type: ignore[union-attr]
        converter = self._converter
        target_format = self._target_format
        ratio = self.sample_rate / self._source_format.sampleRate()
        capacity = int(frames * ratio) + CONVERTER_HEADROOM_FRAMES
        served = False

        def supply(packets: int, status: Any) -> Any:
            # pyobjc's out-parameter convention: the status is *returned*, not
            # written through the pointer, which arrives as None. Writing to it is
            # what made every conversion report Error and produce silence.
            nonlocal served
            if served:
                return (None, _INPUT_RAN_DRY)
            served = True
            return (buffer, _HAVE_DATA)

        pieces: list[bytes] = []
        for _ in range(DRAIN_LIMIT):
            target = avf.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
                target_format, capacity
            )
            status, error = converter.convertToBuffer_error_withInputFromBlock_(
                target, None, supply
            )
            produced = target.frameLength()
            if produced:
                # `as_buffer` takes a sample count and returns that many *elements*
                # worth of bytes, so 16-bit data needs the frame count and not
                # twice it. Passing double read past the valid region and returned
                # audio at the wrong length.
                pieces.append(bytes(target.int16ChannelData()[0].as_buffer(produced)))
            if status != _HAVE_DATA:
                if status != _INPUT_RAN_DRY:
                    logger.warning(
                        "apple audio: converter returned status %s (%s)", status, error
                    )
                break
        return b"".join(pieces)

    # --- speaker ------------------------------------------------------------

    async def play(self, chunk: bytes) -> None:
        """Queue a chunk of 24 kHz mono 16-bit PCM. Returns immediately.

        Scheduled on the player rather than written to a device, so there is no
        writer task and no thread to wedge: `scheduleBuffer` hands the engine a
        buffer and returns, and the engine renders it on its own clock. The rate
        conversion to the hardware's 48 kHz is the connection's job.
        """
        if not chunk or self._closed:
            return
        try:
            # Off the event loop, like every other call into this engine - see
            # `close`. Ordering is preserved because callers await each chunk
            # before handing over the next.
            await asyncio.to_thread(self._play_blocking, chunk)
        except AudioUnavailable as exc:
            self._closed = True
            logger.error("apple audio: playback is unavailable, going mute: %s", exc)
        except Exception:
            # One bad chunk must not end playback for the rest of the conversation.
            logger.exception("apple audio: dropping a chunk the speaker refused")

    def _play_blocking(self, chunk: bytes) -> None:
        """The engine half of `play`, on a worker thread."""
        with self._guard:
            self._start()
            buffer = self._playback_buffer(chunk)
            if buffer is None:
                return
            player = self._player
            if not player.isPlaying():
                # `stop_playback` leaves the player stopped, so this is also
                # how the next answer starts.
                player.play()
            player.scheduleBuffer_completionHandler_(buffer, None)

    def _playback_buffer(self, chunk: bytes) -> Any:
        avf = self._frameworks.avfoundation  # type: ignore[union-attr]
        samples = np.frombuffer(chunk, dtype="<i2")
        if not samples.size:
            return None
        buffer = avf.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(
            self._play_format, samples.size
        )
        buffer.setFrameLength_(samples.size)
        floats = (samples.astype(np.float32) / _FULL_SCALE).astype("<f4")
        buffer.floatChannelData()[0].as_buffer(samples.size)[:] = floats.tobytes()
        return buffer

    async def stop_playback(self) -> None:
        """Drop everything scheduled but not yet played.

        `stop` on the player is the whole of it: unlike PortAudio there is no
        second buffer of ours to empty, because nothing was ever queued on our side
        - `play` hands each chunk straight to the engine. The player is left
        stopped and `play` starts it again.
        """
        await asyncio.to_thread(self._stop_playback_blocking)

    def _stop_playback_blocking(self) -> None:
        with self._guard:
            if self._player is not None and self._started:
                self._player.stop()

    async def close(self) -> None:
        """Stop the engine and release the device.

        On a worker thread, because every call here can block indefinitely inside
        CoreAudio and this one is the worst of them. Measured on the owner's Mac:
        `AudioOutputUnitStop` parked on `HALB_Mutex::Lock` and never returned, and
        because it was awaited straight on the event loop it took the **whole
        daemon** with it - Telegram stopped polling, `/health` stopped answering,
        and the resident sat there alive and mute until it was killed. A wedged
        audio device must cost the conversation, not the process.
        `daemon/voice/audio.py` reached this conclusion first, for PortAudio; this
        engine simply never had it applied.
        """
        await asyncio.to_thread(self._close_blocking)

    def _close_blocking(self) -> None:
        with self._guard:
            self._closed = True
            engine, self._engine = self._engine, None
            source, self._input = self._input, None
            self._player = None
            self._converter = None
            if source is not None and self._tapped:
                source.removeTapOnBus_(0)
                self._tapped = False
            if engine is not None and self._started:
                engine.stop()
            self._started = False
