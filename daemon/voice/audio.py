"""Microphone, speaker, and the speaker-only path - the `AudioIO` half of
daemon/voice/base.py.

`sounddevice` is imported lazily, never at module scope. A text-only install has
no PortAudio, and `import daemon.voice.audio` must still succeed there: this
module is reachable from config and startup code that runs whether or not voice
is on, and an ImportError at the top would turn "voice is off" into "the daemon
does not start".

`LocalSpeaker` is separate from `SoundDeviceAudio` and shares nothing with it.
It is the delivery path from docs/PLAN.md 6.3 - when the user is at the machine, a
proactive utterance goes out of the local speaker, and nothing leaves the device.
On macOS that needs no dependency at all: /usr/bin/say ships with Korean voices.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
"""Set by the provider, not by preference: Gemini Live takes 16kHz in and returns
24kHz out (daemon/voice/gemini_live.py). One rate for both would play the model
back at the wrong pitch."""

CHANNELS = 1
DTYPE = "int16"
BLOCK_FRAMES = 480
"""30ms at 16kHz. Small enough that server-side speech detection reacts promptly,
large enough not to send a websocket frame per millisecond."""

_INSTALL_HINT = "install with: pip install -e '.[voice]'"


class AudioUnavailable(RuntimeError):
    """No usable audio device or backend. Distinct from a session failure: this
    one is fixed by installing something, not by retrying."""


def _sounddevice() -> Any:
    """Import sounddevice on first real use, with an error worth reading.

    OSError, not just ImportError: the wheel installs fine and then fails at
    import time when the PortAudio shared library is missing, which is the
    common case on a fresh Linux box.
    """
    try:
        import sounddevice
    except ImportError:
        raise AudioUnavailable(f"sounddevice is not installed; {_INSTALL_HINT}") from None
    except OSError as exc:
        raise AudioUnavailable(
            f"sounddevice could not load its PortAudio library ({exc}); "
            "install PortAudio (macOS: brew install portaudio, "
            "Debian/Ubuntu: apt install libportaudio2)"
        ) from None
    return sounddevice


class SoundDeviceAudio:
    """Implements the `AudioIO` protocol in daemon/voice/base.py."""

    def __init__(self, *, backend: Any = None, block_frames: int = BLOCK_FRAMES) -> None:
        self.sample_rate = INPUT_SAMPLE_RATE
        """What `record` yields, and what the session must be fed. The protocol
        has one field and the device has two rates; the recording rate is the one
        a caller can get wrong in a way that reaches the provider."""
        self.playback_sample_rate = OUTPUT_SAMPLE_RATE
        # None means "import the real thing on first use"; tests pass a stand-in
        # so no hardware is touched.
        self._backend = backend
        self._block_frames = block_frames
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._writer: asyncio.Task[None] | None = None
        self._out: Any = None
        self._closed = False

    def _module(self) -> Any:
        if self._backend is None:
            self._backend = _sounddevice()
        return self._backend

    async def record(self) -> AsyncIterator[bytes]:
        """Microphone PCM, one block at a time, for as long as it is iterated."""
        sd = self._module()
        loop = asyncio.get_running_loop()
        blocks: asyncio.Queue[bytes] = asyncio.Queue()

        def on_block(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            # Runs on PortAudio's own thread. `bytes(indata)` copies: the buffer
            # is reused for the next block, so keeping the view would hand the
            # provider audio that has already been overwritten.
            if status:
                logger.warning("audio: input stream reported %s", status)
            loop.call_soon_threadsafe(blocks.put_nowait, bytes(indata))

        stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self._block_frames,
            callback=on_block,
        )
        stream.start()
        try:
            while True:
                yield await blocks.get()
        finally:
            # Reached on cancellation and on the consumer breaking out. An open
            # input stream that nobody reads is a microphone light left on.
            stream.stop()
            stream.close()

    async def play(self, chunk: bytes) -> None:
        """Queue a chunk. Returns immediately.

        Deliberately does not wait for the speaker: the model generates faster
        than real time, and blocking here would stall the same loop that has to
        notice the user interrupting.
        """
        if not chunk or self._closed:
            return
        # Started once and never respawned: the one thing that ends it for good
        # is a missing speaker, and restarting into that would log the same
        # failure per chunk, dozens of times a second.
        if self._writer is None:
            self._writer = asyncio.create_task(self._drain())
        self._queue.put_nowait(chunk)

    async def stop_playback(self) -> None:
        """Drop everything queued, in both buffers that hold audio.

        Emptying our queue is not enough on its own - PortAudio has already been
        handed several hundred milliseconds of it, and `abort` is what discards
        that rather than playing it out. Without both, the daemon keeps talking
        over the user for as long as the buffers last.
        """
        while not self._queue.empty():
            self._queue.get_nowait()
        if self._out is not None:
            await asyncio.to_thread(self._out.abort)

    async def close(self) -> None:
        self._closed = True
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.cancel()
            # Suppressed rather than raised: close() runs on the shutdown path,
            # where the cancellation it just caused is not news.
            await asyncio.gather(writer, return_exceptions=True)
        out, self._out = self._out, None
        if out is not None:
            await asyncio.to_thread(out.close)

    async def _drain(self) -> None:
        while True:
            chunk = await self._queue.get()
            try:
                stream = self._output_stream()
                if not stream.active:
                    # A previous stop_playback aborted the stream, which leaves
                    # it stopped.
                    stream.start()
                await asyncio.to_thread(stream.write, chunk)
            except AudioUnavailable as exc:
                # No speaker at all. Nothing to retry, and the caller is not
                # waiting on this task, so say so here and stop.
                logger.error("audio: playback is unavailable, going mute: %s", exc)
                return
            except Exception:
                # One bad write must not silently end playback for the rest of
                # the conversation, which is what an unhandled exception in a
                # background task does.
                logger.exception("audio: dropping a chunk the speaker refused")

    def _output_stream(self) -> Any:
        if self._out is None:
            sd = self._module()
            self._out = sd.RawOutputStream(
                samplerate=self.playback_sample_rate, channels=CHANNELS, dtype=DTYPE
            )
        return self._out


class LocalSpeaker:
    """Say text on this machine's speaker, with no session and no network.

    The path docs/PLAN.md 6.3 routes to when the user is present: it costs
    nothing per minute, and the utterance never leaves the device - which is the
    one privacy claim in docs/PLAN.md 7 that survives voice being switched on.

    Text is written to the process's stdin, never placed in argv: an utterance
    starting with '-' would otherwise be read as an option.
    """

    SAY = "/usr/bin/say"

    def __init__(self, *, voice: str | None = None, platform: str | None = None) -> None:
        self._voice = voice
        """A ko_KR voice such as Yuna or Eddy. macOS ships nine of them, so the
        Korean path needs nothing installed."""
        self._platform = platform if platform is not None else sys.platform
        self._process: asyncio.subprocess.Process | None = None

    def command(self) -> list[str]:
        """The argv for this platform, or an error naming what to install."""
        if self._platform == "darwin":
            # -f - reads the text from stdin. Verified: -f /dev/stdin does not
            # work on macOS ("Bad file descriptor"), so this spelling matters.
            return [self.SAY, *(["-v", self._voice] if self._voice else []), "-f", "-"]
        # Not macOS: no bundled synthesiser exists, so probe for the two a Linux
        # desktop is likely to already have before giving up. Both spellings read
        # the text from stdin; neither is verified on a real Linux box yet, so
        # this branch is the one to check first if a self-hoster reports silence.
        for name, args in (("espeak-ng", ["--stdin"]), ("spd-say", ["--wait", "--pipe-mode"])):
            found = shutil.which(name)
            if found:
                return [found, *args]
        raise AudioUnavailable(
            f"no local text-to-speech on platform {self._platform!r}; "
            "install speech-dispatcher (spd-say) or espeak-ng, "
            "or route proactive utterances to a channel instead"
        )

    @property
    def available(self) -> bool:
        """Whether speaking would work at all - checked before choosing this
        route, so presence routing can fall back to a channel instead of
        discovering the speaker is missing at the moment it wants to talk."""
        try:
            command = self.command()
        except AudioUnavailable:
            return False
        return shutil.which(command[0]) is not None

    async def say(self, text: str) -> None:
        """Speak, and wait until it is finished.

        Sequential on purpose: two overlapping utterances out of one speaker are
        not two messages, they are noise.
        """
        if not text.strip():
            return
        command = self.command()
        await self.stop()
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._process = process
        try:
            _, stderr = await process.communicate(text.encode("utf-8"))
        except asyncio.CancelledError:
            await self.stop()
            raise
        finally:
            if self._process is process:
                self._process = None
        if process.returncode:
            raise AudioUnavailable(
                f"{command[0]} exited {process.returncode}: "
                f"{stderr.decode('utf-8', 'replace').strip() or 'no output'}"
            )

    async def stop(self) -> None:
        """Cut the utterance off mid-word.

        docs/PLAN.md 6.4: a voice coming out of the speaker during a meeting is
        an accident, so being able to stop one is not a nicety.
        """
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return  # finished between the check and the signal
        await process.wait()
