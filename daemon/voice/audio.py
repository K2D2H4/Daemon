"""Microphone, speaker, and the speaker-only path - the `AudioIO` half of
daemon/voice/base.py.

`sounddevice` is imported lazily, never at module scope. A text-only install has
no PortAudio, and `import daemon.voice.audio` must still succeed there: this
module is reachable from config and startup code that runs whether or not voice
is on, and an ImportError at the top would turn "voice is off" into "the daemon
does not start".

Proactive speech at the machine used to live here too, as a second
`LocalSpeaker`. It moved to `daemon/proactivity/speaker.py`, where the rest of
proactivity is: two classes of the same name made `tests/test_reachable.py`
unable to say which one anything constructed, and this one was the dead half.
It is the delivery path from docs/PLAN.md 6.3 - when the user is at the machine, a
proactive utterance goes out of the local speaker, and nothing leaves the device.
On macOS that needs no dependency at all: /usr/bin/say ships with Korean voices.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

from daemon import mic_hold

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

MIC_QUEUE_BLOCKS = 64
"""About 1.9s of microphone audio. Bounded because in real-time audio an old block
has negative value: an unbounded queue meant a consumer that fell behind kept
sending audio from tens of seconds ago, so the session stayed up and the
transcripts kept coming while the conversation drifted into the past. When it is
full the *oldest* block goes."""

MIC_DROP_LOG_EVERY = 32
"""Drops arrive dozens per second when they arrive at all, so they are counted and
reported periodically rather than logged one by one."""

CLOSE_TIMEOUT_SECONDS = 2.0
"""How long `close` waits for the speaker to come back before giving up on
closing the device. See `close`."""

INPUT_RELEASE_TIMEOUT_SECONDS = 2.0
"""How long `wait_for_input_release` waits for a detached microphone release.

Generous rather than tuned: with no other client on the device a `Pa_StopStream`
returns in well under a millisecond, so anything still running after two seconds is
not slow, it is wedged - and the caller needs to be told that rather than kept
waiting. Same value as the speaker's for the same reason, not because the two
measure the same thing."""

_STOP = b""
"""Sentinel that ends the playback writer. Safe as a queue value because `play`
refuses empty chunks, and needed because `to_thread(stream.write)` cannot be
cancelled - see `close`."""

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


def _release_input_stream(stream: Any) -> None:
    """Stop and close a PortAudio input stream, off the event loop.

    Runs on a detached daemon thread from `record`'s finally: the stop can deadlock
    inside CoreAudio, and on the loop thread that froze the whole daemon (see the
    call site). Never raises - the thread has nobody to report to, and a stream that
    will not close is a leaked microphone, not a crash.
    """
    try:
        stream.stop()
        stream.close()
    except Exception:
        logger.exception("audio: could not release the microphone stream")


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
        self._mute = False
        # Serialises everything that touches the output device. PortAudio has no
        # opinion about being aborted or closed from one thread while another is
        # inside write(); at worst that is a native crash, and `to_thread` cannot
        # be cancelled, so exclusion is the only lever we have.
        self._device = asyncio.Lock()
        self.dropped_blocks = 0
        """Microphone blocks thrown away because the consumer fell behind."""
        self._releases: list[threading.Thread] = []
        """Detached release threads nobody has waited for yet. See
        `wait_for_input_release`, which is the only reader and empties it."""

    def _module(self) -> Any:
        if self._backend is None:
            self._backend = _sounddevice()
        return self._backend

    async def record(self) -> AsyncIterator[bytes]:
        """Microphone PCM, one block at a time, for as long as it is iterated."""
        sd = self._module()
        loop = asyncio.get_running_loop()
        blocks: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MIC_QUEUE_BLOCKS)

        def enqueue(block: bytes) -> None:
            # Oldest first, and loudly. Dropping the *newest* would keep feeding
            # the model stale audio, which is the failure this bound exists for:
            # the session stays alive and the transcripts keep arriving while the
            # conversation answers something the user said half a minute ago.
            if blocks.full():
                blocks.get_nowait()
                self.dropped_blocks += 1
                if self.dropped_blocks % MIC_DROP_LOG_EVERY == 1:
                    logger.warning(
                        "audio: microphone queue full, dropped %d block(s) so far; "
                        "the session is not keeping up with the microphone",
                        self.dropped_blocks,
                    )
            blocks.put_nowait(block)

        def on_block(indata: Any, frames: int, time_info: Any, status: Any) -> None:
            # Runs on PortAudio's own thread. `bytes(indata)` copies: the buffer
            # is reused for the next block, so keeping the view would hand the
            # provider audio that has already been overwritten.
            if status:
                logger.warning("audio: input stream reported %s", status)
            loop.call_soon_threadsafe(enqueue, bytes(indata))

        # Tell the rest of the process the microphone is ours, so `presence.py`
        # can subtract it from the CoreAudio probe. Without this the gate reads
        # our own wake listener as somebody on a call and never routes to the
        # local speaker again - see daemon/mic_hold.py. Entered before
        # `RawInputStream` so a backend that fails to construct the stream never
        # leaves the counter incremented, and released in `hold()`'s own
        # `finally` so a stream that dies mid-read - cancelled, or the consumer
        # breaking out - cannot leave it stuck either.
        #
        # It releases when the release *thread is handed the stream*, not when
        # PortAudio has finished letting go, and that is the honest boundary:
        # this counter answers "are we using the microphone", and by then we have
        # stopped. The device itself lingers for about a second afterwards either
        # way (see WAKE_REARM_SETTLE_SECONDS in daemon/app.py), which the probe
        # reads as a brief, self-healing busy rather than as ours.
        def release_off_loop(stream: Any) -> None:
            # Stop and close on a detached daemon thread. `stream.stop()` is
            # `Pa_StopStream` -> CoreAudio `AudioOutputUnitStop` -> a HAL mutex, and
            # that mutex deadlocked once on the wake->voice handover - the session's
            # macOS VoiceProcessing unit and this PortAudio stream contending the same
            # device. On the loop thread it took the whole daemon with it (a `sample`
            # showed `__psynch_mutexwait` under `AudioOutputUnitStop`). A private thread
            # rather than `to_thread` so a wedged release parks one thread instead of
            # burning a pool worker that playback needs. The thread holds `stream`, so
            # it is not collected mid-release.
            thread = threading.Thread(
                target=_release_input_stream,
                args=(stream,),
                name="voice-mic-release",
                daemon=True,
            )
            # Remembered before it is started, so `wait_for_input_release` cannot miss
            # a release that finishes between the two lines.
            self._releases.append(thread)
            thread.start()

        with mic_hold.hold():
            # The open runs off the loop on its own thread, for the same reason the
            # release does. `sd.RawInputStream(...)` is `Pa_OpenStream` and `start()`
            # is `Pa_StartStream`, both of which can wedge on a CoreAudio HAL mutex -
            # the one the wake gate's dead-stream rebuild deadlocked on, contending the
            # detached stop of the stream it had just dropped. On the loop thread that
            # froze the whole daemon for eleven hours: no logs, no scheduler, no wake, a
            # `sample` showing `__psynch_mutexwait` under `Pa_OpenStream` on the uvloop
            # thread. A private thread rather than `to_thread` because this open shares
            # the stop's wedge-forever risk and must not exhaust the pool playback runs
            # on. v0.1.45 moved the stop; this is the open it missed.
            opened: asyncio.Future[Any] = loop.create_future()

            def deliver(stream: Any) -> None:
                # On the loop. If the generator was already closed while the open was
                # still running - the wake gate's 45s dead-stream watchdog returns and
                # `aclose()`s us mid-open - then nobody will ever read this stream, and
                # the `finally` below could not release it because it did not exist yet.
                # So release it here instead of leaving a hot microphone with `mic_hold`
                # already let go. This is the leak daemon/voice/apple_audio.py avoids by
                # keeping its tap on `self`; a local `stream` cannot be reached that way.
                if opened.cancelled():
                    release_off_loop(stream)
                else:
                    opened.set_result(stream)

            def open_stream() -> None:
                try:
                    stream = sd.RawInputStream(
                        samplerate=self.sample_rate,
                        channels=CHANNELS,
                        dtype=DTYPE,
                        blocksize=self._block_frames,
                        callback=on_block,
                    )
                    stream.start()
                except Exception as exc:  # handed to the awaiter, like any open failure
                    loop.call_soon_threadsafe(opened.set_exception, exc)
                else:
                    loop.call_soon_threadsafe(deliver, stream)

            threading.Thread(target=open_stream, name="voice-mic-open", daemon=True).start()
            try:
                await opened
                while True:
                    yield await blocks.get()
            finally:
                # Reached on cancellation and on the consumer breaking out. An open
                # input stream that nobody reads is a microphone light left on. Three
                # states: the open never finished (cancel the future, and `deliver`
                # releases the stream when it finally arrives); it finished with a live
                # stream (release it); or it failed (nothing to release).
                if not opened.done():
                    opened.cancel()
                elif not opened.cancelled() and opened.exception() is None:
                    release_off_loop(opened.result())

    async def wait_for_input_release(
        self, within: float = INPUT_RELEASE_TIMEOUT_SECONDS
    ) -> bool:
        """Wait for every detached microphone release to finish. True if they did.

        `record`'s `finally` hands the stream to a thread and returns, so `aclose()`
        is the moment the daemon *stopped using* the microphone, not the moment
        CoreAudio got it back. Anything that opens a second client on the same device
        needs the later of the two: the wake gate's PortAudio stream and a session's
        macOS VoiceProcessing engine, started while the first was still stopping,
        deadlocked the pair of them on the resident (see `_release_input_stream` and
        `daemon/app.py:_wake_round`).

        Never raises and never blocks the loop - the join runs on a worker, and a
        release that outlives the bound is reported as `False` rather than waited on
        forever, because a stop that has not returned by then is a device already
        lost and the caller has a decision to make about that.

        `within` rather than `timeout`: what is bounded is a `Thread.join`, and an
        `asyncio.timeout` around it - which is what the name `timeout` invites, and
        what ASYNC109 would have this be - cannot cancel a join. It would abandon the
        worker and report a lie.
        """
        pending, self._releases = self._releases, []
        if not pending:
            return True

        def join_all() -> bool:
            deadline = time.monotonic() + within
            for thread in pending:
                thread.join(max(0.0, deadline - time.monotonic()))
            return not any(thread.is_alive() for thread in pending)

        return await asyncio.to_thread(join_all)

    async def play(self, chunk: bytes) -> None:
        """Queue a chunk. Returns immediately.

        Deliberately does not wait for the speaker: the model generates faster
        than real time, and blocking here would stall the same loop that has to
        notice the user interrupting.
        """
        # `_mute` as well as `_closed`: once the writer has given up there is
        # nothing draining this queue, and `put_nowait` kept accepting chunks -
        # about 48 KB a second at 24 kHz 16-bit, forever, on a process meant to run
        # for weeks.
        if not chunk or self._closed or self._mute:
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
        self._discard_queued()
        if self._out is not None:
            # Under the device lock: aborting a stream while the writer thread is
            # inside write() is a race in C, not in Python. Costs at most one
            # block's write, and the queue is already empty by here.
            async with self._device:
                await asyncio.to_thread(self._out.abort)

    async def close(self) -> None:
        """Stop playing and release the device, cooperatively.

        Cancelling the writer is not enough and is actively dangerous: an
        in-flight `to_thread(stream.write)` cannot be cancelled, so cancelling
        only detaches the await and leaves a thread writing into a stream that
        `close()` then closes underneath it - a crash inside PortAudio in the
        worst case. So: drop what is queued, ask the writer to stop, and wait for
        it to come back before touching the device.
        """
        self._closed = True
        writer, self._writer = self._writer, None
        stopped = True
        if writer is not None:
            self._discard_queued()
            self._queue.put_nowait(_STOP)
            try:
                async with asyncio.timeout(CLOSE_TIMEOUT_SECONDS):
                    await writer
            except (TimeoutError, asyncio.CancelledError):
                stopped = False
            except Exception:
                logger.exception("audio: the playback writer ended badly")
        out, self._out = self._out, None
        if out is None:
            return
        if not stopped:
            # Deliberately leaks the stream. A wedged write means the thread is
            # still inside PortAudio, and closing the device under it is the one
            # outcome worse than an unreleased handle on the way out of the
            # process.
            logger.warning(
                "audio: the speaker did not stop within %.0fs; leaving the device "
                "open rather than closing it under a running write",
                CLOSE_TIMEOUT_SECONDS,
            )
            return
        async with self._device:
            await asyncio.to_thread(out.close)

    def _discard_queued(self) -> None:
        while not self._queue.empty():
            self._queue.get_nowait()

    async def _drain(self) -> None:
        while True:
            chunk = await self._queue.get()
            if chunk == _STOP:
                return
            try:
                stream = self._output_stream()
                if not stream.active:
                    # A previous stop_playback aborted the stream, which leaves
                    # it stopped.
                    stream.start()
                async with self._device:
                    await asyncio.to_thread(stream.write, chunk)
            except AudioUnavailable as exc:
                # No speaker at all. Nothing to retry, and the caller is not
                # waiting on this task, so say so here and stop - after shutting
                # the door behind us, or `play` keeps filling a queue that nothing
                # will ever drain.
                self._mute = True
                self._discard_queued()
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