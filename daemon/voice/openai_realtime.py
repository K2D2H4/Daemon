"""OpenAI Realtime - the second hosted native-audio VoiceSession (docs/PLAN.md 6.5).

Structurally a sibling of daemon/voice/gemini_live.py: same connect-with-backoff, the
same API-key log filter, the same permanent-vs-transient close classification, the same
turn-boundary receive(). What differs is the wire - a different socket, JSON events
instead of Gemini's proto-over-JSON, and 24 kHz pcm16 input (so the 16 kHz mic capture is
upsampled here). Event names differ between the GA (gpt-realtime) and older beta surfaces;
the decoder accepts both, and evals/openai_realtime_spike.py pins them against the socket.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import logging
import os
import ssl
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

import certifi
import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedOK,
    InvalidHandshake,
    InvalidStatus,
)

from daemon.llm.base import ToolCall, ToolSpec, decode_tool_arguments, synthesise_call_id
from daemon.tools.base import ToolResult
from daemon.voice.base import Interrupted, Transcript

logger = logging.getLogger(__name__)

WS_URL = "wss://api.openai.com/v1/realtime"

INPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE = 16_000, 24_000  # mic capture -> OpenAI pcm16

SETUP_TIMEOUT_SECONDS = 20.0
"""A server that accepts the socket and never answers setup would otherwise hang
the voice turn forever."""

BACKOFF_START_SECONDS, BACKOFF_MAX_SECONDS, _BACKOFF_MAX_SHIFT = 1.0, 30.0, 5
"""Caps the doubling at 32s. The exponent is clamped, not just the result: in
telegram.py `2 ** (failures - 1)` kept growing behind a min() and eventually
raised OverflowError from inside the retry handler, killing the loop exactly when
the network came back."""

DEFAULT_MAX_ATTEMPTS = 4
"""Bounded on purpose. Unlike an inbound channel, nobody is waiting to be heard
here - a voice turn that cannot open should fail and let the caller fall back to
text, not retry into a per-minute-billed connection forever."""

PARTIAL_BACKLOG = 32
"""How many unread in-progress transcripts to keep. See gemini_live.py's constant
of the same name - the reasoning is unchanged."""

USER_TRANSCRIPT_GRACE_SECONDS = 0.5
"""How long `receive()` waits at the turn boundary for a user transcript that is
still OWED - see `_INPUT_COMMITTED`.

whisper-1 - the only model `_setup_message` requests - delivers
`conversation.item.input_audio_transcription.completed` (the user's own words)
with **non-deterministic** timing relative to `response.done`: measured BEFORE it
on one turn (1.78s vs 2.31s) and ~200ms AFTER it on another, live against
gpt-realtime (docs/design/2026-08-09-openai-realtime-voice-design.md). Releasing
the turn the moment `response.done` arrived, without waiting for a transcript
that was already in flight, recorded the assistant's reply BEFORE the user's
question that prompted it - the exact ordering gemini_live.py's `_flush`
docstring calls out: "user first: that is the order it happened in, and the
order memory should record it in."

Paid only when a transcript is actually owed for this turn:
`input_audio_buffer.committed` (which carries the user audio item's `item_id`)
tells `_decode` one is coming, before `response.created` even arrives. A turn
with no committed user audio - a `send_text`-only turn, or a proactive utterance
- owes nothing and `receive()` does not wait at all. 500ms is the measured
~200ms plus headroom."""

# Server event types (accept GA and beta spellings).
_AUDIO_DELTA = ("response.output_audio.delta", "response.audio.delta")
_ASSISTANT_TR_DELTA = ("response.output_audio_transcript.delta", "response.audio_transcript.delta")
_ASSISTANT_TR_DONE = ("response.output_audio_transcript.done", "response.audio_transcript.done")
_USER_TR_DELTA = ("conversation.item.input_audio_transcription.delta",)
_USER_TR_DONE = ("conversation.item.input_audio_transcription.completed",)
_INPUT_COMMITTED = "input_audio_buffer.committed"
_SPEECH_STARTED = "input_audio_buffer.speech_started"
_RESPONSE_DONE = "response.done"
_FUNC_ARGS_DONE = "response.function_call_arguments.done"
_OUTPUT_ITEM_ADDED = "response.output_item.added"

_PERMANENT_STATUS = frozenset({400, 401, 403, 404})
"""Handshake statuses that retrying cannot fix: a bad, revoked or unauthorised
key, or a wrong endpoint."""

_PERMANENT_CLOSE_CODES = frozenset({1007, 4000})
"""Close codes retrying cannot fix. 1007 is an invalid payload; 4000 is OpenAI's
application-level invalid_request (measured: a beta-shaped session.update on gpt-realtime
closes 4000 invalid_request_error.beta_api_shape_disabled) - a malformed request retries
into the same close. 1008 is deliberately NOT here: like Gemini's, an idle/normal close
can wear it and must be retried; OpenAI's exact 1008 semantics are still unconfirmed."""


def _permanent_close(code: int, reason: str) -> bool:
    return code in _PERMANENT_CLOSE_CODES


CA_BUNDLE_ENV = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
"""Where a corporate proxy's own CA bundle is already configured, if it is.

Read rather than invented: anyone behind TLS interception has set one of these for
httpx, requests or curl already, and a Daemon-specific setting would be a third
place to get it wrong. There is deliberately no way to switch verification off."""


def _ca_bundle() -> str:
    for name in CA_BUNDLE_ENV:
        configured = os.environ.get(name)
        if configured:
            return configured
    return certifi.where()


@functools.cache
def _ssl_context(cafile: str) -> ssl.SSLContext:
    """The trust store for the voice socket: certifi, named explicitly.

    `ssl.create_default_context()` with no arguments is empty on a python.org
    framework build until someone runs "Install Certificates.command", because
    those builds do not read the system keychain. websockets uses that default, so
    voice failed with `CERTIFICATE_VERIFY_FAILED` for every user who installed
    Python that way - while text worked, because httpx bundles certifi. Measured
    on a fresh 3.13 framework build: 0 CAs in the default context, 121 in certifi.

    Cached because building one parses a ~200 kB PEM file, and a session may be
    opened for every proactive utterance.
    """
    return ssl.create_default_context(cafile=cafile)


_RESAMPLE_STEP = INPUT_SAMPLE_RATE / OUTPUT_SAMPLE_RATE
"""Input-sample spacing between consecutive output samples (2/3): a fixed step,
not one recomputed per buffer - see `_resample_16k_to_24k`."""


def _resample_16k_to_24k(pcm: bytes, phase: float, tail: bytes) -> tuple[bytes, float, bytes]:
    """16-bit LE mono, 16 kHz -> 24 kHz by linear interpolation at a fixed step
    (`_RESAMPLE_STEP`), with the fractional phase and any unconsumed trailing
    input sample carried in (`phase`, `tail`) and out (the returned tuple) so
    the interpolation is continuous across chunks rather than restarting at
    every call.

    `send_audio` calls this once per mic chunk (every ~20 ms), so a version
    that re-anchored each chunk's first and last input sample to its own first
    and last output sample - stretching each buffer to fit itself instead of
    stepping at a fixed 16000/24000 - reset the interpolation phase on every
    single call. Measured against this fixed-step version over 20 ms chunks:
    300 Hz 32.9 dB vs 58.0 dB, 1 kHz 22.4 dB vs 37.1 dB, 3 kHz 12.2 dB vs
    18.3 dB SNR. Total duration was already correct either way - there is no
    drift, only the per-chunk phase reset, which is what hurt the SNR.

    `tail` holds the input sample(s) already looked at but not yet fully
    consumed (interpolation needs the sample *after* the last position used),
    so a chunk boundary never drops or duplicates a sample the way starting
    fresh from `pcm` alone would.
    """
    combined = tail + pcm
    n = len(combined) // 2
    if n < 2:
        # Nothing to interpolate between yet - hold it all as tail (drops any
        # trailing odd byte, which is not a whole sample).
        return b"", phase, combined[: n * 2]
    src = [int.from_bytes(combined[i * 2:i * 2 + 2], "little", signed=True) for i in range(n)]
    out = bytearray()
    pos = phase
    while True:
        lo = int(pos)
        if lo + 1 >= n:
            break
        frac = pos - lo
        val = int(round(src[lo] + (src[lo + 1] - src[lo]) * frac))
        val = max(-32768, min(32767, val))
        out += val.to_bytes(2, "little", signed=True)
        pos += _RESAMPLE_STEP
    consumed = int(pos)
    return bytes(out), pos - consumed, combined[consumed * 2:]


class OpenAIRealtimeError(Exception):
    """A realtime session failed. Never carries the API key (see the key filter)."""

    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent
        """True for auth and malformed-setup failures. Retrying those leaves the
        process alive, healthy-looking and permanently mute."""


class _KeyFilter(logging.Filter):
    """Scrubs the API key out of other libraries' log records.

    websockets logs the handshake request - headers included - at DEBUG, so the
    leak happens in someone else's logger and has to be stopped there. Same
    lesson as telegram.py's `_TokenFilter`, which caught a real leak.
    """

    def __init__(self, key: str) -> None:
        super().__init__()
        self._key = key

    def filter(self, record: logging.LogRecord) -> bool:
        # Mapping-style args and a pre-formatted traceback both bypass a
        # tuple-only scrub, and a formatted exception is exactly where a key
        # would surface if any third party logs with exc_info=True while one of
        # ours is active.
        if isinstance(record.args, dict):
            record.args = {
                name: (
                    str(value).replace(self._key, "<key>")
                    if self._key in str(value)
                    else value
                )
                for name, value in record.args.items()
            }
        if record.exc_text and self._key in record.exc_text:
            record.exc_text = record.exc_text.replace(self._key, "<key>")
        message = str(record.msg)
        if self._key in message:
            record.msg = message.replace(self._key, "<key>")
        if isinstance(record.args, tuple):
            record.args = tuple(
                str(arg).replace(self._key, "<key>") if self._key in str(arg) else arg
                for arg in record.args
            )
        return True


async def _sleep(seconds: float) -> None:
    # Indirection so tests can drive the backoff clock without real waiting.
    await asyncio.sleep(seconds)


def _backoff_delay(failures: int) -> float:
    return min(
        BACKOFF_START_SECONDS * 2 ** min(max(failures - 1, 0), _BACKOFF_MAX_SHIFT),
        BACKOFF_MAX_SECONDS,
    )


class OpenAIRealtimeSession:
    """Implements the `VoiceSession` protocol in daemon/voice/base.py."""

    name = "openai-realtime"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        system_instruction: str | None = None,
        voice_name: str | None = None,
        tools: Sequence[ToolSpec] = (),
        connect: Any = None,
        url: str = WS_URL,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty")
        if not model:
            raise ValueError("DAEMON_OPENAI_REALTIME_MODEL is empty")
        self._api_key = api_key
        # No `models/` prefix - unlike Gemini, OpenAI Realtime takes the bare id.
        self._model = model
        self._system_instruction = system_instruction
        self._voice_name = voice_name
        self._tools = tuple(tools)
        self._connect = connect if connect is not None else websockets.connect
        self._url = url
        self._max_attempts = max(1, max_attempts)
        # Injectable so an environment with its own CA - a proxy that re-signs
        # TLS - can be served without a switch that turns verification off.
        self._ssl_context = ssl_context
        self._ws: Any = None
        # Transcripts arrive as incremental fragments with no final/partial flag
        # of their own, so completeness is reconstructed here from turn
        # boundaries - see gemini_live.py's `_decode` for the same reasoning.
        self._said: dict[str, list[str]] = {"user": [], "assistant": []}
        # In-progress user speech, pushed as it arrives for `partial_transcripts`.
        self._partials: asyncio.Queue[Transcript | None] = asyncio.Queue(PARTIAL_BACKLOG)
        # Set at a turn boundary and cleared by `receive()`.
        self._turn_over = False
        # Whether this turn owes a user transcript still - set True by
        # `input_audio_buffer.committed`, cleared by the matching `…completed`.
        # See USER_TRANSCRIPT_GRACE_SECONDS. Reset by `receive()` on entry, same
        # as `_turn_over` - a stale owed flag must never leak into the next turn.
        self._user_transcript_owed = False
        # The `item_id` `input_audio_buffer.committed` carried, if any - used
        # only to log a mismatch; see `_decode`'s `_USER_TR_DONE` branch.
        self._committed_item_id: str | None = None
        self._dropping = False
        # Whether the model is mid-generation. `interrupt()` is only allowed to
        # abandon a turn that exists.
        self._generating = False
        self.ended: str | None = None
        """Why `receive()` finished, set as it finishes. Without it a session
        ending looks exactly like a turn ending, and the caller keeps talking into
        a socket that is gone."""
        self._warned_no_video = False
        """`send_frame` logs once, not per frame - see its docstring."""
        self._resample_phase = 0.0
        self._resample_tail = b""
        """State for `_resample_16k_to_24k`, carried across `send_audio` calls so
        the interpolation does not restart at every mic chunk - see that
        function's docstring. Per-session, starts fresh here and is never reset
        mid-call: a new session (hence a new turn boundary) is the only thing
        that should restart it, and a fresh session object always starts here."""
        self._funcs: dict[str, dict[str, Any]] = {}
        """Function-call items accumulated by id, across `response.output_item.added`
        and `response.function_call_arguments.done` events - see `_decode`."""
        self._log_filter = _KeyFilter(api_key)
        self._filtered: list[logging.Logger | logging.Handler] = [logging.getLogger("websockets")]
        # Handler-level too: logging runs the originating logger's filters, never
        # an ancestor's, so a child logger such as websockets.client would walk
        # straight past a filter installed on "websockets" alone.
        self._filtered.extend(logging.getLogger().handlers)
        for target in self._filtered:
            target.addFilter(self._log_filter)

    async def __aenter__(self) -> OpenAIRealtimeSession:
        # __aexit__ never runs when __aenter__ raises, so cleanup has to happen
        # on the way out of *every* failure, not only the ones we anticipated.
        # Catching just OpenAIRealtimeError was not enough: `websockets.connect`
        # can raise ImportError (a SOCKS proxy in ALL_PROXY without python-socks),
        # InvalidProxy or InvalidURI, none of which are InvalidHandshake, so they
        # escaped and left the log filter - which holds the API key in plain text
        # - installed on the root handlers for the life of the process, one more
        # copy per attempt.
        try:
            return await self._enter()
        except BaseException:
            await self.close()
            raise

    async def _enter(self) -> OpenAIRealtimeSession:
        failures = 0
        while True:
            try:
                self._ws = await self._handshake()
            except OpenAIRealtimeError as exc:
                failures += 1
                if exc.permanent or failures >= self._max_attempts:
                    raise
                delay = _backoff_delay(failures)
                logger.warning("openai-realtime: %s; retrying in %.1fs", exc, delay)
                await _sleep(delay)
                continue
            return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        # Before the socket, and on every exit path including a failed handshake: a
        # partial-transcript consumer parked on the queue would otherwise outlive
        # the session it is listening to.
        self._end_partials()
        for target in self._filtered:
            target.removeFilter(self._log_filter)
        self._filtered = []
        if ws is not None:
            await ws.close()

    async def send_audio(self, chunk: bytes) -> None:
        """One PCM chunk from the microphone: 16-bit little-endian mono at 16kHz,
        resampled to the 24kHz pcm16 OpenAI Realtime input requires.

        The resample phase and trailing sample carry across calls (see
        `_resample_16k_to_24k`) - this is the only caller, and the state is
        per-session, so threading it through here rather than recomputing it
        fresh each time is what keeps the interpolation continuous."""
        if not chunk:
            return
        pcm24, self._resample_phase, self._resample_tail = _resample_16k_to_24k(
            chunk, self._resample_phase, self._resample_tail
        )
        if not pcm24:
            # Fewer than two samples buffered so far - nothing to interpolate
            # yet; held in `self._resample_tail` for the next chunk.
            return
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm24).decode("ascii"),
            }
        )

    async def send_frame(self, jpeg: bytes) -> None:
        """No-op: OpenAI Realtime has no realtime video input channel.

        Screen share is Gemini-only (ADR 0009's `realtimeInput.video`); calling
        this on this provider would otherwise fail silently every frame, so it
        warns once instead - loud enough to notice, quiet enough not to spam a
        log at frame rate.
        """
        if not self._warned_no_video:
            logger.warning(
                "openai-realtime: screen share is unsupported on OpenAI; frames are dropped"
            )
            self._warned_no_video = True

    async def send_image(self, jpeg: bytes, note: str) -> None:
        """No-op, for the same reason `send_frame` is: no image input on this
        provider. Shares that method's warn-once flag rather than adding a second
        one - the owner-facing fact is one fact ("this provider cannot see"), and
        `see_screen` is not offered here anyway (`VIDEO_CAPABLE_VOICE_PROVIDERS`).
        """
        if not self._warned_no_video:
            logger.warning(
                "openai-realtime: images are unsupported on OpenAI realtime; the image is dropped"
            )
            self._warned_no_video = True

    async def send_context(self, text: str) -> None:
        """Put text in the model's history without asking it to answer.

        A `conversation.item.create` message item, and deliberately no
        `response.create` after it: that second message is what asks the model to
        answer, and this is the only way recall reaches a voice turn without the
        daemon narrating old memories nobody asked about (see `send_text`).
        """
        if not text.strip():
            logger.debug("openai-realtime: nothing to seed; skipping send_context")
            return
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )

    async def send_text(self, text: str) -> None:
        """Give the model something to say without any user audio.

        Unlike `send_context`, this follows the item with `response.create`: the
        model answers this text rather than reading it back verbatim.
        """
        if not text.strip():
            logger.warning("openai-realtime: refusing to send empty text")
            return
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            }
        )
        await self._send({"type": "response.create"})

    async def send_tool_response(self, results: Sequence[ToolResult]) -> None:
        """Answer this turn's tool calls, one `function_call_output` item per
        result, then one `response.create` to let the model continue.

        Sends nothing when there is nothing to answer - the same reasoning as
        gemini_live.py's version: a needless message on a per-minute-billed
        socket buys nothing.
        """
        if not results:
            logger.debug("openai-realtime: no tool results to send")
            return
        for result in results:
            await self._send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": result.call_id,
                        "output": result.content,
                    },
                }
            )
        await self._send({"type": "response.create"})

    async def interrupt(self) -> None:
        """The user started talking over us.

        Local only, like gemini_live.py's version: under server VAD the user's own
        audio already stopped generation server-side, so this just refuses to hand
        the caller any more of the abandoned turn's audio. A no-op when nothing is
        being generated - same reasoning as the Gemini sibling.
        """
        if not self._generating:
            logger.debug("openai-realtime: nothing is being generated; interrupt does nothing")
            return
        self._dropping = True

    async def receive(self) -> AsyncIterator[bytes | Transcript | Interrupted | ToolCall]:
        """One turn: audio chunks to play, interleaved with completed transcripts.

        Only `final=True` transcripts are ever yielded, and both roles are
        accumulated and released together at the turn boundary - structurally
        identical to gemini_live.py's `receive()`/`_flush`, not just similar to
        them. **Ends at the turn boundary** (`response.done`), and `ended` stays
        None when it does: the turn is over, the session is not.

        User transcript before assistant, always - `_flush()` guarantees the
        order. Reaching the boundary with a user transcript still OWED
        (`_decode` saw `input_audio_buffer.committed` this turn but not yet the
        matching `…completed`), this waits up to `USER_TRANSCRIPT_GRACE_SECONDS`
        for it - see that constant for why. A turn that owes nothing - no
        `committed` this turn, e.g. a `send_text`-only or proactive turn - does
        not wait at all.
        """
        ws = self._require_open()
        # Cleared on the way in, not on the way out - same reasoning as
        # gemini_live.py's `_turn_over`: a caller that walks away mid-flush must
        # not leave a stale "owed" flag to bleed into the next turn.
        self._turn_over = False
        self._user_transcript_owed = False
        self._committed_item_id = None
        closed: OpenAIRealtimeError | None = None
        try:
            async for raw in ws:
                for item in self._decode(raw):
                    yield item
                if self._turn_over:
                    if self._user_transcript_owed:
                        try:
                            async with asyncio.timeout(USER_TRANSCRIPT_GRACE_SECONDS):
                                async for late_raw in ws:
                                    for item in self._decode(late_raw):
                                        yield item
                                    if not self._user_transcript_owed:
                                        break
                        except TimeoutError:
                            # Measured ~200ms; nothing arrived within the grace
                            # window, so there is nothing left to wait for.
                            pass
                    for item in self._flush():
                        yield item
                    return
        except ConnectionClosedOK:
            pass
        except ConnectionClosed as exc:
            closed = self._closed_error(exc)
        # A dropped connection must not swallow the last thing that was said:
        # flush before surfacing the failure.
        for item in self._flush():
            yield item
        self.ended = str(closed) if closed is not None else "the server closed the stream"
        self._end_partials()
        if closed is not None:
            raise closed

    def pending_transcripts(self) -> list[Transcript]:
        """Take what has been transcribed but not yet released - see
        gemini_live.py's version for why this exists."""
        return list(self._flush())

    async def partial_transcripts(self) -> AsyncIterator[Transcript]:
        """What this yields today, honestly: **one item per turn**, not one per
        delta.

        whisper-1 - the only transcription model `_setup_message` requests -
        never sends `conversation.item.input_audio_transcription.delta`; the
        `_USER_TR_DELTA` branch in `_decode` is kept for a future delta-emitting
        transcription model but does not fire in production today. What
        actually reaches `self._partials` is the *complete* transcript from
        `…completed`, labelled `final=False` - see the push in `_decode`'s
        `_USER_TR_DONE` branch for why a finished transcript is offered as a
        partial. So a caller of this method gets the whole completed utterance
        once per turn, never a growing fragment - unlike gemini_live.py's
        version, which this used to (and no longer does) claim to match.

        `final=False` always, user only, ends when the session does - that
        part is still true and unchanged.
        """
        while True:
            partial = await self._partials.get()
            if partial is None:
                return
            yield partial

    def _require_open(self) -> Any:
        if self._ws is None:
            raise OpenAIRealtimeError("session is not open; use `async with`", permanent=True)
        return self._ws

    async def _send(self, message: dict[str, Any]) -> None:
        ws = self._require_open()
        error: OpenAIRealtimeError | None = None
        try:
            await ws.send(json.dumps(message))
        except ConnectionClosed as exc:
            error = self._closed_error(exc)
        if error is not None:
            # Raised outside the except block on purpose - see `_open`.
            raise error

    async def _handshake(self) -> Any:
        """Connect, configure, and wait to be told the configuration took.

        One unit, because a session is unusable until `session.updated` arrives -
        retrying the socket without the setup would leave a half-open connection
        billing quietly.
        """
        ws = await self._open()
        error: OpenAIRealtimeError | None = None
        try:
            await ws.send(json.dumps(self._setup_message()))
            async with asyncio.timeout(SETUP_TIMEOUT_SECONDS):
                async for raw in ws:
                    message = self._parse(raw)
                    if message is None:
                        continue
                    if message.get("type") == "session.updated":
                        return ws
                    if message.get("type") == "error":
                        # The socket stays open on a rejected `session.update` -
                        # OpenAI reports it as a top-level error event rather than
                        # closing. Left undetected, this loop would wait out the
                        # full SETUP_TIMEOUT_SECONDS and then raise a misleading
                        # "no session.updated" message for what is actually a bad
                        # setup payload - a bug, not weather, so it is permanent.
                        error = OpenAIRealtimeError(
                            self._redact(f"session.update rejected: {self._error_detail(message)}"),
                            permanent=True,
                        )
                        break
                    # Nothing else is required before setup completes.
                    logger.debug("openai-realtime: ignoring pre-setup message %s", sorted(message))
        except ConnectionClosed as exc:
            error = self._closed_error(exc)
        except TimeoutError:
            error = OpenAIRealtimeError(f"no session.updated within {SETUP_TIMEOUT_SECONDS:.0f}s")
        # Outside the except block, so nothing is left for __context__ to point
        # at - see `_open`. A half-open connection is closed either way: it is
        # already billing.
        await self._discard(ws)
        raise error or OpenAIRealtimeError("connection ended before session.updated")

    async def _open(self) -> Any:
        error: OpenAIRealtimeError | None = None
        bundle = _ca_bundle()
        try:
            # Bearer header, not a query param: a key in a URL ends up in every
            # error string that quotes the URI. The model id, unlike the key,
            # carries no secret and is the query param OpenAI's endpoint wants.
            return await self._connect(
                f"{self._url}?model={self._model}",
                # GA drops the `OpenAI-Beta: realtime=v1` header; sending it (plus the
                # old flat session shape) makes gpt-realtime close 4000
                # invalid_request_error.beta_api_shape_disabled. Measured on the live
                # socket, 2026-08-09.
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                # Explicit, never the library default - see `_ssl_context`.
                ssl=self._ssl_context or _ssl_context(bundle),
            )
        except ssl.SSLError as exc:
            # Before the OSError branch, which SSLError is a subclass of. A bare
            # "could not connect" here sent people looking at their own network:
            # the cause is a trust store, and the fix is naming a different one.
            detail = self._redact(f"{type(exc).__name__}: {exc}")
            error = OpenAIRealtimeError(
                f"TLS verification failed: {detail}. Certificates were verified "
                f"against {bundle}, not the system trust store. If you are behind "
                f"a proxy that re-signs TLS, point {CA_BUNDLE_ENV[0]} at its CA "
                "bundle; certificate verification is not optional."
            )
        except InvalidStatus as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            error = OpenAIRealtimeError(
                f"handshake rejected with HTTP {status}: {self._redact(str(exc))}",
                permanent=status in _PERMANENT_STATUS,
            )
        except ConnectionClosed as exc:
            error = self._closed_error(exc)
        except (InvalidHandshake, OSError, TimeoutError) as exc:
            detail = self._redact(f"{type(exc).__name__}: {exc}")
            error = OpenAIRealtimeError(f"could not connect: {detail}")
        except Exception as exc:
            # Deliberately broad. Anything else from the client - a proxy
            # misconfiguration, a bad URI, an optional dependency missing - is
            # still a connect failure, and letting it through unwrapped means an
            # exception whose message or chain may quote the URI escapes
            # redaction entirely. Non-permanent, so the existing backoff applies.
            detail = self._redact(f"{type(exc).__name__}: {exc}")
            error = OpenAIRealtimeError(f"could not connect: {detail}")
        # Raised out here, with no exception active, for the same reason
        # telegram.py does it: `raise ... from None` clears __cause__ but leaves
        # __context__ pointing at the original, and anything that walks the chain
        # - an error reporter, traceback.format_exception, a pytest failure
        # report - would print it, key and all.
        raise error

    async def _discard(self, ws: Any) -> None:
        try:
            await ws.close()
        except (ConnectionClosed, OSError):
            # Already on a failure path; a socket that will not close cleanly
            # must not replace the error that explains why we are here.
            logger.debug("openai-realtime: closing a failed connection did not go cleanly")

    def _setup_message(self) -> dict[str, Any]:
        # The GA (gpt-realtime) session shape, confirmed against the live socket's own
        # `session.created` (2026-08-09): audio config nested under audio.{input,output},
        # `output_modalities` not `modalities`, each format an object {type,rate} not the
        # beta "pcm16" string, and `voice` under audio.output. The old flat beta shape is
        # rejected with a 4000 invalid_request_error.beta_api_shape_disabled close.
        # `rate` is OUTPUT_SAMPLE_RATE (24 kHz) both ways: GA input is 24 kHz, which is
        # exactly what send_audio upsamples the 16 kHz mic capture to.
        audio: dict[str, Any] = {
            "input": {
                "format": {"type": "audio/pcm", "rate": OUTPUT_SAMPLE_RATE},
                "turn_detection": {"type": "server_vad"},
                "transcription": {"model": "whisper-1"},
            },
            "output": {"format": {"type": "audio/pcm", "rate": OUTPUT_SAMPLE_RATE}},
        }
        if self._voice_name:
            audio["output"]["voice"] = self._voice_name
        session: dict[str, Any] = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": audio,
        }
        if self._system_instruction:
            session["instructions"] = self._system_instruction
        if self._tools:
            session["tools"] = self._tool_declarations()
        return {"type": "session.update", "session": session}

    def _tool_declarations(self) -> list[dict[str, Any]]:
        # OpenAI Realtime function tool shape: flat, type "function". `parameters`
        # is passed through unchanged - the same choice llm/providers/openai.py
        # makes for the text path (its Responses payload is flat the same way),
        # not narrowed through a schema helper because none is needed there.
        return [
            {
                "type": "function",
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in self._tools
        ]

    def _parse(self, raw: str | bytes) -> dict[str, Any] | None:
        try:
            message = json.loads(raw)
        except ValueError:
            logger.warning("openai-realtime: dropping a message that is not JSON")
            return None
        return message if isinstance(message, dict) else None

    def _decode(self, raw: str | bytes) -> Iterator[bytes | Transcript | Interrupted | ToolCall]:
        try:
            msg = json.loads(raw)
        except ValueError:
            logger.warning("openai-realtime: dropping a non-JSON message")
            return
        if not isinstance(msg, dict):
            return
        t = msg.get("type")
        if t in _AUDIO_DELTA:
            self._generating = True
            if not self._dropping:
                try:
                    yield base64.b64decode(msg.get("delta") or "", validate=True)
                except Exception:
                    logger.warning("openai-realtime: undecodable audio delta")
            return
        if t == _SPEECH_STARTED:
            if self._generating:
                self._dropping = True
                yield Interrupted()
            return
        if t in _ASSISTANT_TR_DELTA:
            self._said["assistant"].append(msg.get("delta") or "")
            return
        if t in _ASSISTANT_TR_DONE:
            # `transcript` carries the full text; prefer it over accumulated deltas.
            text = msg.get("transcript")
            if isinstance(text, str) and text:
                self._said["assistant"] = [text]
            return
        if t in _USER_TR_DELTA:
            # Kept for a future delta-emitting transcription model. whisper-1 -
            # the only model `_setup_message` requests - never sends this event;
            # it emits only the `completed` event below, so this branch does not
            # fire in production today (measured; see the design doc's
            # wire-mapping table). Left in, not deleted, so `partial_transcripts()`
            # starts working the day the configured model does, with no code
            # change here.
            self._said["user"].append(msg.get("delta") or "")
            self._push_partial()
            return
        if t == _INPUT_COMMITTED:
            # The server committed the user's audio for this turn - measured
            # arriving well before `response.created`. This is the only clean
            # signal that a `…completed` transcript is coming at all: a
            # `send_text`-only or proactive turn never sends this event, and
            # `receive()`'s grace wait is conditioned on it (see
            # USER_TRANSCRIPT_GRACE_SECONDS) rather than paid unconditionally.
            item_id = msg.get("item_id")
            self._user_transcript_owed = True
            self._committed_item_id = item_id if isinstance(item_id, str) else None
            return
        if t in _USER_TR_DONE:
            # Accumulated like the assistant branch above - `transcript` carries
            # the full text, replacing whatever deltas built up - never yielded
            # directly. `receive()`/`_flush()` release both roles together at
            # the turn boundary, user first, exactly like gemini_live.py; a
            # direct yield here was the ordering bug: whisper-1 commonly
            # delivers this ~200ms *after* `response.done`, which decoded it at
            # the START of the NEXT `receive()` call and recorded it as the next
            # turn's user utterance instead of this one's.
            item_id = msg.get("item_id")
            if (
                self._committed_item_id is not None
                and isinstance(item_id, str)
                and item_id != self._committed_item_id
            ):
                # Ids should line up, but a mismatch must not drop the
                # transcript - it is still this turn's answer to "did the user
                # say anything", so it settles whatever was pending anyway.
                logger.debug(
                    "openai-realtime: transcription item_id %s does not match "
                    "committed item_id %s; settling the pending turn anyway",
                    item_id,
                    self._committed_item_id,
                )
            self._user_transcript_owed = False
            self._committed_item_id = None
            text = msg.get("transcript")
            if isinstance(text, str) and text:
                self._said["user"] = [text]
                stripped = text.strip()
                if stripped:
                    # Offered as a partial (`final=False`) even though whisper-1
                    # already delivered the whole utterance here, not a
                    # fragment: `final=True` would let this be recorded as an
                    # utterance twice, once here and once from `_flush()`'s
                    # release at the turn boundary. See `partial_transcripts`'s
                    # docstring - this is the one place that produces what it
                    # yields.
                    self._offer_partial(Transcript(text=stripped, role="user", final=False))
            return
        if t == _OUTPUT_ITEM_ADDED:
            item = msg.get("item") or {}
            if item.get("type") == "function_call":
                cid = item.get("call_id") or item.get("id") or ""
                self._funcs[cid] = {"name": item.get("name")}
            return
        if t == _FUNC_ARGS_DONE:
            cid = msg.get("call_id") or ""
            rec = self._funcs.pop(cid, {"name": None})
            name = rec.get("name")
            if isinstance(name, str) and name:
                if self._tools:
                    yield ToolCall(
                        id=cid or synthesise_call_id(name, 0),
                        name=name,
                        arguments=decode_tool_arguments(msg.get("arguments")),
                    )
                else:
                    # A session that declared nothing cannot legitimately be asked
                    # for anything - see gemini_live.py's `_decode_tool_calls` for
                    # the same drop, logged there for the same reason: silence
                    # here reads as a config mismatch nobody was told about.
                    logger.warning(
                        "openai-realtime: dropping a call to %r - no tool was "
                        "offered in setup",
                        name,
                    )
            else:
                # `output_item.added` was missing, or its `call_id` did not line
                # up with this event's - either way there is no name to run and
                # no way to answer this call, so it is dropped rather than
                # silently ignored.
                logger.warning(
                    "openai-realtime: dropping a function-call-args event with "
                    "no matching name/id"
                )
            return
        if t == "error":
            logger.warning("openai-realtime: server error %s", msg.get("error"))
            return
        if t == _RESPONSE_DONE:
            self._dropping = False
            self._generating = False
            # No flush here - see `receive()`. Whether a user transcript is
            # still owed decides whether receive() waits before flushing; doing
            # the flush here, unconditionally, is the ordering bug this exists
            # to avoid - it would hand the assistant's words over before a user
            # transcript that is already in flight.
            self._turn_over = True
            return

    def _flush(self) -> Iterator[Transcript]:
        """Release the accumulated turn, user first: that is the order it
        happened in, and the order memory should record it in.

        Ported from gemini_live.py's `_flush` - same shape, same reasoning, one
        role at a time, cleared as it is handed over rather than all at once:
        `yield` is a suspension point, so a caller cancelled between the two
        would take the second transcript with it. Left where it is,
        `pending_transcripts` can still recover it."""
        for role in ("user", "assistant"):
            fragments = self._said[role]
            self._said[role] = []
            text = "".join(fragments).strip()
            if text:
                yield Transcript(text=text, role=role, final=True)

    def _push_partial(self) -> None:
        """Offer the utterance so far to whoever is listening for it. Never
        blocks and never raises - see gemini_live.py's version."""
        text = "".join(self._said["user"]).strip()
        if not text:
            return
        self._offer_partial(Transcript(text=text, role="user", final=False))

    def _end_partials(self) -> None:
        """Close the partial stream so `async for` over it finishes rather than
        waiting on a session that is gone."""
        self._offer_partial(None)

    def _offer_partial(self, item: Transcript | None) -> None:
        while self._partials.full():
            # Drop the oldest, which the newest already contains in full. Room for
            # the sentinel matters most of all: without it a consumer would wait
            # forever on a backlog nobody is reading.
            self._partials.get_nowait()
        self._partials.put_nowait(item)

    def _closed_error(self, exc: ConnectionClosed) -> OpenAIRealtimeError:
        received = exc.rcvd
        code = received.code if received is not None else 1006
        reason = self._redact(received.reason) if received is not None else ""
        return OpenAIRealtimeError(
            f"connection closed {code}: {reason or 'no reason given'}",
            permanent=_permanent_close(code, reason),
        )

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "<key>")

    def _error_detail(self, message: dict[str, Any]) -> str:
        """The human-readable part of a top-level `error` event, whatever shape
        it arrives in - OpenAI's documented shape is `{"error": {"message": ...}}`,
        but a detail worth surfacing should not depend on that holding exactly."""
        error = message.get("error")
        if isinstance(error, dict):
            text = error.get("message")
            if isinstance(text, str) and text:
                return text
            return str(error)
        return str(error)
