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
from daemon.voice.base import Interrupted, Transcript

# `ToolResult` is not imported here - nothing in this task's slice (`receive()`'s
# decoder) sends a tool result back yet. That comes with `send_tool_response()` in
# Task 4.

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

# Server event types (accept GA and beta spellings).
_AUDIO_DELTA = ("response.output_audio.delta", "response.audio.delta")
_ASSISTANT_TR_DELTA = ("response.output_audio_transcript.delta", "response.audio_transcript.delta")
_ASSISTANT_TR_DONE = ("response.output_audio_transcript.done", "response.audio_transcript.done")
_USER_TR_DELTA = ("conversation.item.input_audio_transcription.delta",)
_USER_TR_DONE = ("conversation.item.input_audio_transcription.completed",)
_SPEECH_STARTED = "input_audio_buffer.speech_started"
_RESPONSE_DONE = "response.done"
_FUNC_ARGS_DONE = "response.function_call_arguments.done"
_OUTPUT_ITEM_ADDED = "response.output_item.added"

_PERMANENT_STATUS = frozenset({400, 401, 403, 404})
"""Handshake statuses that retrying cannot fix: a bad, revoked or unauthorised
key, or a wrong endpoint."""

_PERMANENT_CLOSE_CODES = frozenset({1007, 1008})  # invalid payload / policy; confirm in spike
"""Confirm in evals/openai_realtime_spike.py once a real key is available - ported
as the closest analogue of gemini_live.py's classification, not yet measured
against OpenAI's own socket."""


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


def _upsample_16k_to_24k(pcm: bytes) -> bytes:
    """16-bit LE mono, 16 kHz -> 24 kHz by linear interpolation (ratio 3/2).

    OpenAI Realtime pcm16 input is fixed at 24 kHz; the mic captures 16 kHz. Only the
    input needs this - output is already 24 kHz (AudioIO.playback_sample_rate). Speech
    band-limits to ~8 kHz either way, so interpolating adds no artefacts a listener hears.
    """
    n = len(pcm) // 2
    if n == 0:
        return b""
    src = [int.from_bytes(pcm[i * 2:i * 2 + 2], "little", signed=True) for i in range(n)]
    out_n = (n * 3) // 2
    out = bytearray()
    for j in range(out_n):
        pos = j * (n - 1) / max(out_n - 1, 1) if out_n > 1 else 0.0
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        val = int(round(src[lo] + (src[hi] - src[lo]) * frac))
        val = max(-32768, min(32767, val))
        out += val.to_bytes(2, "little", signed=True)
    return bytes(out)


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
    """Implements the `VoiceSession` protocol in daemon/voice/base.py.

    `send_tool_response`/`interrupt` land in a later task (docs/PLAN.md Phase B1).
    """

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
        self._dropping = False
        # Whether the model is mid-generation. `interrupt()` is only allowed to
        # abandon a turn that exists.
        self._generating = False
        self.ended: str | None = None
        """Why `receive()` finished, set as it finishes. Without it a session
        ending looks exactly like a turn ending, and the caller keeps talking into
        a socket that is gone."""
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
        upsampled to the 24kHz pcm16 OpenAI Realtime input requires."""
        if not chunk:
            return
        pcm24 = _upsample_16k_to_24k(chunk)
        await self._send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm24).decode("ascii"),
            }
        )

    async def receive(self) -> AsyncIterator[bytes | Transcript | Interrupted | ToolCall]:
        """One turn: audio chunks to play, interleaved with completed transcripts.

        Only `final=True` transcripts are ever yielded - see gemini_live.py's
        `receive()` for why deltas are accumulated rather than handed out as they
        arrive. **Ends at the turn boundary** (`response.done`), and `ended` stays
        None when it does: the turn is over, the session is not.
        """
        ws = self._require_open()
        self._turn_over = False
        closed: OpenAIRealtimeError | None = None
        try:
            async for raw in ws:
                for item in self._decode(raw):
                    yield item
                if self._turn_over:
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
        """The user's utterance as it grows, one item per delta that arrives.

        See gemini_live.py's version - identical reasoning, `final=False` always,
        user only, ends when the session does.
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
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "OpenAI-Beta": "realtime=v1",
                },
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
        session: dict[str, Any] = {
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad"},
            "input_audio_transcription": {"model": "whisper-1"},
        }
        if self._voice_name:
            session["voice"] = self._voice_name
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
            self._said["user"].append(msg.get("delta") or "")
            self._push_partial()
            return
        if t in _USER_TR_DONE:
            text = msg.get("transcript")
            if isinstance(text, str) and text:
                self._said["user"] = [text]
                self._push_partial()
            return
        if t == _OUTPUT_ITEM_ADDED:
            item = msg.get("item") or {}
            if item.get("type") == "function_call":
                cid = item.get("call_id") or item.get("id") or ""
                self._funcs[cid] = {"name": item.get("name"), "args": ""}
            return
        if t == _FUNC_ARGS_DONE:
            cid = msg.get("call_id") or ""
            rec = self._funcs.pop(cid, {"name": None})
            name = rec.get("name")
            if isinstance(name, str) and name and self._tools:
                yield ToolCall(
                    id=cid or synthesise_call_id(name, 0),
                    name=name,
                    arguments=decode_tool_arguments(msg.get("arguments")),
                )
            return
        if t == "error":
            logger.warning("openai-realtime: server error %s", msg.get("error"))
            return
        if t == _RESPONSE_DONE:
            self._dropping = False
            self._generating = False
            yield from self._flush()
            self._turn_over = True
            return

    def _flush(self) -> Iterator[Transcript]:
        """Release the accumulated turn, user first - see gemini_live.py's version
        for why one role at a time, cleared as it is handed over."""
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
