"""Gemini Live - hosted native audio behind `VoiceSession` (docs/PLAN.md 6.5).

Two HTTP-ish operations and one long-lived socket do not justify the google-genai
SDK, so this speaks the BidiGenerateContent WebSocket protocol directly, the same
way daemon/channels/telegram.py speaks the Bot API with httpx. Protocol reference:
https://ai.google.dev/api/live and
https://ai.google.dev/gemini-api/docs/live-api/get-started-websocket

Three things here are load-bearing rather than incidental:

1. **Transcription is always on, both directions.** It is the entire reason
   hosted native audio was chosen over a local cascade: the transcripts are what
   let memory (docs/PLAN.md 4) and persona evolution (5) keep working in voice
   mode. A session without them would still talk and would silently forget.
2. **Sessions are opened per conversation, never held.** Billing is per minute,
   so an idle socket is pure cost - hence no reconnect-forever loop and no
   keepalive. A proactive utterance opens, speaks, and closes.
3. **The API key never reaches a log, an exception, or a traceback.** It is sent
   as a header rather than the documented `?key=` query param so it stays out of
   URLs, and `_KeyFilter` scrubs it from websockets' own DEBUG records, which
   include the handshake headers.
4. **TLS trust comes from certifi, explicitly.** See `_ssl_context`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import functools
import json
import logging
import os
import ssl
from collections.abc import AsyncIterator, Iterator
from typing import Any

import certifi
import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedOK,
    InvalidHandshake,
    InvalidStatus,
)

from daemon.voice.base import Transcript

logger = logging.getLogger(__name__)

WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
"""v1beta is the current version for API-key auth; v1alpha exists only for
ephemeral tokens, which need a different method name entirely."""

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 24_000
"""The two rates differ, and the docs are explicit about both: input is raw
16-bit little-endian PCM at 16kHz, output at 24kHz. Feeding the speaker at the
microphone's rate plays the model back chipmunked, which reads as a broken model
rather than a broken constant."""

INPUT_MIME_TYPE = f"audio/pcm;rate={INPUT_SAMPLE_RATE}"

SETUP_TIMEOUT_SECONDS = 20.0
"""A server that accepts the socket and never answers setup would otherwise hang
the voice turn forever."""

BACKOFF_START_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
_BACKOFF_MAX_SHIFT = 5
"""Caps the doubling at 32s. The exponent is clamped, not just the result: in
telegram.py `2 ** (failures - 1)` kept growing behind a min() and eventually
raised OverflowError from inside the retry handler, killing the loop exactly when
the network came back."""

DEFAULT_MAX_ATTEMPTS = 4
"""Bounded on purpose. Unlike an inbound channel, nobody is waiting to be heard
here - a voice turn that cannot open should fail and let the caller fall back to
text, not retry into a per-minute-billed connection forever."""

_PERMANENT_STATUS = frozenset({400, 401, 403, 404})
"""Handshake statuses that retrying cannot fix: a bad, revoked or unauthorised
key, or a wrong endpoint."""

_PERMANENT_CLOSE_CODES = frozenset({1007, 1008})
"""1008 policy violation is how a rejected or leaked key actually arrives - the
handshake succeeds and the server closes straight after. 1007 invalid argument
means our setup JSON is wrong, which is a bug, not weather."""

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


class GeminiLiveError(Exception):
    """A live session failed. Never carries the API key - see `_redact`."""

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


class GeminiLiveSession:
    """Implements the `VoiceSession` protocol in daemon/voice/base.py."""

    name = "gemini-live"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        system_instruction: str | None = None,
        voice_name: str | None = None,
        connect: Any = None,
        url: str = WS_URL,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty")
        if not model:
            raise ValueError("DAEMON_GEMINI_LIVE_MODEL is empty")
        self._api_key = api_key
        # The wire format is `models/{id}`; accept either so a config value
        # copied straight from the docs works.
        self._model = model if model.startswith("models/") else f"models/{model}"
        self._system_instruction = system_instruction
        # No language code: the native-audio models pick the language themselves
        # and reject being told, which matters for a Korean-first user.
        self._voice_name = voice_name
        self._connect = connect if connect is not None else websockets.connect
        self._url = url
        self._max_attempts = max(1, max_attempts)
        # Injectable so an environment with its own CA - a proxy that re-signs
        # TLS - can be served without a switch that turns verification off.
        self._ssl_context = ssl_context
        self._ws: Any = None
        # Transcripts arrive as incremental fragments with no final/partial flag
        # of their own, so completeness is reconstructed here from turn
        # boundaries. See `_decode`.
        self._said: dict[str, list[str]] = {"user": [], "assistant": []}
        self._dropping = False
        # Whether the model is mid-generation. `interrupt()` is only allowed to
        # abandon a turn that exists - see `interrupt`.
        self._generating = False
        self.going_away = False
        """The server announced it is about to cut the session (`goAway`)."""
        self.ended: str | None = None
        """Why `receive()` finished, set as it finishes. Without it a session
        ending looks exactly like a turn ending, and the caller keeps talking into
        a socket that is gone."""
        self._log_filter = _KeyFilter(api_key)
        self._filtered: list[logging.Logger | logging.Handler] = [logging.getLogger("websockets")]
        # Handler-level too: logging runs the originating logger's filters, never
        # an ancestor's, so a child logger such as websockets.client would walk
        # straight past a filter installed on "websockets" alone.
        self._filtered.extend(logging.getLogger().handlers)
        for target in self._filtered:
            target.addFilter(self._log_filter)

    async def __aenter__(self) -> GeminiLiveSession:
        # __aexit__ never runs when __aenter__ raises, so cleanup has to happen
        # on the way out of *every* failure, not only the ones we anticipated.
        # Catching just GeminiLiveError was not enough: `websockets.connect` can
        # raise ImportError (a SOCKS proxy in ALL_PROXY without python-socks),
        # InvalidProxy or InvalidURI, none of which are InvalidHandshake, so they
        # escaped and left the log filter - which holds the API key in plain text
        # - installed on the root handlers for the life of the process, one more
        # copy per attempt.
        try:
            return await self._enter()
        except BaseException:
            await self.close()
            raise

    async def _enter(self) -> GeminiLiveSession:
        failures = 0
        while True:
            try:
                self._ws = await self._handshake()
            except GeminiLiveError as exc:
                failures += 1
                if exc.permanent or failures >= self._max_attempts:
                    raise
                delay = _backoff_delay(failures)
                logger.warning("gemini-live: %s; retrying in %.1fs", exc, delay)
                await _sleep(delay)
                continue
            return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        for target in self._filtered:
            target.removeFilter(self._log_filter)
        self._filtered = []
        if ws is not None:
            await ws.close()

    async def send_audio(self, chunk: bytes) -> None:
        """One PCM chunk from the microphone: 16-bit little-endian mono at 16kHz."""
        if not chunk:
            return
        await self._send(
            {
                "realtimeInput": {
                    "audio": {
                        # proto bytes over JSON: base64 as an ASCII string.
                        "data": base64.b64encode(chunk).decode("ascii"),
                        "mimeType": INPUT_MIME_TYPE,
                    }
                }
            }
        )

    async def send_text(self, text: str) -> None:
        """Give the model something to say without any user audio.

        Note the honest limitation: the realtime text stream is a *prompt*, not a
        verbatim TTS instruction, so the model answers this text rather than
        reading it out. Proactive utterances that must come out word for word go
        to the local speaker instead (docs/PLAN.md 6.3), which is also the path
        that never leaves the machine.
        """
        if not text.strip():
            # An empty proactive utterance is not rare, and spending a
            # per-minute-billed session on one is worse than skipping it.
            logger.warning("gemini-live: refusing to send empty text")
            return
        await self._send({"realtimeInput": {"text": text}})

    async def receive(self) -> AsyncIterator[bytes | Transcript]:
        """Audio chunks to play, interleaved with completed transcripts.

        Only `final=True` transcripts are ever yielded. Gemini streams
        transcription in fragments - a few syllables at a time - and handing
        those out individually would invite a caller to record half a word as an
        utterance. They are accumulated and released at the turn boundary.
        """
        ws = self._require_open()
        closed: GeminiLiveError | None = None
        self.ended = None
        try:
            async for raw in ws:
                for item in self._decode(raw):
                    yield item
        except ConnectionClosedOK:
            pass
        except ConnectionClosed as exc:
            closed = self._closed_error(exc)
        # A dropped connection must not swallow the last thing that was said:
        # flush before surfacing the failure.
        for item in self._flush():
            yield item
        # Say what ended, and say it before raising, so a caller that only
        # catches the error still learns whether the session is recoverable. A
        # `goAway` in particular arrives *before* the session limit and then the
        # stream simply stops - indistinguishable from a finished turn, which is
        # how the daemon ended up sending audio into a closed socket and, from the
        # user's side, stopping mid-conversation.
        self.ended = (
            str(closed)
            if closed is not None
            else "goAway: the server ended the session, not the turn"
            if self.going_away
            else "the server closed the stream"
        )
        if closed is not None:
            # Raised out here rather than in the handler, so __context__ has no
            # websockets exception to point at - see `_open`.
            raise closed

    async def interrupt(self) -> None:
        """The user started talking over us.

        Local only, and deliberately so: the protocol has no client-side cancel
        message. Under the default server-side activity detection the user's own
        audio is what stops generation, and the server confirms it with
        `serverContent.interrupted`. What this does is refuse to hand the caller
        any more audio from the abandoned turn - audio that is already generated
        and still arriving. Dropping the speaker's own queue is
        `AudioIO.stop_playback`; both are needed.

        A no-op when nothing is being generated. That guard is not defensive
        tidiness: `_dropping` outlived the turn it was set for, so an interrupt
        arriving in a silence dropped *the next* answer's audio in full - while its
        transcript still accumulated and still flushed. Memory then held a reply
        the user never heard, which is worse than either failure alone.
        """
        if not self._generating:
            logger.debug("gemini-live: nothing is being generated; interrupt does nothing")
            return
        self._dropping = True

    def pending_transcripts(self) -> list[Transcript]:
        """Take what has been transcribed but not yet released.

        The escape hatch for cancellation. `receive()` accumulates until the turn
        boundary and an async generator cannot yield from its own `finally`, so a
        shutdown or an upper-layer timeout arriving before `turnComplete` would
        drop the utterance - and in voice mode the transcript is the *only* thing
        memory ever gets. Draining is destructive so a later flush cannot record
        the same words twice.
        """
        return list(self._flush())

    def partial_transcripts(self) -> list[Transcript]:
        """A peek at the turn so far, without consuming it.

        `final=False`, always: these are deltas, and the flag is what stops a
        caller recording a syllable as an utterance. Their one purpose is to let
        recall start embedding while the user is still speaking - the 117 ms
        round trip is free before the utterance ends and unaffordable after it
        (docs/PLAN.md 4.3.1).
        """
        return self._snapshot(final=False)

    def _require_open(self) -> Any:
        if self._ws is None:
            raise GeminiLiveError("session is not open; use `async with`", permanent=True)
        return self._ws

    async def _send(self, message: dict[str, Any]) -> None:
        ws = self._require_open()
        error: GeminiLiveError | None = None
        try:
            await ws.send(json.dumps(message))
        except ConnectionClosed as exc:
            error = self._closed_error(exc)
        if error is not None:
            # Raised outside the except block on purpose - see `_open`.
            raise error

    async def _handshake(self) -> Any:
        """Connect, configure, and wait to be told the configuration took.

        One unit, because a session is unusable until `setupComplete` arrives -
        retrying the socket without the setup would leave a half-open connection
        billing quietly.
        """
        ws = await self._open()
        error: GeminiLiveError | None = None
        try:
            await ws.send(json.dumps(self._setup_message()))
            async with asyncio.timeout(SETUP_TIMEOUT_SECONDS):
                async for raw in ws:
                    message = self._parse(raw)
                    if message is None:
                        continue
                    if "setupComplete" in message:
                        return ws
                    # Nothing else is legal before setup completes; the server
                    # closes on a bad setup rather than replying.
                    logger.debug("gemini-live: ignoring pre-setup message %s", sorted(message))
        except ConnectionClosed as exc:
            error = self._closed_error(exc)
        except TimeoutError:
            error = GeminiLiveError(f"no setupComplete within {SETUP_TIMEOUT_SECONDS:.0f}s")
        # Outside the except block, so nothing is left for __context__ to point
        # at - see `_open`. A half-open connection is closed either way: it is
        # already billing.
        await self._discard(ws)
        raise error or GeminiLiveError("connection ended before setupComplete")

    async def _open(self) -> Any:
        error: GeminiLiveError | None = None
        bundle = _ca_bundle()
        try:
            # Header rather than the documented `?key=` query param: a key in a
            # URL ends up in every error string that quotes the URI.
            return await self._connect(
                self._url,
                additional_headers={"x-goog-api-key": self._api_key},
                # Explicit, never the library default - see `_ssl_context`.
                ssl=self._ssl_context or _ssl_context(bundle),
            )
        except ssl.SSLError as exc:
            # Before the OSError branch, which SSLError is a subclass of. A bare
            # "could not connect" here sent people looking at their own network:
            # the cause is a trust store, and the fix is naming a different one.
            detail = self._redact(f"{type(exc).__name__}: {exc}")
            error = GeminiLiveError(
                f"TLS verification failed: {detail}. Certificates were verified "
                f"against {bundle}, not the system trust store. If you are behind "
                f"a proxy that re-signs TLS, point {CA_BUNDLE_ENV[0]} at its CA "
                "bundle; certificate verification is not optional."
            )
        except InvalidStatus as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            error = GeminiLiveError(
                f"handshake rejected with HTTP {status}: {self._redact(str(exc))}",
                permanent=status in _PERMANENT_STATUS,
            )
        except ConnectionClosed as exc:
            error = self._closed_error(exc)
        except (InvalidHandshake, OSError, TimeoutError) as exc:
            detail = self._redact(f"{type(exc).__name__}: {exc}")
            error = GeminiLiveError(f"could not connect: {detail}")
        except Exception as exc:
            # Deliberately broad. Anything else from the client - a proxy
            # misconfiguration, a bad URI, an optional dependency missing - is
            # still a connect failure, and letting it through unwrapped means an
            # exception whose message or chain may quote the URI escapes
            # redaction entirely. Non-permanent, so the existing backoff applies.
            detail = self._redact(f"{type(exc).__name__}: {exc}")
            error = GeminiLiveError(f"could not connect: {detail}")
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
            logger.debug("gemini-live: closing a failed connection did not go cleanly")

    def _setup_message(self) -> dict[str, Any]:
        generation: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        if self._voice_name:
            generation["speechConfig"] = {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice_name}}
            }
        setup: dict[str, Any] = {
            "model": self._model,
            "generationConfig": generation,
            # Not optional and not configurable. docs/PLAN.md 6.5 rests on these
            # two lines: without transcripts, voice mode still talks but memory
            # and persona evolution get nothing, which is the differentiator
            # switching itself off.
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }
        if self._system_instruction:
            setup["systemInstruction"] = {"parts": [{"text": self._system_instruction}]}
        return {"setup": setup}

    def _parse(self, raw: str | bytes) -> dict[str, Any] | None:
        try:
            message = json.loads(raw)
        except ValueError:
            logger.warning("gemini-live: dropping a message that is not JSON")
            return None
        return message if isinstance(message, dict) else None

    def _decode(self, raw: str | bytes) -> Iterator[bytes | Transcript]:
        message = self._parse(raw)
        if message is None:
            return
        if "goAway" in message:
            # The session is about to be cut. Sessions are short by design, so
            # there is nothing to do about it - but it has to be *distinguishable*
            # from a turn ending, hence the flag as well as the log line.
            left = (message.get("goAway") or {}).get("timeLeft")
            self.going_away = True
            logger.info("gemini-live: server will disconnect, timeLeft=%s", left)
        content = message.get("serverContent")
        if not isinstance(content, dict):
            return

        # Read the interruption flag before yielding audio: one server event may
        # carry several fields at once, and audio from an interrupted turn must
        # not slip out ahead of the flag that condemns it.
        if content.get("interrupted"):
            self._dropping = True

        for part in (content.get("modelTurn") or {}).get("parts") or []:
            data = (part.get("inlineData") or {}).get("data")
            if not data:
                continue
            # Set even when the audio is dropped: a turn that is being abandoned
            # is still a turn in progress, and this is what `interrupt()` checks.
            self._generating = True
            if self._dropping:
                continue
            try:
                yield base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                logger.warning("gemini-live: dropping an undecodable audio part")

        for role, key in (("user", "inputTranscription"), ("assistant", "outputTranscription")):
            text = (content.get(key) or {}).get("text")
            if isinstance(text, str) and text:
                self._said[role].append(text)

        if content.get("generationComplete") or content.get("turnComplete"):
            # An interrupted turn goes interrupted -> turnComplete with no
            # generationComplete, so this is also where dropping ends.
            # An interrupted turn is still recorded: the transcript then covers
            # words that were generated but cut off before they were all heard.
            # The alternative - recording nothing - loses the exchange entirely,
            # and the API gives no way to know how much of it played.
            self._dropping = False
            self._generating = False
            yield from self._flush()

    def _flush(self) -> Iterator[Transcript]:
        """Release the accumulated turn, user first: that is the order it happened
        in, and the order memory should record it in.

        One role at a time, cleared as it is handed over rather than all at once:
        `yield` is a suspension point, so a caller cancelled between the two would
        take the second transcript with it. Left where it is, `pending_transcripts`
        can still recover it.
        """
        for role in ("user", "assistant"):
            fragments = self._said[role]
            self._said[role] = []
            text = "".join(fragments).strip()
            if text:
                yield Transcript(text=text, role=role, final=True)

    def _snapshot(self, *, final: bool) -> list[Transcript]:
        """The accumulated turn as transcripts, without clearing it."""
        found: list[Transcript] = []
        for role in ("user", "assistant"):
            text = "".join(self._said[role]).strip()
            if text:
                found.append(Transcript(text=text, role=role, final=final))
        return found

    def _closed_error(self, exc: ConnectionClosed) -> GeminiLiveError:
        received = exc.rcvd
        code = received.code if received is not None else 1006
        reason = self._redact(received.reason) if received is not None else ""
        return GeminiLiveError(
            f"connection closed {code}: {reason or 'no reason given'}",
            permanent=code in _PERMANENT_CLOSE_CODES,
        )

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "<key>")
