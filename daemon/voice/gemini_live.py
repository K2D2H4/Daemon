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
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
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
from daemon.llm.gemini_schema import gemini_schema
from daemon.tools.base import ToolResult
from daemon.voice.base import Interrupted, Transcript

logger = logging.getLogger(__name__)

WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
"""v1beta is the current version for API-key auth; v1alpha exists only for
ephemeral tokens, which need a different method name entirely.

The default, not the only one: `daemon/voice/vertex.py` supplies the regional
Vertex URI, its bearer-token `auth` provider and its model path, and this class
takes all three as arguments. The protocol below is unchanged by that choice."""

_MAX_REMEMBERED_SECRETS = 4
"""How many rotated credentials the log filter keeps scrubbing. A bearer token
lives about an hour and a session reconnects, so the set has to grow a little -
but unbounded it would be a list of every token this process ever held."""

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

PARTIAL_BACKLOG = 32
"""How many unread in-progress transcripts to keep.

Bounded because a caller need not read them at all - a proactive utterance opens
a session, speaks and closes without ever asking what the user is saying - and an
unbounded queue nobody drains grows for the length of the session. Dropping the
oldest costs nothing: each partial is the whole utterance so far, so the newest
one contains every older one."""

SENSITIVITIES = ("low", "high")
"""What a caller may ask for, in the words a person would write in `.env`.

The wire wants `START_SENSITIVITY_LOW`; nobody should have to type that, and the
short form is what makes the pair below the only place the enum spelling exists.
`UNSPECIFIED` is deliberately absent - it means "the server decides", which is
already what omitting the field does."""

_START_SENSITIVITY = {name: f"START_SENSITIVITY_{name.upper()}" for name in SENSITIVITIES}
_END_SENSITIVITY = {name: f"END_SENSITIVITY_{name.upper()}" for name in SENSITIVITIES}

# --- tool calling -------------------------------------------------------------
# https://ai.google.dev/gemini-api/docs/live-tools
#
# Two knobs, both **off by default**, and the default is the load-bearing part.
#
# A declaration with no `behavior` is a *blocking* function call: the model asks,
# stops, and waits for the answer. Then `scheduling` on the response means nothing
# - it is documented only for `NON_BLOCKING`, where the model kept talking and has
# to be told what to do with an answer that arrived late. So sending neither field
# is not a timid default, it is the only coherent one for a blocking call, and it
# is why `INTERRUPT` is not the default: there is nothing here for it to schedule.
#
# **Measured**, `evals/m1c_voice_tools_spike.py`, 2026-08-05,
# gemini-3.1-flash-live-preview, one session per configuration. The reason to care
# is `daemon/voice/base.py`'s `Interrupted`: `clientContent` mid-answer killed the
# reply on every turn - 2.2 s of audio with recall on, 46.7 s with it off, 38.8 s
# deferred to the turn boundary. A `toolResponse` is a *different* top-level client
# message, and the question was whether the trap extends to it.
#
# It does not, and blocking is better than "not worse":
#
#   blocking (no behavior, no scheduling)   0.0 s of audio before the answer,
#                                           13.69 s after it, 0 interruptions
#
# Two things in that line. `toolResponse` did not interrupt anything - the reply ran
# for 13.69 s after we answered and spoke the value we returned back to us. And the
# 0.0 s says the `toolCall` arrives *before* any audio: a blocking call generates
# nothing while it waits, so there is no generation for a response to land in the
# middle of. The `clientContent` failure needed a mid-answer arrival to happen at
# all, and this shape does not have one.
#
# `NON_BLOCKING` is where the surprise is, and it is the bad kind. Setup **accepted
# it** on a model whose docs say asynchronous function calling is not supported -
# and then nothing came of it, for every scheduling value:
#
#   NON_BLOCKING + INTERRUPT    10.16 s before the answer, 0.0 s after
#   NON_BLOCKING + WHEN_IDLE     8.89 s before,            0.0 s after
#   NON_BLOCKING + SILENT       13.84 s before,            0.0 s after
#
# All three: 0 interruptions, and no second turn within 60 s. The model did keep
# talking while it waited, which is what non-blocking is for - and then the answer
# bought nothing. `INTERRUPT` in particular is documented as making the model break
# off and report, and it did not.
#
# What this run cannot separate: the `toolCall` arrived 9-14 s in, at or near the
# end of the model's own turn, so "asynchronous function calling is inert here" and
# "the answer landed after the turn boundary, leaving scheduling nothing to
# schedule" fit the same numbers. Both point one way, which is why the default is
# not a compromise between them: **a field the server accepts and then ignores is
# worse than one it rejects**, because a rejection fails loudly and this fails while
# looking configured. So neither field is sent, `_warn_about_tool_behavior` says so
# to anyone who sets one, and re-measuring is one spike run.

TOOL_BEHAVIORS = ("NON_BLOCKING",)
"""What a caller may ask for, beyond the default of not sending the field.

Only one value, because the other one *is* the absent field: the API's default
behaviour is blocking, and `"BLOCKING"` is not a spelling it documents.

Still accepted although it was measured inert (above), because the way to find out
that it has started working is to set it and run the spike again. It is not
reachable from configuration - only from code - and setting it logs a warning."""

TOOL_SCHEDULING = ("INTERRUPT", "WHEN_IDLE", "SILENT")
"""What to do with a `NON_BLOCKING` answer: cut the model off, wait for it to
finish, or keep the answer without saying anything.

Accepted here without requiring `NON_BLOCKING` alongside it, on purpose - "does
this field do anything on a blocking call?" is one of the questions the spike
exists to ask, and a constructor that refused the combination would answer it by
assumption. A warning is logged instead."""

_PERMANENT_STATUS = frozenset({400, 401, 403, 404})
"""Handshake statuses that retrying cannot fix: a bad, revoked or unauthorised
key, or a wrong endpoint."""

_PERMANENT_CLOSE_CODES = frozenset({1007})
"""1007 invalid argument means our setup JSON is wrong, which is a bug, not
weather. Measured: hoisting `responseModalities` out of `generationConfig` and
into the top level of `setup` closes the socket with 1007 "Unknown name"."""

_PERMANENT_CLOSE_REASONS = (
    # Measured. `gemini-3.1-flash-live-preview` without the `models/` prefix
    # closes with 1008 "... is not found for API version v1beta". A wrong model id
    # or API version is config, and no amount of retrying fixes config.
    "is not found",
    # Inferred, not measured: the documented text for API_KEY_INVALID, and the
    # shape a revoked key arrives in. A key the server refuses will be refused
    # again, and retrying leaves the process alive, healthy-looking and mute.
    "api key not valid",
    "api_key_invalid",
    "reported as leaked",
)
"""Substrings that make a 1008 close permanent. Everything else about 1008 is
treated as weather.

1008 used to be classified permanent on the code alone, which was wrong in the
direction that costs the most. Measured: an idle session is cut with 1008 "The
operation was aborted." - a plain idle timeout wearing a policy-violation code.
Classified permanent, that ends the voice turn with no retry; classified
transient, a genuinely permanent 1008 costs three retries and 7s of backoff
before failing with the same message. The asymmetry decides the default."""


def _permanent_close(code: int, reason: str) -> bool:
    """Is this close worth retrying?

    The reason string, not just the code: the two 1008s measured against the live
    API mean opposite things, and only the reason tells them apart."""
    if code in _PERMANENT_CLOSE_CODES:
        return True
    if code != 1008:
        return False
    lowered = reason.lower()
    return any(marker in lowered for marker in _PERMANENT_CLOSE_REASONS)


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
    """Scrubs the session's credentials out of other libraries' log records.

    websockets logs the handshake request - headers included - at DEBUG, so the
    leak happens in someone else's logger and has to be stopped there. Same
    lesson as telegram.py's `_TokenFilter`, which caught a real leak.

    Plural, and mutable through `remember`, because the Vertex transport's
    credential is a bearer token fetched per connect attempt rather than a key
    known at construction: a filter that could only hold the constructor's
    argument would scrub nothing on that path.
    """

    def __init__(self, *secrets: str) -> None:
        super().__init__()
        self._secrets: list[str] = [secret for secret in secrets if secret]

    def remember(self, secret: str) -> None:
        """Scrub this too from here on. Short-lived credentials rotate, so the
        newest replaces the previous one rather than piling up."""
        if not secret or secret in self._secrets:
            return
        self._secrets.append(secret)
        del self._secrets[:-_MAX_REMEMBERED_SECRETS]

    def _scrub(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "<key>")
        return text

    def _has_secret(self, text: str) -> bool:
        return any(secret in text for secret in self._secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        # Mapping-style args and a pre-formatted traceback both bypass a
        # tuple-only scrub, and a formatted exception is exactly where a key
        # would surface if any third party logs with exc_info=True while one of
        # ours is active.
        if isinstance(record.args, dict):
            record.args = {
                name: (
                    self._scrub(str(value)) if self._has_secret(str(value)) else value
                )
                for name, value in record.args.items()
            }
        if record.exc_text and self._has_secret(record.exc_text):
            record.exc_text = self._scrub(record.exc_text)
        message = str(record.msg)
        if self._has_secret(message):
            record.msg = self._scrub(message)
        if isinstance(record.args, tuple):
            record.args = tuple(
                self._scrub(str(arg)) if self._has_secret(str(arg)) else arg
                for arg in record.args
            )
        return True


def _warn_about_tool_behavior(behavior: str, scheduling: str) -> None:
    """Say what was measured about the two fields, to whoever just set one.

    Silent for the default, because `daemon voice` constructs a session per
    reconnect attempt and a warning on every ordinary one is a warning nobody
    reads.
    """
    if behavior:
        logger.warning(
            "gemini-live: tool_behavior=%s was measured accepted-and-inert on "
            "gemini-3.1-flash-live-preview - the tool answer produced no further "
            "audio under any scheduling value. Re-check with "
            "evals/m1c_voice_tools_spike.py before relying on it",
            behavior,
        )
    if scheduling and not behavior:
        # Not refused - see `TOOL_SCHEDULING`. Said out loud, because a null result
        # from the spike would otherwise be indistinguishable from the server
        # ignoring a field that needs a companion it did not get.
        logger.warning(
            "gemini-live: tool_scheduling=%s without tool_behavior=NON_BLOCKING; "
            "scheduling is documented only for non-blocking calls, so the server "
            "may ignore it",
            scheduling,
        )


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
        start_sensitivity: str = "",
        end_sensitivity: str = "",
        prefix_padding_ms: int | None = None,
        silence_duration_ms: int | None = None,
        tools: Sequence[ToolSpec] = (),
        tool_behavior: str = "",
        tool_scheduling: str = "",
        connect: Any = None,
        url: str = WS_URL,
        auth: Callable[[], dict[str, str]] | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        # `auth` is the Vertex transport's credential: a provider called per
        # connect attempt, because its bearer token expires while a resident
        # process keeps reconnecting (daemon/voice/vertex.py). With one supplied
        # there is no API key to require - that endpoint does not take one.
        if not api_key and auth is None:
            raise ValueError("GEMINI_API_KEY is empty")
        if not model:
            raise ValueError("DAEMON_GEMINI_LIVE_MODEL is empty")
        self._api_key = api_key
        self._auth = auth
        # The wire format is `models/{id}`; accept either so a config value
        # copied straight from the docs works. A Vertex path
        # (`projects/../publishers/google/models/..`) is already fully qualified
        # and prefixing it would produce `models/projects/...`, which comes back
        # as a 1008 naming a model nobody configured.
        self._model = (
            model
            if model.startswith(("models/", "projects/", "publishers/"))
            else f"models/{model}"
        )
        self._system_instruction = system_instruction
        # No language code: the native-audio models pick the language themselves
        # and reject being told, which matters for a Korean-first user.
        self._voice_name = voice_name
        # Checked here rather than left to the server, because the server's answer
        # is a 1007 close - classified permanent above, so a typo in one of these
        # would not fail the setting, it would end voice mode outright.
        for name in (start_sensitivity, end_sensitivity):
            if name and name not in SENSITIVITIES:
                raise ValueError(
                    f"sensitivity must be one of {SENSITIVITIES} or empty, not {name!r}"
                )
        self._start_sensitivity = start_sensitivity
        self._end_sensitivity = end_sensitivity
        self._prefix_padding_ms = prefix_padding_ms
        self._silence_duration_ms = silence_duration_ms
        # Same reasoning as the sensitivities above, and the same consequence for
        # getting it wrong: an unknown enum value comes back as a 1007 close, which
        # `_permanent_close` classifies permanent, so a typo would not fail the
        # setting - it would end voice mode with a message about invalid arguments.
        if tool_behavior and tool_behavior not in TOOL_BEHAVIORS:
            raise ValueError(
                f"tool behavior must be one of {TOOL_BEHAVIORS} or empty, not {tool_behavior!r}"
            )
        if tool_scheduling and tool_scheduling not in TOOL_SCHEDULING:
            raise ValueError(
                f"tool scheduling must be one of {TOOL_SCHEDULING} or empty, "
                f"not {tool_scheduling!r}"
            )
        _warn_about_tool_behavior(tool_behavior, tool_scheduling)
        self._tools = tuple(tools)
        self._tool_behavior = tool_behavior
        self._tool_scheduling = tool_scheduling
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
        # In-progress user speech, pushed as it arrives for `partial_transcripts`.
        # A queue rather than a snapshot because the contract is a stream: recall
        # has to start while the user is still talking, and polling for it would
        # add up to an interval of silence to every turn.
        self._partials: asyncio.Queue[Transcript | None] = asyncio.Queue(PARTIAL_BACKLOG)
        # Set by `_decode` at a turn boundary and cleared by `receive()`, which is
        # where it ends the iterator. A flag rather than a return value because
        # `_decode` is a generator: the boundary arrives in the same server event
        # as the last transcript, and that transcript has to be yielded first.
        self._turn_over = False
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
        self._secrets: list[str] = []
        """Credentials seen since construction, for `_redact`. Empty on the
        API-key transport, where `_api_key` is the only one and is already known."""
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

    async def send_frame(self, jpeg: bytes) -> None:
        """One JPEG screen frame, as realtime video input (ADR 0009).

        `realtimeInput.video` mirrors the named `realtimeInput.audio` field
        `send_audio` uses above, both part of the same `BidiGenerateContentRealtimeInput`
        message - not the older `mediaChunks` list form. Grounded in that
        symmetry rather than confirmed against the live socket - and since
        measured against it, which retires the "pending confirmation" this
        docstring used to carry and adds a limit that only the socket could say.

        **This works only outside a tool round.** Measured by
        `usageMetadata.promptTokensDetails`, which lists an `IMAGE` entry for a
        picture the prompt actually holds: a frame sent on its own arrives (60
        tokens, and it read a 24px code correctly) but needs ~1s before the next
        client message, and a frame sent between a `toolCall` and its
        `toolResponse` never arrives at all, at any gap tried. That is why the
        `see_screen` path uses `send_image` below and this stays what its name
        says - the live-share pump's transport, which has no tool round in it.
        """
        if not jpeg:
            return
        await self._send(
            {
                "realtimeInput": {
                    "video": {
                        "data": base64.b64encode(jpeg).decode("ascii"),
                        "mimeType": "image/jpeg",
                    }
                }
            }
        )

    async def send_images(self, jpegs: Sequence[bytes], note: str) -> None:
        """The captured JPEGs as `clientContent` image parts, one turn, with their
        framing in the same turn. See `daemon.voice.base.VoiceSession.send_images`
        for the measurements that fix the transport, its position in the exchange,
        and why every image belongs in a single turn.

        `turnComplete: true`, unlike `send_context`: the point is for the model to
        answer the question it is already holding, now that it can see. The
        interrupt that makes `clientContent` dangerous for recall is what makes it
        right here - it replaces an answer composed from a caption alone. It is
        also why one turn rather than one per image: a second turn would interrupt
        the answer the first one asked for.
        """
        parts: list[dict[str, Any]] = [
            {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(jpeg).decode()}}
            for jpeg in jpegs
            if jpeg
        ]
        if not parts:
            return
        if note.strip():
            # Never an empty text part: a part carrying nothing is a field on the
            # wire for no gain, and this file has been closed 1007 for less. Once
            # for the turn, not once per image - the framing is about the turn.
            parts.append({"text": note})
        await self._send(
            {"clientContent": {"turns": [{"role": "user", "parts": parts}], "turnComplete": True}}
        )

    async def send_context(self, text: str) -> None:
        """Put text in the model's history without asking it to answer.

        `clientContent` with `turnComplete: false`, measured against the live API
        rather than inferred from the SDK: the same payload with `turnComplete:
        true` returned 138 kB of audio and a full transcript, and with `false`
        returned no audio and no transcript at all. That asymmetry is the whole
        mechanism - history is seeded and nothing is generated.

        This is the only way recall reaches a voice turn. `send_text` is a prompt,
        so a memory delivered through it makes the daemon narrate an old
        conversation the user never asked about.

        Returns as soon as the frame is written. Nothing is awaited afterwards
        because nothing comes back - a call that waited for a response would hang
        for as long as the server let the session live.
        """
        if not text.strip():
            logger.debug("gemini-live: nothing to seed; skipping send_context")
            return
        await self._send(
            {
                "clientContent": {
                    # `role: "user"` because the two roles Live accepts are "user"
                    # and "model", and this is not the model's own words. The block
                    # the caller hands over says what it is - see the recall
                    # boundary in daemon/loop.py - precisely because the wire has
                    # no role that means "reference material".
                    "turns": [{"role": "user", "parts": [{"text": text}]}],
                    "turnComplete": False,
                }
            }
        )

    async def send_text(self, text: str) -> None:
        """Give the model something to say without any user audio.

        Note the honest limitation: this is a *prompt*, not a verbatim TTS
        instruction, so the model answers this text rather than reading it out.

        That is a statement about the wire, and PR #126 retracts what this
        paragraph used to conclude from it ("proactive utterances that must come
        out word for word go to the local speaker instead"). Asked plainly enough,
        the model answers by saying the sentence and nothing else - measured 8/8
        against 0/8 for an ordinary instruction (`daemon/voice/base.py:send_text`,
        `evals/proactive_verbatim_spike.py`) - so a proactive line does come
        through here now, and `/usr/bin/say` is the fallback.

        **`turnComplete: true` is what makes the answer arrive at all.** This used
        to be `realtimeInput.text`, whose turn-end is "derived from user activity"
        - i.e. left to the server's activity detection, which a third of the time
        never decides the turn is over. Measured live on
        `gemini-3.1-flash-live-preview`, 30 trials per arm against the resident's
        real opening: `realtimeInput.text` never answered **10/30** times (median
        0.69 s when it did), `clientContent` with `turnComplete: true` **0/30**
        (median 0.66 s). Fisher exact p = 0.0008, same median - closing the turn is
        free and is the difference between speaking and going silent.

        `evals/m0_voice_spike` listed this as one of the six things only a live key
        could settle, and closed it on one successful trial. One trial cannot see a
        1-in-3 failure; what the owner saw was a daemon that ignored its own name.

        The frame is now the same one `send_context` uses, and `turnComplete` is
        the whole difference between them: false seeds history silently, true asks
        for an answer. That is the distinction the two methods exist to draw.
        """
        if not text.strip():
            # An empty proactive utterance is not rare, and spending a
            # per-minute-billed session on one is worse than skipping it.
            logger.warning("gemini-live: refusing to send empty text")
            return
        await self._send(
            {
                "clientContent": {
                    "turns": [{"role": "user", "parts": [{"text": text}]}],
                    "turnComplete": True,
                }
            }
        )

    async def send_tool_response(self, results: Sequence[ToolResult]) -> None:
        """Answer this turn's tool calls, all of them in one message.

        `ok` decides the key, not just the text: `{"result": ...}` and
        `{"error": ...}` are what let the model tell "the file said this" from "the
        read was refused". Handing a refusal back under `result` is how a policy
        denial gets reported to the owner as content.

        Sends nothing when there is nothing to answer. An empty frame on a
        per-minute-billed socket buys nothing, and `send_context` has already taught
        this file that a needless client message can cost a turn.
        """
        if not results:
            logger.debug("gemini-live: no tool results to send")
            return
        answers: list[dict[str, Any]] = []
        for result in results:
            answer: dict[str, Any] = {
                "id": result.call_id,
                # Both, though the id alone should pair them: the REST half of this
                # API issues no ids at all and pairs by name (llm/providers/gemini.py),
                # so a server that ignores one of the two still has the other.
                "name": result.name,
                # Must be an object, never a bare string.
                "response": {"result" if result.ok else "error": result.content},
            }
            if self._tool_scheduling:
                answer["scheduling"] = self._tool_scheduling
            answers.append(answer)
        await self._send({"toolResponse": {"functionResponses": answers}})

    async def receive(self) -> AsyncIterator[bytes | Transcript | Interrupted | ToolCall]:
        """One turn: audio chunks to play, interleaved with completed transcripts.

        Only `final=True` transcripts are ever yielded. Gemini streams
        transcription in fragments - a few syllables at a time - and handing
        those out individually would invite a caller to record half a word as an
        utterance. They are accumulated and released at the turn boundary.

        **Ends at that boundary**, and `ended` stays None when it does: the turn is
        over, the session is not, and the next turn is the next call. Measured
        before it ended there - a turn answered in 2.6s, delivered its final
        transcript, and then this iterator blocked until the server cut the idle
        session with 1008 "The operation was aborted.". Both spike runs died that
        way, because `async for` is the only way anyone consumes this.
        """
        ws = self._require_open()
        # Cleared on the way in, not on the way out: a caller that walks away
        # mid-flush - `aclose()` between the user's transcript and the assistant's -
        # leaves the flag set, and the next turn would then end after its first
        # event with the rest of the answer still on the socket.
        self._turn_over = False
        closed: GeminiLiveError | None = None
        try:
            async for raw in ws:
                for item in self._decode(raw):
                    yield item
                if self._turn_over:
                    # The turn is over. `ended` is deliberately left alone: nothing
                    # about the session is.
                    return
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
        # Reaching here means the socket is gone, not the turn, so anyone waiting on
        # the in-progress transcript is waiting on words that will never arrive.
        self._end_partials()
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

    async def partial_transcripts(self) -> AsyncIterator[Transcript]:
        """The user's utterance as it grows, one item per delta that arrives.

        `final=False`, always: the flag is what stops a caller recording a syllable
        as an utterance. Each item is the whole utterance so far rather than the
        delta, because the one consumer is recall - it embeds what was said, not
        what was just added - and because that makes a dropped item free
        (`PARTIAL_BACKLOG`).

        User only. The assistant's own words are not a query, and the model
        answering itself is not work that has to start early.

        Ends when the session does, never at a turn boundary: the point of this
        seam is to be live while `receive()` is yielding nothing (docs/PLAN.md
        4.3.1 - 117 ms of embedding is free before the utterance ends and is
        silence after it).
        """
        while True:
            partial = await self._partials.get()
            if partial is None:
                return
            yield partial

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
        # Outside the try: a credential failure is not a connect failure, and the
        # broad handler below would reclassify a permanent one as retryable -
        # which is how a daemon ends up retrying a revoked credential forever
        # while /health still says running.
        headers = await self._auth_headers()
        try:
            # Header rather than the documented `?key=` query param: a key in a
            # URL ends up in every error string that quotes the URI.
            return await self._connect(
                self._url,
                additional_headers=headers,
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
        if self._tools:
            # Only when there is something to declare. An empty `tools: []` is a
            # new field on the wire for no gain, and this file has already been
            # closed with 1007 "Unknown name" for putting a field in the wrong
            # place - the cost of a wrong field here is the whole session.
            #
            # One tool *object* holding every declaration, not one object each:
            # `llm/providers/gemini.py` says the same thing about the REST API,
            # which is the same proto.
            setup["tools"] = [{"functionDeclarations": self._function_declarations()}]
        detection = self._activity_detection()
        if detection:
            setup["realtimeInputConfig"] = {"automaticActivityDetection": detection}
        return {"setup": setup}

    def _function_declarations(self) -> list[dict[str, Any]]:
        """`ToolSpec` as the wire wants it.

        `parameters` is narrowed by `gemini_schema` first. An MCP server forwards its
        own `inputSchema` untouched, and those carry keywords the Live API rejects
        (`additionalProperties`, `title`, `$schema`) - a single one closes the socket
        1007 and this file treats that as permanent, so one connected MCP server
        killed every voice session. `llm/providers/gemini.py` narrows the REST path
        the same way through the same helper.
        """
        declarations: list[dict[str, Any]] = []
        for spec in self._tools:
            declared: dict[str, Any] = {
                "name": spec.name,
                "description": spec.description,
                "parameters": gemini_schema(spec.parameters),
            }
            if self._tool_behavior:
                declared["behavior"] = self._tool_behavior
            declarations.append(declared)
        return declarations

    def _activity_detection(self) -> dict[str, Any]:
        """How the server decides the user started and stopped talking.

        Sent only for the fields a caller actually set, and the distinction is the
        point: omitting a field leaves the server's own default, and the default
        `silenceDurationMs` is ~800 ms
        (https://ai.google.dev/gemini-api/docs/live-guide). Sending a value for
        everything would mean picking three numbers to get one, and each of the
        other two would then be a guess presented as a decision.

        Nothing here disables detection. `disabled: true` hands turn boundaries to
        the client, which means this session would have to send
        `activityStart`/`activityEnd` itself and something local would have to
        decide when - a different design, not a setting.
        """
        detection: dict[str, Any] = {}
        if self._start_sensitivity:
            detection["startOfSpeechSensitivity"] = _START_SENSITIVITY[self._start_sensitivity]
        if self._end_sensitivity:
            detection["endOfSpeechSensitivity"] = _END_SENSITIVITY[self._end_sensitivity]
        if self._prefix_padding_ms is not None:
            detection["prefixPaddingMs"] = self._prefix_padding_ms
        if self._silence_duration_ms is not None:
            detection["silenceDurationMs"] = self._silence_duration_ms
        return detection

    def _parse(self, raw: str | bytes) -> dict[str, Any] | None:
        try:
            message = json.loads(raw)
        except ValueError:
            logger.warning("gemini-live: dropping a message that is not JSON")
            return None
        return message if isinstance(message, dict) else None

    def _decode(self, raw: str | bytes) -> Iterator[bytes | Transcript | Interrupted | ToolCall]:
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
        if "toolCallCancellation" in message:
            # Ids the server no longer wants answers for. Nothing above this can act
            # on it yet - `VoiceConversation` has no tool loop until PR-2b - so it is
            # reported rather than dropped: a documented server message that vanishes
            # quietly is how an answer gets sent back to a call nobody is waiting for.
            ids = (message.get("toolCallCancellation") or {}).get("ids") or []
            logger.info("gemini-live: the server cancelled tool calls %s", list(ids))
        # `serverContent` first, then `toolCall`. They are siblings on the wire, and
        # audio is the one with somewhere to be: PR-2b's consumer will `await` the
        # tool inside `receive()`'s loop, so a chunk handed over after that is a
        # chunk the speaker got late.
        yield from self._decode_content(message.get("serverContent"))
        yield from self._decode_tool_calls(message.get("toolCall"))

    def _decode_tool_calls(self, block: Any) -> Iterator[ToolCall]:
        """`toolCall.functionCalls` as the neutral `ToolCall` the tool layer runs.

        Not a voice-specific shape: the same dataclass the text path uses, so the
        same runner, the same policy and the same audit row serve both.
        """
        calls = block.get("functionCalls") if isinstance(block, dict) else None
        if not isinstance(calls, list):
            return
        for index, raw in enumerate(calls):
            if not isinstance(raw, dict):
                continue
            name = raw.get("name")
            if not isinstance(name, str) or not name:
                # Nothing could be run and nothing could be answered, so handing it
                # up would only move the failure somewhere with less context.
                logger.warning("gemini-live: dropping a tool call with no name")
                continue
            if not self._tools:
                # A session that declared nothing cannot legitimately be asked for
                # anything. Dropped rather than yielded because a consumer written
                # before tool calls existed routes every non-audio, non-Interrupted
                # item to the transcript path - so this would be recorded as
                # something the owner said.
                logger.warning(
                    "gemini-live: dropping a call to %r - no tool was offered in setup", name
                )
                continue
            call_id = raw.get("id")
            yield ToolCall(
                # Documented as optional. `llm/providers/gemini.py` already had to
                # invent one because the REST half of this API issues none at all,
                # and a result with nothing in `id` cannot be paired with its call.
                id=call_id
                if isinstance(call_id, str) and call_id
                else synthesise_call_id(name, index),
                name=name,
                arguments=decode_tool_arguments(raw.get("args")),
            )

    def _decode_content(self, content: Any) -> Iterator[bytes | Transcript | Interrupted]:
        if not isinstance(content, dict):
            return

        # Read the interruption flag before yielding audio: one server event may
        # carry several fields at once, and audio from an interrupted turn must
        # not slip out ahead of the flag that condemns it.
        if content.get("interrupted"):
            # Only while there is a turn to abandon, and the guard is not tidiness.
            # Measured against the live API, four runs: `interrupted` arrives about
            # 0.25s *after* `generationComplete` on a turn nobody interrupted -
            # every time. Acted on then it drops a complete answer out of the
            # speaker mid-playback, which is the same shape as the bug `interrupt()`
            # below already guards against, one layer earlier.
            if self._generating:
                self._dropping = True
                # Handed to the caller as well as acted on here, because dropping
                # the rest of the stream is only half of a barge-in - the speaker's
                # own buffer has to go too, and only the caller has the speaker.
                # This is the *authoritative* signal: it is the server's own
                # activity detection, and inferring it from transcripts instead
                # killed every turn (daemon/voice/base.py, `Interrupted`).
                yield Interrupted()
            else:
                logger.debug(
                    "gemini-live: ignoring an interruption with nothing being generated"
                )

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
                if role == "user":
                    self._push_partial()

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
            # Either flag is the boundary. `generationComplete` usually arrives
            # first and `turnComplete` in the event after it, so a turn that ends
            # on the former leaves the latter to be read as one more turn that
            # yields nothing - cheap, and cheaper than guessing which of the two a
            # given model sends.
            self._turn_over = True

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

    def _push_partial(self) -> None:
        """Offer the utterance so far to whoever is listening for it.

        Never blocks and never raises: the socket read loop is what calls this, and
        a consumer that has stopped reading must not be able to stall the turn.
        """
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

    def _closed_error(self, exc: ConnectionClosed) -> GeminiLiveError:
        received = exc.rcvd
        code = received.code if received is not None else 1006
        reason = self._redact(received.reason) if received is not None else ""
        return GeminiLiveError(
            f"connection closed {code}: {reason or 'no reason given'}",
            permanent=_permanent_close(code, reason),
        )

    async def _auth_headers(self) -> dict[str, str]:
        """The handshake's credential headers, fetched per attempt.

        In a thread because the Vertex provider refreshes a token over the
        network, and this runs on the loop that is also draining the microphone.
        Every value goes to the log filter and to `_redact` before it reaches
        `websockets`, which logs the handshake headers at DEBUG.
        """
        if self._auth is None:
            return {"x-goog-api-key": self._api_key}
        try:
            headers = await asyncio.to_thread(self._auth)
        except GeminiLiveError:
            raise
        except Exception as exc:
            # Permanent: every credential failure this can reach - no
            # credentials, an expired login, an unreadable key file - needs a
            # person, and a retry loop around it is a silently mute daemon.
            raise GeminiLiveError(
                f"credentials for the voice session failed: {type(exc).__name__}: {exc}",
                permanent=True,
            ) from None
        for value in headers.values():
            self._log_filter.remember(value)
            self._secrets.append(value)
        del self._secrets[:-_MAX_REMEMBERED_SECRETS]
        return headers

    def _redact(self, text: str) -> str:
        for secret in (self._api_key, *self._secrets):
            if secret:
                text = text.replace(secret, "<key>")
        return text
