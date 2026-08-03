"""Telegram channel - the first way Daemon reaches the user (docs/PLAN.md 3.1).

Two HTTP endpoints (`getUpdates`, `sendMessage`) do not justify a framework, so
this talks to the Bot API with httpx directly instead of pulling in
python-telegram-bot.

Kept deliberately thin: no command parsing, no interpretation of inbound text.
Anyone can message a Telegram bot, so the numeric allowlist below is the only
thing standing between a stranger and the user's companion - it is the point of
this module, not a detail. Presence-based routing between this channel and the
local speaker lands later (docs/PLAN.md 6.3); this module knows nothing about it.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Iterable
from datetime import UTC, datetime
from typing import Any

import httpx

from daemon.channels.base import Cursor, InboundMessage, OutboundMessage

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
MAX_TEXT_LEN = 4096
"""Telegram's hard limit on sendMessage text."""

_ASCII_DIGITS = re.compile(r"[0-9]+")

POLL_TIMEOUT_SECONDS = 30
"""Server-side long-poll window; getUpdates returns early when updates exist."""

BACKOFF_START_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 60.0
_BACKOFF_MAX_SHIFT = 6
"""Caps the doubling at 64s so the exponent cannot overflow."""


class TelegramError(Exception):
    """A Bot API call failed. Never carries the bot token - see `_redact`.

    Deliberately not a RuntimeError: listen() treats a bare RuntimeError as
    "the client was closed underneath us", and the two must not be confused.
    """

    def __init__(self, message: str, *, status: int | None = None, retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    @property
    def permanent(self) -> bool:
        """A revoked token, a deleted bot, a bot blocked by the user. Retrying
        these forever leaves the process alive, healthy-looking, and permanently
        deaf - the worst possible failure for a companion that is supposed to be
        listening."""
        return self.status in (401, 403, 404)


class TelegramFatal(RuntimeError):
    """Raised out of listen() when retrying cannot help. The supervisor should
    see the daemon die rather than watch it poll a dead token for days."""


class _TokenFilter(logging.Filter):
    """Scrubs the bot token out of httpx's own log records.

    The token lives in the request path, and httpx logs every request URL at
    INFO. Redacting only in this module would not be enough: the leak happens in
    someone else's logger, so it has to be stopped there.
    """

    def __init__(self, token: str) -> None:
        super().__init__()
        self._token = token

    def filter(self, record: logging.LogRecord) -> bool:
        message = str(record.msg)
        if self._token in message:
            record.msg = message.replace(self._token, "<token>")
        if isinstance(record.args, tuple):
            record.args = tuple(
                str(arg).replace(self._token, "<token>") if self._token in str(arg) else arg
                for arg in record.args
            )
        return True


async def _sleep(seconds: float) -> None:
    # Indirection so tests can drive the backoff clock without real waiting.
    await asyncio.sleep(seconds)


def _received_at(date: Any) -> datetime:
    """Telegram's timestamp, or now if it is unusable.

    `isinstance(date, int)` is not a range check - Python ints are unbounded and
    fromtimestamp raises OverflowError past the platform limit. Telegram sets
    this field so it should always be sane, but this runs inside listen()'s
    loop, where an unhandled exception ends the generator and kills the daemon's
    entire inbound path. A wrong timestamp is worth far less than the loop.
    """
    if isinstance(date, int):
        try:
            return datetime.fromtimestamp(date, tz=UTC)
        except (OverflowError, OSError, ValueError):
            logger.warning("telegram: unusable message date, falling back to now")
    return datetime.now(UTC)


def _parse_user_ids(raw: Iterable[int | str] | str) -> frozenset[int]:
    """Numeric ids only. Usernames and display names are user-changeable."""
    items = raw.replace(",", " ").split() if isinstance(raw, str) else raw
    ids = set()
    for item in items:
        text = str(item).strip()
        # Not int(): it also accepts '+42', '4_2' and unicode decimal digits like
        # '٤٢'. Nothing remote reaches here, but a typo silently allowlisting a
        # *different* account is the failure that matters. Telegram user ids are
        # always positive ASCII integers.
        if not _ASCII_DIGITS.fullmatch(text):
            raise ValueError(f"TELEGRAM_ALLOWED_USER_IDS must be numeric ids, got {item!r}")
        ids.add(int(text))
    return frozenset(ids)


def _split(text: str, limit: int = MAX_TEXT_LEN) -> list[str]:
    """Chunk to Telegram's length limit, preferring line then word boundaries."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        window = rest[:limit]
        cut = window.rfind("\n")
        if cut < limit // 2:
            cut = window.rfind(" ")
        if cut < limit // 2:
            cut = limit  # one long unbroken run: hard cut rather than drop it
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        parts.append(rest)
    return parts


class TelegramChannel:
    """Implements the `Channel` protocol in daemon/channels/base.py."""

    name = "telegram"

    def __init__(
        self,
        token: str,
        allowed_user_ids: Iterable[int | str] | str,
        *,
        client: httpx.AsyncClient | None = None,
        api_base: str = API_BASE,
        poll_timeout: int = POLL_TIMEOUT_SECONDS,
        cursor: Cursor | None = None,
    ) -> None:
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is empty")
        allowed = _parse_user_ids(allowed_user_ids)
        if not allowed:
            raise ValueError(
                "TELEGRAM_ALLOWED_USER_IDS is empty; refusing to start. "
                "Defaulting to allow-all would let anyone who finds the bot talk to it."
            )
        self._token = token
        self._allowed = allowed
        self._api_base = api_base.rstrip("/")
        self._poll_timeout = poll_timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            # Must outlast the server-side long poll, or every poll looks like a failure.
            timeout=httpx.Timeout(poll_timeout + 10.0, connect=10.0)
        )
        self._cursor = cursor
        # Restored so a restart does not re-receive everything Telegram had not
        # yet had confirmed - which would append the same turn to the log again.
        self._offset = cursor.load_cursor(self.name) if cursor is not None else None
        self._closed = False
        self._log_filter = _TokenFilter(token)
        # The token lives in the URL path (Bot API has no other way), and httpx
        # logs the full request URL at INFO. A filter on the "httpx" logger only
        # covers records that logger creates: logging runs the *originating*
        # logger's filters, never an ancestor's, so a child logger such as
        # httpx._client would bypass it entirely - and httpx is unpinned, so one
        # minor release moving the logger silently returns the token to the log.
        # Handler-level scrubbing is the only placement that cannot be bypassed.
        self._filtered: list[logging.Logger | logging.Handler] = [logging.getLogger("httpx")]
        self._filtered.extend(logging.getLogger().handlers)
        for target in self._filtered:
            target.addFilter(self._log_filter)

    async def send(self, message: OutboundMessage) -> None:
        if not message.text.strip():
            # An empty completion is not rare, and Telegram rejects empty text
            # with a 400 - which would turn a harmless empty turn into a failed
            # one, after the empty record was already written to the log.
            logger.warning("telegram: refusing to send an empty message")
            return
        keyboard = self._label_keyboard(message)
        parts = _split(message.text)
        # A reply goes only to whoever asked. Falling back to the whole allowlist
        # would mean adding someone so they *can* talk to Daemon also signs them
        # up to receive every answer to the owner and every proactive utterance.
        # An unaddressed message (proactivity, before any inbound exists) still
        # goes to the allowlist, which for the intended single-user install is
        # exactly the owner.
        if message.recipient_id is not None:
            targets: list[int] = [int(message.recipient_id)]
        else:
            targets = sorted(self._allowed)
        failed: list[TelegramError] = []
        for chat_id in targets:
            try:
                for index, part in enumerate(parts):
                    payload: dict[str, Any] = {"chat_id": chat_id, "text": part}
                    # No parse_mode: model output is plain text, never markup.
                    if keyboard is not None and index == len(parts) - 1:
                        payload["reply_markup"] = keyboard
                    await self._call("sendMessage", payload)
            except TelegramError as exc:
                # One unreachable recipient must not silence the others, and a
                # part failing mid-way must be visible: loop.handle already wrote
                # the full text to the log, so silently delivering half of it
                # would leave the log claiming words the user never saw.
                logger.error("telegram: send to %s failed after %d part(s): %s",
                             chat_id, index, exc)
                failed.append(exc)
        if failed and len(failed) == len(targets):
            raise failed[0]

    async def listen(self) -> AsyncIterator[InboundMessage]:
        failures = 0
        while not self._closed:
            try:
                updates = await self._call(
                    "getUpdates",
                    {
                        "timeout": self._poll_timeout,
                        "allowed_updates": ["message"],
                        **({} if self._offset is None else {"offset": self._offset}),
                    },
                )
            except TelegramError as exc:
                if self._closed:
                    return
                if exc.permanent:
                    raise TelegramFatal(f"telegram rejected us permanently: {exc}") from None
                failures += 1
                # Clamp the *exponent*, not just the result. min() bounded the
                # delay but 2 ** (failures - 1) kept growing, so a long outage -
                # about 17 hours at the 60s ceiling - overflowed float and threw
                # OverflowError from inside this handler, killing the generator
                # and the daemon's whole inbound path exactly when the network
                # came back.
                delay = min(
                    BACKOFF_START_SECONDS * 2 ** min(failures - 1, _BACKOFF_MAX_SHIFT),
                    BACKOFF_MAX_SECONDS,
                )
                if exc.retry_after:
                    delay = max(delay, float(exc.retry_after))
                if exc.status == 409:
                    # Two instances are polling the same bot. Backing off quietly
                    # would hide a misconfiguration that never resolves itself.
                    logger.error("telegram: conflict, another instance is polling: %s", exc)
                else:
                    logger.warning("telegram: %s; retrying in %.1fs", exc, delay)
                await _sleep(delay)
                continue
            except RuntimeError:
                # close() can pull the client out from under an in-flight poll.
                if not self._closed:
                    raise
                return
            failures = 0
            for update in updates or []:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    # Advance for every update seen, including dropped ones, so the
                    # same update is never handed to us twice.
                    self._offset = update_id + 1
                inbound = self._to_inbound(update)
                if inbound is None:
                    # Nothing to hand over, so it is finished: confirm it now.
                    self._save_cursor()
                    continue
                yield inbound
                # Only now, after the consumer has come back from the yield and
                # the turn is on disk. Saving before would trade duplicates for
                # silently losing what the user said, which is the worse half of
                # the trade.
                self._save_cursor()

    def _save_cursor(self) -> None:
        if self._cursor is not None and self._offset is not None:
            self._cursor.save_cursor(self.name, self._offset)

    async def close(self) -> None:
        self._closed = True
        for target in self._filtered:
            target.removeFilter(self._log_filter)
        if self._owns_client:
            await self._client.aclose()

    def _label_keyboard(self, message: OutboundMessage) -> dict[str, Any] | None:
        """Thumbs up/down for proactive utterances (docs/PLAN.md 8.3).

        Handling the callback query is M3; attaching the buttons now is free.
        """
        if not message.labelable:
            return None
        if not message.utterance_id:
            logger.warning("telegram: labelable message without utterance_id, sending unlabelled")
            return None
        return {
            "inline_keyboard": [
                [
                    {"text": "\U0001f44d", "callback_data": f"label:up:{message.utterance_id}"},
                    {"text": "\U0001f44e", "callback_data": f"label:down:{message.utterance_id}"},
                ]
            ]
        }

    def _to_inbound(self, update: dict[str, Any]) -> InboundMessage | None:
        message = update.get("message")
        if not isinstance(message, dict):
            return None  # edits, channel posts, callback queries: not M1a
        sender_id = (message.get("from") or {}).get("id")
        if not isinstance(sender_id, int):
            logger.warning("telegram: update %s has no numeric sender id", update.get("update_id"))
            return None
        if sender_id not in self._allowed:
            # Log who tried; never the message body.
            logger.warning("telegram: dropped message from non-allowlisted sender id=%d", sender_id)
            return None
        text = message.get("text")
        if not isinstance(text, str) or not text:
            logger.info("telegram: ignoring non-text message from id=%d", sender_id)
            return None
        received_at = _received_at(message.get("date"))
        # Text is passed through verbatim - untrusted data, never a command.
        # But record whether the sender actually wrote it: a forward, an
        # inline-bot result or a quoted third party carries someone else's words
        # under an allowlisted `from.id`, and vouching for those as the owner's
        # own would launder injected text into origin='owner'.
        relayed = any(message.get(key) for key in ("forward_origin", "forward_from", "via_bot"))
        update_id = update.get("update_id")
        return InboundMessage(
            text=text,
            sender_id=str(sender_id),
            external_id=str(update_id) if isinstance(update_id, int) else None,
            received_at=received_at,
            channel=self.name,
            authored_by_sender=not relayed,
        )

    async def _call(self, method: str, payload: dict[str, Any]) -> Any:
        url = f"{self._api_base}/bot{self._token}/{method}"
        detail: str | None = None
        try:
            response = await self._client.post(url, json=payload)
        except httpx.HTTPError as exc:
            # httpx exceptions carry the request URL, and the URL carries the
            # token. `raise ... from None` is not enough: it only clears
            # __cause__, while __context__ keeps the original exception - and
            # anything that walks the chain unconditionally (an error reporter,
            # traceback.format_exception(chain=True), a pytest failure report)
            # would surface the token. A bot token is full account takeover with
            # no recovery but revocation, so capture the redacted text here and
            # raise outside the block, where there is no active exception for
            # __context__ to point at.
            detail = self._redact(str(exc))
        if detail is not None:
            raise TelegramError(f"{method} failed: {detail}")
        if response.status_code != 200:
            retry_after = None
            try:
                retry_after = (response.json().get("parameters") or {}).get("retry_after")
            except ValueError:
                pass  # not every error response is JSON
            raise TelegramError(
                f"{method} returned HTTP {response.status_code}",
                status=response.status_code,
                retry_after=retry_after if isinstance(retry_after, int) else None,
            )
        body = response.json()
        if not body.get("ok"):
            raise TelegramError(
                f"{method} rejected: {self._redact(str(body.get('description')))}",
                status=body.get("error_code") if isinstance(body.get("error_code"), int) else None,
            )
        return body.get("result")

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "<token>")
