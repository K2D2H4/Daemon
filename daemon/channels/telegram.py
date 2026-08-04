"""Telegram channel - the first way Daemon reaches the user (docs/PLAN.md 3.1).

Two HTTP endpoints (`getUpdates`, `sendMessage`) do not justify a framework, so
this talks to the Bot API with httpx directly instead of pulling in
python-telegram-bot.

Kept deliberately thin: no command parsing, no interpretation of inbound text.
Anyone can message a Telegram bot, so the numeric allowlist below is the only
thing standing between a stranger and the user's companion - it is the point of
this module, not a detail. Presence-based routing between this channel and the
local speaker lands later (docs/PLAN.md 6.3); this module knows nothing about it.

How that allowlist gets populated is `dm_policy`:

  * `allowlist` - ids come from configuration, and an empty list is a
    misconfiguration worth refusing to start over.
  * `pairing` - an unknown sender is answered with a code the owner approves from
    their terminal (`channels/pairing.py`), so nobody transcribes a numeric id by
    hand. Here an empty list is the *normal* first-run state: no owner yet.

Either way the decision is made on the numeric id alone - and that includes the
👍/👎 taps on a proactive utterance (docs/PLAN.md 8.3), which arrive as
`callback_query` updates carrying their *own* sender id: whoever pressed the
button, not whoever the message was sent to.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import httpx

from daemon import clock
from daemon.channels.base import Cursor, InboundMessage, OutboundMessage
from daemon.channels.pairing import DENY, Pairing

logger = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
MAX_TEXT_LEN = 4096
"""Telegram's hard limit on sendMessage text."""

_ASCII_DIGITS = re.compile(r"[0-9]+")

POLL_TIMEOUT_SECONDS = 30
"""Server-side long-poll window; getUpdates returns early when updates exist."""

MIN_POLL_INTERVAL_SECONDS = 1.0
"""Floor between polls when a poll came back empty.

Pacing was left entirely to Telegram's server-side long poll, which holds the
request open for `timeout` seconds when there is nothing to send. That is true of
Telegram and not true of everything in front of it: a proxy that does not honour
long-polling, or any path that returns immediately, turns this loop into a busy
spin. Measured with an instant transport: **481,299 requests in 30 seconds**,
roughly 16,000 a second, starving the rest of the event loop and hammering the
API into rate-limiting us.

Only empty polls are paced. When updates arrive the next poll is immediate,
because the whole point of long-polling is that a reply is not made to wait.
"""

BACKOFF_START_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 60.0
_BACKOFF_MAX_SHIFT = 6
"""Caps the doubling at 64s so the exponent cannot overflow."""

DM_POLICIES = ("allowlist", "pairing")
"""What to do with a DM from an id that is not configured. See the module docstring."""

LABEL_PREFIX = "label"
"""First field of the `callback_data` `_label_keyboard` puts on the buttons."""

_LABEL_VERDICTS = {"up": "good", "down": "bad"}
"""Button verb -> the two values `proactive_utterances.label` admits.

A *mapping* rather than a pass-through, because the column has a CHECK constraint:
handing sqlite a forged verb would raise an IntegrityError from inside the poll
loop instead of being quietly ignored, which is the wrong end to find out at.
"""

_LABEL_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
"""The charset an `utterance_id` we ourselves emit can use - a uuid, and nothing
wider than one. Narrow and bounded on purpose: a forged id is rejected before it
reaches storage or a log line, and Telegram caps the whole `callback_data` at 64
bytes, so nothing legitimate is longer than this either."""

LABEL_ANSWERS = {
    "good": "\U0001f44d 좋았다고 기록했어. 고마워. / Noted as good.",
    "bad": "\U0001f44e 방해였다고 기록했어. / Noted as bad.",
}
LABEL_UNKNOWN = (
    "이 발화를 찾을 수 없어서 기록하지 못했어. / No such utterance; nothing was recorded."
)
LABEL_UNUSABLE = (
    "알 수 없는 버튼이라 아무것도 기록하지 않았어. / Unrecognised button; nothing was recorded."
)
LABEL_NO_STORE = "지금은 라벨을 저장할 수 없어. / Labels are not being stored right now."
LABEL_FAILED = "저장하다가 문제가 생겼어. 기록되지 않았어. / Storing the label failed."
"""What the tap says back. Telegram caps this text at 200 characters.

The failing ones are shown as an alert rather than a toast (`show_alert`), because
the difference between "counted" and "lost" must not be a notification the user
happened to miss - a label the owner believes is in the precision numbers and is
not is worse than no label at all.
"""


@runtime_checkable
class LabelStore(Protocol):
    """The storage a label needs - see `daemon/memory/store.py`.

    A protocol for the same reason as `base.Cursor` and `pairing.PairingStore`:
    the channel is handed something that can remember a verdict rather than
    importing storage.

    `is_allowed` is here rather than reusing `Pairing.screen()` because screen()
    *mints a pairing code* for an unknown sender. A button press is an
    unauthenticated update, so three forged presses would fill the pending cap and
    lock the real owner out of onboarding for an hour. This reads the same
    `channel_pairing` rows pairing writes - not a second allowlist.
    """

    def is_allowed(self, channel: str, sender_id: str) -> bool: ...

    def label_utterance(self, utterance_id: str, label: str, *, now: datetime) -> bool: ...


@dataclass(frozen=True, slots=True)
class _LabelPress:
    """A 👍/👎 tap, on its way from the poll loop to the store.

    Returned out of `_to_inbound` next to `_PairingNotice`, and for both of the
    same reasons. The sender is screened in one pass, before the payload is parsed.
    And `Channel.listen()` is frozen: it yields `InboundMessage`, which is
    conversation - a label is not something the owner *said*, and handing one over
    as a message would append a button press to the log as their words.

    `data` is left raw: it is remote input, and parsing it is `_handle_label`'s job,
    after the sender has already been found acceptable.
    """

    query_id: str
    sender_id: str
    data: Any


@dataclass(frozen=True, slots=True)
class _PairingNotice:
    """A code to send back instead of handling the message. Returned out of
    `_to_inbound` so the sender is screened in one pass, before the message body
    is ever looked at, let alone logged."""

    recipient_id: str
    text: str


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


class TelegramNoRecipient(TelegramError):
    """Nothing was sent, because nobody is configured to receive it.

    Its own class so `TelegramError`'s docstring stays true - no Bot API call was
    attempted here, this is a configuration state - and a *subclass* so everything
    already catching `TelegramError` keeps working unchanged.

    Raised rather than logged, and that distinction cost something to learn: a
    `logger.error` is not a signal a caller can act on. Proactivity is the caller
    that has to act. `ProactiveDelivery` deletes the utterance row and leaves the
    candidate live when a send fails, which is the difference between "not said"
    and "said, spent one of the day's three, and left an utterance nobody can
    label". With a quiet return it recorded a delivered utterance and marked the
    candidate fired while the words reached no one.
    """


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


def _monotonic() -> float:
    # Indirection for the same reason as _sleep: a test can pin the clock.
    return time.monotonic()


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


def _parse_label(data: Any) -> tuple[str, str] | None:
    """`label:up:<utterance_id>` -> `(utterance_id, 'good')`, or None.

    Validates rather than destructures, because this is the one place remote input
    picks a database write. A missing field, a wrong verb, a 200-character id, one
    that is not a uuid: each is a None, never an exception - the caller is inside
    the poll loop.

    `maxsplit=2` keeps a payload with extra colons in one piece, where the id
    charset then rejects it, instead of silently reading the first 36 characters.
    """
    if not isinstance(data, str):
        return None
    parts = data.split(":", 2)
    if len(parts) != 3 or parts[0] != LABEL_PREFIX:
        return None
    verdict = _LABEL_VERDICTS.get(parts[1])
    if verdict is None:
        return None
    if not _LABEL_ID.fullmatch(parts[2]):
        return None
    return parts[2], verdict


class TelegramChannel:
    """Implements the `Channel` protocol in daemon/channels/base.py."""

    name = "telegram"

    def __init__(
        self,
        token: str,
        allowed_user_ids: Iterable[int | str] | str,
        *,
        dm_policy: str = "allowlist",
        pairing: Pairing | None = None,
        client: httpx.AsyncClient | None = None,
        api_base: str = API_BASE,
        poll_timeout: int = POLL_TIMEOUT_SECONDS,
        cursor: Cursor | None = None,
        labels: LabelStore | None = None,
    ) -> None:
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is empty")
        if dm_policy not in DM_POLICIES:
            raise ValueError(
                f"unknown dm_policy {dm_policy!r}; expected one of {', '.join(DM_POLICIES)}"
            )
        allowed = _parse_user_ids(allowed_user_ids)
        if dm_policy == "pairing":
            if pairing is None:
                raise ValueError(
                    "dm_policy='pairing' needs a Pairing instance; without one every "
                    "unknown sender would be dropped with no way to ever be approved."
                )
        elif not allowed:
            # Under `pairing` an empty list means "no owner yet", which is the
            # first run. Under `allowlist` it can only mean the configuration is
            # wrong, and defaulting to allow-all would let anyone who finds the
            # bot talk to it.
            raise ValueError(
                "TELEGRAM_ALLOWED_USER_IDS is empty; refusing to start. "
                "Defaulting to allow-all would let anyone who finds the bot talk to it."
            )
        self._token = token
        self._allowed = allowed
        self._pairing = pairing if dm_policy == "pairing" else None
        self._api_base = api_base.rstrip("/")
        self._poll_timeout = poll_timeout
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            # Must outlast the server-side long poll, or every poll looks like a failure.
            timeout=httpx.Timeout(poll_timeout + 10.0, connect=10.0)
        )
        self._cursor = cursor
        # Optional so a channel can be built without storage (tests, `daemon
        # doctor`). When it is missing a press is answered honestly rather than
        # thanked for - see `_handle_label`.
        self._labels = labels
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
        if not targets:
            # Reachable only under dm_policy='pairing' before anyone is approved,
            # and only for an unaddressed message. Raised rather than logged: an
            # utterance that goes nowhere looks identical to one that was never
            # generated, and only the caller knows what to undo. Nothing has been
            # sent at this point, so there is nothing to half-abandon.
            raise TelegramNoRecipient("no configured recipient for an unaddressed message")
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
        # Only when *every* target failed. Partial delivery is a success on
        # purpose: the words are in somebody's chat, so a caller that undid the
        # send would retry into a duplicate for them, and proactivity would delete
        # the utterance row out from under a label button they can still press.
        if failed and len(failed) == len(targets):
            raise failed[0]

    async def listen(self) -> AsyncIterator[InboundMessage]:
        failures = 0
        while not self._closed:
            poll_started = _monotonic()
            try:
                updates = await self._call(
                    "getUpdates",
                    {
                        "timeout": self._poll_timeout,
                        # callback_query is not optional decoration: this filter is
                        # server-side, so without it Telegram never delivers a
                        # button press at all and the label clock (PLAN 8.1) reads
                        # as "nobody ever labels anything".
                        "allowed_updates": ["message", "callback_query"],
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
            if not updates:
                # Nothing to do, so make sure the next poll is not immediate.
                elapsed = _monotonic() - poll_started
                if elapsed < MIN_POLL_INTERVAL_SECONDS:
                    await _sleep(MIN_POLL_INTERVAL_SECONDS - elapsed)
                continue
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
                if isinstance(inbound, _LabelPress):
                    # Handled here rather than yielded: the consumer of listen()
                    # runs the conversation, and a label is not a turn in it.
                    # Cursor after handling, like the notice below - a press
                    # re-delivered because we died mid-handling only repeats the
                    # same UPDATE.
                    await self._handle_label(inbound)
                    self._save_cursor()
                    continue
                if isinstance(inbound, _PairingNotice):
                    # An unknown sender: the message itself is dropped, and only a
                    # code goes back. Still finished, so the cursor advances.
                    await self._send_pairing_notice(inbound)
                    self._save_cursor()
                    continue
                yield inbound
                # Only now, after the consumer has come back from the yield and
                # the turn is on disk. Saving before would trade duplicates for
                # silently losing what the user said, which is the worse half of
                # the trade.
                self._save_cursor()

    async def _send_pairing_notice(self, notice: _PairingNotice) -> None:
        try:
            await self.send(OutboundMessage(text=notice.text, recipient_id=notice.recipient_id))
        except TelegramError as exc:
            # A stranger who messages and then blocks the bot makes this
            # sendMessage fail with a permanent error. Letting that out of
            # listen() would hand anyone a two-message kill switch for the
            # daemon's whole inbound path. The code stays stored, so they also do
            # not earn a fresh one by retrying.
            logger.warning(
                "telegram: could not deliver a pairing code to id=%s: %s",
                notice.recipient_id,
                exc,
            )

    def _save_cursor(self) -> None:
        if self._cursor is not None and self._offset is not None:
            self._cursor.save_cursor(self.name, self._offset)

    async def close(self) -> None:
        self._closed = True
        for target in self._filtered:
            target.removeFilter(self._log_filter)
        if self._owns_client:
            await self._client.aclose()

    async def _handle_label(self, press: _LabelPress) -> None:
        """Record a verdict, and tell Telegram the query is answered.

        **Every exit answers the callback.** An unanswered `callback_query` leaves
        the spinner turning on the user's button until their client gives up, which
        is exactly what a dead daemon looks like from the outside.

        Nothing here may raise: this runs inside `listen()`, where an exception ends
        the generator and leaves a process that is alive, healthy-looking and
        completely deaf.

        The buttons are deliberately left in place, so a second press overwrites the
        first. 👍 and 👎 are adjacent targets on a phone, the label clock wants the
        verdict the owner meant rather than the one their thumb landed on, and
        correcting a mis-tap has to cost one tap - removing the keyboard after the
        first press would freeze the wrong answer into the precision numbers.
        """
        parsed = _parse_label(press.data)
        if parsed is None:
            # Forged, truncated, or a verb we never emitted. Logged without the
            # payload: it is remote text, and a long callback_data in a log line is
            # somebody else writing our logs.
            logger.warning("telegram: unusable label callback from id=%s", press.sender_id)
            await self._answer_callback(press.query_id, LABEL_UNUSABLE, alert=True)
            return
        utterance_id, verdict = parsed
        if self._labels is None:
            # Loud: a label that vanishes is the M3 gate quietly losing its only
            # source of truth, and the daemon would look perfectly healthy.
            logger.error("telegram: a label arrived with no label store wired; dropping it")
            await self._answer_callback(press.query_id, LABEL_NO_STORE, alert=True)
            return
        try:
            recorded = self._labels.label_utterance(utterance_id, verdict, now=clock.now())
        except Exception:
            # A locked or full sqlite is plausible, and must cost one label rather
            # than the whole inbound path.
            logger.exception("telegram: could not record a label from id=%s", press.sender_id)
            await self._answer_callback(press.query_id, LABEL_FAILED, alert=True)
            return
        if not recorded:
            # No such row: a forged id, or an utterance deleted because nothing was
            # ever delivered. Said out loud rather than thanked for - the owner
            # believing a label was counted when it was not is the worse failure.
            logger.warning(
                "telegram: label for an unknown utterance from id=%s", press.sender_id
            )
            await self._answer_callback(press.query_id, LABEL_UNKNOWN, alert=True)
            return
        logger.info("telegram: recorded a %s label from id=%s", verdict, press.sender_id)
        await self._answer_callback(press.query_id, LABEL_ANSWERS[verdict])

    async def _answer_callback(self, query_id: str, text: str, *, alert: bool = False) -> None:
        try:
            await self._call(
                "answerCallbackQuery",
                {"callback_query_id": query_id, "text": text, "show_alert": alert},
            )
        except TelegramError as exc:
            # A query id is only good for about a minute, so answering a stale one
            # fails. The label is already stored by then; letting this out of
            # listen() would trade the daemon's inbound path for a spinner.
            logger.warning("telegram: could not answer a callback query: %s", exc)

    def _label_keyboard(self, message: OutboundMessage) -> dict[str, Any] | None:
        """Thumbs up/down for proactive utterances (docs/PLAN.md 8.3).

        The press comes back as a `callback_query` and is handled by
        `_handle_label`; `_parse_label` must keep accepting whatever this emits.
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

    def _to_inbound(
        self, update: dict[str, Any]
    ) -> InboundMessage | _PairingNotice | _LabelPress | None:
        message = update.get("message")
        if not isinstance(message, dict):
            callback = update.get("callback_query")
            if isinstance(callback, dict):
                return self._to_label_press(callback, update.get("update_id"))
            return None  # edits, channel posts, inline queries: not ours
        sender_id = (message.get("from") or {}).get("id")
        if not isinstance(sender_id, int):
            logger.warning("telegram: update %s has no numeric sender id", update.get("update_id"))
            return None
        if sender_id not in self._allowed:
            # Screened on the numeric id alone, and before the text is read: a
            # display name or username saying "4242" proves nothing, and a
            # stranger's words must not reach the log even as a dropped record.
            decision = DENY if self._pairing is None else self._pairing.screen(str(sender_id))
            if not decision.allowed:
                # Log who tried; never the message body.
                logger.warning(
                    "telegram: dropped message from non-allowlisted sender id=%d", sender_id
                )
                if decision.notice is None:
                    return None
                return _PairingNotice(recipient_id=str(sender_id), text=decision.notice)
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

    def _to_label_press(self, callback: dict[str, Any], update_id: Any) -> _LabelPress | None:
        """Screen a button press. `callback_data` is not looked at here."""
        sender_id = (callback.get("from") or {}).get("id")
        if not isinstance(sender_id, int):
            logger.warning("telegram: callback query in update %s has no numeric sender id",
                           update_id)
            return None
        if not self._may_label(sender_id):
            # The allowlist boundary, and the reason it is *here*: a callback_query
            # carries its own `from` id - whoever pressed the button, not whoever
            # the message was addressed to - and anyone who can reach the bot can
            # send arbitrary callback_data. Screened on the numeric id alone and
            # before the payload is parsed, for the same reason as a message body.
            #
            # Nothing is sent back, not even an empty answer: a press we did not
            # authorise is indistinguishable from a forgery, and answering would
            # let an unauthenticated update make us call the API on demand. The
            # spinner is on a button they should not be holding.
            logger.warning(
                "telegram: dropped a label from non-allowlisted sender id=%d", sender_id
            )
            return None
        query_id = callback.get("id")
        if not isinstance(query_id, str) or not query_id:
            # There is nothing to answer, so there is nothing worth going on for.
            logger.warning("telegram: callback query from id=%d has no id", sender_id)
            return None
        return _LabelPress(query_id=query_id, sender_id=str(sender_id), data=callback.get("data"))

    def _may_label(self, sender_id: int) -> bool:
        """Whether this id is one Daemon listens to.

        Under `allowlist` the configured ids are the whole answer, exactly as in
        `_to_inbound`. Under `pairing` the approved set lives in storage, so it is
        read - see `LabelStore` for why it is read rather than asked of
        `Pairing.screen()`.
        """
        if sender_id in self._allowed:
            return True
        if self._pairing is None or self._labels is None:
            return False
        try:
            return self._labels.is_allowed(self.name, str(sender_id))
        except Exception:
            # Fail closed, and stay alive: a storage error here must not decide the
            # allowlist, and must not end the poll loop either.
            logger.exception("telegram: could not check whether id=%d may label", sender_id)
            return False

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
