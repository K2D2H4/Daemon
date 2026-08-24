"""Telegram channel tests. No network: httpx.MockTransport stands in for the Bot API.

The allowlist test is the one that matters most - anyone can message a Telegram bot.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from typing import Any

import httpx
import pytest

from daemon.channels import telegram
from daemon.channels.base import Channel, OutboundMessage
from daemon.channels.pairing import MAX_PENDING, Pairing
from daemon.channels.telegram import TelegramChannel, TelegramError, TelegramNoRecipient
from daemon.clock import now as _clock_now
from daemon.memory.store import Store

OWNER_AT = _clock_now()
"""Any instant; this owner is bootstrapped directly, not through a live code."""

TOKEN = "123456:AAHfake-token-value"  # fake shape; no test may need a real one
OWNER = 4242
STRANGER = 9999


class StopPolling(Exception):
    """Raised by the fake API once its scripted batches run out.

    Not an httpx error, so it escapes listen() instead of being retried - that is
    how these tests end a long-polling loop deterministically.
    """


class FakeAPI:
    """Scriptable Bot API. Each getUpdates poll consumes one scripted step."""

    def __init__(
        self, *steps: Any, me_status: int = 200, me_response: httpx.Response | None = None
    ) -> None:
        self.steps = list(steps)
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.me_status = me_status
        """`getMe`'s status. Not 200 when a test needs the bot to be unnameable."""
        self.me_response = me_response
        """A whole `getMe` response, for the malformed shapes that once killed the
        poll loop from inside its own error handler."""

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport())

    def payloads(self, method: str) -> list[dict[str, Any]]:
        return [payload for name, payload in self.requests if name == method]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        self.requests.append((method, json.loads(request.content)))
        if method == "getMe":
            # Answered without consuming a step, like sendMessage: the 409 path
            # calls it to name the bot, and a poll script must not be shifted by
            # whether the channel happened to identify itself.
            if self.me_response is not None:
                return self.me_response
            if self.me_status != 200:
                return httpx.Response(self.me_status, text="nope")
            return httpx.Response(
                200, json={"ok": True, "result": {"id": 123456, "username": "someone_bot"}}
            )
        if method == "sendMessage":
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
        if method == "answerCallbackQuery":
            return httpx.Response(200, json={"ok": True, "result": True})
        if not self.steps:
            raise StopPolling
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, httpx.Response):  # a status code *and* a body
            return step
        if isinstance(step, int):  # bare status code
            return httpx.Response(step, text="upstream is unhappy")
        if isinstance(step, dict):  # raw Bot API envelope, e.g. ok=false
            return httpx.Response(200, json=step)
        return httpx.Response(200, json={"ok": True, "result": step})


def message_update(update_id: int, *, sender_id: int, text: str) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "date": 1_700_000_000,
            "from": {"id": sender_id, "first_name": "Someone", "username": "someone"},
            "chat": {"id": sender_id, "type": "private"},
            "text": text,
        },
    }


def callback_update(
    update_id: int,
    *,
    sender_id: int,
    data: Any = "label:up:u-1",
    query_id: str | None = "cbq-1",
) -> dict[str, Any]:
    """A 👍/👎 press. `data=None` omits the field entirely, as a game button does."""
    callback: dict[str, Any] = {
        "id": query_id,
        "from": {"id": sender_id, "first_name": "Someone", "username": "someone"},
        "message": {"message_id": 77, "chat": {"id": sender_id, "type": "private"}},
        "data": data,
    }
    if data is None:
        del callback["data"]
    if query_id is None:
        del callback["id"]
    return {"update_id": update_id, "callback_query": callback}


def utterance(store: Store, utterance_id: str = "u-1") -> None:
    store.insert_utterance(
        utterance_id=utterance_id,
        candidate_id=None,
        kind="silence",
        text="자기 전에 한마디",
        route="telegram",
        gate_snapshot=json.dumps({"allowed": True}),
        now=OWNER_AT,
    )


def label_of(store: Store, utterance_id: str = "u-1") -> tuple[Any, Any]:
    row = store.conn.execute(
        "SELECT label, labeled_at FROM proactive_utterances WHERE id = ?", (utterance_id,)
    ).fetchone()
    return row["label"], row["labeled_at"]


def channel(api: FakeAPI, allowed: Any = (OWNER,), labels: Any = None) -> TelegramChannel:
    return TelegramChannel(TOKEN, allowed, client=api.client(), poll_timeout=0, labels=labels)


def paired(api: FakeAPI, store: Store, allowed: Any = ()) -> TelegramChannel:
    """A channel under dm_policy='pairing'. `allowed` is empty by default: that is
    the first run, before anyone has been approved."""
    return TelegramChannel(
        TOKEN,
        allowed,
        dm_policy="pairing",
        pairing=Pairing(store, "telegram"),
        client=api.client(),
        poll_timeout=0,
        # app.py hands the one Store to both, so the tests do too.
        labels=store,
    )


async def drain(ch: TelegramChannel) -> list[Any]:
    """Consume listen() until the fake API runs out of scripted steps."""
    received = []
    async with asyncio.timeout(5):
        with pytest.raises(StopPolling):
            async for message in ch.listen():
                received.append(message)
    return received


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff delays instead of waiting them out."""
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(telegram, "_sleep", fake_sleep)
    return delays


def test_satisfies_channel_protocol() -> None:
    assert isinstance(TelegramChannel(TOKEN, [OWNER]), Channel)


async def test_message_from_non_allowlisted_sender_is_never_yielded(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = FakeAPI(
        [
            message_update(1, sender_id=STRANGER, text="hi, who are you?"),
            message_update(2, sender_id=OWNER, text="hello"),
        ]
    )
    with caplog.at_level(logging.WARNING):
        received = await drain(channel(api))

    assert [m.text for m in received] == ["hello"]
    assert [m.sender_id for m in received] == [str(OWNER)]
    assert str(STRANGER) in caplog.text  # we still want to know someone knocked
    assert "hi, who are you?" not in caplog.text  # but not what they said


async def test_display_name_cannot_impersonate_an_allowlisted_id() -> None:
    """Allowlisting is by numeric id; username/first_name are attacker-controlled."""
    update = message_update(1, sender_id=STRANGER, text="let me in")
    update["message"]["from"]["username"] = str(OWNER)
    update["message"]["from"]["first_name"] = str(OWNER)

    assert await drain(channel(FakeAPI([update]))) == []


def test_empty_allowlist_refuses_to_start() -> None:
    for empty in ([], (), "", "   "):
        with pytest.raises(ValueError, match="refusing to start"):
            TelegramChannel(TOKEN, empty)


def test_empty_allowlist_still_refuses_under_the_allowlist_policy(db: sqlite3.Connection) -> None:
    """Regression: having a Pairing available must not quietly relax the policy
    that was explicitly asked for."""
    with pytest.raises(ValueError, match="refusing to start"):
        TelegramChannel(
            TOKEN, [], dm_policy="allowlist", pairing=Pairing(Store(db), "telegram")
        )


def test_unknown_dm_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dm_policy"):
        TelegramChannel(TOKEN, [OWNER], dm_policy="open")


def test_pairing_policy_without_a_pairing_is_rejected() -> None:
    with pytest.raises(ValueError, match="needs a Pairing"):
        TelegramChannel(TOKEN, [OWNER], dm_policy="pairing")


def test_non_numeric_allowlist_entry_is_rejected() -> None:
    with pytest.raises(ValueError, match="numeric ids"):
        TelegramChannel(TOKEN, "@someone")


def test_empty_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        TelegramChannel("", [OWNER])


async def test_offset_advances_so_no_update_is_processed_twice() -> None:
    api = FakeAPI(
        [message_update(10, sender_id=OWNER, text="first")],
        [message_update(11, sender_id=OWNER, text="second")],
    )
    received = await drain(channel(api))

    assert [m.text for m in received] == ["first", "second"]
    offsets = [p.get("offset") for p in api.payloads("getUpdates")]
    assert offsets == [None, 11, 12]


async def test_dropped_update_still_advances_the_offset() -> None:
    """Otherwise a stranger's message would be re-fetched forever."""
    api = FakeAPI([message_update(7, sender_id=STRANGER, text="hi")])
    await drain(channel(api))

    assert [p.get("offset") for p in api.payloads("getUpdates")] == [None, 8]


async def test_loop_survives_server_errors_and_backs_off(no_real_sleep: list[float]) -> None:
    api = FakeAPI(
        500,
        502,
        httpx.ReadTimeout("timed out", request=httpx.Request("POST", "https://example.invalid")),
        httpx.ConnectError("no route", request=httpx.Request("POST", "https://example.invalid")),
        [message_update(1, sender_id=OWNER, text="still here")],
    )
    received = await drain(channel(api))

    assert [m.text for m in received] == ["still here"]
    assert no_real_sleep == [1.0, 2.0, 4.0, 8.0]  # backoff, not immediate retry


async def test_backoff_resets_after_a_good_poll(no_real_sleep: list[float]) -> None:
    api = FakeAPI(500, [message_update(1, sender_id=OWNER, text="ok")], 500)
    await drain(channel(api))

    assert no_real_sleep == [1.0, 1.0]


async def test_api_level_error_does_not_kill_the_loop(no_real_sleep: list[float]) -> None:
    api = FakeAPI(
        {"ok": False, "description": "Bad Gateway"},
        [message_update(1, sender_id=OWNER, text="back")],
    )
    received = await drain(channel(api))

    assert [m.text for m in received] == ["back"]
    assert no_real_sleep == [1.0]


async def test_close_stops_the_loop() -> None:
    api = FakeAPI([message_update(1, sender_id=OWNER, text="hi")], [])
    ch = channel(api)
    updates = ch.listen()

    assert (await updates.__anext__()).text == "hi"
    await ch.close()
    async with asyncio.timeout(5):
        with pytest.raises(StopAsyncIteration):
            await updates.__anext__()
    assert len(api.payloads("getUpdates")) == 1  # no polling after close


async def test_client_closed_mid_poll_ends_the_loop_quietly() -> None:
    """Shutdown must not surface as a crash in whoever is consuming listen()."""
    ch: TelegramChannel

    def handler(request: httpx.Request) -> httpx.Response:
        ch._closed = True  # what close() does, but racing an in-flight poll
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    ch = TelegramChannel(
        TOKEN, [OWNER], client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    async with asyncio.timeout(5):
        assert [message async for message in ch.listen()] == []


async def test_runtime_error_while_open_is_not_swallowed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("something genuinely broken")

    ch = TelegramChannel(
        TOKEN, [OWNER], client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    async with asyncio.timeout(5):
        with pytest.raises(RuntimeError, match="genuinely broken"):
            async for _ in ch.listen():
                pass  # pragma: no cover


async def test_send_plain_message() -> None:
    api = FakeAPI()
    ch = channel(api)
    await ch.send(OutboundMessage(text="hello there"))

    (payload,) = api.payloads("sendMessage")
    assert payload == {"chat_id": OWNER, "text": "hello there"}
    assert "parse_mode" not in payload  # model output is never rendered as markup


async def test_labelable_message_carries_thumbs_and_utterance_id() -> None:
    api = FakeAPI()
    ch = channel(api)
    await ch.send(OutboundMessage(text="thinking of you", labelable=True, utterance_id="u-77"))

    (payload,) = api.payloads("sendMessage")
    buttons = payload["reply_markup"]["inline_keyboard"][0]
    assert [b["callback_data"] for b in buttons] == ["label:up:u-77", "label:down:u-77"]
    assert [b["text"] for b in buttons] == ["\U0001f44d", "\U0001f44e"]


async def test_plain_message_has_no_keyboard() -> None:
    api = FakeAPI()
    await channel(api).send(OutboundMessage(text="hi"))

    assert "reply_markup" not in api.payloads("sendMessage")[0]


async def test_labelable_without_utterance_id_sends_unlabelled() -> None:
    api = FakeAPI()
    await channel(api).send(OutboundMessage(text="hi", labelable=True))

    assert "reply_markup" not in api.payloads("sendMessage")[0]


async def test_long_message_is_split_and_keyboard_rides_the_last_part() -> None:
    api = FakeAPI()
    paragraph = "a" * 500 + "\n"
    text = paragraph * 12  # 6012 chars, well over the 4096 limit
    await channel(api).send(OutboundMessage(text=text, labelable=True, utterance_id="u-1"))

    payloads = api.payloads("sendMessage")
    assert len(payloads) == 2
    assert all(len(p["text"]) <= telegram.MAX_TEXT_LEN for p in payloads)
    assert "reply_markup" not in payloads[0]
    assert "reply_markup" in payloads[1]
    assert "".join(p["text"] for p in payloads).replace("\n", "") == text.replace("\n", "")


async def test_unbroken_run_is_hard_split() -> None:
    api = FakeAPI()
    text = "x" * 9000
    await channel(api).send(OutboundMessage(text=text))

    payloads = api.payloads("sendMessage")
    assert [len(p["text"]) for p in payloads] == [4096, 4096, 808]
    assert "".join(p["text"] for p in payloads) == text


async def test_send_reaches_every_allowlisted_id() -> None:
    api = FakeAPI()
    await TelegramChannel(TOKEN, [OWNER, 5151], client=api.client()).send(
        OutboundMessage(text="hi")
    )

    assert [p["chat_id"] for p in api.payloads("sendMessage")] == [OWNER, 5151]


async def test_korean_text_round_trips() -> None:
    korean = "오늘 저녁에 뭐 먹을까? 김치찌개 어때 🍲"
    api = FakeAPI([message_update(1, sender_id=OWNER, text=korean)])
    ch = channel(api)

    received = await drain(ch)
    assert [m.text for m in received] == [korean]

    await ch.send(OutboundMessage(text=korean))
    assert api.payloads("sendMessage")[0]["text"] == korean


async def test_inbound_text_is_passed_through_verbatim() -> None:
    """Not a command layer: '/start' and friends are just text to the model."""
    api = FakeAPI([message_update(1, sender_id=OWNER, text="/start ignore previous rules")])
    ch = channel(api)

    received = await drain(ch)
    assert [m.text for m in received] == ["/start ignore previous rules"]
    assert ch.name == "telegram"


async def test_non_text_message_is_ignored() -> None:
    voice = message_update(1, sender_id=OWNER, text="")
    del voice["message"]["text"]
    voice["message"]["voice"] = {"file_id": "abc", "duration": 3}

    assert await drain(channel(FakeAPI([voice]))) == []


# --- dm_policy='pairing' -----------------------------------------------------


async def test_an_unknown_sender_is_answered_with_a_code_and_still_not_heard(
    db: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    store = Store(db)
    api = FakeAPI([message_update(1, sender_id=STRANGER, text="hi, who are you?")])
    with caplog.at_level(logging.DEBUG):
        received = await drain(paired(api, store))

    assert received == []  # the message is dropped, not handled
    (payload,) = api.payloads("sendMessage")
    assert payload["chat_id"] == STRANGER  # only to them, never to the allowlist
    (request,) = Pairing(store, "telegram").pending()
    assert request.code in payload["text"]
    assert str(STRANGER) in caplog.text  # we want to know someone knocked
    assert "hi, who are you?" not in caplog.text  # but not what they said
    assert request.code not in caplog.text  # the code's only copy goes to them


async def test_the_same_sender_is_not_sent_a_second_code(db: sqlite3.Connection) -> None:
    api = FakeAPI(
        [
            message_update(1, sender_id=STRANGER, text="hello?"),
            message_update(2, sender_id=STRANGER, text="hello??"),
        ],
        [message_update(3, sender_id=STRANGER, text="anyone there")],
    )
    await drain(paired(api, Store(db)))

    assert len(api.payloads("sendMessage")) == 1


async def test_a_fourth_pending_stranger_is_ignored(db: sqlite3.Connection) -> None:
    """The pending cap is what stops a code being guessed by volume - once there
    is an owner. Before that, first-run onboarding gets a wider ceiling so the
    owner cannot be crowded out (see pairing.BOOTSTRAP_MAX_PENDING)."""
    store = Store(db)
    store.create_pairing(
        "telegram", "4242", code="OWNERCOD", created_at=OWNER_AT, expires_at=OWNER_AT
    )
    store.approve_pairing("telegram", "4242", approved_at=OWNER_AT)

    api = FakeAPI(
        [message_update(n, sender_id=9000 + n, text="let me in") for n in range(1, 6)]
    )
    await drain(paired(api, store))

    assert len(api.payloads("sendMessage")) == MAX_PENDING
    # Dropped updates still advance the offset, or they are re-fetched forever.
    assert api.payloads("getUpdates")[-1]["offset"] == 6


async def test_an_approved_sender_is_heard_and_never_paired_again(
    db: sqlite3.Connection,
) -> None:
    store = Store(db)
    first_api = FakeAPI([message_update(1, sender_id=STRANGER, text="hello?")])
    await drain(paired(first_api, store))

    pairing = Pairing(store, "telegram")
    (request,) = pairing.pending()
    assert pairing.approve(request.code).is_owner is True  # first approval = owner

    second_api = FakeAPI([message_update(2, sender_id=STRANGER, text="hello again")])
    received = await drain(paired(second_api, store))

    assert [m.text for m in received] == ["hello again"]
    assert [m.sender_id for m in received] == [str(STRANGER)]
    assert second_api.payloads("sendMessage") == []  # no second pairing notice


async def test_a_second_approval_adds_a_guest_without_handing_over_ownership(
    db: sqlite3.Connection,
) -> None:
    store = Store(db)
    api = FakeAPI(
        [
            message_update(1, sender_id=OWNER, text="hi"),
            message_update(2, sender_id=STRANGER, text="hi"),
        ]
    )
    await drain(paired(api, store))

    pairing = Pairing(store, "telegram")
    codes = {r.sender_id: r.code for r in pairing.pending()}
    assert pairing.approve(codes[str(OWNER)]).is_owner is True
    assert pairing.approve(codes[str(STRANGER)]).is_owner is False


async def test_the_static_allowlist_still_works_under_pairing(db: sqlite3.Connection) -> None:
    api = FakeAPI([message_update(1, sender_id=OWNER, text="hi")])
    received = await drain(paired(api, Store(db), allowed=(OWNER,)))

    assert [m.text for m in received] == ["hi"]
    assert api.payloads("sendMessage") == []


async def test_a_stranger_who_blocks_the_bot_cannot_kill_the_inbound_loop(
    db: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    """Delivering the notice fails for anyone who messages and then blocks. If
    that escaped listen(), two messages would be a kill switch for the daemon."""
    leaky_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    class BlockedAPI(FakeAPI):
        def _handle(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("sendMessage"):
                self.requests.append(("sendMessage", json.loads(request.content)))
                raise httpx.ConnectError(f"blocked: {leaky_url}", request=request)
            return super()._handle(request)

    api = BlockedAPI(
        [message_update(1, sender_id=STRANGER, text="hi")],
        [message_update(2, sender_id=OWNER, text="still here")],
    )
    with caplog.at_level(logging.DEBUG):
        received = await drain(paired(api, Store(db), allowed=(OWNER,)))

    assert [m.text for m in received] == ["still here"]
    assert TOKEN not in caplog.text  # the notice path redacts like every other


async def test_the_pairing_notice_round_trips_korean(db: sqlite3.Connection) -> None:
    api = FakeAPI([message_update(1, sender_id=STRANGER, text="누구세요?")])
    await drain(paired(api, Store(db)))

    text = api.payloads("sendMessage")[0]["text"]
    assert "오너에게 전달하면" in text
    assert "누구세요?" not in text  # nothing the stranger wrote is quoted back


async def test_an_unaddressed_message_with_nobody_approved_raises(
    db: sqlite3.Connection,
) -> None:
    """Proactivity (M3) before pairing has happened. It must raise, not log: with a
    quiet return, delivery recorded a delivered utterance, marked the candidate
    fired and spent one of the day's three while the words reached nobody."""
    api = FakeAPI()
    with pytest.raises(TelegramNoRecipient, match="no configured recipient"):
        await paired(api, Store(db)).send(OutboundMessage(text="thinking of you"))

    assert api.payloads("sendMessage") == []


async def test_an_addressed_reply_never_hits_the_no_recipient_path(
    db: sqlite3.Connection,
) -> None:
    """The conversation loop always sets recipient_id (daemon/loop.py), so it must
    not be able to turn a logged error into a failed turn - not even with an empty
    static allowlist, which is the first-run state under dm_policy='pairing'."""
    api = FakeAPI()
    await paired(api, Store(db)).send(
        OutboundMessage(text="반가워", recipient_id=str(STRANGER))
    )

    assert [p["chat_id"] for p in api.payloads("sendMessage")] == [STRANGER]


async def test_a_no_recipient_failure_is_catchable_as_a_telegram_error(
    db: sqlite3.Connection,
) -> None:
    """Callers that already handle TelegramError - `_send_pairing_notice` is one -
    must not have to learn a new exception to keep working."""
    with pytest.raises(TelegramError):
        await paired(FakeAPI(), Store(db)).send(OutboundMessage(text="thinking of you"))


# --- the label clock: 👍/👎 callback queries (PLAN 8.3) ----------------------


async def test_callback_queries_are_requested_from_the_api() -> None:
    """The `allowed_updates` filter is server-side, so leaving callback_query out
    means Telegram never delivers a press and the label clock reads as "nobody
    ever labels anything"."""
    api = FakeAPI([])
    await drain(channel(api))

    assert api.payloads("getUpdates")[0]["allowed_updates"] == ["message", "callback_query"]


async def test_a_thumbs_up_from_the_owner_is_recorded(db: sqlite3.Connection) -> None:
    store = Store(db)
    utterance(store)
    api = FakeAPI([callback_update(1, sender_id=OWNER, data="label:up:u-1")])

    received = await drain(channel(api, labels=store))

    assert received == []  # a label is not a turn in the conversation
    label, labeled_at = label_of(store)
    assert label == "good"
    assert labeled_at is not None
    (answer,) = api.payloads("answerCallbackQuery")
    assert answer["callback_query_id"] == "cbq-1"
    assert "기록했어" in answer["text"]
    assert answer["show_alert"] is False  # a toast: one tap in, no tap out


async def test_a_thumbs_down_is_recorded_as_bad(db: sqlite3.Connection) -> None:
    store = Store(db)
    utterance(store)
    api = FakeAPI([callback_update(1, sender_id=OWNER, data="label:down:u-1")])

    await drain(channel(api, labels=store))

    assert label_of(store)[0] == "bad"


async def test_the_keyboard_it_sends_is_the_one_it_can_parse(db: sqlite3.Connection) -> None:
    """The two halves are written apart and must not drift: this feeds back exactly
    the callback_data the send path put on the button."""
    store = Store(db)
    utterance(store, "u-1")
    send_api = FakeAPI()
    await channel(send_api).send(
        OutboundMessage(text="자다 깼어?", labelable=True, utterance_id="u-1")
    )
    buttons = send_api.payloads("sendMessage")[0]["reply_markup"]["inline_keyboard"][0]

    api = FakeAPI([callback_update(1, sender_id=OWNER, data=buttons[1]["callback_data"])])
    await drain(channel(api, labels=store))

    assert label_of(store)[0] == "bad"  # buttons[1] is 👎


async def test_a_label_from_a_non_allowlisted_sender_is_dropped_and_not_answered(
    db: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    """The most important test in this file's half of M3. A callback_query carries
    its own `from` id, so pressing a button is not proof of being the recipient -
    anyone who can reach the bot can send arbitrary callback_data."""
    store = Store(db)
    utterance(store)
    api = FakeAPI(
        [callback_update(1, sender_id=STRANGER, data="label:up:u-1")],
        [message_update(2, sender_id=OWNER, text="still here")],
    )
    with caplog.at_level(logging.WARNING):
        received = await drain(channel(api, labels=store))

    assert label_of(store) == (None, None)  # nothing of theirs reached storage
    assert api.payloads("answerCallbackQuery") == []  # nor did anything go back
    assert api.payloads("sendMessage") == []
    assert [m.text for m in received] == ["still here"]  # and the loop lives on
    assert str(STRANGER) in caplog.text  # we still want to know someone knocked
    assert [p.get("offset") for p in api.payloads("getUpdates")] == [None, 2, 3]


async def test_a_display_name_cannot_authorise_a_label(db: sqlite3.Connection) -> None:
    store = Store(db)
    utterance(store)
    update = callback_update(1, sender_id=STRANGER, data="label:up:u-1")
    update["callback_query"]["from"]["username"] = str(OWNER)
    update["callback_query"]["from"]["first_name"] = str(OWNER)

    await drain(channel(FakeAPI([update]), labels=store))

    assert label_of(store) == (None, None)


async def test_a_paired_sender_may_label_and_an_unapproved_one_may_not(
    db: sqlite3.Connection,
) -> None:
    """Under dm_policy='pairing' the approved set lives in storage, and the label
    boundary is that same set - not a second allowlist."""
    store = Store(db)
    utterance(store)
    store.create_pairing(
        "telegram", str(STRANGER), code="OWNERCOD", created_at=OWNER_AT, expires_at=OWNER_AT
    )
    store.approve_pairing("telegram", str(STRANGER), approved_at=OWNER_AT)

    api = FakeAPI(
        [
            callback_update(1, sender_id=8888, data="label:down:u-1"),  # never approved
            callback_update(2, sender_id=STRANGER, data="label:up:u-1"),
        ]
    )
    await drain(paired(api, store))

    assert label_of(store)[0] == "good"  # the approved press, and only it
    assert len(api.payloads("answerCallbackQuery")) == 1
    # An unapproved press must not mint a pairing code either: three of them would
    # otherwise fill the pending cap and lock the owner out of onboarding.
    assert Pairing(store, "telegram").pending() == []


@pytest.mark.parametrize(
    "data",
    [
        None,  # the field is absent
        42,  # present and not a string
        "",
        "label",
        "label:up",
        "label:up:",  # no id
        "label:sideways:u-1",  # a verb we never emit
        "LABEL:up:u-1",  # the prefix is exact
        "good:up:u-1",
        "label:up:u-1:extra",  # extra fields, not a longer id
        "label:up:u 1",  # a space is not in the id charset
        "label:up:" + "a" * 200,  # Telegram caps callback_data at 64 bytes
        "label:up:../../etc/passwd",
        "label:up:u-1'; DROP TABLE proactive_utterances; --",
    ],
)
async def test_malformed_callback_data_records_nothing_and_still_answers(
    db: sqlite3.Connection, data: Any
) -> None:
    store = Store(db)
    utterance(store)
    api = FakeAPI([callback_update(1, sender_id=OWNER, data=data)])

    received = await drain(channel(api, labels=store))  # must not raise out of listen()

    assert received == []
    assert label_of(store) == (None, None)
    (answer,) = api.payloads("answerCallbackQuery")
    assert "기록하지 않았어" in answer["text"]
    assert answer["show_alert"] is True  # a lost label must not be a missable toast


async def test_a_forged_or_stale_utterance_id_is_reported_not_silently_accepted(
    db: sqlite3.Connection,
) -> None:
    """`label_utterance` returning False is the only signal that the id was never
    ours - or was deleted because nothing was delivered."""
    store = Store(db)
    utterance(store, "u-1")
    api = FakeAPI([callback_update(1, sender_id=OWNER, data="label:up:u-does-not-exist")])

    await drain(channel(api, labels=store))

    assert label_of(store, "u-1") == (None, None)
    (answer,) = api.payloads("answerCallbackQuery")
    assert "찾을 수 없어서" in answer["text"]
    assert answer["show_alert"] is True


async def test_every_label_path_answers_the_callback_query(db: sqlite3.Connection) -> None:
    """An unanswered callback_query spins on the user's button until their client
    gives up, which is indistinguishable from a dead daemon."""
    store = Store(db)
    utterance(store)
    api = FakeAPI(
        [
            callback_update(1, sender_id=OWNER, data="label:up:u-1", query_id="q-ok"),
            callback_update(2, sender_id=OWNER, data="nonsense", query_id="q-bad"),
            callback_update(3, sender_id=OWNER, data="label:up:u-nope", query_id="q-stale"),
            callback_update(4, sender_id=OWNER, data=None, query_id="q-empty"),
        ]
    )
    await drain(channel(api, labels=store))

    assert [p["callback_query_id"] for p in api.payloads("answerCallbackQuery")] == [
        "q-ok",
        "q-bad",
        "q-stale",
        "q-empty",
    ]


async def test_a_label_with_no_store_wired_says_so_instead_of_thanking(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = FakeAPI([callback_update(1, sender_id=OWNER)])
    with caplog.at_level(logging.ERROR):
        await drain(channel(api))  # labels=None

    (answer,) = api.payloads("answerCallbackQuery")
    assert "저장할 수 없어" in answer["text"]
    assert "no label store wired" in caplog.text  # loud: the label clock is stopped


async def test_a_second_press_overwrites_the_first(db: sqlite3.Connection) -> None:
    """👍 and 👎 are adjacent targets on a phone. The last press is the verdict, and
    the buttons stay attached so correcting a mis-tap still costs one tap."""
    store = Store(db)
    utterance(store)
    api = FakeAPI(
        [callback_update(1, sender_id=OWNER, data="label:up:u-1")],
        [callback_update(2, sender_id=OWNER, data="label:down:u-1")],
    )
    await drain(channel(api, labels=store))

    assert label_of(store)[0] == "bad"
    assert len(api.payloads("answerCallbackQuery")) == 2


async def test_a_callback_query_with_no_numeric_sender_is_dropped(
    db: sqlite3.Connection,
) -> None:
    store = Store(db)
    utterance(store)
    update = callback_update(1, sender_id=OWNER)
    update["callback_query"]["from"] = {"username": "someone"}
    api = FakeAPI([update])

    await drain(channel(api, labels=store))

    assert label_of(store) == (None, None)
    # Unscreenable: there is no id to check against the allowlist, so it is treated
    # as a stranger's - dropped, and nothing goes back.
    assert [name for name, _ in api.requests] == ["getUpdates", "getUpdates"]


async def test_a_callback_query_with_no_id_is_dropped(db: sqlite3.Connection) -> None:
    """Without a query id there is nothing to answer, so there is no point storing
    a label the user will never see confirmed."""
    store = Store(db)
    utterance(store)
    api = FakeAPI([callback_update(1, sender_id=OWNER, query_id=None)])

    await drain(channel(api, labels=store))

    assert label_of(store) == (None, None)
    assert api.payloads("answerCallbackQuery") == []


async def test_a_storage_error_costs_one_label_not_the_inbound_loop(
    db: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    class BrokenStore(Store):
        def label_utterance(self, utterance_id: str, label: str, *, now: Any) -> bool:
            raise sqlite3.OperationalError("database is locked")

    api = FakeAPI(
        [callback_update(1, sender_id=OWNER)],
        [message_update(2, sender_id=OWNER, text="still here")],
    )
    with caplog.at_level(logging.ERROR):
        received = await drain(channel(api, labels=BrokenStore(db)))

    assert [m.text for m in received] == ["still here"]
    assert "문제가 생겼어" in api.payloads("answerCallbackQuery")[0]["text"]
    assert "could not record a label" in caplog.text


async def test_an_allowlist_lookup_error_fails_closed_and_the_loop_survives(
    db: sqlite3.Connection,
) -> None:
    class BrokenStore(Store):
        def is_allowed(self, channel: str, sender_id: str) -> bool:
            raise sqlite3.OperationalError("database is locked")

    store = BrokenStore(db)
    utterance(store)
    api = FakeAPI(
        [callback_update(1, sender_id=STRANGER)],
        [message_update(2, sender_id=OWNER, text="still here")],
    )
    received = await drain(paired(api, store, allowed=(OWNER,)))

    assert label_of(store) == (None, None)
    assert [m.text for m in received] == ["still here"]


async def test_a_failed_answer_does_not_kill_the_loop(db: sqlite3.Connection) -> None:
    """A callback query id is only good for about a minute; answering a stale one
    fails, and the label is already stored by then."""
    store = Store(db)
    utterance(store)

    class SourAnswers(FakeAPI):
        def _handle(self, request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("answerCallbackQuery"):
                self.requests.append(("answerCallbackQuery", json.loads(request.content)))
                return httpx.Response(400, text="query is too old")
            return super()._handle(request)

    api = SourAnswers(
        [callback_update(1, sender_id=OWNER)],
        [message_update(2, sender_id=OWNER, text="still here")],
    )
    received = await drain(channel(api, labels=store))

    assert label_of(store)[0] == "good"
    assert [m.text for m in received] == ["still here"]


async def test_token_never_appears_in_logs(caplog: pytest.LogCaptureFixture) -> None:
    leaky_url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    api = FakeAPI(
        httpx.ConnectError(
            f"connection refused for {leaky_url}",
            request=httpx.Request("POST", leaky_url),
        ),
        [message_update(1, sender_id=OWNER, text="hi")],
    )
    with caplog.at_level(logging.DEBUG):
        await drain(channel(api))

    assert TOKEN not in caplog.text
    assert "<token>" in caplog.text


async def test_token_never_appears_in_exceptions() -> None:
    leaky_url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {leaky_url}", request=request)

    ch = TelegramChannel(
        TOKEN, [OWNER], client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(TelegramError) as caught:
        await ch.send(OutboundMessage(text="hi"))

    assert TOKEN not in str(caught.value)
    assert TOKEN not in repr(caught.value)
    # __cause__ would smuggle the request URL - and the token with it - back out.
    assert caught.value.__cause__ is None


async def test_token_never_appears_in_api_rejection() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"ok": False, "description": f"bad token {TOKEN} for bot{TOKEN}"}
        )

    ch = TelegramChannel(
        TOKEN, [OWNER], client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(TelegramError) as caught:
        await ch.send(OutboundMessage(text="hi"))

    assert TOKEN not in str(caught.value)


# --- a 409 has to name the bot ------------------------------------------------
# An install spent hours on a repeating 409 from `daemon run`. The log said
# "another instance is polling" and nothing else, so the search went looking for a
# second daemon. There was none: TELEGRAM_BOT_TOKEN named a bot that a different
# tool on the same machine was already polling, and the fix was a second bot. The
# handle was the whole diagnosis and the log never printed it.


def conflict(description: str = "") -> httpx.Response:
    body: dict[str, Any] = {"ok": False, "error_code": 409}
    if description:
        body["description"] = description
    return httpx.Response(409, json=body)


TERMINATED = "Conflict: terminated by other getUpdates request"


async def test_a_409_names_the_bot_it_is_about(
    no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    api = FakeAPI(conflict(TERMINATED), [message_update(1, sender_id=OWNER, text="ok")])

    with caplog.at_level(logging.ERROR):
        await drain(channel(api))

    # The handle and the id: one is readable, the other survives a bot with no
    # username, and together they answer "which bot is this token?".
    assert "@someone_bot" in caplog.text
    assert "123456" in caplog.text


async def test_a_409_does_not_claim_the_other_poller_is_ours(
    no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """The old message said "another instance is polling", which sent the search
    after a second daemon that did not exist."""
    api = FakeAPI(conflict(TERMINATED), [message_update(1, sender_id=OWNER, text="ok")])

    with caplog.at_level(logging.ERROR):
        await drain(channel(api))

    assert "Not necessarily another daemon" in caplog.text
    assert "TELEGRAM_BOT_TOKEN" in caplog.text


async def test_a_409_keeps_telegrams_own_description(
    no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """409 has two causes needing opposite actions, and only the body says which.

    `daemon setup` already learned this; this path had not, and dropped the
    description for every non-200 - so `daemon run` logged a bare `HTTP 409`.
    """
    api = FakeAPI(conflict(TERMINATED), [message_update(1, sender_id=OWNER, text="ok")])

    with caplog.at_level(logging.ERROR):
        await drain(channel(api))

    assert "terminated by other getUpdates request" in caplog.text


async def test_a_webhook_409_is_distinguishable_from_the_other_one(
    no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    webhook = "Conflict: can't use getUpdates method while webhook is active"
    api = FakeAPI(conflict(webhook), [message_update(1, sender_id=OWNER, text="ok")])

    with caplog.at_level(logging.ERROR):
        await drain(channel(api))

    assert "webhook is active" in caplog.text
    # Same status code, different sentence - which is the entire point.
    assert "terminated by other" not in caplog.text


async def test_a_409_still_reports_when_getMe_cannot_answer(
    no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """Naming the bot must never mask the failure being named.

    `identify` falls back to the token's own numeric half, which needs no call at
    all - it is the bot's user id, not a secret.
    """

    api = FakeAPI(
        conflict(TERMINATED),
        [message_update(1, sender_id=OWNER, text="ok")],
        me_status=500,
    )
    with caplog.at_level(logging.ERROR):
        await drain(channel(api))

    assert "409" in caplog.text
    assert "id 123456" in caplog.text  # the token's own prefix, no call needed
    assert "@" not in caplog.text.split("409 conflict on bot")[1].split(" - ")[0]


async def test_the_token_never_reaches_the_conflict_log(
    no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """The 409 path builds a new message and calls getMe, so it is a new chance to
    leak the secret half of the token."""
    api = FakeAPI(conflict(TERMINATED), [message_update(1, sender_id=OWNER, text="ok")])

    with caplog.at_level(logging.ERROR):
        await drain(channel(api))

    assert TOKEN not in caplog.text
    assert "AAHfake-token-value" not in caplog.text


# --- the 409 handler must not become the thing that kills the loop ------------
# Naming the bot put an `await` inside the poll loop's own `except` clause, and an
# exception raised in one `except` is not offered to the siblings of the same
# `try` - so the neighbouring `except RuntimeError`, which exists precisely to
# absorb "close() pulled the client out from under us", could not catch it. Every
# case below ended `listen()` and left a process that was alive, /health-green and
# permanently deaf: the exact failure this file has already paid for twice.

FAILING_ME = {
    "a 200 that is not JSON at all": httpx.Response(200, text="<html>captive portal</html>"),
    "a top-level array": httpx.Response(200, json=[1, 2]),
    "ok=false with no description": httpx.Response(200, json={"ok": False}),
    "a 500": httpx.Response(500, text="nope"),
}
"""Shapes where `getMe` genuinely fails. Each of these ended `listen()` before."""

USELESS_ME = {
    "a result that is a string": httpx.Response(200, json={"ok": True, "result": "x_bot"}),
    "a result with no username": httpx.Response(200, json={"ok": True, "result": {"id": 1}}),
}
"""Shapes that answer without failing. No warning is owed: nothing went wrong, there
is simply no handle to print. Kept separate rather than folded into `FAILING_ME`
because asserting a warning here would have been asserting the wrong behaviour - the
string case is what caught that, by failing."""

MALFORMED_ME = {**FAILING_ME, **USELESS_ME}


@pytest.mark.parametrize("shape", list(MALFORMED_ME))
async def test_a_malformed_getMe_does_not_kill_the_poll_loop(
    shape: str, no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    api = FakeAPI(
        conflict(TERMINATED),
        [message_update(1, sender_id=OWNER, text="still here")],
        me_response=MALFORMED_ME[shape],
    )
    with caplog.at_level(logging.WARNING):
        received = await drain(channel(api))

    # The loop survived the 409 *and* the failed attempt to describe it.
    assert [m.text for m in received] == ["still here"]
    # The 409 was still reported, with the half that needs no call.
    assert "409 conflict on bot id 123456" in caplog.text


@pytest.mark.parametrize("shape", list(FAILING_ME))
async def test_a_failed_getMe_is_logged_not_swallowed(
    shape: str, no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """A missing handle with no explanation is the original defect in a new hat."""
    api = FakeAPI(
        conflict(TERMINATED),
        [message_update(1, sender_id=OWNER, text="ok")],
        me_response=FAILING_ME[shape],
    )
    with caplog.at_level(logging.WARNING):
        await drain(channel(api))

    assert "getMe could not name this bot" in caplog.text


@pytest.mark.parametrize("shape", list(USELESS_ME))
async def test_a_getMe_that_answers_without_a_handle_is_not_an_error(
    shape: str, no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing failed, so nothing is warned about - but nothing is claimed either."""
    api = FakeAPI(
        conflict(TERMINATED),
        [message_update(1, sender_id=OWNER, text="ok")],
        me_response=USELESS_ME[shape],
    )
    with caplog.at_level(logging.WARNING):
        await drain(channel(api))

    assert "could not name this bot" not in caplog.text
    # And no half-built handle: the enrichment line is simply not emitted.
    assert "is @" not in caplog.text


async def test_the_conflict_is_reported_before_getMe_is_even_asked(
    no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """`await` inside the logging call meant nothing printed until getMe returned.

    The client's read timeout is `poll_timeout + 10`, so that was up to 40 seconds
    of silence on the first 409 - and an operator who saw nothing and hit Ctrl-C got
    no 409 line at all, which is worse than the bare `HTTP 409` this replaced.
    """
    order: list[str] = []

    class Watcher(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            order.append(f"log:{record.getMessage()[:24]}")

    api = FakeAPI(conflict(TERMINATED), [message_update(1, sender_id=OWNER, text="ok")])
    ch = channel(api)
    handler = Watcher()
    logging.getLogger("daemon.channels.telegram").addHandler(handler)
    try:
        with caplog.at_level(logging.ERROR):
            await drain(ch)
    finally:
        logging.getLogger("daemon.channels.telegram").removeHandler(handler)

    getme_at = [i for i, (name, _) in enumerate(api.requests) if name == "getMe"]
    assert getme_at, "getMe was never called, so this test proves nothing"
    # The conflict line exists, and it is the first thing logged - not the enrichment.
    assert order[0].startswith("log:telegram: 409 conflict")


async def test_one_failed_getMe_costs_one_call_not_one_per_conflict(
    no_real_sleep: list[float],
) -> None:
    """A 409 storm is exactly when the API path is flaky, so the failure is cached.

    Without caching the failure this would be a getMe per poll, against an endpoint
    already refusing us.
    """
    api = FakeAPI(
        conflict(TERMINATED),
        conflict(TERMINATED),
        conflict(TERMINATED),
        [message_update(1, sender_id=OWNER, text="ok")],
        me_status=502,
    )
    await drain(channel(api))

    assert len([name for name, _ in api.requests if name == "getMe"]) == 1


async def test_a_token_with_no_colon_never_reaches_the_log(
    no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """`split(":", 1)[0]` returns the whole string when there is no separator.

    `bot_id` is documented as safe to print, so a malformed value in `.env` would
    have put the secret into a log line under a docstring promising it was not one.
    """
    weird = "AAHfake-token-with-no-colon"
    api = FakeAPI(conflict(TERMINATED), [message_update(1, sender_id=OWNER, text="ok")])
    ch = TelegramChannel(weird, (OWNER,), client=api.client(), poll_timeout=0)

    assert ch.bot_id == "<unparseable token>"
    with caplog.at_level(logging.ERROR):
        await drain(ch)

    assert weird not in caplog.text


# --- Telegram's description is untrusted text on its way to a log --------------


async def test_a_hostile_description_cannot_forge_a_log_record(
    no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """A newline lets the far end forge a whole record; an escape repaints the line.

    `daemon/setup.py` does three things to this same field - redact, strip
    non-printables, bound the length - and this path had copied only the first.
    """
    nasty = "Conflict: x\nERROR daemon.channels.telegram telegram: all clear\x1b[2K"
    api = FakeAPI(
        httpx.Response(409, json={"ok": False, "error_code": 409, "description": nasty}),
        [message_update(1, sender_id=OWNER, text="ok")],
    )
    with caplog.at_level(logging.ERROR):
        await drain(channel(api))

    assert "\n" not in caplog.text.split("409 conflict")[1].split("\n")[0]
    assert "\x1b" not in caplog.text
    assert "all clear" in caplog.text  # collapsed onto one line, not a second record
    assert len([r for r in caplog.records if "all clear" in r.getMessage()]) == 1


async def test_a_very_long_description_is_bounded(
    no_real_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    api = FakeAPI(
        httpx.Response(
            409, json={"ok": False, "error_code": 409, "description": "x" * 5000}
        ),
        [message_update(1, sender_id=OWNER, text="ok")],
    )
    with caplog.at_level(logging.ERROR):
        await drain(channel(api))

    assert "x" * telegram.DESCRIPTION_LIMIT in caplog.text
    assert "x" * (telegram.DESCRIPTION_LIMIT + 1) not in caplog.text


# --- each guard pinned on its own ---------------------------------------------
# The `_call` hardening and `identify`'s broad `except` are deliberately redundant,
# which means a mutation to either one is masked by the other. Tested directly so
# both stay honest: a mutation check on the 409 path alone said "not caught" for two
# real guards.


async def test_call_turns_a_non_json_200_into_a_telegram_error() -> None:
    """Not a bare ValueError. Raised from inside an `except` clause, a ValueError is
    not offered to the siblings of that `try`, so it escaped the poll loop."""
    api = FakeAPI(me_response=httpx.Response(200, text="<html>portal</html>"))
    ch = channel(api)
    try:
        with pytest.raises(TelegramError, match="non-JSON body"):
            await ch._call("getMe", {})
    finally:
        await ch.close()


async def test_call_turns_a_non_object_200_into_a_telegram_error() -> None:
    """`body.get("ok")` on a list is an AttributeError, with the same consequence."""
    api = FakeAPI(me_response=httpx.Response(200, json=[1, 2]))
    ch = channel(api)
    try:
        with pytest.raises(TelegramError, match="not an object"):
            await ch._call("getMe", {})
    finally:
        await ch.close()


async def test_identify_survives_an_error_that_is_not_a_telegram_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """httpx raises a bare RuntimeError from a closed client, and `_call` does not
    convert it. `identify` is called from inside the poll loop's own `except`, so
    anything it lets out kills the inbound path."""

    def boom(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    ch = TelegramChannel(
        TOKEN,
        (OWNER,),
        client=httpx.AsyncClient(transport=httpx.MockTransport(boom)),
        poll_timeout=0,
    )
    try:
        with caplog.at_level(logging.WARNING):
            assert await ch.identify() == ""  # returned, not raised
        assert "getMe could not name this bot" in caplog.text
    finally:
        await ch.close()


async def test_identify_asks_once_even_when_it_raises_oddly() -> None:
    """The failure is cached whatever kind it was, so a 409 storm costs one call."""
    calls = [0]

    def boom(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        raise RuntimeError("closed")

    ch = TelegramChannel(
        TOKEN,
        (OWNER,),
        client=httpx.AsyncClient(transport=httpx.MockTransport(boom)),
        poll_timeout=0,
    )
    try:
        for _ in range(4):
            assert await ch.identify() == ""
        assert calls[0] == 1
    finally:
        await ch.close()


async def test_an_unaddressed_message_goes_to_the_paired_owner(
    db: sqlite3.Connection,
) -> None:
    """Proactivity (M3) and the delegation report both send with recipient_id=None,
    which channels/base.py documents as "goes to the configured owner". Under
    dm_policy='pairing' the static allowlist is empty and the owner lives in
    `channel_pairing`, so reading only the allowlist lost every one of them - 397 in
    the resident's log between 2026-08-14 and 2026-08-20, each swallowed by its caller.
    """
    store = Store(db)
    store.create_pairing(
        "telegram", str(OWNER), code="OWNERCOD", created_at=OWNER_AT, expires_at=OWNER_AT
    )
    store.approve_pairing("telegram", str(OWNER), approved_at=OWNER_AT)

    api = FakeAPI()
    await paired(api, store).send(OutboundMessage(text="thinking of you"))

    assert [p["chat_id"] for p in api.payloads("sendMessage")] == [OWNER]


async def test_an_unaddressed_message_prefers_a_configured_allowlist(
    db: sqlite3.Connection,
) -> None:
    """The owner lookup is a fallback, not a replacement: an install that names ids
    in the env keeps addressing exactly those, so turning this on cannot re-route an
    existing allowlist install to a single paired id."""
    store = Store(db)
    store.create_pairing(
        "telegram", str(OWNER), code="OWNERCOD", created_at=OWNER_AT, expires_at=OWNER_AT
    )
    store.approve_pairing("telegram", str(OWNER), approved_at=OWNER_AT)

    api = FakeAPI()
    await paired(api, store, allowed=(STRANGER,)).send(OutboundMessage(text="hi"))

    assert [p["chat_id"] for p in api.payloads("sendMessage")] == [STRANGER]


async def test_an_unaddressed_message_raises_when_the_owner_lookup_fails(
    db: sqlite3.Connection,
) -> None:
    """Fail closed, exactly as `_may_label` does: a storage error must not be allowed
    to invent a recipient, and the caller must still hear that nothing was sent."""
    store = Store(db)

    class Broken:
        def is_allowed(self, channel: str, sender_id: str) -> bool:
            return False

        def owner_id(self, channel: str) -> str | None:
            raise sqlite3.OperationalError("database is locked")

        def label_utterance(self, utterance_id: str, label: str, *, now: Any) -> bool:
            return False

    api = FakeAPI()
    ch = TelegramChannel(
        TOKEN,
        (),
        dm_policy="pairing",
        pairing=Pairing(store, "telegram"),
        client=api.client(),
        poll_timeout=0,
        labels=Broken(),
    )
    with pytest.raises(TelegramNoRecipient, match="no configured recipient"):
        await ch.send(OutboundMessage(text="thinking of you"))

    assert api.payloads("sendMessage") == []
