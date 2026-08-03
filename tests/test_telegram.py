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
from daemon.channels.telegram import TelegramChannel, TelegramError
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

    def __init__(self, *steps: Any) -> None:
        self.steps = list(steps)
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport())

    def payloads(self, method: str) -> list[dict[str, Any]]:
        return [payload for name, payload in self.requests if name == method]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        self.requests.append((method, json.loads(request.content)))
        if method == "sendMessage":
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
        if not self.steps:
            raise StopPolling
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
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


def channel(api: FakeAPI, allowed: Any = (OWNER,)) -> TelegramChannel:
    return TelegramChannel(TOKEN, allowed, client=api.client(), poll_timeout=0)


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


async def test_an_unaddressed_message_with_nobody_approved_is_not_sent(
    db: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    """Proactivity (M3) before pairing has happened. Loud rather than silent: an
    utterance that goes nowhere is indistinguishable from one never generated."""
    api = FakeAPI()
    with caplog.at_level(logging.ERROR):
        await paired(api, Store(db)).send(OutboundMessage(text="thinking of you"))

    assert api.payloads("sendMessage") == []
    assert "no configured recipient" in caplog.text


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
