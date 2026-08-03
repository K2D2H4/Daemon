"""Telegram channel tests. No network: httpx.MockTransport stands in for the Bot API.

The allowlist test is the one that matters most - anyone can message a Telegram bot.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest

from daemon.channels import telegram
from daemon.channels.base import Channel, OutboundMessage
from daemon.channels.telegram import TelegramChannel, TelegramError

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
