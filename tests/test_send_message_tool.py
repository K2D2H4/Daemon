"""`send_message`: the voice path's one way to put text on the owner's channel."""

import pytest

from daemon.tools.base import ToolError
from daemon.tools.message import SendMessage
from daemon.tools.schema import is_flat_schema


class FakeChannel:
    name = "telegram"

    def __init__(self, error: Exception | None = None) -> None:
        self.sent: list = []
        self._error = error

    async def send(self, message) -> None:
        if self._error is not None:
            raise self._error
        self.sent.append(message)


async def test_run_addresses_the_message_to_the_paired_owner():
    """Addressed, not unaddressed. Measured, not assumed: under `dm_policy=pairing`
    the approved owner lives in storage and `TelegramChannel._allowed` only ever
    holds the *env* list, so an unaddressed send reaches nobody - which is what
    silently dropped 397 proactive utterances in this install between 2026-08-14
    and 2026-08-20 (`no configured recipient for an unaddressed message`)."""
    channel = FakeChannel()
    tool = SendMessage(channel, recipient=lambda: "8675309")
    out = await tool.run({"text": "https://example.com/그거"})
    assert len(channel.sent) == 1
    assert channel.sent[0].text == "https://example.com/그거"
    assert channel.sent[0].recipient_id == "8675309"
    assert channel.name in out


async def test_no_paired_owner_falls_back_to_the_unaddressed_route():
    """An `allowlist`-policy install has no pairing row and does not need one: its
    ids are configured, and None is the route channels/base.py defines for them."""
    channel = FakeChannel()
    tool = SendMessage(channel, recipient=lambda: None)
    await tool.run({"text": "hi"})
    assert channel.sent[0].recipient_id is None


async def test_a_failing_recipient_lookup_degrades_the_route_it_does_not_lose_the_send():
    """Same shape as `deliver_result`: a storage error costs the address, not the
    message. If the channel then has nobody either, the send fails loudly below."""
    channel = FakeChannel()

    def broken() -> str:
        raise RuntimeError("database is locked")

    tool = SendMessage(channel, recipient=broken)
    await tool.run({"text": "hi"})
    assert channel.sent[0].recipient_id is None


async def test_run_rejects_empty_text_without_sending():
    channel = FakeChannel()
    tool = SendMessage(channel, recipient=lambda: "1")
    with pytest.raises(ToolError):
        await tool.run({"text": "   "})
    assert channel.sent == []


async def test_a_failed_send_is_a_tool_error_not_a_silent_success():
    # The whole point of the tool: the audio model confabulates "보냈어" when it
    # has no way to send. A send that failed must come back as a failure.
    tool = SendMessage(
        FakeChannel(error=RuntimeError("no configured recipient")), recipient=lambda: None
    )
    with pytest.raises(ToolError) as caught:
        await tool.run({"text": "hi"})
    assert "no configured recipient" in str(caught.value)


def test_spec_is_flat_so_the_audio_model_can_call_it():
    tool = SendMessage(FakeChannel(), recipient=lambda: None)
    assert tool.spec.name == "send_message"
    assert tool.risk == "safe"
    assert is_flat_schema(tool.spec.parameters)


def test_preview_names_this_call():
    tool = SendMessage(FakeChannel(), recipient=lambda: None)
    assert "example.com" in tool.preview({"text": "https://example.com"})
