"""Putting text on the owner's channel from inside a turn.

Telegram is a door the daemon answers, not a place it can address: the only code
that sends is the conversation loop replying to a sender, the proactive delivery,
and the delegation report. So "send me that link on Telegram" had nothing to call,
and the native-audio model answers a missing capability by confabulating the
result (evals/voice_write_nudge_spike.py). This is the tool that makes the claim
true - one string argument, so the audio model can actually fill it.

Registered for the voice surface only (`daemon/app.py`): on the text path the
reply already lands on the channel, and a send tool there would only produce a
second copy of it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from daemon.channels.base import Channel, OutboundMessage
from daemon.llm.base import ToolSpec
from daemon.tools.base import ToolError

logger = logging.getLogger(__name__)

SEND_TOOL_NAME = "send_message"


class SendMessage:
    """Implements the `Tool` protocol in daemon/tools/base.py."""

    risk = "safe"
    """Local in the sense that matters: the only recipient is the owner's own
    channel, which is where this turn's words were going anyway."""

    def __init__(
        self, channel: Channel, *, recipient: Callable[[], str | None]
    ) -> None:
        self._channel = channel
        self._recipient = recipient
        self.spec = ToolSpec(
            name=SEND_TOOL_NAME,
            description=(
                "Send text to the owner as a message on their chat channel "
                "(Telegram), so they have it in writing. Use it when they ask for "
                "a link, a name or anything else they need to keep rather than "
                "just hear. For a notification on this computer instead, use "
                "`notify`."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "The message to send. Include the actual content - a "
                            "link in full, not a description of it."
                        ),
                    }
                },
                "required": ["text"],
            },
        )

    def preview(self, arguments: Mapping[str, Any]) -> str:
        text = str(arguments.get("text", "")).strip()
        return f"send_message({text[:80]})"

    async def run(self, arguments: Mapping[str, Any]) -> str:
        text = str(arguments.get("text", "")).strip()
        if not text:
            raise ToolError("send_message needs the text to send")
        try:
            await self._channel.send(
                OutboundMessage(text=text, recipient_id=self._addressee())
            )
        except Exception as exc:
            # Converted rather than propagated, and never swallowed: the model must
            # hear that the message did not go, or it will report a send that never
            # happened. Broad because the failure is the channel's to name (an
            # unpaired owner, a Telegram API error) and this layer must not import
            # one channel's exceptions to catch them.
            raise ToolError(f"could not send the message: {exc}") from exc
        return f"sent on {self._channel.name}"

    def _addressee(self) -> str | None:
        """Who to name, or None for the channel's own unaddressed route.

        Naming the owner rather than leaving it None is not a preference: under
        `dm_policy=pairing` the approved owner lives in storage and the channel's
        allowlist holds only the configured env ids, which a paired install leaves
        empty - so an unaddressed message reaches nobody and raises. An
        `allowlist`-policy install has no pairing row and needs none; None is
        correct there. A lookup that fails degrades the route rather than losing
        the message (the shape `delegation.deliver_result` already uses): if the
        channel then has nobody either, `run` reports the failure.
        """
        try:
            return self._recipient()
        except Exception:
            logger.exception("send_message: could not look up the owner; sending unaddressed")
            return None
