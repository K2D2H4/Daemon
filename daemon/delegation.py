"""Running an owner's delegated request in the background, and reporting it back.

A voice turn hands work off through `delegate_task`; this is where that work is
actually run - through the same text `ConversationLoop` the Telegram path uses, so
a nested-schema tool the voice model could not call is called here where it can be -
and reported by presence. See docs/superpowers/specs/2026-08-14-async-delegation-design.md.
"""

from __future__ import annotations

import logging
from typing import Any

from daemon.channels.base import OutboundMessage

logger = logging.getLogger(__name__)


async def deliver_result(
    text: str, *, presence: Any, speaker: Any, channel: Any, recipient_id: str | None
) -> str:
    """Route a finished result to the owner by presence. Never raises.

    At the keyboard: speak it and send it. Away: send it. A speak or send that
    fails degrades the route rather than losing the result - the reply already
    happened, and raising here would strand it.
    """
    at_keyboard = False
    if presence is not None:
        try:
            reading = await presence.read()
            at_keyboard = bool(reading.at_keyboard)
        except Exception:
            logger.exception("delegation: presence read failed; treating as away")

    spoke = False
    if at_keyboard and speaker is not None:
        try:
            spoke = await speaker.say(text)
        except Exception:
            logger.exception("delegation: could not speak the result")

    sent = False
    if channel is not None:
        try:
            await channel.send(OutboundMessage(text=text, recipient_id=recipient_id))
            sent = True
        except Exception:
            logger.exception("delegation: could not send the result to the channel")

    if spoke and sent:
        return "both"
    if sent:
        return "telegram"
    if spoke:
        return "local_speaker"
    logger.warning("delegation: result reached nobody: %s", text[:80])
    return "none"
