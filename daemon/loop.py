"""The conversation loop: inbound message in, recorded exchange and reply out.

M1a is deliberately the whole of it - no recall, no reflection, no proactivity
(docs/PLAN.md 8.2). The point of shipping it first is to start the log clock,
because persona evolution later needs weeks of real accumulated conversation
and that wall-clock time cannot be compressed (docs/PLAN.md 8.1).

Dependencies arrive through the constructor as protocols only: this module must
not know that the channel is Telegram or that memory is markdown.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from daemon import clock
from daemon.channels.base import Channel, InboundMessage, OutboundMessage
from daemon.llm.base import Message
from daemon.llm.gateway import LLMGateway
from daemon.memory.base import LoggedMessage, MemoryWriter
from daemon.tasks import Task

logger = logging.getLogger(__name__)

FAILURE_NOTICE = "Something went wrong on my side, so I could not answer that one."
"""Said to the user when a turn fails. Silence would read as being ignored,
which is worse than an admission."""


class ConversationLoop:
    def __init__(
        self,
        channel: Channel,
        gateway: LLMGateway,
        memory: MemoryWriter,
        *,
        data_dir: Path,
        context_turns: int = 20,
    ) -> None:
        self._channel = channel
        self._gateway = gateway
        self._memory = memory
        self._seed_path = Path(data_dir) / "persona" / "seed.md"
        self._context_turns = context_turns

    async def run(self) -> None:
        """Consume the channel forever. One bad turn must not end the loop."""
        async for inbound in self._channel.listen():
            try:
                await self.handle(inbound)
            except Exception:
                logger.exception("turn failed sender=%s", inbound.sender_id)
                try:
                    await self._channel.send(
                        OutboundMessage(text=FAILURE_NOTICE, recipient_id=inbound.sender_id)
                    )
                except Exception:
                    # The channel itself is down; nothing left to do but record it.
                    logger.exception("could not deliver the failure notice")

    async def handle(self, inbound: InboundMessage) -> None:
        if inbound.external_id is not None and await self._memory.seen(
            inbound.channel, inbound.external_id
        ):
            # A restart can land between handling a message and the channel
            # confirming it, so the same message arrives twice. The markdown is
            # append-only, so the duplicate has to be refused here rather than
            # reconciled later - and answering twice is its own annoyance.
            logger.info(
                "skipping already-handled message channel=%s id=%s",
                inbound.channel,
                inbound.external_id,
            )
            return

        session_kind = "voice" if inbound.modality == "voice" else "interactive"

        # Recorded before the model is called: if the process dies mid-call the
        # user's words are already on disk. Markdown is the source of truth.
        await self._memory.record(
            LoggedMessage(
                ts=inbound.received_at,
                role="user",
                content=inbound.text,
                # An allowlisted sender relaying someone else's words - a
                # forward, an inline-bot result - is not the owner speaking.
                # Recording it as 'owner' would let injected text reach the
                # curated tier and, through reflection, persona rules.
                origin="owner" if inbound.authored_by_sender else "untrusted",
                session_kind=session_kind,
                modality=inbound.modality,
                channel=inbound.channel,
                sender_id=inbound.sender_id,
                external_id=inbound.external_id,
            )
        )

        messages = await self._assemble(inbound)
        # M1a routes every turn as text; voice is a later milestone and needs a
        # native-audio provider rather than this text path (docs/PLAN.md 6.5).
        completion = await self._gateway.complete(Task.CHAT_TEXT, messages)

        await self._memory.record(
            LoggedMessage(
                ts=clock.now(),
                role="assistant",
                content=completion.text,
                origin="agent",
                session_kind=session_kind,
                modality=inbound.modality,
                channel=inbound.channel,
            )
        )
        await self._channel.send(
            OutboundMessage(text=completion.text, recipient_id=inbound.sender_id)
        )

    async def _assemble(self, inbound: InboundMessage) -> list[Message]:
        """Persona seed as the system turn, then the recent window.

        M1b replaces `recent()` with real recall; M4 replaces the seed read with
        persona/loader.py assembling seed.md + learned.md.
        """
        messages: list[Message] = []
        seed = await self._read_seed()
        if seed:
            messages.append(Message(role="system", content=seed))

        history = await self._memory.recent(limit=self._context_turns)
        messages.extend(Message(role=item.role, content=item.content) for item in history)

        # The user turn above was recorded first, so whether it is already in
        # `recent()` depends on the writer's mirror timing. Append only if absent
        # so the model never sees the same words twice.
        if not history or history[-1].content != inbound.text:
            messages.append(Message(role="user", content=inbound.text))
        return messages

    async def _read_seed(self) -> str:
        """Read every turn, not once at startup: seed.md is human-owned and an
        edit should take effect without a restart (docs/PLAN.md 5.1)."""
        try:
            return (await asyncio.to_thread(self._seed_path.read_text, encoding="utf-8")).strip()
        except FileNotFoundError:
            return ""
        except OSError:
            logger.exception("could not read %s", self._seed_path)
            return ""
