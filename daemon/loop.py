"""The text conversation: inbound message in, recorded exchange and reply out.

What the daemon *can* do lives in `daemon/companion.py` - the persona, the tool
rules, recall, and writing an exchange down. This file is the text transport: it
assembles a `list[Message]` per turn, runs the tool loop, handles `/approve`, and
sends. The voice endpoint carries the same companion over a stream instead
(`daemon/voice/conversation.py`), and the two are deliberately not one pipeline;
`daemon/companion.py` says why.

Recall and tools are both optional, and that is deliberate - but the loop no
longer sees that: they are injected into the `Companion`, and a half-finished
layer (an embedder that will not load, a tool policy still being written) degrades
inside it to exactly the behaviour that already works instead of taking the log
clock down with it (docs/PLAN.md 8.1). What arrives here is the channel, the
gateway and that companion; this module still must not know that the channel is
Telegram, that memory is markdown, or which of recall and tools the companion has.
"""

from __future__ import annotations

import logging

from daemon import clock
from daemon.channels.base import Channel, InboundMessage, OutboundMessage
from daemon.companion import Companion
from daemon.llm.base import Message
from daemon.llm.gateway import LLMGateway
from daemon.memory.base import LoggedMessage
from daemon.tasks import Task
from daemon.tools.policy import Command, parse_command
from daemon.tools.runner import Outcome
from daemon.tools.screen import screen_note

logger = logging.getLogger(__name__)

FAILURE_NOTICE = "Something went wrong on my side, so I could not answer that one."
"""Said to the user when a turn fails. Silence would read as being ignored,
which is worse than an admission."""

MAX_TOOL_ROUNDS = 6
"""Model call, tools, model call again - how many times round before the turn is
made to answer with what it has. A bound rather than a target: without one, a
model that keeps re-reading the same file spends the user's money in a loop no one
is watching."""

ROUND_LIMIT_NOTICE = (
    "You have used every tool call available for this turn. Answer now with what you "
    "already know, and say what is still unresolved."
)

INCOMPLETE_NOTICE = (
    "I went back and forth on that one and couldn't wrap it up cleanly. Could you try "
    "again, or ask it a little differently?"
)
"""Sent when a turn ends with no answer text at all - the model kept asking for
tools until the round cap and then, even with none on offer, returned another
(often hallucinated) tool call instead of prose. The channel refuses an empty
message, so without this the whole turn is silence the owner reads as being
ignored, and they have to poke it again to get anything (measured on
gemini-3.6-flash: a 'next week's weather' turn that no tool could satisfy)."""

APPROVAL_REQUEST = (
    "That needs your say-so:\n\n{preview}\n\n"
    "Reply `/approve {code}` to let it run once, `/approve {code} always` to stop "
    "asking about exactly this, or `/deny {code}`. It lapses in {minutes} minutes."
)

APPROVAL_UNKNOWN = (
    "That is not a code I am waiting on - it may have lapsed, or already been used."
)
APPROVAL_NO_CODE = (
    "Add the code from the request, like `/approve A3F2K9QT` (or `/deny A3F2K9QT`). "
    "`/approve` on its own does not say which pending request you mean."
)
APPROVAL_DENIED = "Understood, I have not run it."
APPROVAL_NOT_OWNER = (
    "Approvals only count when they come from you directly, not forwarded on."
)


class ConversationLoop:
    def __init__(
        self,
        channel: Channel,
        gateway: LLMGateway,
        companion: Companion,
        *,
        context_turns: int = 20,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self._channel = channel
        self._gateway = gateway
        self._companion = companion
        self._context_turns = context_turns
        self._max_tool_rounds = max_tool_rounds

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
        if inbound.external_id is not None and await self._companion.seen(
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

        origin = "owner" if inbound.authored_by_sender else "untrusted"

        if self._companion.has_tools:
            command = parse_command(inbound.text)
            if command is not None:
                # Control plane, not conversation: deliberately handled before the
                # markdown write, so `/approve A3F2K9QT` does not become a memory
                # that recall surfaces next week. What it authorised is recorded in
                # `tool_calls` instead, where it belongs. A replay after a restart
                # is harmless because spending a code is single-use (memory/store.py).
                await self._approve(inbound, command, origin=origin)
                return

        session_kind = "voice" if inbound.modality == "voice" else "interactive"

        # Recorded before the model is called: if the process dies mid-call the
        # user's words are already on disk. Markdown is the source of truth.
        await self._companion.record(
            LoggedMessage(
                ts=inbound.received_at,
                role="user",
                content=inbound.text,
                # An allowlisted sender relaying someone else's words - a
                # forward, an inline-bot result - is not the owner speaking.
                # Recording it as 'owner' would let injected text reach the
                # curated tier and, through reflection, persona rules.
                origin=origin,
                session_kind=session_kind,
                modality=inbound.modality,
                channel=inbound.channel,
                sender_id=inbound.sender_id,
                external_id=inbound.external_id,
            )
        )

        messages = await self._assemble(inbound, origin=origin)
        # M1a routes every turn as text; voice is a later milestone and needs a
        # native-audio provider rather than this text path (docs/PLAN.md 6.5).
        text, outcome = await self._answer(
            messages, origin=origin, channel=inbound.channel, sender_id=inbound.sender_id
        )

        await self._companion.record(
            LoggedMessage(
                ts=clock.now(),
                role="assistant",
                content=text,
                origin="agent",
                session_kind=session_kind,
                modality=inbound.modality,
                channel=inbound.channel,
            )
        )

        await self._channel.send(OutboundMessage(text=text, recipient_id=inbound.sender_id))
        await self._ask_approvals(outcome, inbound.sender_id)
        # After the reply, never before: embedding costs a round trip to the local
        # model, and the vector index is regenerable from the markdown while the
        # user's wait is not recoverable. Both turns are indexed, because a vector
        # index holding only one side of the conversation makes "what did you
        # suggest?" unanswerable while looking like it works.
        await self._companion.index_recorded()

    async def _answer(
        self, messages: list[Message], *, origin: str, channel: str, sender_id: str | None
    ) -> tuple[str, Outcome]:
        """The model's reply, running whatever tools it asks for on the way.

        With nothing on offer this is one call and nothing else, which is exactly the
        behaviour before tools existed. `Companion.specs` decides that, and a turn
        that is not the owner's own words is one of the cases where it says nothing.
        """
        specs = self._companion.specs(origin=origin)
        if not specs:
            completion = await self._gateway.complete(Task.CHAT_TEXT, messages)
            return completion.text, Outcome()

        outcome = Outcome()
        completion = await self._gateway.complete(Task.CHAT_TEXT, messages, tools=specs)

        rounds = 0
        while completion.tool_calls:
            if rounds >= self._max_tool_rounds:
                logger.warning(
                    "tool round limit (%d) reached; making the turn answer",
                    self._max_tool_rounds,
                )
                # Asked again with no tools on offer, so the answer cannot be
                # another tool call. Breaking without this would reply with the
                # empty text that came back alongside the calls.
                completion = await self._gateway.complete(
                    Task.CHAT_TEXT,
                    [*messages, Message(role="system", content=ROUND_LIMIT_NOTICE)],
                )
                break
            rounds += 1

            round_outcome = await self._companion.run_tools(
                completion.tool_calls, origin=origin, channel=channel, sender_id=sender_id
            )
            outcome.approvals.extend(round_outcome.approvals)

            messages.append(
                Message(
                    role="assistant",
                    content=completion.text,
                    tool_calls=completion.tool_calls,
                )
            )
            messages.extend(
                Message(role="tool", content=result.content, tool_call_id=result.call_id)
                for result in round_outcome.results
            )
            # A captured image rides on its own plain `user` turn, never inside the
            # `tool`-role message above: a user turn holding an image is the one
            # shape all four providers accept (Task 1.2-1.5), where a `tool`-role
            # image is not. The note is the untrusted-data framing (security stance
            # A) - the same fencing spirit as recall/browser, adapted because
            # pixels cannot be nonce-fenced the way text can (`screen.screen_note`).
            messages.extend(
                Message(role="user", content=screen_note("the screen"), images=result.images)
                for result in round_outcome.results
                if result.images
            )
            completion = await self._gateway.complete(Task.CHAT_TEXT, messages, tools=specs)

        # Just the model's answer. What actually ran is not folded into the reply -
        # a companion that narrates every `run`/`write`/`rm` reads as clutter, and the
        # owner's ground-truth record lives in the `tool_calls` audit (`daemon tools
        # log`) rather than in a line the model's prose sits on top of.
        if not completion.text.strip():
            # No answer text - the model spent the round cap on tool calls and then
            # returned another one instead of prose, even with no tools offered. The
            # channel refuses an empty message, so returning "" is silence; say
            # something the owner can act on instead.
            logger.warning(
                "turn produced no answer text after tools; sending the incomplete notice"
            )
            return INCOMPLETE_NOTICE, outcome
        return completion.text, outcome

    async def _ask_approvals(self, outcome: Outcome, recipient_id: str | None) -> None:
        """Send one approval request per parked call, after the reply.

        Separate messages rather than appended to the reply: the code has to be
        copied, and burying it in a paragraph is how a person ends up sending the
        wrong one.
        """
        for approval in outcome.approvals:
            minutes = max(1, round((approval.expires_at - clock.now()).total_seconds() / 60))
            await self._channel.send(
                OutboundMessage(
                    text=APPROVAL_REQUEST.format(
                        preview=approval.preview, code=approval.code, minutes=minutes
                    ),
                    recipient_id=recipient_id,
                )
            )

    async def _approve(
        self, inbound: InboundMessage, command: Command, *, origin: str
    ) -> None:
        """Handle `/approve` or `/deny`, then answer with what came of it."""

        async def say(text: str) -> None:
            await self._channel.send(
                OutboundMessage(text=text, recipient_id=inbound.sender_id)
            )

        if origin != "owner":
            # A forwarded `/approve CODE` is someone else's instruction wearing the
            # owner's account. The code is not even looked up, so a guess costs
            # nothing and reveals nothing.
            logger.warning("refusing a relayed approval from sender=%s", inbound.sender_id)
            await say(APPROVAL_NOT_OWNER)
            return

        if not command.code:
            # A bare `/approve` with no code. Answered here, in the control plane,
            # rather than handed to the model: the model would treat it as ordinary
            # conversation and re-issue the guarded call, minting a fresh code and
            # asking again - the loop this whole branch exists to close. No claim, no
            # model call, and the pending code stays live for a real `/approve CODE`.
            await say(APPROVAL_NO_CODE)
            return

        claimed = self._companion.claim(command, sender_id=inbound.sender_id)
        if claimed is None:
            await say(APPROVAL_UNKNOWN)
            return
        if claimed.denied:
            await say(APPROVAL_DENIED)
            return

        result = await self._companion.resume(
            claimed, origin=origin, channel=inbound.channel, sender_id=inbound.sender_id
        )

        # One call, with no tools offered: the approval authorised this and nothing
        # further, so the turn ends in an answer rather than in another request.
        messages = await self._assemble_after_tool(
            claimed.preview, result.content, said=inbound.text
        )
        completion = await self._gateway.complete(Task.CHAT_TEXT, messages)
        text = completion.text

        await self._companion.record(
            LoggedMessage(
                ts=clock.now(),
                role="assistant",
                content=text,
                origin="agent",
                session_kind="interactive",
                modality=inbound.modality,
                channel=inbound.channel,
            )
        )
        await say(text)
        # Indexed like any other reply. Recording without indexing left this one turn
        # with no vector until the next restart's backfill, and it is the turn most
        # likely to be asked about later ("what did you change in my todo?") - the
        # same "looks like it works" gap `handle` argues against above.
        await self._companion.index_recorded()

    async def _assemble_after_tool(
        self, preview: str, output: str, *, said: str
    ) -> list[Message]:
        """Context for the turn that resumes after an approval.

        The tool result arrives as a system note rather than as a replayed
        `tool` turn: the original request was made in an earlier turn that is no
        longer in hand, and reconstructing a matching tool-call transcript would
        mean persisting one. The model has the conversation and the outcome, which
        is what it needs to say something useful about it.

        It ends on the owner's `/approve` as a user turn, and that is load-bearing
        rather than tidy. The system note is hoisted into a top-level field by three
        of the four providers, so without a real user turn the list would end on an
        *assistant* turn - which Anthropic reads as a prefill to continue, making the
        answer a continuation of "I have asked you to approve that" instead of a new
        sentence. The approval is also genuinely the last thing the owner said; it is
        left out of the markdown log, not out of the conversation.
        """
        messages: list[Message] = []
        seed = await self._companion.persona()
        if seed:
            messages.append(Message(role="system", content=seed))
        history = await self._companion.recent(limit=self._context_turns)
        messages.extend(Message(role=item.role, content=item.content) for item in history)
        messages.append(
            Message(
                role="system",
                content=(
                    f"The owner approved `{preview}` and it has now run. Result:\n\n{output}\n\n"
                    "Tell them what came of it, in your own voice. Do not mention the "
                    "approval code."
                ),
            )
        )
        messages.append(Message(role="user", content=said))
        return messages

    async def _assemble(self, inbound: InboundMessage, *, origin: str) -> list[Message]:
        """Persona and recalled memory, then the recent window.

        Recall and the recent window are both here and do different jobs: the
        window is the thread being spoken right now, recall is everything older
        that turned out to matter. Which blocks there are, and in what order, is
        `Companion.context`; this decides that each becomes a system turn ahead of
        the conversation.
        """
        history = await self._companion.recent(limit=self._context_turns)
        blocks = await self._companion.context(
            inbound.text,
            already={item.content for item in history},
            origin=origin,
        )
        messages = [Message(role="system", content=block) for block in blocks]
        messages.extend(Message(role=item.role, content=item.content) for item in history)

        # The user turn above was recorded first, so whether it is already in
        # `recent()` depends on the writer's mirror timing. Append only if absent
        # so the model never sees the same words twice.
        if not history or history[-1].content != inbound.text:
            messages.append(Message(role="user", content=inbound.text))
        return messages


