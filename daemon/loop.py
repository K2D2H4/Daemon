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

import json
import logging
from datetime import datetime

from daemon import clock, timesense
from daemon.channels.base import Channel, InboundMessage, OutboundMessage
from daemon.companion import Companion
from daemon.face import FaceBus, split_mood
from daemon.llm.base import Message, ProviderError, ToolCall
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

MAX_TOOL_ROUNDS = 25
"""Model call, tools, model call again - the last-resort ceiling before the turn is
made to answer with what it has. Set generously high on purpose: honest agentic
work - write a script, register it, test it - runs well past a handful of rounds,
and six cut real builds short mid-way (the launchd-scheduler turn that shipped this
change). The thing this actually guards against - a model re-issuing the same call
forever, spending the owner's money in a loop no one is watching - is caught far
earlier and far more precisely by LOOP_REPEAT_LIMIT below; this number is only the
net for a turn that keeps making *different*, plausible-looking calls without ever
converging."""

LOOP_REPEAT_LIMIT = 3
"""How many times the exact same call, fired back to back, counts as a stuck loop
rather than progress. Consecutive is the whole point: a productive turn that
re-runs `git status` between edits is not looping, so a total-occurrence counter
would cut it off wrongly; the identical call three rounds straight with nothing
else in between is the real pathology, and this catches it without waiting for the
generous MAX_TOOL_ROUNDS ceiling."""

ROUND_LIMIT_NOTICE = (
    "You have used every tool call available for this turn. Answer now with what you "
    "already know, and say what is still unresolved."
)

INCOMPLETE_NOTICE = (
    "I went back and forth on that one and couldn't wrap it up cleanly. Could you try "
    "again, or ask it a little differently?"
)
"""Last resort, only when a limited-out turn ran *nothing* worth reporting: the model
kept asking for tools until the cap and then, even with none on offer, returned
another (often hallucinated) tool call instead of prose. The channel refuses an
empty message, so without this the whole turn is silence the owner reads as being
ignored. When real work *did* run, `LIMIT_PROGRESS_NOTICE` goes out instead - this
bare version stranded the owner on top of a plist that had actually been written
(measured on gemini-3.6-flash: the 'build me a scheduler' turn)."""

SALVAGE_INSTRUCTION = (
    "You were working on the user's request and reached your step limit before you "
    "could finish. Do not call any tools now. Reply to the user in plain prose: say "
    "what you got done and what is still left, then offer to continue.{steps}"
)
"""The trimmed re-ask. The escape call that carries the whole tool transcript primes
the model to emit yet another tool call instead of summarising (measured: that is why
the owner got the generic notice on top of finished work). Re-asking with only the
request and a short list of what ran - no transcript, no tools - is what actually
gets prose back."""

LIMIT_PROGRESS_NOTICE = (
    "I hit my step limit before I could wrap that up. The last thing I got done was "
    "{last}. Want me to keep going from there?"
)
"""The deterministic floor when even the trimmed re-ask stays silent: built from what
actually ran, so the owner always gets how far it got and a next step rather than a
dead end. Narrating a tool this way is the one place the loop does - reserved for the
turn that produced no prose at all, where a bare apology on top of real work is worse
than the clutter."""

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


def _round_signature(tool_calls: tuple[ToolCall, ...]) -> tuple[tuple[str, str], ...]:
    """A stable fingerprint of one round's calls: name plus canonicalised arguments,
    order-independent. Two rounds asking for the very same thing share a signature,
    which is how the loop tells a stuck repeat from progress. Keys are sorted so the
    same calls in a different order still match; arguments are JSON with sorted keys
    so `{a, b}` and `{b, a}` do too."""
    return tuple(
        sorted(
            (call.name, json.dumps(call.arguments, sort_keys=True, ensure_ascii=False))
            for call in tool_calls
        )
    )


def _session_breaks_or_empty(
    history: list[LoggedMessage], moment: datetime
) -> list[tuple[int, str]]:
    """`timesense.session_breaks`, wrapped so a raise costs the break lines and not
    the turn - the same never-fail contract as `Companion.context`'s own time
    helpers (`daemon/companion.py`), applied here because this call sits in
    `_assemble`, outside `Companion`."""
    try:
        return timesense.session_breaks(history, moment)
    except Exception:
        logger.exception("timesense: could not compute session breaks")
        return []


def _call_label(call: ToolCall) -> str:
    """A one-line, human-facing tag for a call that ran: the tool plus its most telling
    argument (a path or a command). Used only to tell the owner how far a limited-out
    turn got - the full record is the `tool_calls` audit."""
    arg = call.arguments.get("path") or call.arguments.get("command") or ""
    arg = str(arg).strip().splitlines()[0][:80] if arg else ""
    return f"`{call.name}` {arg}".strip()


class ConversationLoop:
    def __init__(
        self,
        channel: Channel,
        gateway: LLMGateway,
        companion: Companion,
        *,
        context_turns: int = 20,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
        face: FaceBus | None = None,
    ) -> None:
        self._channel = channel
        self._gateway = gateway
        self._companion = companion
        self._context_turns = context_turns
        self._max_tool_rounds = max_tool_rounds
        # None is a complete no-op (daemon/face.py) - a text-only install pays
        # nothing for this module existing.
        self._face = face

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
        if self._face is not None:
            # Set before the early-return branches below too: whichever way this
            # turn exits, the `finally` is what brings the face back, so it does
            # not matter here that a duplicate or a `/approve` skips straight
            # past the reply.
            self._face.set_activity("thinking")
        try:
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

            # The strip happens here, before either of the two ways this text can
            # escape (`record` below, and `_approve`'s own record/say below it):
            # both write to the markdown log recall replays into later prompts, or
            # put a tag on the wire. Either would be read back to the model next
            # turn as something it said, laundering an instruction into the
            # personality that `data/persona/seed.md` being human-owned exists to
            # prevent (daemon/face.py: split_mood).
            text = self._speak(text)

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
        finally:
            if self._face is not None:
                # Without this, one failed turn leaves the face stuck on
                # "thinking" for the rest of the process's life.
                self._face.set_activity("idle")

    def _speak(self, text: str) -> str:
        """Strip a leading mood tag off a reply and, if a face is attached,
        publish it before the text is used anywhere.

        Both places a reply can leave this object - the ordinary turn's
        `record`/`send` in `handle`, and the approval-resume turn's own
        `record`/`say` in `_approve` - go through here first, so there is one
        strip point rather than two copies of the same three lines to keep in
        sync (and one place to trust that the tag never reaches the log or the
        wire either way).
        """
        text, mood = split_mood(text)
        if self._face is not None:
            if mood is not None:
                # Before the activity flips to speaking, not after: a face
                # moves just ahead of the words it is reacting to.
                self._face.one_shot(mood)
            self._face.set_activity("speaking")
        return text

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

        # Captured before the loop mutates `messages`: the salvage re-ask needs the
        # owner's own words, and by the end the list is mostly tool turns.
        user_request = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )

        outcome = Outcome()
        did: list[str] = []
        completion = await self._gateway.complete(Task.CHAT_TEXT, messages, tools=specs)

        rounds = 0
        repeats = 0
        last_signature: tuple[tuple[str, str], ...] | None = None
        while completion.tool_calls:
            signature = _round_signature(completion.tool_calls)
            repeats = repeats + 1 if signature == last_signature else 1
            last_signature = signature

            capped = rounds >= self._max_tool_rounds
            stalled = repeats >= LOOP_REPEAT_LIMIT
            if capped or stalled:
                # Either the last-resort ceiling, or the same call fired back to back
                # with no progress. Both end the same way: answer with what we have.
                reason = (
                    f"tool round limit ({self._max_tool_rounds}) reached"
                    if capped
                    else f"the same tool call repeated {repeats} rounds with no progress"
                )
                logger.warning("%s; making the turn answer", reason)
                # Asked again with no tools on offer, so the answer cannot be
                # another tool call. Breaking without this would reply with the
                # empty text that came back alongside the calls.
                try:
                    completion = await self._gateway.complete(
                        Task.CHAT_TEXT,
                        [*messages, Message(role="system", content=ROUND_LIMIT_NOTICE)],
                    )
                except ProviderError:
                    # A reasoning model can spend this whole call's output budget on
                    # reasoning tokens and return neither text nor a tool call - the
                    # provider contract (llm/base.py) requires raising for exactly
                    # that emptiness rather than returning it. The `not
                    # completion.text.strip()` check below already forgives that
                    # emptiness when it comes back as an empty completion; this call
                    # is one place it can arrive as a raise instead, because it is
                    # made with no tools on offer. A first-call or mid-loop
                    # ProviderError is a genuine failure on an ordinary turn and must
                    # still reach run()'s FAILURE_NOTICE - only this last call, past
                    # the cap and just trying to summarise, degrades to the salvage.
                    logger.warning(
                        "%s, and the escape call raised; salvaging a progress note",
                        reason,
                    )
                    return await self._finish_without_prose(user_request, did), outcome
                break
            rounds += 1

            round_outcome = await self._companion.run_tools(
                completion.tool_calls, origin=origin, channel=channel, sender_id=sender_id
            )
            outcome.approvals.extend(round_outcome.approvals)
            # What actually ran, for the salvage note if this turn never gives prose.
            # Only calls that succeeded: "how far I got" should not list a failure.
            by_id = {call.id: call for call in completion.tool_calls}
            did.extend(
                _call_label(by_id[result.call_id])
                for result in round_outcome.results
                if result.ok and result.call_id in by_id
            )

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
            # channel refuses an empty message, so returning "" is silence; salvage a
            # reply from what ran instead.
            logger.warning(
                "turn produced no answer text after tools; salvaging a progress note"
            )
            return await self._finish_without_prose(user_request, did), outcome
        return completion.text, outcome

    async def _finish_without_prose(self, user_request: str, did: list[str]) -> str:
        """A reply for a turn that ran tools but never gave prose.

        First a trimmed re-ask - just the request and a short list of what ran, no
        tool transcript and no tools on offer - because the transcript is what primes
        the model to emit yet another tool call instead of summarising. If even that
        stays silent, a deterministic note built from what ran, so the owner always
        gets how far it got and a next step rather than a dead end.
        """
        steps = "; ".join(did[-8:])
        instruction = SALVAGE_INSTRUCTION.format(
            steps=f" Steps you completed: {steps}." if steps else ""
        )
        try:
            completion = await self._gateway.complete(
                Task.CHAT_TEXT,
                [
                    Message(role="system", content=instruction),
                    Message(role="user", content=user_request),
                ],
            )
        except ProviderError:
            # The re-ask is best-effort: a provider that fails it still owes the owner
            # the deterministic note below, not run()'s generic failure.
            completion = None
        if completion is not None and completion.text.strip():
            return completion.text
        if did:
            return LIMIT_PROGRESS_NOTICE.format(last=did[-1])
        return INCOMPLETE_NOTICE

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
        # A genuine model-generated reply, same as the ordinary turn's - so it
        # goes through the same strip point before either `record` or `say`.
        text = self._speak(completion.text)

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

        The clock is read exactly once, here, and threaded into both
        `Companion.context` and `timesense.session_breaks` below. Two separate reads
        used to straddle a minute - and, near local midnight, a date - so one prompt
        named today in `[현재 시각]` and tomorrow in `[대화 단절]`.
        """
        history = await self._companion.recent(limit=self._context_turns)
        moment = clock.now()
        blocks = await self._companion.context(
            inbound.text,
            history=history,
            already={item.content for item in history},
            origin=origin,
            now=moment,
        )
        messages = [Message(role="system", content=block) for block in blocks]
        turns = [Message(role=item.role, content=item.content) for item in history]
        # Descending, so an earlier insertion does not shift a later index.
        for index, line in reversed(_session_breaks_or_empty(history, moment)):
            turns.insert(index, Message(role="system", content=line))
        messages.extend(turns)

        # The user turn above was recorded first, so whether it is already in
        # `recent()` depends on the writer's mirror timing. Append only if absent
        # so the model never sees the same words twice.
        if not history or history[-1].content != inbound.text:
            messages.append(Message(role="user", content=inbound.text))
        return messages


