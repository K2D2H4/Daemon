"""One voice conversation: microphone in, speaker out, transcripts recorded.

The text path is daemon/loop.py and this is shaped after it on purpose -
dependencies arrive as protocols only, what the user said is written before
anything else can lose it, and recall never fails a turn. What differs is that in
voice mode the audio model *is* the brain (docs/PLAN.md 6.5): there is no gateway
call and no assembled prompt, so the session is the turn.

Four things here are load-bearing rather than incidental:

1. **Only `final=True` transcripts are recorded.** Gemini streams transcription
   as deltas, so recording anything else would leave single syllables in the log
   as though they were utterances (daemon/voice/base.py).
2. **Recall is prefetched from partial transcripts, and delivered from the same
   place.** An embedder round trip is 117 ms at p50, ~105 ms of it fixed overhead
   (docs/PLAN.md 4.3.1) - a cost that is unaffordable after the user stops talking
   and free while they are still talking. So the search starts on the partial text,
   the final transcript reuses that result when it covers enough of what was
   actually said, and what came back goes to `send_context` while the user is still
   mid-sentence: the final transcript only lands after the model has answered, so
   anything sent then is context for the next turn rather than this one.
3. **A barge-in does two things or it does nothing.** `session.interrupt()` stops
   the abandoned turn's audio from arriving, `audio.stop_playback()` drops what is
   already queued. Either one alone leaves the daemon talking over the user.
4. **One `receive()` is one turn, so a conversation is a loop.** `receive()` ends at
   the turn boundary (daemon/voice/base.py). The single call this replaced delivered
   the first answer and then blocked until the server cut the idle session with
   1008 - measured, twice.

The session is opened for one conversation and closed after it, with an idle
timeout instead of a socket held open between exchanges: Live sessions bill per
minute, so an idle connection is pure cost (docs/PLAN.md 6.5).

Not here: choosing between the local speaker and a channel by presence
(docs/PLAN.md 6.3). That is a routing decision for *proactive* speech and belongs
to M3; this module is the conversation the user starts.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
from collections.abc import AsyncIterator

from daemon import clock

# The recall block comes from the text loop, privates included, rather than being
# rendered again here. It is not formatting: the nonce boundary is what stops a
# recalled memory posing as an instruction, and `_label` is what stops relayed text
# posing as the owner's own words - a fix an earlier audit made in the loop and
# then found undone one layer up. A second copy is a second place for that to
# happen, and voice is the weaker position to start from: the wire has no role that
# means "reference material", so the block arrives as a *user* turn.
from daemon.loop import render_recall
from daemon.memory.base import LoggedMessage, MemoryWriter, Recall, RecalledItem
from daemon.voice.base import AudioIO, Transcript, VoiceSession

logger = logging.getLogger(__name__)

_IDENTITY_NONCE = "same-memories"
"""Nonce used only to compare two recall payloads for equality, never sent."""

VOICE_CHANNEL = "voice"
"""`channel` for a recorded voice turn. Not the provider's name: the column says
how the words reached us, and a session that moves from Gemini to OpenAI Realtime
is the same channel (docs/PLAN.md 6.5 keeps two providers behind one seam)."""

IDLE_TIMEOUT_SECONDS = 30.0
"""How long a session may hear nothing before it is closed. Billing is per
minute, so silence is spending."""

EMPTY_TURNS_ALLOWED = 1
"""How many turns in a row may yield nothing before the session is treated as
over.

One, because one is expected: a turn that ends on `generationComplete` leaves the
`turnComplete` behind it to be read as an empty turn. Two in a row means
`receive()` is returning without ever awaiting anything, and looping on that would
spin the event loop hard enough that even the idle timeout could not fire."""

PREFETCH_MIN_CHARS = 4
"""Below this a query is mostly noise, and one embedder call per syllable buys
nothing."""

REUSE_COVERAGE = 0.7
"""How much of the final utterance a prefetched query must cover to be reused.

The tail of a sentence is where Korean puts the question, so a prefetch made from
the first third of an utterance is a different query and gets thrown away;
covering most of it embeds to nearly the same vector and is worth the whole 117 ms
it saves."""


class VoiceConversation:
    """One live voice exchange, driven by injected protocols.

    Nothing concrete is imported: the session is a `VoiceSession`, the hardware is
    an `AudioIO`, and both are faked in tests so the suite needs no key, no
    network and no microphone.
    """

    def __init__(
        self,
        session: VoiceSession,
        audio: AudioIO,
        memory: MemoryWriter,
        *,
        recall: Recall | None = None,
        recall_limit: int = 6,
        channel: str = VOICE_CHANNEL,
        idle_timeout: float = IDLE_TIMEOUT_SECONDS,
    ) -> None:
        self._session = session
        self._audio = audio
        self._memory = memory
        self._recall = recall
        self._recall_limit = recall_limit
        self._channel = channel
        self._idle_timeout = idle_timeout

        self.recalled: list[RecalledItem] = []
        """What recall had ready for the last completed utterance.

        Sent as well as exposed, now that there is a seam for it:
        `session.send_context` puts it in the model's history without asking for an
        answer, which `send_text` cannot do - that is a prompt, and a memory
        delivered through it makes the daemon narrate an old conversation the user
        never asked about. Still public because it is what a caller inspects to see
        what the answer was allowed to draw on."""

        self.ended: str | None = None
        """Why the conversation finished, for a caller that has to tell "the turn
        is over" from "the session is gone" (a `goAway` looks like the former and
        is the latter)."""

        self.interruptions = 0
        self._playing = False
        self._speculative: tuple[str, asyncio.Task[list[RecalledItem]]] | None = None
        self._offered: str | None = None
        """The last block put in front of the model, so a prefetch that is reused
        rather than redone does not seed the same memories twice."""

    async def run(self) -> None:
        """Hold one conversation, then close the session.

        Returns when the session ends, when nothing has been heard for
        `idle_timeout`, or when the caller cancels. Raises whatever the session
        raises - a voice turn that cannot run should surface so the caller can
        fall back to text.
        """
        async with self._session as session:
            microphone = self._audio.record()
            # The microphone keeps feeding the session while the model talks. It
            # has to: server-side activity detection is what notices a barge-in,
            # and it can only notice it in audio we actually send.
            pump = asyncio.create_task(
                self._forward_microphone(session, microphone), name="voice-microphone"
            )
            watch = asyncio.create_task(self._watch_partials(session), name="voice-prefetch")
            try:
                await self._receive(session)
            finally:
                for task in (pump, watch):
                    task.cancel()
                await asyncio.gather(pump, watch, return_exceptions=True)
                await _aclose(microphone)
                await self._cancel_speculative()
                # Reached on cancellation too, which is the point: the transcript
                # is the only record voice mode produces, and a shutdown arriving
                # before `turnComplete` would otherwise drop the utterance from
                # the markdown and the mirror both. Shielded so a second
                # cancellation - a shutdown that does not wait - cannot lose it
                # either.
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(self._record_pending(session))

    # --- the two streams ----------------------------------------------------

    async def _receive(self, session: VoiceSession) -> None:
        """Take turns until the session ends or the silence does.

        A loop because `receive()` is one turn: it ends at the turn boundary
        (daemon/voice/base.py), so a conversation is as many calls as there were
        turns. The single call it replaces is the measured defect - it delivered the
        first answer and then blocked until the server cut the idle session.

        The idle budget spans the whole conversation rather than each turn: what
        bills per minute is the session, and the gap between turns is exactly where
        it is spent on nothing.
        """
        empty = 0
        try:
            async with asyncio.timeout(self._idle_timeout) as budget:
                while self.ended is None:
                    if await self._one_turn(session, budget):
                        empty = 0
                        continue
                    empty += 1
                    if empty > EMPTY_TURNS_ALLOWED and self.ended is None:
                        self.ended = "the session stopped producing turns"
                        logger.info("voice: %s", self.ended)
        except TimeoutError:
            self.ended = f"nothing heard for {self._idle_timeout:.0f}s"
            logger.info("voice: %s; closing a session that bills per minute", self.ended)
        finally:
            if self.ended is None:
                self.ended = "the session stream ended"

    async def _one_turn(self, session: VoiceSession, budget: asyncio.Timeout) -> bool:
        """One `receive()`. True if anything came out of it.

        Whether the *session* also ended is a separate question and only the
        provider can answer it - a `goAway` arrives before the session limit and
        then the stream simply stops, which reads exactly like a turn that
        finished. `getattr` because that answer is not in the protocol: the three
        methods this module calls are, `ended` is not.
        """
        loop = asyncio.get_running_loop()
        produced = False
        stream = session.receive()
        try:
            async for item in stream:
                produced = True
                budget.reschedule(loop.time() + self._idle_timeout)
                if isinstance(item, bytes):
                    self._playing = True
                    await self._audio.play(item)
                else:
                    await self._on_transcript(session, item)
        finally:
            await _aclose(stream)
        self.ended = getattr(session, "ended", None)
        return produced

    async def _forward_microphone(
        self, session: VoiceSession, microphone: AsyncIterator[bytes]
    ) -> None:
        try:
            async for chunk in microphone:
                await session.send_audio(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Either the socket went or the microphone did, and the receive side
            # is already reporting which. Logged rather than raised so this task
            # dying does not replace the error the caller needs to see.
            logger.exception("voice: stopped feeding the session")

    # --- transcripts --------------------------------------------------------

    async def _on_transcript(self, session: VoiceSession, transcript: Transcript) -> None:
        if not transcript.final:
            # base.py: Gemini never yields these, and one recorded as an utterance
            # leaves a syllable in the log forever. Nothing partial gets past here.
            logger.debug("voice: ignoring a partial %s transcript", transcript.role)
            return
        # Both roles are flushed at the turn boundary, so a final transcript is
        # also the signal that this turn's audio is done.
        self._playing = False
        if transcript.role == "user":
            await self._settle_recall(session, transcript.text)
        await self._record(transcript)

    async def _record(self, transcript: Transcript) -> None:
        if transcript.role not in ("user", "assistant"):
            logger.warning("voice: dropping a transcript with role %r", transcript.role)
            return
        text = transcript.text.strip()
        if not text:
            return
        await self._memory.record(
            LoggedMessage(
                # The wall clock, because a transcript carries no time of its own
                # and the turn just happened.
                ts=clock.now(),
                role=transcript.role,
                content=text,
                # A microphone has no relay path - nobody forwards a third party's
                # words through it the way a Telegram forward does - so the user
                # side is the owner speaking.
                origin="owner" if transcript.role == "user" else "agent",
                session_kind="voice",
                modality="voice",
                channel=self._channel,
            )
        )

    async def _record_pending(self, session: VoiceSession) -> None:
        """Record what was said but never released.

        A generator cannot yield from its own `finally`, so a `receive()` cancelled
        mid-turn takes the accumulated transcript with it. The accumulation is
        therefore drained from outside instead - a protocol call, not a hopeful
        `getattr`: in voice mode the transcript is the only record there is.
        """
        try:
            for transcript in session.pending_transcripts():
                await self._record(transcript)
        except Exception:
            logger.exception("voice: could not record the unfinished turn")

    # --- recall -------------------------------------------------------------

    async def _watch_partials(self, session: VoiceSession) -> None:
        """Prefetch recall, and notice a barge-in, from the in-progress transcript.

        Both come from the same signal because both are about the user speaking,
        and the provider's own activity detection is the only thing that knows -
        docs/PLAN.md's reference list is explicit that VAD and endpointing are not
        ours to build.

        Pushed, not polled: each item arrives when the provider transcribes another
        few syllables, so the search starts then rather than up to an interval
        later, and a barge-in is noticed on the delta that proves it.
        """
        seen = ""
        partials = session.partial_transcripts()
        try:
            async for partial in partials:
                said = partial.text.strip()
                if said == seen:
                    continue
                if self._playing and len(said) > len(seen):
                    await self._barge_in(session)
                seen = said
                self._prefetch(session, said)
        finally:
            # Cancelled at the end of every conversation, so the stream is closed
            # here rather than left to whenever the generator is collected.
            await _aclose(partials)

    def _prefetch(self, session: VoiceSession, query: str) -> None:
        """Start a search for what has been said so far, if none is running.

        One in flight is enough: it lands in about 117 ms, and starting another on
        every delta would spend a round trip per syllable to arrive at the same
        answer.
        """
        if self._recall is None or len(query) < PREFETCH_MIN_CHARS:
            return
        if self._speculative is not None and not self._speculative[1].done():
            return
        self._speculative = (
            query,
            asyncio.create_task(self._prepare(session, query), name="voice-recall-prefetch"),
        )

    async def _prepare(self, session: VoiceSession, query: str) -> list[RecalledItem]:
        """Search, and put what comes back in front of the model.

        Delivered here - inside the prefetch, while the user is still speaking -
        because that is the only window where it can reach the answer to *this*
        question. The final transcript arrives at the turn boundary, and by then the
        model has already spoken; context that lands then is context for the next
        turn. The prefetch, at ~117 ms into an utterance, lands before the model
        starts generating.
        """
        items = await self._search(query)
        await self._offer(session, items)
        return items

    async def _offer(self, session: VoiceSession, items: list[RecalledItem]) -> None:
        """Put recalled memory in the model's history without asking for an answer.

        Never fails the turn, for the same reason recall itself does not: in voice
        mode an exception is silence, and answering with less memory beats not
        answering.
        """
        # Identity is the same rendering under a fixed nonce, never sent. The real
        # nonce differs every time by design, so comparing sent payloads directly
        # would make two blocks carrying identical memories always look different -
        # and the model would be handed the same facts on every partial transcript.
        identity = render_recall(items, _IDENTITY_NONCE)
        if not identity or identity == self._offered:
            return
        self._offered = identity
        try:
            await session.send_context(render_recall(items, secrets.token_hex(4)))
        except Exception:
            logger.exception("voice: recall could not be put in front of the model")

    async def _settle_recall(self, session: VoiceSession, said: str) -> None:
        """Reuse the prefetched search if it was for close enough to this, else
        redo it. Reuse is the whole point - it is what makes recall cost the voice
        turn nothing (docs/PLAN.md 4.3.1)."""
        if self._recall is None:
            return
        prepared, task = self._speculative or (None, None)
        self._speculative = None
        if task is not None and prepared is not None:
            if _covers(prepared, said):
                self.recalled = await task
                return
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        # Late for this answer and still worth sending: the utterance the prefetch
        # missed is the one the user actually made, and its memories are in front of
        # the model for the next thing they say.
        self.recalled = await self._prepare(session, said)

    async def _search(self, query: str) -> list[RecalledItem]:
        """Lane 1, with the same rule as the text loop: recall never fails a turn.

        An implementation that raises anyway is swallowed here. Answering with
        less memory beats answering with an apology, and in voice mode an
        exception is silence.
        """
        recall = self._recall
        if recall is None:
            return []
        try:
            return await recall.search(query, limit=self._recall_limit)
        except Exception:
            logger.exception("voice: recall failed; the turn goes on without it")
            return []

    async def _cancel_speculative(self) -> None:
        speculative, self._speculative = self._speculative, None
        if speculative is None:
            return
        speculative[1].cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await speculative[1]

    # --- barge-in -----------------------------------------------------------

    async def _barge_in(self, session: VoiceSession) -> None:
        """The user started talking over us.

        Both calls, always. Refusing to hand over more audio does not empty the
        speaker's buffer, and emptying the buffer does not stop the stream; one
        without the other keeps the daemon talking (daemon/voice/base.py).
        """
        self.interruptions += 1
        self._playing = False
        await session.interrupt()
        await self._audio.stop_playback()


def _covers(prepared: str, said: str) -> bool:
    """Is a prefetched query close enough to the finished utterance to reuse?

    Transcripts grow by appending, so containment is the natural test, and the
    length ratio is what stops a two-word prefix from standing in for a sentence
    whose point is at the end.
    """
    if not prepared or not said:
        return False
    return said.startswith(prepared) and len(prepared) >= REUSE_COVERAGE * len(said)


async def _aclose(stream: object) -> None:
    """Close an async iterator if it is one. The protocols promise `AsyncIterator`,
    not a generator, so the closer is asked for rather than assumed - and a
    microphone nobody closes is a recording light left on."""
    closer = getattr(stream, "aclose", None)
    if closer is None:
        return
    with contextlib.suppress(Exception):
        await closer()
