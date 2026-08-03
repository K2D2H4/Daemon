"""One voice conversation: microphone in, speaker out, transcripts recorded.

The text path is daemon/loop.py and this is shaped after it on purpose -
dependencies arrive as protocols only, what the user said is written before
anything else can lose it, and recall never fails a turn. What differs is that in
voice mode the audio model *is* the brain (docs/PLAN.md 6.5): there is no gateway
call and no assembled prompt, so the session is the turn.

Three things here are load-bearing rather than incidental:

1. **Only `final=True` transcripts are recorded.** Gemini streams transcription
   as deltas, so recording anything else would leave single syllables in the log
   as though they were utterances (daemon/voice/base.py).
2. **Recall is prefetched from partial transcripts.** An embedder round trip is
   117 ms at p50, ~105 ms of it fixed overhead (docs/PLAN.md 4.3.1) - a cost that
   is unaffordable after the user stops talking and free while they are still
   talking. So the search starts on the partial text and the final transcript
   reuses that result when it covers enough of what was actually said.
3. **A barge-in does two things or it does nothing.** `session.interrupt()` stops
   the abandoned turn's audio from arriving, `audio.stop_playback()` drops what is
   already queued. Either one alone leaves the daemon talking over the user.

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
from collections.abc import AsyncIterator

from daemon import clock
from daemon.memory.base import LoggedMessage, MemoryWriter, Recall, RecalledItem
from daemon.voice.base import AudioIO, Transcript, VoiceSession

logger = logging.getLogger(__name__)

VOICE_CHANNEL = "voice"
"""`channel` for a recorded voice turn. Not the provider's name: the column says
how the words reached us, and a session that moves from Gemini to OpenAI Realtime
is the same channel (docs/PLAN.md 6.5 keeps two providers behind one seam)."""

IDLE_TIMEOUT_SECONDS = 30.0
"""How long a session may hear nothing before it is closed. Billing is per
minute, so silence is spending."""

PREFETCH_INTERVAL_SECONDS = 0.15
"""How often the in-progress transcript is checked. Just above the 117 ms
embedder round trip, so a prefetch usually lands before the next check."""

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
        prefetch_interval: float = PREFETCH_INTERVAL_SECONDS,
    ) -> None:
        self._session = session
        self._audio = audio
        self._memory = memory
        self._recall = recall
        self._recall_limit = recall_limit
        self._channel = channel
        self._idle_timeout = idle_timeout
        self._prefetch_interval = prefetch_interval

        self.recalled: list[RecalledItem] = []
        """What recall had ready for the last completed utterance.

        Read rather than sent: `VoiceSession` has no way to put text in front of
        the model without the model answering it - `realtimeInput.text` is a
        prompt, and `clientContent` (which would seed history silently) is not in
        the protocol. So the prefetch is proven and exposed here, and the delivery
        seam is a protocol change, not a workaround. See the report in
        daemon/voice/base.py on why that file is frozen."""

        self.ended: str | None = None
        """Why the conversation finished, for a caller that has to tell "the turn
        is over" from "the session is gone" (a `goAway` looks like the former and
        is the latter)."""

        self.interruptions = 0
        self._playing = False
        self._speculative: tuple[str, asyncio.Task[list[RecalledItem]]] | None = None

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
            watch = asyncio.create_task(
                self._watch_partials(session), name="voice-prefetch"
            )
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
        stream = session.receive()
        loop = asyncio.get_running_loop()
        try:
            async with asyncio.timeout(self._idle_timeout) as budget:
                async for item in stream:
                    budget.reschedule(loop.time() + self._idle_timeout)
                    if isinstance(item, bytes):
                        self._playing = True
                        await self._audio.play(item)
                    else:
                        await self._on_transcript(session, item)
        except TimeoutError:
            self.ended = f"nothing heard for {self._idle_timeout:.0f}s"
            logger.info("voice: %s; closing a session that bills per minute", self.ended)
        finally:
            await _aclose(stream)
            if self.ended is None:
                # `getattr` because only the provider knows, and the protocol has
                # no field for it: a session that ended on `goAway` reads exactly
                # like one whose turn finished, and a caller that cannot tell them
                # apart keeps sending audio into a socket that is gone.
                self.ended = getattr(session, "ended", None) or "the session stream ended"

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
            await self._settle_recall(transcript.text)
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
        therefore drained from outside instead.
        """
        drain = getattr(session, "pending_transcripts", None)
        if drain is None:
            return
        try:
            for transcript in drain():
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
        """
        peek = getattr(session, "partial_transcripts", None)
        if peek is None:
            logger.debug(
                "voice: session %r exposes no partial transcripts; recall will wait "
                "for the utterance to end and cost the turn a round trip",
                getattr(session, "name", "?"),
            )
            return
        seen = ""
        while True:
            await asyncio.sleep(self._prefetch_interval)
            said = _role_text(peek(), "user")
            if said == seen:
                continue
            if self._playing and len(said) > len(seen):
                await self._barge_in(session)
            seen = said
            self._prefetch(said)

    def _prefetch(self, query: str) -> None:
        """Start a search for what has been said so far, if none is running.

        One in flight is enough: it lands in about 117 ms, and replacing it every
        150 ms would spend a round trip per syllable to arrive at the same answer.
        """
        if self._recall is None or len(query) < PREFETCH_MIN_CHARS:
            return
        if self._speculative is not None and not self._speculative[1].done():
            return
        self._speculative = (
            query,
            asyncio.create_task(self._search(query), name="voice-recall-prefetch"),
        )

    async def _settle_recall(self, said: str) -> None:
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
        self.recalled = await self._search(said)

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


def _role_text(transcripts: list[Transcript], role: str) -> str:
    return "".join(t.text for t in transcripts if t.role == role).strip()


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
