"""One voice conversation: microphone in, speaker out, transcripts recorded.

What the daemon can do is `daemon/companion.py` - recalled memory, and writing an
exchange down with its vector - and this file is the voice transport for it. The
text endpoint carries the same companion over request/response
(`daemon/loop.py`); what differs is that in voice mode the audio model *is* the
brain (docs/PLAN.md 6.5): there is no gateway call and no assembled prompt, so the
session is the turn, and a message sent at the wrong moment kills the answer. That
asymmetry is why there is no shared single-turn pipeline; `daemon/companion.py` says
so at more length.

Four things here are load-bearing rather than incidental:

1. **Only `final=True` transcripts are recorded.** Gemini streams transcription
   as deltas, so recording anything else would leave single syllables in the log
   as though they were utterances (daemon/voice/base.py).
2. **Recall is prefetched from partial transcripts, and delivered from the same
   place** - but against Gemini it does not arrive in time, and saying so is more
   use than the design it was written for. An embedder round trip is 117 ms at p50,
   ~105 ms of it fixed overhead (docs/PLAN.md 4.3.1), which is free while the user
   is still talking and silence afterwards. That only works if the provider
   transcribes mid-utterance, and this one does not: `inputTranscription` arrives
   at the turn boundary, in the same server event as the first audio chunk of the
   answer. Measured, twice. So the prefetch fires when the answer has already
   started and what it finds is context for the *next* turn. The machinery stays -
   it would be right for a provider that streams partials, which as it turns out
   neither hosted provider does today: OpenAI Realtime's `partial_transcripts()`
   also yields one item per turn, the completed utterance, not a growing
   fragment (daemon/voice/openai_realtime.py, whisper-1 sends no deltas) - and
   the claim that this machinery makes recall free is retracted.
3. **A barge-in is the provider's call, not ours, and it does two things or it does
   nothing.** `session.interrupt()` stops the abandoned turn's audio from arriving,
   `audio.stop_playback()` drops what is already queued; either alone leaves the
   daemon talking over the user. What decides it is `Interrupted` from `receive()`,
   never a guess made here: inferring it from a transcript growing while audio played
   ruled *every* turn a barge-in for the reason in point 2, and threw away complete
   answers unheard. But `Interrupted` is not only the user, either - it is also raised
   by anything *we* send mid-generation, which is why recall waits for the turn
   boundary (`_offer`). Both readings of one flag, and both cost an answer.
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
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass

from daemon import clock

# The recall block is rendered by the companion, not again here. It is not
# formatting: the nonce boundary is what stops a recalled memory posing as an
# instruction, and the origin label is what stops relayed text posing as the owner's
# own words - a fix an earlier audit made in the loop and then found undone one layer
# up. A second copy is a second place for that to happen, and voice is the weaker
# position to start from: the wire has no role that means "reference material", so
# the block arrives as a *user* turn.
from daemon.companion import Companion
from daemon.llm.base import ToolCall
from daemon.memory.base import LoggedMessage, RecalledItem
from daemon.tools.base import ToolResult
from daemon.voice.base import AudioIO, Interrupted, Transcript, VoiceSession
from daemon.voice.screen_share import ScreenShareController, ScreenSharePump

logger = logging.getLogger(__name__)

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

CALLED_BY_NAME = (
    "The owner just called your name to get your attention. Greet them briefly, in "
    "character, and wait for what they want. Do not read this instruction aloud."
)
"""What a session opened by the wake word alone is asked to answer.

Not the transcript. The wake gate matches on what the *recognizer* returns, which
is routinely not the name at all - this owner's aliases are `연락,벨라` because
"벨라" reliably comes back as "연락", an ordinary Korean word meaning "contact".
Sending that through as the owner's words got exactly what it deserved: measured
against the live model, "벨라" produced "네, 부르셨어요?" and "연락" produced
"...에? 연락?" - the daemon puzzling over a word nobody said.

So the wake word is delivered as what it *means* rather than as what it sounded
like. Phrased as an instruction the model answers rather than a line it reads:
measured over repeated trials, this wording never leaked into the reply, while a
Korean-language variant of it leaked the word "context" into one.
"""

OPENING_ANSWER_HOLD_SECONDS = 6.0
"""How long the microphone is held while the model owes an answer to the wake
word. Long enough to cover a slow first turn (measured: 1.1 s with no microphone
attached, 11.77 s with one), short enough that a model which never answers hands
the room back well inside the idle budget."""

PLAYBACK_BYTES_PER_FRAME = 2
"""16-bit mono, which is what every `AudioIO` here plays (daemon/voice/audio.py).
Only used to turn a byte count into seconds for the session report."""


@dataclass(frozen=True, slots=True)
class VoiceStats:
    """What one conversation actually did, as numbers.

    Exists because `interruptions` was counted and never reported, so the one
    failure this module has in the field - the daemon cutting itself off when its
    own speaker leaks into the microphone - was invisible from the outside. A
    change to echo cancellation or to the server's endpointing can only be argued
    for against these.
    """

    turns: int
    """Turns that produced anything. An empty `receive()` is not a turn (see
    `EMPTY_TURNS_ALLOWED`)."""

    interruptions: int
    """Barge-ins acted on. With no echo cancellation this counts the daemon
    interrupting itself, which is the point of reporting it."""

    first_audio_seconds: float | None
    """From the conversation starting to the first audio chunk handed to the
    speaker: handshake, setup, the first utterance and the first response, all in
    one number. None if it never spoke at all."""

    played_seconds: float
    """Audio handed to the speaker. Not audio heard - a barge-in drops part of
    what is already queued - so this is what was generated and paid for.

    The one to read against `interruptions`: a turn count with seconds of audio and
    no interruptions is a conversation, and turns with almost no audio are the
    daemon throwing its own answers away."""

    # Deliberately no per-turn response latency. It was tried, from the last
    # in-progress transcript to that turn's first audio, and the measurement that
    # justified this whole change killed it: the provider delivers that transcript
    # in the *same server event* as the first audio, so the interval is always about
    # zero and says nothing about endpointing. `first_audio_seconds` is the honest
    # end-to-end number until something local decides turn boundaries.

    def describe(self) -> str:
        """One line, for the end of a session. An absent measurement says so rather
        than reading as a zero."""
        spoke = self.first_audio_seconds
        first = "never spoke" if spoke is None else f"{spoke:.2f}s"
        return (
            f"{self.turns} turn(s), {self.interruptions} interruption(s), "
            f"first audio {first}, {self.played_seconds:.1f}s of audio played"
        )


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
        companion: Companion,
        *,
        channel: str = VOICE_CHANNEL,
        idle_timeout: float = IDLE_TIMEOUT_SECONDS,
        opening_audio: bytes = b"",
        opening_text: str = "",
        screen_share: ScreenShareController | None = None,
        screen_pump_factory: Callable[[VoiceSession], ScreenSharePump] | None = None,
        barge_in: bool = True,
    ) -> None:
        self._session = session
        self._audio = audio
        self._companion = companion
        self._channel = channel
        self._idle_timeout = idle_timeout
        self._barge_in_enabled = barge_in
        """Whether the microphone streams while the daemon talks (config.py,
        DAEMON_VOICE_BARGE_IN). False is half-duplex: answers always play to the
        end, and speaking over one changes nothing until it finishes."""
        self._opening_audio = opening_audio
        self._opening_text = opening_text
        """What the owner said, as text, when the audio is not worth sending.

        The wake-word-only case: the local recognizer has already decided the
        segment is the name and nothing else, so the *words* are settled and only
        the audio is ambiguous - sent as audio, "벨라" reached the model as "별로"
        and it answered a sentence nobody said. Sent as text there is nothing left
        to mishear, and the model answers being called by name the way a person
        does. Skipping it entirely is not the third option: that leaves the owner
        calling into silence, waiting, and calling again."""
        """Audio to hand the session before the microphone is even open.

        The wake gate's own segment, normally: it heard the owner say "루시 뭐 해",
        matched the alias, and used to throw the sound away - so the session began
        deaf to the question it was opened for and the owner said it again. At
        `AudioIO.sample_rate`, because that is what a session must be fed."""

        self._screen_share = screen_share
        """Owns the live screen-share pump, if screen sharing is on at all
        (Task 2.3). `None` for the text path and for a voice conversation with
        `DAEMON_SCREEN_ENABLED` off - both cases where there is nothing to bind."""
        self._screen_pump_factory = screen_pump_factory
        """Builds a fresh `ScreenSharePump` bound to *this* session, once it is
        live. A factory rather than a pump handed in directly, because
        `run_voice`'s reconnect loop (`_voice_attempts`) opens a new session per
        attempt and each one needs its own pump - never the previous, dropped
        session's."""

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
        self.turns = 0
        """Turns that produced something. Reported, along with the rest of
        `stats`, because none of these numbers used to leave this object."""

        # Monotonic, from the event loop's own clock: these are durations, and
        # `clock.now()` is the wall clock the records are stamped with.
        self._started_at: float | None = None
        self._first_audio_at: float | None = None
        self._played_bytes = 0
        self._playback_until = 0.0
        """Loop-clock instant the speaker runs dry, so the idle budget can start
        counting silence when the room actually falls silent (see `_on_audio`)."""

        self._generating = False
        """Whether the model is mid-turn.

        Named for what it gates: `clientContent` interrupts generation, so this is
        what keeps recall off the wire until the answer is finished. It used to be
        called `_playing` and used to decide barge-ins, and neither survived contact
        with the provider - see `_offer` and `_watch_partials`."""

        self._opening_answer_until = 0.0
        """Loop-clock deadline for holding the microphone after an opening the model
        still owes an answer to.

        Measured, and the measurement is the whole justification: the same opening
        text answered in **1.1 s** against a session with no microphone attached, and
        took **11.77 s** in the resident, where the mic starts streaming the room the
        instant the session is up. Incoming audio at the moment a turn begins reads
        as the user starting to speak, so the server cancels the answer it was about
        to give and waits for an utterance that never comes - and it does that
        without emitting `interrupted`, which is why the session reported zero
        barge-ins while the owner sat through eleven seconds of nothing and concluded
        the daemon was deaf. The owner has just said the wake word and is waiting;
        holding the room out of the socket until the answer starts costs nothing they
        were going to say. Bounded, because a model that never answers must not take
        the microphone with it."""

        self._answering_tool = False
        """Whether a tool call is between "the model asked" and "the model spoke".

        The blind spot `_generating` alone had: a blocking tool call arrives before
        any audio (measured - the provider module's tool-calling notes),
        so nothing had set `_generating` when the user's transcript settled recall
        - and the recall block's `clientContent` interrupted the answer the model
        was about to give for the tool result. Live symptom: `tool.ran ok=True`
        followed milliseconds later by `gemini-live: the server cancelled tool
        calls [...]` on every single tool call, and the daemon opened Finder and
        said nothing about it."""

        self._deferred: list[RecalledItem] | None = None
        """Recall that arrived mid-answer and is waiting for the turn to end."""

        self._speculative: tuple[str, asyncio.Task[list[RecalledItem]]] | None = None
        self._offered: str | None = None
        """The last block put in front of the model, so a prefetch that is reused
        rather than redone does not seed the same memories twice."""

    @property
    def stats(self) -> VoiceStats:
        """What this conversation did. Safe to read before, during and after
        `run`; a caller reports it in a `finally`, so it must not depend on the
        conversation having finished cleanly."""
        seconds_per_byte = 1.0 / (self._audio.playback_sample_rate * PLAYBACK_BYTES_PER_FRAME)
        first = None
        if self._first_audio_at is not None and self._started_at is not None:
            first = self._first_audio_at - self._started_at
        return VoiceStats(
            turns=self.turns,
            interruptions=self.interruptions,
            first_audio_seconds=first,
            played_seconds=self._played_bytes * seconds_per_byte,
        )

    async def run(self) -> None:
        """Hold one conversation, then close the session.

        Returns when the session ends, when nothing has been heard for
        `idle_timeout`, or when the caller cancels. Raises whatever the session
        raises - a voice turn that cannot run should surface so the caller can
        fall back to text.
        """
        # Before the session opens, so `first_audio_seconds` includes the handshake.
        # A provider is allowed 20s of setup budget before it gives up, and a report
        # that started the clock afterwards would hide exactly that.
        self._started_at = asyncio.get_running_loop().time()
        try:
            async with self._session as session:
                # As soon as the session is live and before the turn loop starts:
                # the pump needs a live session to send frames on, and binding it
                # here (rather than earlier) is what gives a reconnect a fresh pump
                # for its fresh session. Not started - `start_screen_share` does
                # that, and only if the owner asks.
                if self._screen_share is not None and self._screen_pump_factory is not None:
                    self._screen_share.bind(self._screen_pump_factory(session))
                # Before the opening audio: the tail of the conversation this wake
                # word interrupted, so a fresh session does not greet an amnesiac's
                # greeting. Safe to send here and only here - nothing is generating
                # yet, and `send_context` mid-generation kills the answer (measured,
                # module docstring). The date goes first: the tail is read *in* the
                # present, so the present has to already be established.
                await self._send_time(session)
                await self._send_continuity(session)
                # Before the microphone, so the utterance that opened the session is
                # the first thing the model hears rather than racing live audio for
                # its place in the turn.
                await self._send_opening(session)
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
        finally:
            # A share must never outlive its session: reached on every exit path,
            # including cancellation and a session that never finished opening (in
            # which case `bind` never ran and this is a no-op).
            if self._screen_share is not None:
                await self._screen_share.stop_and_unbind()

    async def _send_opening(self, session: VoiceSession) -> None:
        """Hand over what was already said, if anything was.

        Never fails the conversation: an opening that cannot be delivered costs the
        user one repeated sentence, and raising here would cost them the whole turn.
        """
        try:
            if self._opening_audio:
                await session.send_audio(self._opening_audio)
            elif self._opening_text:
                # A prompt, so the model answers it - which is the point: being
                # called by name deserves an answer, not silence.
                await session.send_text(self._opening_text)
                loop = asyncio.get_running_loop()
                self._opening_answer_until = loop.time() + OPENING_ANSWER_HOLD_SECONDS
        except Exception:
            logger.exception("voice: could not hand over the utterance that opened the session")

    async def _send_time(self, session: VoiceSession) -> None:
        """Tell the model what day it is before anything else reaches it.

        Same never-fails contract as `_send_continuity`: the time lost costs one
        wrong "아까", raising would cost the turn.
        """
        try:
            block = await self._companion.time_block()
        except Exception:
            logger.exception("voice: could not assemble the current time")
            return
        if not block:
            return
        try:
            await session.send_context(block)
        except Exception:
            logger.exception("voice: could not hand over the current time")

    async def _send_continuity(self, session: VoiceSession) -> None:
        """Hand over the conversation this wake word is continuing, if it is fresh.

        Every session used to start empty - the server holds voice history, and a
        new socket has none - so calling the daemon twice in three minutes met a
        stranger the second time, while the same two messages over Telegram were one
        thread (the text loop carries `recent()` in every request). Same never-fails
        contract as `_send_opening`: continuity lost costs one "what were we
        saying?", raising would cost the turn.
        """
        try:
            block = await self._companion.continuity_block()
        except Exception:
            logger.exception("voice: could not assemble the recent conversation")
            return
        if not block:
            return
        try:
            await session.send_context(block)
        except Exception:
            logger.exception("voice: could not hand over the recent conversation")

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
                if isinstance(item, bytes):
                    self._generating = True
                    self._on_audio(loop.time(), len(item))
                    await self._audio.play(item)
                elif isinstance(item, Interrupted):
                    await self._barge_in(session)
                elif isinstance(item, ToolCall):
                    await self._run_tool_call(session, item)
                else:
                    await self._on_transcript(session, item)
                # After the item is handled, not before: a chunk has to be able to
                # extend the budget by *its own* playing time, and `_on_audio` is
                # what knows that. Rescheduling first counted every answer as if it
                # were heard the instant it arrived - the ten seconds `_on_audio`
                # describes. From when the room falls silent, not when the packet
                # landed.
                budget.reschedule(
                    max(loop.time(), self._playback_until) + self._idle_timeout
                )
        finally:
            await _aclose(stream)
        # The turn is over, whatever ended it, so nothing is generating and anything
        # held back is now safe to send. Both matter: leaving the flag set would hold
        # recall forever after a turn that ended without a final transcript, and
        # never flushing would mean a memory searched for and then silently dropped.
        self._generating = False
        self._answering_tool = False
        await self._flush_deferred(session)
        self.ended = getattr(session, "ended", None)
        if produced:
            self.turns += 1
        return produced

    async def _forward_microphone(
        self, session: VoiceSession, microphone: AsyncIterator[bytes]
    ) -> None:
        try:
            async for chunk in microphone:
                if self._answering_tool and not self._generating:
                    # The floor belongs to the tool. Between "the model asked" and
                    # its first audio back, the server's activity detection reads
                    # any mic sound - a breath, a mumble, the room - as the user
                    # interrupting and cancels the pending call, so the daemon ran
                    # the tool and never spoke the result (measured live: `tool.ran
                    # ok=True` then `the server cancelled tool calls`, with zero
                    # real interruptions in the session). Nothing is playing yet,
                    # so there is nothing to barge into; the chunk is dropped, not
                    # queued. The moment audio flows, `_generating` is set and the
                    # microphone resumes - barge-in over the spoken result works
                    # exactly as before.
                    continue
                if self._opening_answer_until and not self._generating:
                    if asyncio.get_running_loop().time() < self._opening_answer_until:
                        # The answer to the wake word has not started yet; see
                        # `_opening_answer_until`. Dropped, not queued - stale room
                        # audio helps nobody once the answer is under way.
                        continue
                    # Bound reached: the model is not answering, so the microphone
                    # goes back to the owner rather than staying held.
                    self._opening_answer_until = 0.0
                if not self._barge_in_enabled and (self._generating or self._answering_tool):
                    # Half-duplex, by the owner's choice (DAEMON_VOICE_BARGE_IN=false):
                    # while the daemon is speaking, the microphone yields entirely, so
                    # an answer always plays to the end - no echo leak, no "응" of
                    # agreement, no room noise can kill it mid-sentence. The price is
                    # stated in config.py: interrupting by voice does nothing until
                    # the answer finishes.
                    continue
                await session.send_audio(chunk)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Either the socket went or the microphone did, and the receive side
            # is already reporting which. Logged rather than raised so this task
            # dying does not replace the error the caller needs to see.
            logger.exception("voice: stopped feeding the session")

    def _on_audio(self, at: float, size: int) -> None:
        """Note one chunk on its way to the speaker, and when it will finish playing.

        The second half is what keeps the idle budget honest. The model generates
        faster than real time, so a whole answer *arrives* long before it is *heard*:
        measured on the owner's Mac, 28.4 s of audio landed in about 19 s, and the
        30 s silence budget - rescheduled on arrival - had already been running for
        ten of them while the daemon was still talking. The owner got 20.7 s of turn
        instead of 30, was cut off mid-reply, and the log said "nothing heard for
        30s" about a stretch it had spent speaking.
        """
        self._played_bytes += size
        # The answer has started, so the microphone goes back to the room.
        self._opening_answer_until = 0.0
        if self._first_audio_at is None:
            self._first_audio_at = at
        seconds = size / (self._audio.playback_sample_rate * PLAYBACK_BYTES_PER_FRAME)
        # Chunks queue behind each other, so playback ends after whatever is already
        # queued - not `at + seconds`, which would forget the backlog.
        self._playback_until = max(self._playback_until, at) + seconds

    # --- transcripts --------------------------------------------------------

    async def _on_transcript(self, session: VoiceSession, transcript: Transcript) -> None:
        if not transcript.final:
            # base.py: Gemini never yields these, and one recorded as an utterance
            # leaves a syllable in the log forever. Nothing partial gets past here.
            logger.debug("voice: ignoring a partial %s transcript", transcript.role)
            return
        # Both roles are flushed at the turn boundary, so a final transcript is
        # also the signal that this turn's audio is done.
        self._generating = False
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
        await self._companion.record(
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
        # Embedded now, in the session, rather than left to the next restart's
        # backfill. Nothing indexed a voice turn at all until the companion owned
        # both halves of this: the words were in the markdown and the mirror, and the
        # vector lane simply did not have them - which for Korean is the difference
        # between 50% keyword-only recall and 93% hybrid (daemon/CLAUDE.md). Here
        # rather than at some later boundary because a voice turn has no wait left to
        # protect: the transcript arrives in the same server event as the answer's
        # first audio chunk, so the model is already talking.
        await self._companion.index_recorded()

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
        """Prefetch recall from the in-progress transcript.

        **A barge-in is not decided here, and used to be.** The test was "audio is
        playing and the user's transcript grew", which reads as sound reasoning and
        is wrong for this provider: Gemini delivers `inputTranscription` at the turn
        boundary, in the same server event as the first audio chunk of the answer
        (measured - daemon/voice/base.py, `Interrupted`). So the question's own
        transcript always arrived while its answer was playing, every turn was
        ruled a barge-in, and a complete reply was generated and then thrown away
        unheard. The signal now comes from the provider's own activity detection,
        through `receive()`.

        The same measurement costs this method its other justification, honestly
        stated: a partial that arrives with the answer cannot make recall free. See
        `_prefetch`.
        """
        seen = ""
        partials = session.partial_transcripts()
        try:
            async for partial in partials:
                said = partial.text.strip()
                if said == seen:
                    continue
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
        if not self._companion.has_recall or len(query) < PREFETCH_MIN_CHARS:
            return
        if self._speculative is not None and not self._speculative[1].done():
            return
        self._speculative = (
            query,
            asyncio.create_task(self._prepare(session, query), name="voice-recall-prefetch"),
        )

    async def _prepare(self, session: VoiceSession, query: str) -> list[RecalledItem]:
        """Search, and put what comes back in front of the model.

        Delivered from inside the prefetch because that is the earliest moment there
        is - not because it is early enough. Against a provider that transcribes
        mid-utterance this lands before generation starts; against Gemini the first
        partial and the first audio arrive in the same server event, so this reaches
        the model one turn late (see point 2 of the module docstring). Still worth
        sending: one turn late is what the memory is for the *next* question, and the
        alternative is not sending it at all.
        """
        items = await self._companion.search(query)
        await self._offer(session, items)
        return items

    async def _offer(self, session: VoiceSession, items: list[RecalledItem]) -> None:
        """Put recalled memory in the model's history without asking for an answer.

        **Never while the model is generating.** `send_context` is `clientContent`,
        and the Live API is explicit that "a message here will interrupt any current
        model generation" - so seeding a memory mid-answer kills the answer. Measured
        against the live API: `interrupted` arrived **90 ms** after the recall block
        went out, and over one conversation the same room and microphone delivered
        2.2s of audio with recall on and 46.7s with it off. Held to the turn boundary
        instead, which costs nothing that was not already lost: the prefetch fires on
        a partial that arrives with the answer, so its memories were only ever going
        to reach the *next* turn anyway.

        This is also what `serverContent.interrupted` really means. It is documented
        as "a client message has interrupted current model generation" - not "the user
        spoke - so it was never the pure user-VAD signal an earlier version of this
        module took it for, and the daemon was reading its own interruption as the
        owner talking over it.

        Never fails the turn, for the same reason recall itself does not: in voice
        mode an exception is silence, and answering with less memory beats not
        answering.
        """
        # Identity is the same memories under a fixed nonce, never sent. The real
        # nonce differs every time by design, so comparing sent payloads directly
        # would make two blocks carrying identical memories always look different -
        # and the model would be handed the same facts on every partial transcript.
        identity = self._companion.recall_key(items)
        if not identity or identity == self._offered:
            return
        if self._generating or self._answering_tool:
            # Held, not dropped, and `_offered` is deliberately left alone so the
            # flush does not mistake this for something already sent. A later
            # prefetch landing in the same turn simply replaces it - the newest
            # search is the better one.
            self._deferred = items
            logger.debug("voice: holding recall until the turn ends, so it cannot cut it short")
            return
        self._deferred = None
        self._offered = identity
        try:
            await session.send_context(self._companion.recall_block(items))
        except Exception:
            logger.exception("voice: recall could not be put in front of the model")

    async def _flush_deferred(self, session: VoiceSession) -> None:
        """Send what was held back while the model was talking.

        Called at the turn boundary, which is where `clientContent` is safe: there is
        no generation for it to interrupt.
        """
        held, self._deferred = self._deferred, None
        if held is not None:
            await self._offer(session, held)

    async def _settle_recall(self, session: VoiceSession, said: str) -> None:
        """Reuse the prefetched search if it was for close enough to this, else
        redo it. Reuse is the whole point - it is what makes recall cost the voice
        turn nothing (docs/PLAN.md 4.3.1)."""
        if not self._companion.has_recall:
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

    async def _cancel_speculative(self) -> None:
        speculative, self._speculative = self._speculative, None
        if speculative is None:
            return
        speculative[1].cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await speculative[1]

    # --- barge-in -----------------------------------------------------------

    async def _barge_in(self, session: VoiceSession) -> None:
        """The provider says the user talked over us.

        Both calls, always. Refusing to hand over more audio does not empty the
        speaker's buffer, and emptying the buffer does not stop the stream; one
        without the other keeps the daemon talking (daemon/voice/base.py).
        """
        # Logged, not just counted. A session report saying "3 interruptions" does
        # not say whether a person cut in three times or the daemon cut itself off
        # three times, and those need opposite fixes - which is exactly the
        # confusion that let a self-interruption on every single turn go unnoticed.
        logger.info(
            "voice: barge-in %d, reported by the provider (the user cut in, or "
            "something we sent did)",
            self.interruptions + 1,
        )
        self.interruptions += 1
        self._generating = False
        await session.interrupt()
        await self._audio.stop_playback()

    # --- tools ---------------------------------------------------------------

    async def _run_tool_call(self, session: VoiceSession, call: ToolCall) -> None:
        """Run one tool the model asked for and hand the result back on the socket.

        The same `Companion.run_tools` the text loop drives, so a spoken call goes
        through the same policy, the same registry and the same `tool_calls` audit
        row (docs/CONTRACTS.md 12) - `receive()` yields the neutral `ToolCall`
        precisely so there is one tool path, not a voice-shaped copy of it.

        `origin='owner'` without asking who is speaking: a microphone has no relay
        path, so a spoken turn is the owner's own words (see `_record`). The gate in
        `daemon/companion.py` only offered these tools because it agreed, and
        `daemon/app.py` pins voice to `allowlist` mode - so the runner either runs
        the call or refuses it with a reason the model can speak, and never parks it
        for an approval nobody is watching a spoken turn to give.

        Answered from inside `receive()`'s loop because a blocking call waits: the
        tool call arrives before any audio and the model generates nothing until the
        response goes back, measured against the live API (daemon/voice/base.py,
        `send_tool_response`). The runner returns one result per call it was given,
        so there is always something to send.
        """
        # From this moment until the model has spoken (or the turn ends), a
        # `send_context` would land in the middle of the tool exchange and the
        # server cancels the call - see `_answering_tool`. Set before running the
        # tool, because the user's own transcript settles recall in this same
        # window.
        self._answering_tool = True
        outcome = await self._companion.run_tools(
            [call], origin="owner", channel=self._channel, sender_id=None
        )
        # What ran is not narrated back - not spoken, and no longer logged here
        # either (docs/CONTRACTS.md 12). The owner's ground-truth record is the
        # `tool_calls` audit row the runner writes for every executed call
        # (`daemon tools log`); the runner also logs each run at INFO. A spoken turn
        # has no reply line to fold a notice into, and reading raw commands aloud is
        # exactly the clutter this removal is about.
        results = await self._deliver_images(session, outcome.results)
        await session.send_tool_response(results)

    async def _deliver_images(
        self, session: VoiceSession, results: Sequence[ToolResult]
    ) -> Sequence[ToolResult]:
        """Put what a tool captured in front of the model, not just its caption.

        `see_screen` returns a caption *and* pixels (`ToolResult.images`). The text
        loop attaches the pixels as their own user turn (daemon/loop.py); voice
        dropped them on the floor and sent only the caption - "captured 1
        display(s)" - so the model was asked what is on screen while being shown
        nothing, and did what a model does with a question it cannot answer: it
        invented one. The owner's screen held a photo of food and the daemon called
        it a picture of a dog, confidently, while the text path described the same
        screen correctly.

        Frames go over `realtime_input.video`, the channel the live share already
        uses, and they go *before* the tool response so the image is in history when
        the model composes its answer. Verified against the live socket rather than
        inferred: a probe image carrying the digits 7392 came back read aloud
        correctly, which is also what retires the "pending confirmation" note this
        transport used to carry.

        Never fails the turn: an image that cannot be delivered downgrades the
        caption to say so, which is the one honest thing to tell a model that is
        about to be asked what it can see.
        """
        from dataclasses import replace

        from daemon.tools.screen import screen_note

        delivered: list[ToolResult] = []
        for result in results:
            if not result.images:
                delivered.append(result)
                continue
            sent = 0
            for image in result.images:
                try:
                    await session.send_frame(image.data)
                    sent += 1
                except Exception:
                    logger.exception("voice: could not hand over a captured image")
            if not sent:
                delivered.append(
                    replace(
                        result,
                        content=(
                            f"{result.content}\n\nThe image could not be delivered, so "
                            "you cannot see it. Say that rather than describing it."
                        ),
                    )
                )
                continue
            # The untrusted-data framing rides in the caption, because a frame
            # carries no text of its own - same stance as the text path, which puts
            # `screen_note` on the turn holding the image (security stance A).
            delivered.append(
                replace(result, content=f"{result.content}\n\n{screen_note('the screen')}")
            )
        return delivered


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
