"""What the daemon is doing, for the face to look like it.

Two event kinds, and the split is the design rather than a detail. `activity` is a
*state* the code already knows - listening, thinking, speaking, running a tool - and
it drives a looping clip. A mood is an *event*: give a feeling a start and an end time
and you have invented a question with no answer (a timer? the next turn? a change of
subject?), so it does not get one. A laugh is not a state, it is a thing that happens.

Nothing here imports anything else in `daemon/` (CONTRACTS 4). Publishers are handed a
bus; `daemon/app.py` is the only file that builds one.

**With nobody subscribed, publishing is a comparison.** A text-only install that never
opens the face page pays nothing for this module existing.
"""

from __future__ import annotations

import array
import asyncio
import math
import re
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

Activity = Literal["idle", "listening", "thinking", "speaking", "working"]
Mood = Literal["amused", "sulky", "curious"]

SHOT_BACKLOG = 4
"""One-shots queue, unlike `level`, because a dropped laugh is a missing expression
rather than a stale one. The cap exists anyway: a subscriber far enough behind to owe
five expressions is better off skipping the oldest than replaying a backlog of
feelings the conversation has moved past."""


@dataclass(frozen=True, slots=True)
class FaceState:
    """The current continuous state. `level` is only meaningful while speaking."""

    activity: Activity = "idle"
    level: float = 0.0


@dataclass(frozen=True, slots=True)
class OneShot:
    """A clip to play once and forget - a mood, or an idle flourish."""

    clip: str


Event = FaceState | OneShot


class _Sub:
    """One subscriber's mailbox: a slot for state, a small queue for one-shots."""

    __slots__ = ("closed", "shots", "state", "wake")

    def __init__(self) -> None:
        self.state: FaceState | None = None
        self.shots: deque[OneShot] = deque(maxlen=SHOT_BACKLOG)
        self.wake = asyncio.Event()
        self.closed = False


class FaceBus:
    """Fan-out of `FaceState` and `OneShot` to whoever is watching.

    No back pressure anywhere. A slow subscriber loses intermediate `level` values
    instead of delaying them, because the mouth position that matters is the current
    one - a queue would make the mouth lag the sound by however far behind the reader
    is, which is worse than dropping frames of movement.
    """

    def __init__(self) -> None:
        self._state = FaceState()
        self._subs: set[_Sub] = set()

    @property
    def state(self) -> FaceState:
        """The snapshot a new subscriber starts from."""
        return self._state

    def set_activity(self, activity: Activity) -> None:
        if activity == self._state.activity:
            return
        self._state = FaceState(activity=activity, level=self._state.level)
        self._fan(self._state)

    def set_level(self, level: float) -> None:
        level = 0.0 if level < 0.0 else 1.0 if level > 1.0 else level
        if level == self._state.level:
            # Same guard `set_activity` already has, and for the same reason
            # (design spec §2: the socket is open and the traffic is zero while
            # nothing is happening). Without it `SpeechClock.pump` republishes an
            # identical `level: 0.0` on every one of its 25Hz ticks for the whole
            # of a voice conversation - forty events a second down every open
            # stream saying nothing changed.
            return
        self._state = FaceState(activity=self._state.activity, level=level)
        if self._subs:
            self._fan(self._state)

    def one_shot(self, clip: str) -> None:
        self._fan(OneShot(clip=clip))

    def subscribe(self) -> AsyncIterator[Event]:
        return self._events()

    async def _events(self) -> AsyncIterator[Event]:
        sub = _Sub()
        # The snapshot first, always. Without it a reconnect during a reply leaves the
        # face sitting on idle while the daemon is audibly speaking.
        sub.state = self._state
        sub.wake.set()
        self._subs.add(sub)
        try:
            while True:
                # Before the wait, and only with the mailbox empty. After the wait it
                # deadlocks (the wake that carried the close has been cleared and
                # nothing will set it again); after the batch it drops whatever was
                # published in the same tick as the close - measured, a one-shot
                # queued just before `close()` never reached the page.
                if sub.closed and not sub.shots and sub.state is None:
                    return
                await sub.wake.wait()
                sub.wake.clear()
                # Drained into a batch before anything is yielded, and state goes
                # first in it.
                #
                # State first because a mood is published *with* the `speaking` it
                # belongs to, in one synchronous block, so both always land in the
                # same wake and no publisher can separate them - only this order
                # can. The other way round the page starts the mood and the
                # `speaking` right behind it cuts it (spec 3.2: `speaking` is the
                # one transition allowed to), so the expression was on screen for
                # about 0ms. This way it plays over the speaking loop and the page
                # hands back to that loop when the arc ends.
                #
                # Batched because yielding straight from the mailbox made the
                # order depend on where this generator happened to be suspended:
                # resuming from a `yield` re-enters mid-body and skips whatever
                # came before it. A consumer that asks for the next event
                # immediately (daemon/face_routes.py does) is almost always
                # parked on the `wait()` above, which is why that read the right
                # way round nearly all of the time rather than always.
                batch: list[Event] = []
                if sub.state is not None:
                    batch.append(sub.state)
                    sub.state = None
                while sub.shots:
                    batch.append(sub.shots.popleft())
                for event in batch:
                    yield event
        finally:
            self._subs.discard(sub)

    def close(self) -> None:
        """End every open subscription, so an SSE response can actually finish.

        `/face/stream` is a response that by design never completes, and uvicorn
        cannot close a connection whose response is still open - it clears
        `keep_alive` and waits. One open face page was therefore enough to pin the
        whole process in `Waiting for connections to close` (daemon/MEASURED.md),
        which is the restart the admin console was waiting for. This is the release:
        wake each subscriber with `closed` set so its generator returns, the
        `StreamingResponse` completes, and the connection closes on its own.

        A snapshot of the set, because each generator discards itself from `_subs`
        in its own `finally` as it unwinds. Idempotent and safe with no subscribers:
        `daemon run` with nothing watching is the common case.
        """
        for sub in tuple(self._subs):
            sub.closed = True
            sub.wake.set()

    def _fan(self, event: Event) -> None:
        for sub in self._subs:
            if isinstance(event, OneShot):
                sub.shots.append(event)
            else:
                sub.state = event
            sub.wake.set()


class SpeechClock:
    """`speaking` and `level`, timed to when audio is *heard* rather than queued.

    `play()` on either audio backend is a queue put, so the loudness of the chunk
    being handed over is the loudness of something the speaker has not reached yet.
    Published as-is it runs the mouth ahead of the sound by the whole backlog - which
    is the difference between a face that looks alive and one that looks dubbed.

    So chunks are stamped with the moment they will be audible, using exactly the
    arithmetic `daemon/voice/conversation.py` already does for barge-in:

        audible_until = max(audible_until, arrival) + len(chunk) / (rate * width)

    `speaking` is then true for `now < audible_until` and nothing else, so the instant
    it passes is exact rather than guessed. No debounce, and there still is none.

    That arithmetic alone was claimed here to make flicker impossible, and it does
    not: it holds mid-utterance, where the model runs ahead of real time and the
    backlog never empties, but not at the *start*, where one chunk shorter than the
    caller's tick interval is the whole queue. Measured off a live session's SSE
    stream: 34 of 79 activity changes were sub-second noise. `pump(generating=...)`
    is what closes that, and its docstring carries the rest.
    """

    __slots__ = ("_bus", "_pending", "_rate", "_until", "_width")

    def __init__(self, bus: FaceBus, *, sample_rate: int, bytes_per_frame: int) -> None:
        self._bus = bus
        self._rate = sample_rate
        self._width = bytes_per_frame
        self._until = 0.0
        self._pending: deque[tuple[float, float]] = deque()

    def fed(self, chunk: bytes, at: float) -> None:
        """One chunk handed to the speaker at wall-clock `at`."""
        if not chunk:
            return
        seconds = len(chunk) / (self._rate * self._width)
        starts = max(self._until, at)
        self._until = starts + seconds
        self._pending.append((self._until, _rms(chunk)))

    def pump(self, at: float, *, generating: bool = False, resting: Activity = "idle") -> None:
        """Publish whatever is audible at `at`. Call this on a timer.

        `generating` is "the model is still producing this turn". While it is true
        a dry queue is a *gap between chunks*, not the end of speech, so the falling
        edge is held: level drops to zero, `speaking` stays.

        Without it the face flickers at the start of every answer, which is not a
        hypothetical - measured off a live session's own SSE stream, 34 of 79
        activity changes were sub-second noise, `idle` the most common. The first
        chunk of a turn is routinely shorter than the 40ms pump interval, so `fed()`
        puts `_until` barely ahead of `at` and the very next tick finds the queue
        dry: `speaking -> idle -> speaking` inside one second. Mid-utterance the
        queue does not dry out, because the model generates faster than real time
        (28.4s of audio in about 19s - `daemon/voice/conversation.py`), which is why
        the start is where this bites.

        Deliberately not a debounce. A timed hold would make the *end* of speech
        late by however long the hold is, and this class exists to publish an exact
        instant rather than a guessed one. Holding on a flag the caller already
        maintains keeps the falling edge exact whenever it is real.

        `resting` is what "not speaking any more" means to the caller, and the
        default is only right for the text path. In an open voice conversation the
        microphone is live, so the answer ending means the daemon is *listening* -
        publishing `idle` there put a zero-length blip between `speaking` and the
        `listening` that the very next microphone chunk set microseconds later.
        Measured on a live 6-turn session: that happened at the end of **every one
        of the six turns**, half of all the sub-second noise left in it, and each
        one costs the page two neutral-wait cycles at exactly the moment the owner
        is watching for the mouth to stop.
        """
        while len(self._pending) > 1 and self._pending[0][0] <= at:
            self._pending.popleft()
        if at < self._until:
            self._bus.set_activity("speaking")
            self._bus.set_level(self._pending[0][1] if self._pending else 0.0)
            return
        if generating:
            # Between chunks: silent, but still this turn. `_pending` is left alone -
            # the loop above already trimmed it to at most one entry.
            self._bus.set_level(0.0)
            return
        self._pending.clear()
        self._bus.set_level(0.0)
        if self._bus.state.activity == "speaking":
            self._bus.set_activity(resting)


def _rms(chunk: bytes) -> float:
    """0..1 loudness of one 16-bit mono PCM chunk.

    Hand-rolled because `audioop` was removed in Python 3.13 (PEP 594), and a
    dependency for one square root is not worth it.
    """
    samples = array.array("h")
    samples.frombytes(chunk[: len(chunk) - (len(chunk) % 2)])
    if not samples:
        return 0.0
    total = math.fsum(float(s) * float(s) for s in samples)
    return min(1.0, math.sqrt(total / len(samples)) / 32768.0)


_MOOD_TAG = re.compile(r"^\s*\[mood:(amused|sulky|curious)\]\s*", re.IGNORECASE)

MOOD_TOOL = "set_mood"
"""The name of the one model-invoked value CONTRACTS 12 exempts from an audit row.

**Not a tool, and it must never become one.** It changes a facial expression and
nothing else - it does not touch the machine, so it has nothing to put in the row
rule 12 exists to write (see docs/adr/0018 and the rule itself). It is deliberately
absent from the tool registry, so there is no path by which `ToolRunner` could ever
run it; `daemon/voice/conversation.py` answers it inline, before the runner is
reached, and `tests/test_voice_conversation.py` fails if that ever stops being true.

Voice-only, because voice is the only path that needs it: the text path has a reply
to prepend a tag to (`MOOD_INSTRUCTION` below), and voice has no text we own.
"""

MOOD_VOICE_INSTRUCTION = (
    "표정: 정말로 그렇게 느껴질 때만 `set_mood`를 호출해서 얼굴 표정을 바꾼다 - "
    "amused(재미있음/웃김), sulky(서운함/삐침), curious(궁금함). 느껴지는 게 없으면 "
    "호출하지 않는다. **이 도구나 표정에 대해 절대 소리 내어 말하지 않는다.** 호출은 "
    "조용히 하고, 대답은 원래 하려던 말만 한다."
)
"""What makes the voice model reach for `MOOD_TOOL`, measured before it shipped.

`evals/voice_set_mood_spike.py`, `gemini-3.1-flash-live-preview`, 48 live audio
sessions over 81 flat-filtered declarations: **call rate 24/24, mood correct 24/24,
false positives 0/8, spoken aloud 0/32.**

The last of those is why the final sentence is in here and is not decoration. Spec
section 5 rejected putting the tag in the transcript because the model reads it out;
this instruction has to earn that not happening, and 0/32 is the number that says it
did. Editing this string re-opens that question - re-run the spike.

Note what it does *not* share with the text path's wording: nothing about neutral
turns. It did not need it. The text tag over-fires `curious` on 11 of 15 deliberately
neutral prompts and this over-fires on 0 of 8, which is the difference between a tag
that costs nothing to prepend and a call the model has to decide to make.
"""

MOOD_INSTRUCTION = (
    "표현 방식: 정말로 그렇게 느껴질 때만 답장 맨 앞에 다음 세 가지 중 하나를 붙인다 - "
    "[mood:amused] (재미있음/웃김), [mood:sulky] (서운함/삐침), [mood:curious] (궁금함). "
    "형식은 정확히 이대로 쓴다: 대괄호, mood, 콜론, 소문자 영어 단어, 그 뒤에 공백 하나와 "
    "답장 본문. 느껴지는 게 없으면 아무것도 붙이지 않고 답만 쓴다. 태그를 언급하거나 "
    "설명하지 않는다."
)
"""What makes a model write the tag `_MOOD_TAG` above reads. Here, next to the
parser, because the two are one contract: change the syntax in one and the other
silently stops matching, and the failure mode is invisible - replies keep
arriving, the face just never reacts.

Measured before shipping rather than assumed (`evals/face_mood_tag_spike.py`,
which imports this exact string so a later edit re-measures the thing that ships).
gemini-3.6-flash, 2026-08-26, 60 replies: **zero malformed attempts** and 45/45 on
prompts aimed at a mood. The known weakness is the other direction - 11 of 15
deliberately neutral prompts still got `[mood:curious]`, every false positive that
one word, because the model counts its own follow-up question as curiosity. So
`curious` fires often and `amused`/`sulky` stay honest. Tightening the wording is
a change to a *measured* string: re-run the spike, do not just reword it.

Text only. Spec section 5 keeps mood off the voice path, so this is attached in
`daemon/loop.py` and never in `Companion.context`, which both paths share.
"""


def split_mood(text: str) -> tuple[str, Mood | None]:
    """Pull a leading `[mood:...]` tag off a model reply, returning both halves.

    **The tag must never reach the wire or the markdown log.** Recall replays the log
    into later prompts, so a tag left in place would be read back to the model as
    something it says, and from there laundered into the personality - the one thing
    `data/persona/seed.md` being human-owned exists to prevent.

    Only a *leading* tag counts. A model that writes the syntax mid-sentence is
    quoting it, not declaring a mood, and an unknown mood name is left alone rather
    than guessed at.
    """
    match = _MOOD_TAG.match(text)
    if match is None:
        return text, None
    mood: Mood = match.group(1).lower()  # type: ignore[assignment]
    return text[match.end() :], mood
