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

    __slots__ = ("shots", "state", "wake")

    def __init__(self) -> None:
        self.state: FaceState | None = None
        self.shots: deque[OneShot] = deque(maxlen=SHOT_BACKLOG)
        self.wake = asyncio.Event()


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

    `speaking` is then true for `now < audible_until` and nothing else. No debounce:
    a queue that momentarily empties mid-utterance leaves that instant in the future,
    so the face does not flicker, and the instant it passes is exact rather than
    guessed.
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

    def pump(self, at: float) -> None:
        """Publish whatever is audible at `at`. Call this on a timer."""
        while len(self._pending) > 1 and self._pending[0][0] <= at:
            self._pending.popleft()
        if at < self._until:
            self._bus.set_activity("speaking")
            self._bus.set_level(self._pending[0][1] if self._pending else 0.0)
            return
        self._pending.clear()
        self._bus.set_level(0.0)
        if self._bus.state.activity == "speaking":
            self._bus.set_activity("idle")


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
