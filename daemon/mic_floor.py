"""Asking the wake loop for the microphone, so a proactive line can be spoken.

The daemon cannot listen and talk at the same time: the wake gate holds the
capture stream for as long as it is listening, and `daemon/proactivity/speaker.py`
refuses to speak while this process holds the microphone, because a speaker
talking into a live gate is the gate hearing the daemon's own voice.

Measured on the owner's machine, 2026-08-26, the first time proactivity ever
spoke: the gate correctly saw the owner at the keyboard and chose `both`, and
the utterance went to Telegram alone with
`speaker: refusing to speak while this process holds the microphone` in the log.
Not a bug in either half - the two halves had no way to talk to each other. This
module is that way.

**It is a mailbox, not a lock.** The proactive side posts what it wants said and
waits; the wake side takes it, does the speaking on its own terms, and answers.
Neither imports the other, which is the same arrangement - and the same reason -
as `daemon/mic_hold.py`: `presence.py` cannot import the voice layer without
making a text-only install unable to read presence at all.

Why the wake side does the speaking rather than being asked to step aside:
releasing the microphone is not a thing that can be done to the gate from
outside. `daemon/app.py`'s `_wake_round` already owns the whole sequence - close
the gate, hold one spoken thing, let the caller reopen - because that is what
being called by name does. A proactive line is another kind of spoken thing, so
it goes through the same sequence rather than a second one written beside it.
The alternative, cancelling the wake task from the scheduler to prise the device
loose, is the CoreAudio contention that froze this daemon for eleven hours
(`daemon/voice/audio.py`'s `release_off_loop`).

Not thread-safe, deliberately: everything here runs on the one event loop
(CONTRACTS non-negotiable 9), and a lock here would be a claim about concurrency
this process does not have.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

logger = logging.getLogger(__name__)

Outcome = Literal["spoke", "not-spoken", "no-listener"]
"""What became of a line handed to this module.

Three, not two, because the caller does something different in each. `spoke` and
`not-spoken` both come from a wake loop that took the request and ran it - it is
in charge, and the caller's only job is to record what happened. `no-listener` is
the different thing: **nobody took it**, so nothing in this process ever will, and
the caller is free to use the speaker directly.

Note what does *not* separate them: whether the microphone is held. `mic_hold` is
zero for the whole of `_speak_unprompted` - `listen()`'s `finally` closes the
capture stream, which exits `record()`'s hold, before `_wake_round` closes the
gate and long before anything is spoken. An earlier version of this paragraph said
`not-spoken` meant the wake loop still had the device (PR #115 review). It does
not, and the real reasons are in `ProactiveDelivery._say`.

Collapsing the last two into a single `False` is what the first version did, and
it was wrong in a way that would only show on installs where the wake loop is
configured but not running (PR #115 review): the speaker would have worked and was
never tried, so proactivity went quiet on the local machine and nothing said why."""

REPLY_CEILING_SECONDS = 150.0
"""How long to wait for an answer *after* the wake loop has taken the request.

Distinct from the wait for someone to take it, and deliberately far larger,
because these two measure different things. Once the request is taken,
`daemon/app.py`'s `_speak_unprompted` owes an answer in a `finally` and will give
one on every path including cancellation, so this ceiling is not a fallback - it is
a backstop against a contract this module cannot verify.

Sized above what the speaking itself can legitimately take. `speaker.py` caps a
line at 240 characters and gives its own subprocess `SPAWN_OVERHEAD_SECONDS` plus
0.5 s per character, so its own worst case is 125 s; anything past that is the
speaker's timeout firing, not speech still running.

The first version used one flat 10 s for both waits, which is **shorter than the
speech it was waiting on**: at the measured 0.145 s/char a line anywhere near the
judge's 120-character ceiling takes ~18 s. The requester gave up mid-sentence, the
row recorded `route='telegram'` and `modality='text'` for a line the room had
heard, and a Telegram failure on top of that would have deleted the utterance and
left the candidate live to be said aloud a second time."""

DEFAULT_TIMEOUT_SECONDS = 10.0
"""How long a line waits for *someone to take it* before giving up on the speaker.

The gate answers on its own schedule - it notices between audio blocks, roughly
every 30 ms while healthy - so this is not sized for a healthy gate. It is sized
for the ways there is nobody to answer: a live voice session (the round is inside
`run_voice`, not inside `listen`, and will not look at this mailbox until the
conversation ends), a capture stream that has stopped delivering, a wake loop
sleeping out `WAKE_RETRY_SECONDS` after a failed round. In every one of those the
answer is the same and the cost of learning it slowly is a late local line, so this
is far longer than the mechanism needs and far shorter than the tick it sits
inside."""

_waiting: tuple[str, asyncio.Future[bool], asyncio.Event] | None = None


def pending() -> bool:
    """Whether a line is waiting to be spoken.

    Read by the wake gate once per audio block, so it must stay this cheap.
    """
    return _waiting is not None


async def request(text: str, *, wait_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> Outcome:
    """Ask for the microphone and wait. See `Outcome` for the three answers.

    Two waits, because there are two different things to wait for. Until the wake
    loop takes the request, `wait_seconds` bounds how long to believe one is
    coming. After it takes it, the wait is for the speaking itself, which is as
    long as the line is - `REPLY_CEILING_SECONDS` is only a backstop.

    A second request while one is outstanding is refused rather than queued. Two
    proactive lines cannot be in flight at once (the tick judges one candidate and
    breaks), so a second one here means something is wrong, and a queue would hold
    a line until a moment nobody chose to say it at.
    """
    global _waiting
    if _waiting is not None:
        logger.warning("mic floor: a line is already waiting; refusing to queue another")
        return "not-spoken"

    loop = asyncio.get_running_loop()
    future: asyncio.Future[bool] = loop.create_future()
    taken = asyncio.Event()
    _waiting = (text, future, taken)
    try:
        try:
            await asyncio.wait_for(taken.wait(), timeout=wait_seconds)
        except TimeoutError:
            if not taken.is_set():
                logger.info(
                    "mic floor: nothing took the line in %.0fs; no wake loop is listening",
                    wait_seconds,
                )
                return "no-listener"
            # Re-checked rather than argued. `no-listener` is the one answer that
            # sends the caller to the speaker itself (`ProactiveDelivery._say`), so
            # returning it after `take` succeeded would say the same sentence into
            # the room twice. `take` sets this event and then returns synchronously
            # to a `_wake_round` that is about to speak, and whether a timeout
            # firing on the same loop iteration wins is a question about asyncio's
            # scheduling that this module should not be relying on either way.
            logger.info("mic floor: the take deadline and the take itself raced; taken wins")
        try:
            spoke = await asyncio.wait_for(
                asyncio.shield(future), timeout=REPLY_CEILING_SECONDS
            )
        except TimeoutError:
            # The taker owes an answer in a `finally`, so this is a broken contract
            # rather than a slow line - loud, and reported as taken, because
            # something may well have been said and saying it twice is worse than
            # not recording it.
            logger.error(
                "mic floor: the wake loop took a line and never answered in %.0fs",
                REPLY_CEILING_SECONDS,
            )
            return "not-spoken"
        return "spoke" if spoke else "not-spoken"
    finally:
        # Only if it is still ours. `take` clears the slot when it accepts the
        # request, and clearing it again here would drop a *later* request that
        # arrived while this one was being spoken.
        if _waiting is not None and _waiting[1] is future:
            _waiting = None


def take() -> tuple[str, asyncio.Future[bool]] | None:
    """Accept the waiting request, or `None` if there is not one.

    The caller owes the future exactly one `answer` - `request` is sitting on it,
    and a taker that drops it turns a fallback into a two-and-a-half-minute stall.
    `daemon/app.py` answers it in a `finally`.
    """
    global _waiting
    taken = _waiting
    _waiting = None
    if taken is None:
        return None
    text, future, taken_event = taken
    # Before returning, so the requester stops counting down the take timeout the
    # moment it is accepted rather than when the speaking finishes.
    taken_event.set()
    return text, future


def answer(future: asyncio.Future[bool], spoke: bool) -> None:
    """Tell the waiting request how it went. Safe to call twice, and safe after
    the requester has already stopped listening."""
    if not future.done():
        future.set_result(spoke)
