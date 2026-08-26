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
(CONTRACTS non-negotiable 9), and a lock would be a claim about concurrency this
process does not have.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10.0
"""How long a proactive line waits for the microphone before giving up on the
speaker and letting the channel carry it alone.

The wait exists because the gate answers on its own schedule - it notices the
request between audio blocks, roughly every 30 ms while healthy. Ten seconds is
therefore not sized for a healthy gate; it is sized for the ways the gate is
*not* healthy and nobody is coming: a live voice session (the round is inside
`run_voice`, not inside `listen`, and will not look at this mailbox until the
conversation ends), a capture stream that has stopped delivering, a wake loop
that failed its round and is sleeping out `WAKE_RETRY_SECONDS`. In every one of
those the answer is the same and the cost of learning it slowly is a late
Telegram message, so this is deliberately far longer than the mechanism needs
and still far shorter than the five-minute tick it sits inside."""

_waiting: tuple[str, asyncio.Future[bool]] | None = None


def pending() -> bool:
    """Whether a line is waiting to be spoken.

    Read by the wake gate once per audio block, so it must stay this cheap.
    """
    return _waiting is not None


async def request(text: str, *, wait_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> bool:
    """Ask for the microphone and wait for an answer. True if the line was spoken.

    False covers every way it was not - nobody took the request, the speaker
    refused, the synthesiser failed - because the caller does the same thing in
    all of them: let the channel carry the words instead. A caller that wants to
    know *why* has the log.

    A second request while one is outstanding is refused rather than queued. Two
    proactive lines cannot be in flight at once (the tick judges one candidate and
    breaks), so a second one here means something is wrong, and a queue would hold
    a line long enough to be said at a moment nobody chose.
    """
    global _waiting
    if _waiting is not None:
        logger.warning("mic floor: a line is already waiting; refusing to queue another")
        return False

    loop = asyncio.get_running_loop()
    future: asyncio.Future[bool] = loop.create_future()
    _waiting = (text, future)
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout=wait_seconds)
    except TimeoutError:
        logger.info(
            "mic floor: no answer in %.0fs; the channel carries this one alone",
            wait_seconds,
        )
        return False
    finally:
        # Only if it is still ours. `take` clears the slot when it accepts the
        # request, and clearing it again here would drop a *later* request that
        # arrived while this one was being spoken.
        if _waiting is not None and _waiting[1] is future:
            _waiting = None


def take() -> tuple[str, asyncio.Future[bool]] | None:
    """Accept the waiting request, or `None` if there is not one.

    The caller owes the future exactly one `answer` - `request` is sitting on it,
    and a taker that drops it turns a ten-second fallback into a ten-second delay
    for no reason. `daemon/app.py` answers it in a `finally`.
    """
    global _waiting
    taken = _waiting
    _waiting = None
    return taken


def answer(future: asyncio.Future[bool], spoke: bool) -> None:
    """Tell the waiting request how it went. Safe to call twice, and safe after
    the requester has already timed out and stopped listening."""
    if not future.done():
        future.set_result(spoke)
