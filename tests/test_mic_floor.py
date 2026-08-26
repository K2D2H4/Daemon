"""The mailbox between the proactive tick and the wake loop.

Every test resets the module global, because it is a module global on purpose
(see `daemon/mic_floor.py` on why a lock would be a lie about this process).
"""

from __future__ import annotations

import asyncio

import pytest

from daemon import mic_floor


@pytest.fixture(autouse=True)
def _empty_mailbox():
    mic_floor._waiting = None
    yield
    mic_floor._waiting = None


@pytest.mark.asyncio
async def test_nobody_takes_it_so_the_caller_may_use_the_speaker_itself() -> None:
    """`no-listener` and not-spoken are different answers because the caller does
    different things with them. Nobody took the line, so nothing in this process is
    holding the microphone either, and the local speaker will work - PR #115 review
    found the first version collapsed these into one `False` and left an install
    with a configured-but-dead wake loop unable to speak aloud at all."""
    outcome = await mic_floor.request("요즘 llm-wiki 쪽은 잘 돼가요?", wait_seconds=0.05)

    assert outcome == "no-listener"
    assert mic_floor.pending() is False, "a timed-out request must not linger in the mailbox"


@pytest.mark.asyncio
async def test_a_taker_that_speaks_it_is_reported_back() -> None:
    async def waiter() -> str:
        return await mic_floor.request("한마디", wait_seconds=5.0)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)  # let `request` post before we look

    assert mic_floor.pending() is True
    taken = mic_floor.take()
    assert taken is not None
    text, future = taken
    assert text == "한마디"
    assert mic_floor.pending() is False, "taking must empty the mailbox"

    mic_floor.answer(future, True)
    assert await task == "spoke"


@pytest.mark.asyncio
async def test_a_taker_that_could_not_speak_is_not_a_missing_listener() -> None:
    """A wake loop that took the line and could not say it gets one answer of its
    own, distinct from nobody taking it: the caller must *not* go around it to the
    speaker.

    Not because the microphone is still held - it is not, `daemon/mic_floor.py`
    retracts that reading - but for the two reasons in `ProactiveDelivery._say`: a
    second attempt moments later hits whatever made the first one fail, and the
    wake loop is already rebuilding its gate."""

    async def waiter() -> str:
        return await mic_floor.request("한마디", wait_seconds=5.0)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    taken = mic_floor.take()
    assert taken is not None
    mic_floor.answer(taken[1], False)

    assert await task == "not-spoken"


@pytest.mark.asyncio
async def test_the_take_deadline_stops_counting_once_the_line_is_taken() -> None:
    """The two waits measure different things, and the first version used one flat
    10 s for both - **shorter than the speech it was waiting on**. At the measured
    0.145 s/char a line near the judge's 120-character ceiling takes ~18 s, so the
    requester gave up mid-sentence and the row recorded `route='telegram'` for a
    line the room had heard (PR #115 review).

    Driven with a take deadline far shorter than the speaking, which is the shape
    that used to fail: taken at once, answered long after the deadline is gone."""

    async def waiter() -> str:
        return await mic_floor.request("긴 문장", wait_seconds=0.05)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    taken = mic_floor.take()
    assert taken is not None

    await asyncio.sleep(0.2)  # four times the take deadline, still speaking
    assert not task.done(), "the requester gave up while the line was being spoken"

    mic_floor.answer(taken[1], True)
    assert await task == "spoke"


@pytest.mark.asyncio
async def test_a_taker_that_never_answers_is_loud_rather_than_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The taker owes an answer in a `finally`, so this is a broken contract rather
    than a slow line. Reported as taken, not as a missing listener: something may
    well have been said, and saying it twice is worse than not recording it."""
    monkeypatch.setattr(mic_floor, "REPLY_CEILING_SECONDS", 0.05)

    async def waiter() -> str:
        return await mic_floor.request("한마디", wait_seconds=5.0)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    assert mic_floor.take() is not None  # taken, then dropped on the floor

    assert await task == "not-spoken"


@pytest.mark.asyncio
async def test_answering_after_the_requester_gave_up_is_not_an_error() -> None:
    """A taker that has already started speaking must be able to report back
    without raising into the wake loop, which would cost the round."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[bool] = loop.create_future()
    future.cancel()

    mic_floor.answer(future, True)  # must not raise

    future2: asyncio.Future[bool] = loop.create_future()
    mic_floor.answer(future2, True)
    mic_floor.answer(future2, False)
    assert future2.result() is True, "the first answer stands"


@pytest.mark.asyncio
async def test_a_second_line_is_refused_rather_than_queued() -> None:
    """Two proactive lines cannot legitimately be in flight - the tick judges one
    candidate and breaks - so a second request means something is wrong. A queue
    would hold it until a moment nobody chose to say it at.

    Refused as `not-spoken` rather than `no-listener`: a caller told there is no
    listener goes to the speaker, and the outstanding line may be coming out of it
    right now."""

    async def waiter() -> str:
        return await mic_floor.request("첫 번째", wait_seconds=5.0)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)

    assert await mic_floor.request("두 번째", wait_seconds=5.0) == "not-spoken"
    taken = mic_floor.take()
    assert taken is not None and taken[0] == "첫 번째", "the first line must survive the second"
    mic_floor.answer(taken[1], True)
    assert await task == "spoke"


@pytest.mark.asyncio
async def test_a_finished_request_does_not_clear_a_later_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`request`'s `finally` clears the mailbox, but only if the slot is still its
    own. Clearing unconditionally would let a finishing request delete a line
    posted while it was still being spoken, and the wake loop would find an empty
    mailbox it had just been told was full.

    The interleaving has to be built deliberately - the first version of this test
    let the earlier request finish *before* the later one was ever posted, so it
    passed with the guard removed (PR #115 review)."""
    monkeypatch.setattr(mic_floor, "REPLY_CEILING_SECONDS", 0.05)

    first = asyncio.create_task(mic_floor.request("먼저", wait_seconds=5.0))
    await asyncio.sleep(0)
    taken = mic_floor.take()
    assert taken is not None

    # Posted while the first is still outstanding, which is the case the guard is for.
    later = asyncio.create_task(mic_floor.request("나중", wait_seconds=5.0))
    await asyncio.sleep(0)
    assert mic_floor.pending() is True

    assert await first == "not-spoken", "the first should give up on its own"
    assert mic_floor.pending() is True, "the finishing request cleared a later one"

    taken2 = mic_floor.take()
    assert taken2 is not None and taken2[0] == "나중"
    mic_floor.answer(taken2[1], True)
    assert await later == "spoke"


@pytest.mark.asyncio
async def test_a_take_that_races_the_deadline_never_reports_no_listener() -> None:
    """`no-listener` is the one answer that sends the caller to the speaker itself
    (`ProactiveDelivery._say`), so returning it after `take` succeeded would say the
    same sentence into the room twice - once by the wake loop that took it, once by
    the caller that was told nobody had.

    `take` sets the event and returns synchronously to a `_wake_round` that is about
    to speak. Whether a timeout firing on the same loop iteration wins is a question
    about asyncio's scheduling, and the answer must not depend on it: the event is
    re-checked after the timeout, so the dangerous outcome is unreachable rather
    than merely unlikely. Driven here with a deadline of zero, which is that race
    with the timing removed."""
    async def waiter() -> str:
        return await mic_floor.request("한마디", wait_seconds=0)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    taken = mic_floor.take()
    assert taken is not None, "the request must have been posted before the deadline ran"

    mic_floor.answer(taken[1], True)
    outcome = await task

    assert outcome != "no-listener", "the caller would speak a line already being spoken"
    assert outcome == "spoke"
