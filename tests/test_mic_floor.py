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
async def test_nobody_takes_it_so_the_channel_carries_it_alone() -> None:
    """The wake loop is inside a voice session, or its capture stream is dead, or
    it is sleeping off a failed round. Whatever the reason, the utterance must not
    be lost - `False` is the caller's signal to let Telegram carry it."""
    spoke = await mic_floor.request("요즘 llm-wiki 쪽은 잘 돼가요?", wait_seconds=0.05)

    assert spoke is False
    assert mic_floor.pending() is False, "a timed-out request must not linger in the mailbox"


@pytest.mark.asyncio
async def test_a_taker_that_speaks_it_is_reported_back() -> None:
    async def waiter() -> bool:
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
    assert await task is True


@pytest.mark.asyncio
async def test_a_taker_that_could_not_speak_is_also_reported_back() -> None:
    """The distinction the caller needs is spoken/not, never why: a speaker that
    refused and a synthesiser that failed both mean the same thing downstream."""

    async def waiter() -> bool:
        return await mic_floor.request("한마디", wait_seconds=5.0)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    taken = mic_floor.take()
    assert taken is not None
    mic_floor.answer(taken[1], False)

    assert await task is False


@pytest.mark.asyncio
async def test_answering_after_the_requester_gave_up_is_not_an_error() -> None:
    """The race is real: the gate can notice the request on the same loop turn the
    ten seconds run out. A taker that has already started speaking must be able to
    report back without raising into the wake loop, which would cost the round."""
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
    would hold it until a moment nobody chose to say it at."""

    async def waiter() -> bool:
        return await mic_floor.request("첫 번째", wait_seconds=5.0)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)

    assert await mic_floor.request("두 번째", wait_seconds=5.0) is False
    taken = mic_floor.take()
    assert taken is not None and taken[0] == "첫 번째", "the first line must survive the second"
    mic_floor.answer(taken[1], True)
    assert await task is True


@pytest.mark.asyncio
async def test_a_timeout_does_not_clear_a_later_request() -> None:
    """`request`'s `finally` clears the mailbox, but only if the slot is still its
    own. Clearing unconditionally would let a line that timed out at 10.0s delete
    the line posted at 10.001s, and the wake loop would find an empty mailbox it
    had just been told was full."""

    async def waiter() -> bool:
        return await mic_floor.request("먼저", wait_seconds=0.05)

    task = asyncio.create_task(waiter())
    await asyncio.sleep(0)
    # The first request is still outstanding; take it the way the wake loop would,
    # which is what frees the mailbox for the next one.
    taken = mic_floor.take()
    assert taken is not None
    assert await task is False, "taken but never answered still times out"

    later = asyncio.create_task(mic_floor.request("나중", wait_seconds=5.0))
    await asyncio.sleep(0.1)  # past the first request's own timeout

    assert mic_floor.pending() is True, "the later request was cleared by the earlier one"
    taken2 = mic_floor.take()
    assert taken2 is not None and taken2[0] == "나중"
    mic_floor.answer(taken2[1], True)
    assert await later is True
