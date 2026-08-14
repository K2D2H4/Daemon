"""Running an owner's delegated request in the background, and reporting it back.

A voice turn hands work off through `delegate_task`; this is where that work is
actually run - through the same text `ConversationLoop` the Telegram path uses, so
a nested-schema tool the voice model could not call is called here where it can be -
and reported by presence. See docs/superpowers/specs/2026-08-14-async-delegation-design.md.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from daemon.channels.base import InboundMessage, OutboundMessage
from daemon.clock import now as clock_now
from daemon.loop import ConversationLoop

logger = logging.getLogger(__name__)

FAILURE_PREFIX = "그 작업을 하려다 실패했어"

EMPTY_REPLY_MESSAGE = "작업은 처리했는데 돌려줄 결과 메시지가 비어 있어. 노션/로그를 확인해줘."


async def deliver_result(
    text: str, *, presence: Any, speaker: Any, channel: Any, recipient_id: str | None
) -> str:
    """Route a finished result to the owner by presence. Never raises.

    At the keyboard: speak it and send it. Away: send it. A speak or send that
    fails degrades the route rather than losing the result - the reply already
    happened, and raising here would strand it.
    """
    at_keyboard = False
    if presence is not None:
        try:
            reading = await presence.read()
            at_keyboard = bool(reading.at_keyboard)
        except Exception:
            logger.exception("delegation: presence read failed; treating as away")

    spoke = False
    if at_keyboard and speaker is not None:
        try:
            spoke = await speaker.say(text)
        except Exception:
            logger.exception("delegation: could not speak the result")

    sent = False
    if channel is not None:
        try:
            await channel.send(OutboundMessage(text=text, recipient_id=recipient_id))
            sent = True
        except Exception:
            logger.exception("delegation: could not send the result to the channel")

    if spoke and sent:
        return "both"
    if sent:
        return "telegram"
    if spoke:
        return "local_speaker"
    logger.warning("delegation: result reached nobody: %s", text[:80])
    return "none"


class CaptureChannel:
    """A `Channel` that keeps the loop's reply instead of sending it.

    `ConversationLoop.handle` ends by `channel.send(...)`; the worker wants that text,
    not a delivery. `listen()` is never driven - the worker calls `handle` directly.
    """

    name = "delegate"

    def __init__(self) -> None:
        self.reply: str | None = None

    async def send(self, message: Any) -> None:
        self.reply = message.text

    async def listen(self) -> AsyncIterator[InboundMessage]:
        return
        yield  # pragma: no cover - makes this an async generator; never driven


def build_run_request(
    *, gateway: Any, companion_factory: Callable[[], Any], owner_id: str = "owner"
) -> Callable[[str], Awaitable[str]]:
    """A `run_request` that runs one task through the real text ConversationLoop.

    A fresh CaptureChannel and companion per call: the loop records and answers as
    the text path does, and the captured reply is what gets delivered. Origin is the
    owner (`authored_by_sender=True`) - the request came from the owner by voice.
    """

    async def run_request(request: str) -> str:
        capture = CaptureChannel()
        loop = ConversationLoop(capture, gateway, companion_factory())
        await loop.handle(
            InboundMessage(
                text=request,
                sender_id=owner_id,
                received_at=clock_now(),
                channel="delegate",
                authored_by_sender=True,
            )
        )
        return capture.reply or ""

    return run_request


class DelegationWorker:
    """Runs queued delegated tasks one at a time and reports each result."""

    def __init__(
        self,
        store: Any,
        run_request: Callable[[str], Awaitable[str]] | None,
        deliver: Callable[[str, Any], Awaitable[None]] | None,
        *,
        wake: asyncio.Event,
        poll_seconds: float = 5.0,
    ) -> None:
        self._store = store
        self._run_request = run_request
        self._deliver = deliver
        self._wake = wake
        self._poll_seconds = poll_seconds

    async def drain_once(self) -> bool:
        """Claim and finish exactly one task. Returns False if the queue was empty."""
        row = self._store.claim_next_queued()
        if row is None:
            return False
        task_id, request = row["id"], row["request"]
        try:
            assert self._run_request is not None
            reply = await self._run_request(request)
        except Exception as exc:
            logger.exception("delegation: task %s failed", task_id)
            reply = f"{FAILURE_PREFIX}: {exc}"
            self._store.mark_task_failed(task_id, str(exc))
        else:
            # The work already happened - a failure to record it must not turn a
            # real result into a reported failure. Log it, deliver the real reply.
            if not reply.strip():
                reply = EMPTY_REPLY_MESSAGE
            try:
                self._store.mark_task_done(task_id, reply)
            except Exception:
                logger.exception("delegation: could not mark task %s done", task_id)
        if self._deliver is not None:
            try:
                await self._deliver(reply, row)
            except Exception:
                logger.exception("delegation: could not report task %s", task_id)
        return True

    async def run(self) -> None:
        """Drain the queue, then wait for a wake signal or the poll timeout."""
        while True:
            try:
                while await self.drain_once():
                    pass
            except Exception:
                logger.exception("delegation: worker loop error; continuing")
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                pass
