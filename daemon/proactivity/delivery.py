"""Getting a proactive utterance to the user, and writing down that it happened.

docs/PLAN.md 6.3 routes by presence and delivers to **both** places when the user
is at the machine: the speaker says it, and the same words go to Telegram. Two
reasons, both from that section - being spoken to by a speaker and answering on a
keyboard is awkward without a thread to answer in, and a line nobody heard is lost
otherwise.

## Order, and why it is not the usual one

Everywhere else in this codebase the markdown goes first (non-negotiable 1). Here
the sqlite row goes first, and the reason is the label button:

    insert row ──► send ──► achieved? ──► log markdown, mark fired
                              │
                              └─ nothing sent ──► delete the row

The `utterance_id` is on the button before the message leaves, so the row has to
exist before the send or a fast tap resolves to nothing and the user is told their
label was stale. Non-negotiable 1 is about never losing *user data*; this row is
our own bookkeeping and is rebuildable from the markdown that follows it.

If nothing was delivered the row is deleted and the candidate stays live. An
utterance that reached nobody was not said: keeping it would spend the day's budget
on silence and put an unlabelable message into the precision numbers that M3's own
gate is judged on.

## The utterance is logged as `proactive`, and that matters twice

It goes in the conversation log because the user's reply is meaningless without
it - "잘 됐어" answering a question that is not in the history reads as a non
sequitur to the next turn. And it is logged with `session_kind='proactive'`, which
is what hygiene rule 1 (PLAN 4.2) filters on: the daemon's own speech must not
become evidence for the next reflection, or for the silence clock that decides
whether to speak again. Speaking would otherwise be its own excuse to speak.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from daemon.channels.base import Channel, OutboundMessage
from daemon.clock import now as clock_now
from daemon.memory.base import LoggedMessage, MemoryWriter
from daemon.memory.store import Store
from daemon.proactivity.base import Candidate, Delivery, Speaker, Utterance, Verdict

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Delivered:
    """What actually happened. `route` is None when nothing reached the user."""

    utterance_id: str
    text: str
    route: Delivery | None
    spoke_locally: bool = False
    sent_to_channel: bool = False

    def __bool__(self) -> bool:
        return self.route is not None


class ProactiveDelivery:
    """Sends one utterance and records it. Owns the ordering above."""

    def __init__(
        self,
        store: Store,
        memory: MemoryWriter,
        *,
        channel: Channel | None = None,
        speaker: Speaker | None = None,
        ask_for_the_floor: Callable[[str], Awaitable[bool]] | None = None,
    ) -> None:
        self._store = store
        self._memory = memory
        self._channel = channel
        self._speaker = speaker
        # `daemon/mic_floor.py`'s `request`, when a wake loop is running. Injected
        # rather than imported so this module keeps knowing nothing about the voice
        # layer, and so an install with no wake loop - voice off, or `daemon
        # proactive --speak` from a terminal - still reaches the speaker directly.
        self._ask_for_the_floor = ask_for_the_floor

    async def deliver(
        self,
        candidate: Candidate,
        utterance: Utterance,
        verdict: Verdict,
        *,
        now: datetime | None = None,
    ) -> Delivered:
        if not utterance:
            raise ValueError("deliver() needs something to say; check `if utterance` first")
        moment = now or clock_now()
        utterance_id = str(uuid.uuid4())

        self._store.insert_utterance(
            utterance_id=utterance_id,
            candidate_id=candidate.id,
            kind=candidate.kind,
            text=utterance.text,
            route=verdict.delivery,
            gate_snapshot=json.dumps(verdict.as_snapshot(), ensure_ascii=False),
            now=moment,
        )

        wants_speaker = verdict.delivery in {"local_speaker", "both"}
        wants_channel = verdict.delivery in {"telegram", "both"}
        spoke = await self._say(utterance.text) if wants_speaker else False
        sent = await self._send(utterance.text, utterance_id) if wants_channel else False

        achieved = _route_of(spoke=spoke, sent=sent)
        if achieved is None:
            # Nothing reached the user, so nothing was said. The candidate is left
            # live and un-fired: the next tick tries again, which is the right
            # answer for a dead network or a busy audio device.
            self._store.delete_utterance(utterance_id)
            logger.warning(
                "proactive: %s could not be delivered (%s); candidate stays live",
                candidate.kind,
                verdict.delivery,
            )
            return Delivered(utterance_id=utterance_id, text=utterance.text, route=None)

        if achieved != verdict.delivery:
            self._store.set_utterance_route(utterance_id, achieved)

        await self._log(utterance.text, achieved, moment, spoke=spoke)
        if candidate.id is not None:
            self._store.mark_candidate_fired(candidate.id, now=moment)
        return Delivered(
            utterance_id=utterance_id,
            text=utterance.text,
            route=achieved,
            spoke_locally=spoke,
            sent_to_channel=sent,
        )

    async def _say(self, text: str) -> bool:
        """Speak it here at the machine, if anything can.

        Through the mic floor when a wake loop is running, because it is holding
        the capture stream and `Speaker.say` refuses outright while this process
        does (`daemon/proactivity/speaker.py`). Measured 2026-08-26, the first time
        proactivity ever spoke: the gate saw the owner at the keyboard, chose
        `both`, and the line went to Telegram alone because nothing could ask the
        gate to stand down. `daemon/mic_floor.py` is the ask; the wake loop does
        the speaking and answers `False` if it could not, which lands here exactly
        like a speaker that refused.
        """
        if self._ask_for_the_floor is not None:
            return await self._ask_for_the_floor(text)
        if self._speaker is None:
            # The gate should not have chosen a speaker route without one, but a
            # mismatch here must not lose the Telegram half.
            logger.warning("proactive: speaker route chosen with no speaker configured")
            return False
        return await self._speaker.say(text)

    async def _send(self, text: str, utterance_id: str) -> bool:
        if self._channel is None:
            logger.warning("proactive: channel route chosen with no channel configured")
            return False
        try:
            await self._channel.send(
                OutboundMessage(
                    text=text,
                    # `labelable` is what makes the channel attach the thumbs
                    # buttons, and `utterance_id` is what a press comes back with.
                    # Without both, this utterance never enters the precision
                    # numbers and the label clock (PLAN 8.1) does not start.
                    labelable=True,
                    utterance_id=utterance_id,
                    # None on purpose: an unsolicited utterance answers no request,
                    # so the channel delivers it to its configured owner rather
                    # than to whoever spoke last (channels/base.py).
                    recipient_id=None,
                )
            )
        except Exception:
            logger.exception("proactive: channel refused the utterance")
            return False
        return True

    async def _log(self, text: str, route: Delivery, moment: datetime, *, spoke: bool) -> None:
        """Put it in the conversation log, marked as the daemon's own initiative.

        Never fails the delivery: the words already reached the user, and losing the
        log entry costs context on the next turn rather than the utterance itself.
        Reported loudly because it also means `daemon reindex` cannot restore it.
        """
        try:
            await self._memory.record(
                LoggedMessage(
                    ts=moment,
                    role="assistant",
                    content=text,
                    origin="agent",
                    session_kind="proactive",
                    # Keyed on whether it was actually spoken, not on the route:
                    # `both` used the speaker too, and the column exists to record
                    # that a paralinguistic channel was involved (PLAN 4.2).
                    modality="voice" if spoke else "text",
                    channel=route,
                )
            )
        except Exception:
            logger.exception("proactive: spoke but could not log it; the next turn loses context")


def _route_of(*, spoke: bool, sent: bool) -> Delivery | None:
    if spoke and sent:
        return "both"
    if spoke:
        return "local_speaker"
    if sent:
        return "telegram"
    return None
