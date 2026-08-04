"""Shapes for proactivity - M3. FROZEN once the pieces are built against it.

The three stages in docs/PLAN.md 6.1 are deliberately separate objects, and the
separation is the design rather than tidiness:

    candidates  ->  gate  ->  one LLM call  ->  delivery
    deterministic   deterministic  the only model call   presence-routed

docs/CONTRACTS.md non-negotiable 7 states it as a rule: **silence is the
default.** Candidate generation and the gate make zero model calls, there is
exactly one LLM call, and it happens only for a candidate that already passed the
gate. Asking a model "should I speak?" as an open question gets "yes" almost every
time, so the model is never asked that - it is asked what to say, about a specific
reason, at a moment already judged safe.

## Why presence is a reading and not a boolean

PLAN 6.4: an ignored notification costs nothing, and **a voice coming out of the
speaker during a meeting is an accident.** So the gate needs the actual numbers to
put in `proactive_utterances.gate_snapshot`, not a verdict someone else already
collapsed - a bad call has to be diagnosable afterwards instead of guessed at.

And when a probe cannot answer, that is a third state, not a `False`. `Reading`
carries `unknown` for exactly that: unknown presence must never route to the
speaker, because the failure it risks is the expensive one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

CandidateKind = Literal["open_loop", "emotional", "silence", "pattern_time", "association"]
"""The five in `daemon/memory/schema.sql`. Type A-E of PLAN 6.1, in the same order."""

Delivery = Literal["local_speaker", "telegram", "both"]
"""Where an utterance went. Matches `proactive_utterances.route`."""


@dataclass(frozen=True, slots=True)
class Reading:
    """What the machine can say about whether the user is here, right now.

    Every field is what a probe measured, not what a policy concluded. The gate
    owns the thresholds; this owns the facts.
    """

    at: datetime
    idle_seconds: float | None = None
    """Seconds since the last HID event. `None` means the probe could not answer -
    which is not "the user is here" and not "the user is away"."""
    foreground_app: str | None = None
    audio_busy: bool | None = None
    """Something else is holding the audio device - a call, a recording."""
    unknown: tuple[str, ...] = ()
    """Which probes failed, and why, for the gate snapshot."""

    @property
    def at_keyboard(self) -> bool | None:
        """Whether the user is *at* the machine. `None` when it cannot be known.

        A separate concept from `idle_seconds` being small, because the caller
        needs the three-way answer and Python's truthiness would flatten it.
        """
        if self.idle_seconds is None:
            return None
        return self.idle_seconds < AT_KEYBOARD_SECONDS

    def as_snapshot(self) -> dict[str, object]:
        """The reading as JSON for `proactive_utterances.gate_snapshot`."""
        return {
            "at": self.at.isoformat(),
            "idle_seconds": self.idle_seconds,
            "foreground_app": self.foreground_app,
            "audio_busy": self.audio_busy,
            "unknown": list(self.unknown),
        }


AT_KEYBOARD_SECONDS = 120.0
"""Idle below this and the user is treated as present at the machine.

Two minutes rather than seconds because the question is "are they sitting here",
not "did they just type". Reading a long message on screen is not absence.
"""


@runtime_checkable
class Presence(Protocol):
    """Reads the machine. Platform-specific; the fallback answers `unknown`."""

    async def read(self) -> Reading: ...


@dataclass(frozen=True, slots=True)
class Candidate:
    """A reason it might be worth speaking. Mirrors a `proactive_candidates` row.

    `reason` is human-readable because it goes into the LLM prompt verbatim: the
    model is told *why* this surfaced and asked what to say about it, which is the
    narrow question it can actually answer well.
    """

    kind: CandidateKind
    reason: str
    payload: dict[str, object] = field(default_factory=dict)
    due_at: datetime | None = None
    expires_at: datetime | None = None
    fire_budget: int = 1
    cooldown_secs: int = 86_400
    id: int | None = None
    """Set once mirrored. `None` for one this tick just generated."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """The gate's answer for one candidate.

    Carries the reading it decided from, so a wrong call can be read back off
    `proactive_utterances.gate_snapshot` rather than reconstructed.
    """

    allowed: bool
    why: str
    """Which rule decided, in words. `"ok"` when nothing blocked."""
    reading: Reading
    delivery: Delivery = "telegram"
    """Where this would go if it is spoken. `telegram` is the safe default:
    unknown presence must not reach the speaker (PLAN 6.4)."""

    def as_snapshot(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "why": self.why,
            "delivery": self.delivery,
            **self.reading.as_snapshot(),
        }


# --- stage 3: the one model call, and what carries it ------------------------


@dataclass(frozen=True, slots=True)
class Utterance:
    """What the judge decided. `text` empty means it chose not to speak.

    Declining is a first-class answer, not a failure. The model is asked what to
    say about a specific reason at a moment already judged safe - and "nothing
    worth saying" is the correct answer most of the time, so it has to be
    expressible without an exception.
    """

    text: str = ""
    why_not: str = ""
    """Why it declined, for the log. Empty when it spoke."""

    def __bool__(self) -> bool:
        return bool(self.text.strip())


@runtime_checkable
class Judgement(Protocol):
    """Stage 3. One model call, for a candidate that already passed the gate.

    A seam so the tick does not import the judge: `daemon/proactivity/tick.py`
    orchestrates and must stay testable without a gateway. The narrow signature is
    the point - everything about *when* was decided before this is called, so the
    only question left is what to say.
    """

    async def decide(self, candidate: Candidate) -> Utterance: ...


@runtime_checkable
class Speaker(Protocol):
    """Says a line out loud at this machine.

    A separate seam from `VoiceSession` and not a special case of it: PLAN 6.5
    records that Live API has **no verbatim TTS path** - `realtimeInput.text` is a
    prompt, so the model answers the text instead of reading it. Speaking a
    sentence we already chose is therefore a local job, which is also why this
    path leaves no data on the machine at all (PLAN 6.3).
    """

    async def say(self, text: str) -> bool:
        """True if it was spoken. False - never an exception - if it could not be:
        a failed utterance must not lose the Telegram copy that went with it."""
        ...

    async def aclose(self) -> None: ...
