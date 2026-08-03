"""Voice contract.

docs/PLAN.md 6.5: voice is hosted native audio (Gemini Live first, OpenAI
Realtime second). Not a local STT -> LLM -> TTS pipeline - a cascade structurally
discards the paralinguistics that make a companion sound human, and the
open-weight native-audio models cannot generate audio on Apple Silicon today.

Two seams, because they fail for different reasons and are mocked differently:
`VoiceSession` is the network, `AudioIO` is the hardware.

Nothing here is imported unless voice is enabled: the audio dependencies live in
the `voice` extra so a text-only install does not need PortAudio.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Transcript:
    """What was said, in text.

    The hosted realtime APIs return transcripts for both directions, which is
    what lets memory (docs/PLAN.md 4) and persona evolution (5) keep working in
    voice mode - a pure audio-to-audio model with no text would leave them
    nothing to work with.
    """

    text: str
    role: str
    """'user' or 'assistant'."""
    final: bool
    """Only final transcripts are recorded.

    Gemini Live has no wire equivalent: `BidiGenerateContentTranscription` carries
    a single `text` field and streams incremental *deltas*, not growing
    hypotheses. So the Gemini implementation accumulates deltas and emits one
    `final=True` transcript at the turn boundary, and never yields `final=False` -
    yielding a delta would let a caller record a syllable as an utterance. The
    field stays because OpenAI Realtime, the second provider, does distinguish
    the two, and because a live-captions consumer would need its own seam rather
    than a loosening of this one."""


@runtime_checkable
class VoiceSession(Protocol):
    """One live conversation with a hosted native-audio model.

    Opened per conversation, not held permanently: a proactive utterance opens a
    session, speaks, and only opens the inbound stream if the user answers.
    Sessions are billed per minute, so an idle open connection is pure cost.
    """

    name: str

    async def __aenter__(self) -> VoiceSession: ...

    async def __aexit__(self, *exc: object) -> None: ...

    async def send_audio(self, chunk: bytes) -> None:
        """One PCM chunk from the microphone."""
        ...

    async def send_text(self, text: str) -> None:
        """Send text as a turn, with no user audio.

        **Not verbatim text-to-speech.** `realtimeInput.text` is a prompt: the
        model answers it rather than reading it out, and Live API has no
        verbatim path at all (`clientContent` is restricted to seeding history on
        current models). Say-this-exactly belongs to `AudioIO`'s local speaker,
        which is also the path a proactive utterance takes when the user is at
        the machine (docs/PLAN.md 6.3) - and which never leaves the device."""
        ...

    def receive(self) -> AsyncIterator[bytes | Transcript]:
        """Interleaved output: audio chunks to play, transcripts to record."""
        ...

    async def interrupt(self) -> None:
        """The user started talking over us: stop handing out this turn's audio.

        Local only. The protocol has no client-side cancel - under server-side
        VAD the user's own audio is what stops generation, reported back as
        `serverContent.interrupted`. So this refuses to yield more audio from the
        abandoned turn, and `AudioIO.stop_playback()` drops what is already
        queued. Neither alone is enough: without the first the daemon keeps
        streaming, without the second it keeps talking out of the buffer."""
        ...


@runtime_checkable
class AudioIO(Protocol):
    """Microphone and speaker. Separate from the session so tests can drive a
    conversation with no hardware, and so the local speaker used for proactive
    utterances (docs/PLAN.md 6.3) can exist without a session at all."""

    sample_rate: int
    """Capture rate, which is also what a session must be fed - 16 kHz for
    Gemini Live. Named as the plain `sample_rate` because it is the one a caller
    can get wrong in a way that reaches the provider."""

    playback_sample_rate: int
    """Output rate, which differs: Gemini returns 24 kHz. Playing 24 kHz output
    through a 16 kHz device is the chipmunk bug, so the two cannot share a
    field."""

    def record(self) -> AsyncIterator[bytes]: ...

    async def play(self, chunk: bytes) -> None: ...

    async def stop_playback(self) -> None:
        """Drop anything queued. Called on interruption - without it the daemon
        keeps talking over the user for as long as the buffer lasts."""
        ...

    async def close(self) -> None: ...
