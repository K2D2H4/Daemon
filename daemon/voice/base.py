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
    """False for a partial hypothesis. Only final transcripts are recorded."""


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
        """Speak this. Used by proactive utterances, which have no user audio."""
        ...

    def receive(self) -> AsyncIterator[bytes | Transcript]:
        """Interleaved output: audio chunks to play, transcripts to record."""
        ...

    async def interrupt(self) -> None:
        """The user started talking over us. Stop generating and drop queued audio."""
        ...


@runtime_checkable
class AudioIO(Protocol):
    """Microphone and speaker. Separate from the session so tests can drive a
    conversation with no hardware, and so the local speaker used for proactive
    utterances (docs/PLAN.md 6.3) can exist without a session at all."""

    sample_rate: int

    def record(self) -> AsyncIterator[bytes]: ...

    async def play(self, chunk: bytes) -> None: ...

    async def stop_playback(self) -> None:
        """Drop anything queued. Called on interruption - without it the daemon
        keeps talking over the user for as long as the buffer lasts."""
        ...

    async def close(self) -> None: ...
