"""Voice contract.

docs/PLAN.md 6.5: voice is hosted native audio (Gemini Live first, OpenAI
Realtime second). Not a local STT -> LLM -> TTS pipeline - a cascade structurally
discards the paralinguistics that make a companion sound human, and the
open-weight native-audio models cannot generate audio on Apple Silicon today.

Two seams, because they fail for different reasons and are mocked differently:
`VoiceSession` is the network, `AudioIO` is the hardware.

Two more for the wake gate, which exists because the session above bills per
minute. Holding one open on the chance of being spoken to costs about 48x what
30 minutes a day costs (docs/PLAN.md 6.5), so something free has to decide when
to open it: `VoiceActivityDetector` says whether a frame is speech at all, and
`SpeechRecognizer` says what a segment of speech was. Both are local and both are
seams for the same reason as the two above - one is arithmetic on audio, the other
is an OS service that can be absent, and they are mocked differently.

Nothing here is imported unless voice is enabled: the audio dependencies live in
the `voice` extra so a text-only install does not need PortAudio.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from daemon.llm.base import ToolCall
from daemon.tools.base import ToolResult

# `ToolCall` and `ToolResult` rather than voice-shaped copies of them. A spoken
# tool call is the same event as a typed one - the model asking for a named thing
# with decoded arguments - and it goes to the same `ToolRunner`, through the same
# `ToolPolicy`, leaving the same `tool_calls` row. A second pair of dataclasses
# would be a second place for the origin gate to be got wrong, and the whole
# reason that gate is trustworthy is that there is one of it.
# Both modules imported here are protocol files themselves, so this is a sideways
# import and not a layering break: `tools/base.py` already imports `llm/base.py`
# for `ToolSpec`.


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


@dataclass(frozen=True, slots=True)
class Interrupted:
    """The provider's activity detection says the user talked over us.

    A separate item in `receive()` rather than something a caller works out for
    itself, because working it out is what broke. `VoiceConversation` used to infer
    a barge-in from the user's transcript growing while audio played, and Gemini
    delivers `inputTranscription` *at the turn boundary* - measured, in the same
    server event as the first audio chunk of the answer. So the arrival of the
    question's own transcript looked exactly like someone interrupting the answer to
    it, and every turn was killed: a full, fluent reply generated and 0.0s of it
    played, twice reproduced against the live API.

    What it does **not** mean is "the user spoke". The Live API documents the flag as
    "a client message has interrupted current model generation", and a client message
    is something *we* sent: seeding recall through `send_context` mid-answer raised
    this flag 90 ms later and killed the answer. So a caller must both act on it -
    the speaker has to be emptied - and avoid causing it, which is
    `VoiceConversation._offer`'s job. Two different failures wearing one flag.
    """


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

    async def send_context(self, text: str) -> None:
        """Put text in front of the model without asking it to respond.

        This is how recall reaches a voice turn. `send_text` cannot do it: that is
        a prompt, so delivering a memory through it makes the daemon narrate old
        conversations unprompted.

        Measured against the live API rather than inferred - `clientContent` with
        `turnComplete: false` produced no audio and no transcript at all, while
        the same payload with `turnComplete: true` produced a full answer. That
        asymmetry is the seam.
        """
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

    async def send_tool_response(self, results: Sequence[ToolResult]) -> None:
        """Answer the tool calls this turn asked for, in one message.

        One message per round rather than one per call: the server pairs a result
        to its request by `ToolResult.call_id`, and every extra client message is
        another chance to interrupt generation - which is not a theoretical cost
        here. `send_context` is `clientContent`, the Live API says a message there
        "will interrupt any current model generation", and one recall seed
        mid-answer cut a 46.7 s turn down to 2.2 s of audio (`Interrupted`).

        A `toolResponse` is a different top-level message from `clientContent` and
        the docs claim nothing about it interrupting. That is inference, not
        measurement: `evals/m1c_voice_tools_spike.py` is what settles it, and
        `daemon/voice/gemini_live.py` records what it found.

        Whether a caller may answer at all is not this seam's question. The
        origin gate (`daemon/tools/policy.py`) decides that, and it decides it for
        a spoken turn exactly as for a typed one.
        """
        ...

    def receive(self) -> AsyncIterator[bytes | Transcript | Interrupted | ToolCall]:
        """Interleaved output for one turn: audio to play, transcripts to record,
        the provider saying the user cut in, and tools the model wants run.

        A `ToolCall` is only ever yielded by a session that was given tool specs
        to declare. A caller that offered none cannot receive one, which is what
        makes it safe for `VoiceConversation` to have routed everything that is
        neither audio nor `Interrupted` to the transcript path for as long as it
        has.

        **Ends at the turn boundary.** Measured: a turn that answered in 2.6 s
        and delivered its final transcript then blocked forever, and the server
        eventually aborted the session - reported as close 1008 "The operation was
        aborted", which reads like a policy violation and is really an idle
        timeout. An iterator that never ends cannot be consumed with `async for`,
        which is the only way anyone will consume it.
        """
        ...

    def pending_transcripts(self) -> list[Transcript]:
        """Whatever accumulated but was never yielded, drained destructively.

        In the protocol because it is the only cancellation-safe path memory has
        in voice mode: transcripts are how anything is remembered from a spoken
        turn, and a cancel between the last delta and the turn boundary otherwise
        loses the utterance from both the markdown and the mirror.
        """
        ...

    def partial_transcripts(self) -> AsyncIterator[Transcript]:
        """In-progress user transcripts, for work that must start before the user
        stops talking.

        Separate from `receive()` because `receive()` yields nothing while the
        user speaks, so there is no event to hang that work off - and because a
        partial must never reach the code that records utterances. Recall uses it:
        embedding costs ~117 ms of mostly fixed overhead, which is silence if it
        starts when the user finishes and free if it starts while they talk.
        """
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


# --- the wake gate ------------------------------------------------------------
# Measured on this project's target machine (Apple Silicon, macOS 26, Python 3.13),
# because both numbers decide whether the gate is allowed to exist:
#   Silero VAD  0.155 ms per 32 ms frame - 0.49% of one core, 206x realtime
#   Apple on-device ko-KR  partial at ~700 ms, final at 760 ms, no network
# And one measurement that shapes the design rather than justifying it: the VAD
# calls a 3-note chord with vibrato speech in 46.8% of frames. Music is speech to
# a VAD, so the VAD cannot be the whole gate - a real `daemon voice` run against
# ambient music recorded "nela o trecho da musica" as though the owner had said it.


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Is this frame speech? Cheap enough to run forever.

    Stateful: the reference implementation carries an LSTM state and the last 64
    samples of context between frames, so frames must arrive in order and `reset`
    must be called between unrelated streams. Getting that wrong is quiet - it
    returns plausible-looking probabilities near zero for real speech.
    """

    frame_samples: int
    """Exactly how many samples a call expects, and the implementation must check it.

    An earlier version of this line claimed the model raises on anything else. It
    does not, and the correction matters more than the original claim: measured on
    the vendored model, widths 256, 300, 512, 576 and 600 all *run* and return
    plausible-looking near-zero probabilities, while only 128, 1024 and 1600 raise.
    So a wrong frame size is a silent wrong answer - exactly the failure this
    project treats as the dangerous one - and the length check in `probability` is
    the only thing that turns it into an error.
    """

    sample_rate: int

    def probability(self, frame: bytes) -> float:
        """Speech likelihood in [0, 1] for one frame of 16-bit little-endian PCM."""
        ...

    def reset(self) -> None:
        """Forget the stream. Required between segments, or the tail of the last
        one biases the head of the next."""
        ...


@runtime_checkable
class SpeechRecognizer(Protocol):
    """What did this segment of speech say?

    Only ever asked about audio a `VoiceActivityDetector` already accepted, and
    only to decide whether the owner said the wake phrase - never to hold a
    conversation. That is `VoiceSession`'s job, and the reason is quality: a
    cascade discards paralinguistics (docs/PLAN.md 6.5, and the module docstring
    above). This seam exists to spend nothing while nobody is talking.

    `available` is separate from the call because absence is normal and must be
    reportable: an OS speech service can be missing, unauthorised, or lack the
    locale, and each looks identical from a failed transcription.
    """

    @property
    def available(self) -> bool:
        """Whether this recognizer can answer at all, checked without asking."""
        ...

    async def transcribe(self, pcm: bytes) -> str:
        """Best guess at the words in one segment, or `""` if it cannot say.

        Must not raise for ordinary failure. It is called from a loop that has to
        outlive an unavailable recognizer - a wake gate that dies takes the
        always-listening promise with it, and does so silently.
        """
        ...


@dataclass(frozen=True, slots=True)
class WakeEvent:
    """The owner called. Carries what was heard, not just that something was.

    `heard` is the recognizer's actual output and `matched` is the alias it hit,
    which are different strings on purpose: an on-device recognizer will not emit
    a coined name, so "루시야" arrives as "루시" and "헤이 데몬" as "헤이 대문".
    Both are stable per speaker, which is what makes matching on them work, and
    keeping the raw text is what lets `daemon doctor` show why a phrase does or
    does not fire instead of leaving the owner to guess.
    """

    heard: str
    matched: str
    confidence: float
    """Mean VAD probability over the segment. Not the recognizer's confidence -
    it does not report one on-device - so it says "this was speech", not "these
    were the words"."""

    pcm: bytes = b""
    """The audio that fired the gate, at `AudioIO.sample_rate`, so the conversation
    can begin with what was actually said.

    Carried because throwing it away cost the owner a whole utterance. The gate
    consumed "루시 뭐 해", matched on `루시`, and then discarded the sound - so the
    session opened having never heard "뭐 해" and the owner had to say it again.
    Measured on a real run: 14.79 s from the wake word to the first audio out, most
    of it a person repeating themselves into a microphone that had just changed
    hands.

    A whole segment, wake phrase included, not the tail after the phrase: there is no
    reliable boundary between them - the recognizer returns text, not alignment - and
    a model hearing "루시 뭐 해" is being addressed by name, which is what happened.

    Empty by default, so a gate that has no audio to offer and a caller that does not
    want any both keep working. **It only helps a phrase spoken in one breath.** A
    pause after the wake word ends the segment (`hangover_ms`), and whatever is said
    during the handover belongs to nobody - which is what `AudioIO`'s ready cue is
    for."""
