"""The wake gate: does it open a paid session exactly when it should?

Fakes at the four protocol edges in `daemon/voice/base.py` - no microphone, no
speech service, no network. Two properties are worth more than the rest and are
tested from both sides:

  * **the VAD alone never fires.** It calls a 3-note chord with vibrato speech in
    46.8% of frames, and a real `daemon voice` run against ambient music recorded
    `nela o trecho da música` as though the owner had said it.
  * **the gate never dies.** A raising VAD or recognizer leaves a process that is
    alive and permanently deaf, which this repo has shipped three times.

Korean throughout, because the matching is against measured Korean transcriptions
(`헤이 데몬` -> `헤이 대문`) and CJK normalisation has broken things here before.
"""

from __future__ import annotations

import asyncio
import unicodedata
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daemon.config import ConfigError, Settings
from daemon.voice import wake
from daemon.voice.base import WakeEvent
from daemon.voice.wake import WakeGate, match_alias, normalise

FRAME_SAMPLES = 512
"""What Silero was exported for: 512 samples, 32 ms at 16 kHz."""

FRAME_BYTES = FRAME_SAMPLES * 2

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "wake"

# Markers so a test can tell which audio reached the recognizer.
QUIET = 0x00
LEAD = 0x11
"""Frames before the VAD says speech - the pre-roll, in the tests that need one."""
VOICE = 0x22


def frames(marker: int, count: int) -> list[bytes]:
    """`count` distinguishable frames of PCM. Content is arbitrary: the fake VAD is
    scripted rather than measuring anything, which is what lets a test assert *which*
    audio arrived at the recognizer."""
    return [bytes((marker, index % 251)) * FRAME_SAMPLES for index in range(count)]


def script(*runs: tuple[int, float, int]) -> tuple[list[bytes], list[float]]:
    """`(marker, probability, count)` runs -> the audio, and the VAD's answers."""
    blocks: list[bytes] = []
    probabilities: list[float] = []
    for marker, probability, count in runs:
        blocks.extend(frames(marker, count))
        probabilities.extend([probability] * count)
    return blocks, probabilities


# --- fakes at the protocol edges ---------------------------------------------


class FakeAudio:
    """The hardware seam. Ends when its blocks run out, so nothing here can hang."""

    playback_sample_rate = 24_000

    def __init__(
        self,
        blocks: list[bytes],
        *,
        sample_rate: int = 16_000,
        block_bytes: int | None = None,
        fail: BaseException | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self._fail = fail
        self.closed = False
        if block_bytes is None:
            self.blocks = list(blocks)
        else:
            # Re-chunked to a size that does not divide the VAD's frame, which is
            # the real case: the microphone's block is 480 samples and the model's
            # is 512, so they never line up.
            stream = b"".join(blocks)
            self.blocks = [
                stream[at : at + block_bytes] for at in range(0, len(stream), block_bytes)
            ]

    async def record(self) -> AsyncIterator[bytes]:
        for block in self.blocks:
            yield block
        if self._fail is not None:
            raise self._fail

    async def play(self, chunk: bytes) -> None: ...

    async def stop_playback(self) -> None: ...

    async def close(self) -> None:
        self.closed = True


class FakeVad:
    """Scripted speech probabilities, one per frame, in arrival order."""

    frame_samples = FRAME_SAMPLES

    def __init__(
        self,
        probabilities: list[float],
        *,
        sample_rate: int = 16_000,
        tail: float = 0.0,
        raise_on: set[int] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self._probabilities = list(probabilities)
        self._tail = tail
        self._raise_on = raise_on or set()
        self.seen = 0
        self.resets = 0

    def probability(self, frame: bytes) -> float:
        assert len(frame) == FRAME_BYTES, "the real model raises on any other size"
        index = self.seen
        self.seen += 1
        if index in self._raise_on:
            raise RuntimeError("the model threw on a frame")
        if index < len(self._probabilities):
            return self._probabilities[index]
        return self._tail

    def reset(self) -> None:
        self.resets += 1


class EnergyVad:
    """Loudness, for the one test that feeds real recorded speech. Still a fake -
    it is arithmetic on samples, not Silero - but it is not scripted, so frame
    alignment against a real file is genuinely exercised."""

    frame_samples = FRAME_SAMPLES
    sample_rate = 16_000

    def __init__(self, *, floor: int = 900) -> None:
        self.floor = floor
        self.resets = 0

    def probability(self, frame: bytes) -> float:
        samples = memoryview(frame).cast("h")
        peak = max(abs(sample) for sample in samples)
        return min(1.0, peak / (self.floor * 2))

    def reset(self) -> None:
        self.resets += 1


class FakeRecognizer:
    """Returns scripted transcripts, and records exactly what audio it was paid for."""

    def __init__(
        self,
        *texts: str,
        available: bool = True,
        fail_times: int = 0,
        available_raises: bool = False,
    ) -> None:
        self._texts = list(texts)
        self._available = available
        self._available_raises = available_raises
        self._fail_times = fail_times
        self.calls: list[bytes] = []

    @property
    def available(self) -> bool:
        if self._available_raises:
            raise RuntimeError("the speech service could not be asked")
        return self._available

    async def transcribe(self, pcm: bytes) -> str:
        self.calls.append(pcm)
        if self._fail_times > 0:
            self._fail_times -= 1
            raise RuntimeError("the recognizer threw")
        index = len(self.calls) - 1
        if index < len(self._texts):
            return self._texts[index]
        return self._texts[-1] if self._texts else ""


class Clock:
    """A monotonic clock that advances `step` seconds every time it is read. `0.0`
    freezes it, which is how the cooldown is held open."""

    def __init__(self, step: float = 0.0) -> None:
        self.step = step
        self.now = 1000.0

    def __call__(self) -> float:
        self.now += self.step
        return self.now


@dataclass
class Rig:
    """A gate and the two fakes behind it, so a test can assert on either."""

    gate: WakeGate
    audio: FakeAudio
    vad: Any

    async def collect(self) -> list[WakeEvent]:
        """Drain the gate. Bounded, because the fake microphone ends."""
        return [event async for event in self.gate.listen()]


def build(
    blocks: list[bytes],
    probabilities: list[float],
    recognizer: FakeRecognizer,
    *,
    aliases: tuple[str, ...] = ("헤이 대문",),
    vad: Any = None,
    audio: Any = None,
    **knobs: Any,
) -> Rig:
    the_vad = vad if vad is not None else FakeVad(probabilities)
    the_audio = audio if audio is not None else FakeAudio(blocks)
    return Rig(WakeGate(the_audio, the_vad, recognizer, aliases, **knobs), the_audio, the_vad)


# One utterance: a little silence, ~400 ms of speech, then enough silence to close.
ONE_PHRASE = ((LEAD, 0.02, 4), (VOICE, 0.9, 13), (QUIET, 0.02, 25))


# --- the phrase fires, and only the phrase ------------------------------------


async def test_a_matched_alias_fires_exactly_one_event() -> None:
    blocks, probabilities = script(*ONE_PHRASE)
    recognizer = FakeRecognizer("헤이 대문")
    rig = build(blocks, probabilities, recognizer)

    events = await rig.collect()

    assert len(events) == 1
    assert events[0].heard == "헤이 대문"
    assert events[0].matched == "헤이 대문"
    # Mean over the frames the VAD called speech - it says "this was speech", not
    # "these were the words".
    assert events[0].confidence == pytest.approx(0.9)
    assert rig.gate.counters.fired == 1
    assert rig.gate.counters.transcribed == 1


async def test_the_event_carries_the_audio_that_fired_it() -> None:
    """Discarding it cost the owner a whole utterance: the gate consumed
    "루시 뭐 해", matched on the alias, and the session then opened having never heard
    "뭐 해" - so it was said again, into a microphone that had just changed hands.
    Measured on a real run at 14.79s from wake word to first audio out."""
    blocks, probabilities = script(*ONE_PHRASE)
    recognizer = FakeRecognizer("헤이 대문")
    rig = build(blocks, probabilities, recognizer)

    events = await rig.collect()

    assert events[0].pcm, "the audio that fired the gate was thrown away"
    # Exactly what the recognizer was asked about, so the session hears what the
    # match was made on rather than a re-slice of it.
    assert events[0].pcm == recognizer.calls[-1]


async def test_the_wake_phrase_followed_by_a_question_still_fires() -> None:
    """`헤이 데몬, 지금 뭐 하고 있어` is one breath, so the alias arrives with the
    question attached. Requiring the whole transcript to equal an alias would make
    the natural way of speaking the one thing that does not work."""
    blocks, probabilities = script((VOICE, 0.9, 20), (QUIET, 0.02, 25))
    recognizer = FakeRecognizer("헤이 대문 지금 뭐 하고 있어")
    rig = build(blocks, probabilities, recognizer)

    events = await rig.collect()

    assert [event.matched for event in events] == ["헤이 대문"]


async def test_ordinary_korean_speech_fires_nothing() -> None:
    blocks, probabilities = script(*ONE_PHRASE)
    recognizer = FakeRecognizer("오늘 날씨가 참 좋네요")
    rig = build(blocks, probabilities, recognizer)

    events = await rig.collect()

    assert events == []
    # And the second stage really ran: this is a rejection, not a gate that never
    # got as far as looking.
    assert rig.gate.counters.transcribed == 1
    assert rig.gate.counters.fired == 0


async def test_music_the_vad_calls_speech_does_not_open_a_session() -> None:
    """The 46.8% case, and the whole reason the recognizer stage exists. The VAD
    accepts every frame of a chord with vibrato; the transcript is the one a real
    run produced from ambient music, and it is not a wake phrase."""
    blocks, probabilities = script((VOICE, 0.95, 40), (QUIET, 0.02, 25))
    recognizer = FakeRecognizer("nela o trecho da música")
    rig = build(blocks, probabilities, recognizer)

    events = await rig.collect()

    assert events == []
    assert rig.gate.counters.segments_closed == 1, "the VAD did accept it as speech"
    assert rig.gate.counters.transcribed == 1
    assert rig.gate.counters.fired == 0


async def test_an_alias_in_the_middle_of_a_sentence_does_not_open_a_session() -> None:
    """Head-anchored on purpose. The recognizer's rendering of `데몬` is `질문`, an
    ordinary Korean word, so matching anywhere in a sentence would open a paid
    session in the middle of a conversation with somebody else."""
    blocks, probabilities = script((VOICE, 0.9, 20), (QUIET, 0.02, 25))
    recognizer = FakeRecognizer("어제 질문 하나 했잖아")
    rig = build(blocks, probabilities, recognizer, aliases=("질문",))

    assert await rig.collect() == []


# --- matching: spacing, punctuation, Unicode form -----------------------------


@pytest.mark.parametrize(
    ("heard", "aliases", "expected"),
    [
        # Spacing is the recognizer's, not the owner's.
        ("헤이자기", ("헤이 자기",), "헤이 자기"),
        ("헤이 자기", ("헤이자기",), "헤이자기"),
        ("헤이  대문", ("헤이 대문",), "헤이 대문"),
        # Punctuation the owner never said.
        ("헤이 대문, 뭐해?", ("헤이 대문",), "헤이 대문"),
        # NFD Korean, which is what macOS hands out, against an NFC alias.
        (unicodedata.normalize("NFD", "루시야"), ("루시",), "루시"),
        # And the other way round.
        ("루시야", (unicodedata.normalize("NFD", "루시"),), unicodedata.normalize("NFD", "루시")),
        # Latin aliases fold case.
        ("Hey Lucy, 오늘 어때", ("hey lucy",), "hey lucy"),
        # The longest configured alias wins, so `matched` names the phrase said.
        ("헤이 루시야", ("헤이", "헤이 루시"), "헤이 루시"),
        # Non-matches.
        ("오늘 날씨가 참 좋네요", ("헤이 대문",), None),
        ("", ("헤이 대문",), None),
        ("헤이 대문", (), None),
        ("헤이 대문", ("",), None),
    ],
)
async def test_matching_tolerates_spacing_and_unicode_form(
    heard: str, aliases: tuple[str, ...], expected: str | None
) -> None:
    assert match_alias(heard, aliases) == expected


async def test_a_spacing_variant_fires_through_the_whole_gate() -> None:
    """The same tolerance, through `listen` rather than the helper - a matcher that
    is only correct in isolation is the defect class this repo keeps finding."""
    blocks, probabilities = script(*ONE_PHRASE)
    rig = build(blocks, probabilities, FakeRecognizer("헤이자기"), aliases=("헤이 자기",))

    events = await rig.collect()

    assert [event.matched for event in events] == ["헤이 자기"]


async def test_normalise_keeps_letters_and_digits_only() -> None:
    assert normalise(" 헤이 대문, 2번! ") == "헤이대문2번"
    assert normalise(unicodedata.normalize("NFD", "루시")) == "루시"


# --- the pre-roll --------------------------------------------------------------


async def test_the_recognizer_receives_audio_from_before_speech_started() -> None:
    """Without this the first syllable is clipped, and a one-syllable head is the
    whole match: `루시야` heard from its second syllable is not `루시`."""
    blocks, probabilities = script((LEAD, 0.02, 6), (VOICE, 0.9, 13), (QUIET, 0.02, 25))
    recognizer = FakeRecognizer("헤이 대문")
    # 300 ms of pre-roll is 9 frames at 32 ms, more than the 6 quiet frames here, so
    # every one of them must arrive.
    rig = build(blocks, probabilities, recognizer, pre_roll_ms=300)

    await rig.collect()

    assert len(recognizer.calls) == 1
    lead = b"".join(frames(LEAD, 6))
    assert recognizer.calls[0].startswith(lead), (
        "the segment must begin before the VAD said speech, not at the frame it did"
    )


async def test_the_pre_roll_is_bounded_to_what_was_configured() -> None:
    """It is a head, not a recording of the room. One frame of pre-roll must not
    drag in the previous minute."""
    blocks, probabilities = script((LEAD, 0.02, 20), (VOICE, 0.9, 13), (QUIET, 0.02, 25))
    recognizer = FakeRecognizer("헤이 대문")
    rig = build(blocks, probabilities, recognizer, pre_roll_ms=64)  # 2 frames

    await rig.collect()

    # 2 pre-roll + 13 speech + the hangover that closed the segment.
    assert len(recognizer.calls[0]) < len(b"".join(blocks))
    assert recognizer.calls[0].startswith(b"".join(frames(LEAD, 20)[-2:]))


# --- bounds --------------------------------------------------------------------


async def test_a_blip_never_reaches_the_recognizer() -> None:
    blocks, probabilities = script((LEAD, 0.02, 3), (VOICE, 0.9, 2), (QUIET, 0.02, 25))
    recognizer = FakeRecognizer("헤이 대문")
    rig = build(blocks, probabilities, recognizer, min_speech_ms=200)

    events = await rig.collect()

    assert events == []
    assert recognizer.calls == [], "64 ms of speech is not a wake word"
    assert rig.gate.counters.skipped_short == 1
    assert rig.gate.counters.transcribed == 0


async def test_a_monologue_is_cut_into_bounded_segments() -> None:
    """A minute of talking must not accumulate a minute of audio in memory, and must
    not arrive at the recognizer as one blob."""
    blocks, probabilities = script((VOICE, 0.9, 300), (QUIET, 0.02, 25))
    recognizer = FakeRecognizer("오늘 회사에서 있었던 일 얘기해 줄게")
    rig = build(
        blocks, probabilities, recognizer, max_segment_ms=640, pre_roll_ms=64
    )  # 20 frames, 2 frames

    events = await rig.collect()

    assert events == []
    assert len(recognizer.calls) >= 10, "the cap must cut, not wait for silence"
    assert max(len(call) for call in recognizer.calls) <= 20 * FRAME_BYTES
    assert rig.gate.counters.fired == 0


async def test_a_segment_cut_by_the_cap_leaves_a_head_on_the_next_one() -> None:
    """The cap can land in the middle of a word. The next segment is seeded from the
    pre-roll for the same reason a first segment is: a phrase whose head was cut off
    matches nothing, and the cut is ours rather than the speaker's."""
    blocks, probabilities = script((VOICE, 0.9, 60), (QUIET, 0.02, 25))
    recognizer = FakeRecognizer("계속 얘기하는 중")
    rig = build(blocks, probabilities, recognizer, max_segment_ms=640, pre_roll_ms=64)

    await rig.collect()

    assert len(recognizer.calls) >= 2
    tail = recognizer.calls[0][-2 * FRAME_BYTES :]
    assert tail in recognizer.calls[1][: 4 * FRAME_BYTES], (
        "the audio either side of the cut must overlap, or a phrase the cap split is lost"
    )


async def test_a_wake_phrase_after_a_long_monologue_still_fires() -> None:
    """The cap flushes the segment rather than dropping the stream, so the gate is
    still listening when the owner finally calls it."""
    blocks, probabilities = script(
        (VOICE, 0.9, 60), (QUIET, 0.02, 25), (VOICE, 0.9, 13), (QUIET, 0.02, 25)
    )
    recognizer = FakeRecognizer("계속 얘기하는 중", "계속 얘기하는 중", "헤이 대문")
    rig = build(blocks, probabilities, recognizer, max_segment_ms=960, pre_roll_ms=64)

    events = await rig.collect()

    assert [event.matched for event in events] == ["헤이 대문"]


# --- the cooldown --------------------------------------------------------------


async def test_the_cooldown_suppresses_a_second_immediate_fire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One `루시야` must not open two sessions - and the second segment must not even
    cost a transcription."""
    monkeypatch.setattr(wake, "_monotonic", Clock(step=0.0))
    blocks, probabilities = script(*ONE_PHRASE, *ONE_PHRASE)
    recognizer = FakeRecognizer("루시", "루시")
    rig = build(blocks, probabilities, recognizer, aliases=("루시",), cooldown_seconds=5.0)

    events = await rig.collect()

    assert len(events) == 1
    assert rig.gate.counters.segments_closed == 2
    assert len(recognizer.calls) == 1, "the suppressed segment must not be transcribed"
    assert rig.gate.counters.skipped_cooldown == 1


async def test_a_call_after_the_cooldown_fires_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """A window, not a one-shot. A gate that fired once per process would be worse
    than one that fired twice."""
    monkeypatch.setattr(wake, "_monotonic", Clock(step=60.0))
    blocks, probabilities = script(*ONE_PHRASE, *ONE_PHRASE)
    recognizer = FakeRecognizer("루시", "루시")
    rig = build(blocks, probabilities, recognizer, aliases=("루시",), cooldown_seconds=5.0)

    events = await rig.collect()

    assert len(events) == 2
    assert rig.gate.counters.skipped_cooldown == 0


# --- it must not die -----------------------------------------------------------


async def test_an_unavailable_recognizer_skips_transcription_entirely() -> None:
    """Absence is normal - an OS speech service can be missing, unauthorised or
    lack the locale - and it must be reported rather than guessed at."""
    blocks, probabilities = script(*ONE_PHRASE)
    recognizer = FakeRecognizer("헤이 대문", available=False)
    rig = build(blocks, probabilities, recognizer)

    events = await rig.collect()

    assert events == []
    assert recognizer.calls == [], "an unavailable recognizer must not be called"
    assert rig.gate.counters.skipped_unavailable == 1
    assert rig.gate.counters.frames_seen > 0, "the gate itself is still alive"


async def test_a_recognizer_that_cannot_say_whether_it_is_available_is_survived() -> None:
    blocks, probabilities = script(*ONE_PHRASE)
    recognizer = FakeRecognizer("헤이 대문", available_raises=True)
    rig = build(blocks, probabilities, recognizer)

    assert await rig.collect() == []
    assert rig.gate.counters.skipped_unavailable == 1
    assert rig.gate.counters.errors == 1


async def test_a_stream_that_delivers_nothing_at_all_is_dead_too() -> None:
    """The other face of a dead capture, and the one the zero-frame detector cannot
    see: the callback never fires, the queue stays empty, and the gate parks on a
    block that never comes - `/health` said `running` while a launchd resident heard
    nothing (measured live). The watchdog on the wait is what turns that into a
    bounded deaf spell."""

    class Mute(FakeAudio):
        async def record(self) -> AsyncIterator[bytes]:
            yield b""  # opens fine, then never delivers again
            await asyncio.Event().wait()

    rig = build([], [], FakeRecognizer(), audio=Mute([]), dead_stream_ms=150)

    events = await rig.collect()

    assert events == []
    assert rig.gate.counters.frames_seen == 0, "nothing ever arrived - and the gate still ended"


async def test_a_stream_that_goes_mute_mid_listen_is_dead_too() -> None:
    """Same watchdog, after real audio has flowed: a stream that stops delivering is
    as dead as one that never started."""

    class GoesMute(FakeAudio):
        async def record(self) -> AsyncIterator[bytes]:
            for block in self.blocks:
                yield block
            await asyncio.Event().wait()

    blocks, probabilities = script(*ONE_PHRASE)
    audio = GoesMute(blocks)
    rig = build(blocks, probabilities, FakeRecognizer("루시"), aliases=("루시",), audio=audio,
                dead_stream_ms=150)

    events = await rig.collect()

    assert [event.matched for event in events] == ["루시"], "the phrase before death still fired"
    assert rig.gate.counters.frames_seen == len(blocks)


async def test_an_all_zero_stream_is_treated_as_dead_and_ends_the_gate() -> None:
    """A capture opened right after a voice session has come up dead: all-zero blocks,
    no exception, so the `except` that catches a vanished device never fires and the
    gate stays `running` and permanently deaf (measured live). It is caught here
    instead, as impossibly perfect silence, and the gate ends so daemon/app.py rebuilds
    the stream rather than the owner having to restart the process."""
    # 15 frames of pure zero; the threshold below is 10 (320 ms / 32 ms).
    dead = b"\x00" * (FRAME_BYTES * 15)
    rig = build([dead], [], FakeRecognizer(), dead_stream_ms=320)

    events = await rig.collect()

    assert events == []
    # Ended at the threshold rather than draining all 15 frames: it bailed early.
    assert rig.gate.counters.frames_seen == 10


async def test_a_quiet_but_live_stream_is_not_mistaken_for_dead() -> None:
    """A real microphone in a silent room still has a nonzero noise floor, so a single
    nonzero sample per frame keeps the gate listening. Only *exact* zero across the
    whole window is a dead stream; a quiet house must never trip it."""
    floor = b"\x01\x00" * FRAME_SAMPLES  # every sample == 1: quiet, but never zero
    rig = build([floor * 15], [], FakeRecognizer(), dead_stream_ms=320)

    events = await rig.collect()

    assert events == []
    assert rig.gate.counters.frames_seen == 15, "a live-but-quiet stream is read to the end"


async def test_a_raising_recognizer_leaves_the_loop_alive() -> None:
    """A dead listener leaves a process that is alive and permanently deaf."""
    blocks, probabilities = script(*ONE_PHRASE, *ONE_PHRASE)
    recognizer = FakeRecognizer("헤이 대문", "헤이 대문", fail_times=1)
    rig = build(blocks, probabilities, recognizer, cooldown_seconds=0.0)

    events = await rig.collect()

    assert [event.heard for event in events] == ["헤이 대문"], "the later call still fires"
    assert rig.gate.counters.errors == 1
    assert rig.gate.counters.transcribed == 1


async def test_a_raising_vad_leaves_the_loop_alive() -> None:
    """The VAD is stateful and can throw on a frame. Those frames count as silence -
    the honest reading - and the next phrase is still heard."""
    blocks, probabilities = script((VOICE, 0.9, 6), (QUIET, 0.02, 25), *ONE_PHRASE)
    recognizer = FakeRecognizer("헤이 대문")
    vad = FakeVad(probabilities, raise_on={0, 1, 2, 3, 4, 5})
    rig = build(blocks, probabilities, recognizer, vad=vad, cooldown_seconds=0.0)

    events = await rig.collect()

    assert [event.matched for event in events] == ["헤이 대문"]
    assert rig.gate.counters.errors == 6


async def test_a_vad_that_cannot_reset_is_survived() -> None:
    class UnresettableVad(FakeVad):
        def reset(self) -> None:
            raise RuntimeError("the model would not reset")

    blocks, probabilities = script(*ONE_PHRASE)
    recognizer = FakeRecognizer("헤이 대문")
    rig = build(blocks, probabilities, recognizer, vad=UnresettableVad(probabilities))

    assert len(await rig.collect()) == 1
    assert rig.gate.counters.errors >= 1


async def test_the_vad_is_reset_between_segments() -> None:
    """Its state is an LSTM plus 64 samples of context, and the tail of one segment
    biases the head of the next - quietly, with plausible probabilities near zero."""
    blocks, probabilities = script(*ONE_PHRASE, *ONE_PHRASE)
    rig = build(blocks, probabilities, FakeRecognizer("오늘 뭐 했어"))

    await rig.collect()

    assert rig.vad.resets >= 2, "once per closed segment, plus the start of the stream"


async def test_a_microphone_that_fails_is_reported_rather_than_pretended_away() -> None:
    """The one thing not swallowed: a device that has gone away is fixed by plugging
    it back in, and a gate that retried forever would report itself healthy while
    hearing nothing."""
    blocks, probabilities = script(*ONE_PHRASE)
    audio = FakeAudio(blocks, fail=OSError("the input device disappeared"))
    rig = build(blocks, probabilities, FakeRecognizer("헤이 대문"), audio=audio)

    with pytest.raises(OSError, match="input device disappeared"):
        await rig.collect()


async def test_a_vad_at_the_wrong_rate_is_refused_at_construction() -> None:
    """A VAD fed at the wrong sample rate returns plausible probabilities near zero
    for real speech, so the gate would hear nothing and nothing would say why. The
    two rates are the one pair a caller can mismatch: `AudioIO` has two of its own
    (16 kHz in, 24 kHz out) and the wrong field is right there to pass."""
    with pytest.raises(ValueError, match="24000 Hz"):
        WakeGate(FakeAudio([]), FakeVad([], sample_rate=24_000), FakeRecognizer(), ("루시",))


# --- frames, blocks and counters ----------------------------------------------


async def test_microphone_blocks_that_do_not_align_with_the_vad_frame_still_fire() -> None:
    """The microphone's block is 480 samples and the model's frame is 512, so they
    never line up. The VAD raises on any other size, which makes this silent-failure
    territory: `FakeVad` asserts the size it is handed."""
    blocks, probabilities = script(*ONE_PHRASE)
    audio = FakeAudio(blocks, block_bytes=960)  # 30 ms at 16 kHz, as sounddevice sends
    recognizer = FakeRecognizer("헤이 대문")
    rig = build(blocks, probabilities, recognizer, audio=audio)

    events = await rig.collect()

    assert [event.matched for event in events] == ["헤이 대문"]


async def test_an_odd_length_block_does_not_desynchronise_the_stream() -> None:
    blocks, probabilities = script(*ONE_PHRASE)
    audio = FakeAudio(blocks, block_bytes=777)
    rig = build(blocks, probabilities, FakeRecognizer("헤이 대문"), audio=audio)

    assert len(await rig.collect()) == 1


async def test_the_counters_report_what_happened() -> None:
    """`daemon doctor` has to tell "nobody said anything" from "we have been deaf
    since Tuesday", and neither raises."""
    blocks, probabilities = script(
        (VOICE, 0.9, 2),  # a blip: too short
        (QUIET, 0.02, 25),
        (VOICE, 0.9, 13),  # music the recognizer rejects
        (QUIET, 0.02, 25),
        (VOICE, 0.9, 13),  # the wake phrase
        (QUIET, 0.02, 25),
    )
    recognizer = FakeRecognizer("nela o trecho da música", "헤이 대문")
    rig = build(blocks, probabilities, recognizer, cooldown_seconds=0.0)

    await rig.collect()

    counters = rig.gate.counters
    assert counters.frames_seen == len(blocks)
    assert counters.segments_closed == 3
    assert counters.skipped_short == 1
    assert counters.transcribed == 2
    assert counters.fired == 1
    assert counters.skipped_cooldown == 0
    assert counters.skipped_unavailable == 0
    assert counters.errors == 0


async def test_nothing_at_all_costs_nothing() -> None:
    """A silent room: the VAD runs forever for 0.49% of a core, and nothing else
    happens. No transcription, no session, no error."""
    blocks, probabilities = script((QUIET, 0.02, 200))
    recognizer = FakeRecognizer("헤이 대문")
    rig = build(blocks, probabilities, recognizer)

    assert await rig.collect() == []
    assert rig.gate.counters.frames_seen == 200
    assert rig.gate.counters.segments_closed == 0
    assert recognizer.calls == []


async def test_the_consumer_can_stop_the_gate_by_stopping_the_iteration() -> None:
    """An async generator, like `Channel.listen`: closing the iteration is how the
    caller stops listening, and it must not hang."""
    blocks, probabilities = script(*ONE_PHRASE, *ONE_PHRASE)
    recognizer = FakeRecognizer("루시", "루시")
    rig = build(blocks, probabilities, recognizer, aliases=("루시",), cooldown_seconds=0.0)

    stream = rig.gate.listen()
    first = await anext(stream)
    await stream.aclose()

    assert first.matched == "루시"
    # And nothing is left running that would have kept consuming audio.
    assert rig.gate.counters.fired == 1


# --- real recorded Korean, with a fake recognizer -----------------------------


def read_wav(name: str) -> bytes:
    with wave.open(str(FIXTURES / name), "rb") as handle:
        assert handle.getframerate() == 16_000, "the fixture is what AudioIO captures"
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        return handle.readframes(handle.getnframes())


@pytest.mark.parametrize(
    ("name", "heard", "fires"),
    [
        # What the on-device recognizer actually returned for these clips, measured
        # three times each: `헤이 데몬` -> `헤이 대문`.
        ("wake-alone.wav", "헤이 대문", True),
        ("wake-and-question.wav", "헤이 대문 지금 뭐 하고 있어", True),
        ("no-wake-word.wav", "오늘 날씨가 참 좋네요", False),
    ],
)
async def test_real_korean_speech_through_an_energy_vad(
    name: str, heard: str, fires: bool
) -> None:
    """Real 16 kHz speech rather than scripted probabilities, so the frame
    arithmetic, the odd tail of a real file and the segment logic are exercised
    against audio nobody tuned. The recognizer is still a fake: the measured
    transcript is data, not something a test may go and ask a machine for."""
    pcm = read_wav(name) + b"\x00" * (FRAME_BYTES * 25)  # trailing silence closes it
    audio = FakeAudio([pcm], block_bytes=960)
    recognizer = FakeRecognizer(heard)
    rig = Rig(WakeGate(audio, EnergyVad(), recognizer, ("헤이 대문",)), audio, None)

    events = await rig.collect()

    assert rig.gate.counters.segments_closed >= 1, "the VAD found speech in real speech"
    assert recognizer.calls, "and it was long enough to be worth transcribing"
    assert bool(events) is fires
    if fires:
        assert events[0].heard == heard
        assert 0.0 < events[0].confidence <= 1.0


# --- the seam the product actually uses ---------------------------------------


def app_module() -> Any:
    """Imported inside the tests that use it, like everything else that reaches the
    composition root: `daemon/app.py` is the only file allowed to import an
    implementation, and importing it at module scope would drag the voice extra into
    every test in this file."""
    from daemon import app

    return app


def fake_machine(monkeypatch: pytest.MonkeyPatch) -> tuple[FakeRecognizer, FakeAudio]:
    """Swap the three implementations `build_wake_gate` imports for fakes.

    Patched where they are *defined*, not where they are used: the imports are inside
    the function, so the names resolve when it runs. No microphone, no model file, no
    speech service."""
    from daemon.voice import apple_speech
    from daemon.voice import audio as audio_module
    from daemon.voice import vad as vad_module

    blocks, probabilities = script(*ONE_PHRASE)
    recognizer = FakeRecognizer("루시")
    microphone = FakeAudio(blocks)
    monkeypatch.setattr(vad_module, "SileroVad", lambda: FakeVad(probabilities))
    monkeypatch.setattr(apple_speech, "AppleSpeechRecognizer", lambda: recognizer)
    monkeypatch.setattr(audio_module, "SoundDeviceAudio", lambda: microphone)
    return recognizer, microphone


async def test_the_app_seam_builds_a_gate_that_fires(monkeypatch: pytest.MonkeyPatch) -> None:
    """`app.build_wake_gate` is the only place a real gate is constructed, and
    `daemon wake test` is its consumer. Driven here with fakes swapped in for the
    three implementations, because `tests/test_reachable.py` can only see *that*
    something calls `WakeGate(...)` - never whether it calls it with arguments the
    class actually has. It did not, briefly, while this module and `app.py` were
    written in parallel, and nothing failed anywhere except at runtime.

    The implementations are patched where they are defined rather than where they are
    used: `build_wake_gate` imports them inside the function, so the names resolve
    when it runs. No microphone, no model file, no speech service.
    """
    recognizer, microphone = fake_machine(monkeypatch)
    settings = voice_settings(DAEMON_WAKE_ENABLED="true", DAEMON_WAKE_ALIASES="루시")

    gate, close = await app_module().build_wake_gate(settings)
    events = [event async for event in gate.listen()]
    await close()

    assert [event.matched for event in events] == ["루시"]
    assert recognizer.calls, "the segment reached the recognizer"
    assert microphone.closed, "the closer app hands back is what releases the device"


async def test_the_app_seam_passes_the_tuning_settings_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config validates six numbers beyond the aliases, and .env.example documents
    them. A gate built with only the aliases would leave all six as documented
    no-ops - the owner tunes a value and nothing reads it, which is this project's
    signature failure with a knob attached.

    Asserted through behaviour rather than by reading the gate's fields: a minimum
    longer than the phrase must stop the phrase, un-transcribed."""
    recognizer, microphone = fake_machine(monkeypatch)
    settings = voice_settings(
        DAEMON_WAKE_ENABLED="true",
        DAEMON_WAKE_ALIASES="루시",
        DAEMON_WAKE_MIN_SPEECH_MS="2000",
        DAEMON_WAKE_MAX_SEGMENT_MS="5000",
    )

    gate, close = await app_module().build_wake_gate(settings)
    events = [event async for event in gate.listen()]
    await close()

    assert events == []
    assert recognizer.calls == [], "a 2 s minimum must drop a 416 ms phrase unheard"
    assert gate.counters.skipped_short == 1


# --- configuration a person can actually write --------------------------------


def wake_settings(**kwargs: Any) -> Settings:
    """The offline preset, so no case here needs a key. Voice stays off - anything
    that turns it on wants `voice_settings` below."""
    return Settings(
        _env_file=None,
        DAEMON_PRESET="offline",
        DAEMON_OLLAMA_MODEL="gemma3:4b",
        DAEMON_DATA_DIR="/tmp/daemon-wake",
        **kwargs,
    )


def voice_settings(**kwargs: Any) -> Settings:
    """A configuration where voice is genuinely available.

    `offline` deliberately routes no voice task - that absence is what makes the
    privacy claim in docs/PLAN.md 7 true - so turning voice on there fails for a
    reason that has nothing to do with the wake gate, and a test asserting a wake
    message would pass on somebody else's error."""
    return Settings(
        _env_file=None,
        DAEMON_PRESET="balanced",
        DAEMON_HOSTED_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="k",
        GEMINI_API_KEY="k",
        DAEMON_OLLAMA_MODEL="gemma3:4b",
        DAEMON_DATA_DIR="/tmp/daemon-wake",
        DAEMON_VOICE_ENABLED="true",
        DAEMON_GEMINI_LIVE_MODEL="m",
        **kwargs,
    )


@pytest.mark.parametrize("source", ["keyword", "environment"])
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("헤이 대문", ("헤이 대문",)),
        ("헤이 대문,루씨", ("헤이 대문", "루씨")),
        ("헤이 대문, 루씨", ("헤이 대문", "루씨")),
        ("  헤이 대문 ,, 루씨  ", ("헤이 대문", "루씨")),
        ('["헤이 대문", "루씨"]', ("헤이 대문", "루씨")),
        ("", ()),
    ],
)
def test_the_alias_list_accepts_what_a_person_would_type(
    source: str, raw: str, expected: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same trap as TELEGRAM_ALLOWED_USER_IDS: pydantic-settings JSON-decodes a
    complex field before field validators run, so without NoDecode a plain comma list
    raises and the documented form is impossible to write.

    **Both delivery paths, because only one of them has the trap in it.** Passing the
    value as a keyword argument skips the decoding entirely, so a kwarg-only test
    passes with NoDecode removed - measured, by removing it. `.env` and the process
    environment are how a person actually sets this, and that path raises
    `SettingsError` before any validator here runs.

    And unlike the numeric allowlist, splitting on whitespace would be wrong:
    `헤이 대문` is one phrase, and two wake words neither of which was ever said is a
    gate that never fires with nothing anywhere saying why.
    """
    if source == "environment":
        monkeypatch.setenv("DAEMON_WAKE_ALIASES", raw)
        settings = wake_settings()
    else:
        settings = wake_settings(DAEMON_WAKE_ALIASES=raw)

    assert settings.wake_aliases == expected


def test_the_wake_gate_is_off_by_default() -> None:
    settings = wake_settings()
    assert settings.wake_enabled is False
    assert settings.wake_aliases == ()


def test_wake_enabled_without_aliases_says_to_calibrate() -> None:
    """And names the command that fixes it. `daemon wake calibrate` is the only way to
    learn the strings that work, because they depend on the owner's voice."""
    with pytest.raises(ConfigError, match="daemon wake calibrate"):
        voice_settings(DAEMON_WAKE_ENABLED="true")


def test_wake_enabled_without_voice_says_which_switch_to_flip() -> None:
    """The gate exists only to open a voice session. On with voice off, it would
    listen forever and have nowhere to hand a match."""
    with pytest.raises(ConfigError, match="DAEMON_VOICE_ENABLED"):
        wake_settings(DAEMON_WAKE_ENABLED="true", DAEMON_WAKE_ALIASES="루씨")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"DAEMON_WAKE_VAD_THRESHOLD": "0"}, "every frame of silence is speech"),
        ({"DAEMON_WAKE_VAD_THRESHOLD": "1.5"}, "within"),
        ({"DAEMON_WAKE_HANGOVER_MS": "0"}, "greater than 0"),
        ({"DAEMON_WAKE_MIN_SPEECH_MS": "-1"}, "greater than 0"),
        ({"DAEMON_WAKE_PRE_ROLL_MS": "-1"}, "cannot be negative"),
        ({"DAEMON_WAKE_COOLDOWN_SECONDS": "-1"}, "cannot be negative"),
        # A cap below the pre-roll plus the minimum cuts every segment short of the
        # length that would have made it worth transcribing.
        ({"DAEMON_WAKE_MAX_SEGMENT_MS": "400"}, "leaves no room"),
    ],
)
def test_incoherent_wake_settings_fail_at_startup(
    kwargs: dict[str, str], message: str
) -> None:
    """Not at the first wake word. Every one of these produces a gate that looks
    configured and hears nothing."""
    with pytest.raises(ConfigError, match=message):
        wake_settings(**kwargs)


def test_the_settings_defaults_are_the_ones_the_gate_uses() -> None:
    """Two literals for each number, because config.py is foundation and may not
    import the voice layer. Drift between them is a real defect: the owner would
    tune a value that nothing reads."""
    settings = wake_settings()
    assert settings.wake_vad_threshold == wake.DEFAULT_THRESHOLD
    assert settings.wake_hangover_ms == wake.DEFAULT_HANGOVER_MS
    assert settings.wake_pre_roll_ms == wake.DEFAULT_PRE_ROLL_MS
    assert settings.wake_min_speech_ms == wake.DEFAULT_MIN_SPEECH_MS
    assert settings.wake_max_segment_ms == wake.DEFAULT_MAX_SEGMENT_MS
    assert settings.wake_cooldown_seconds == wake.DEFAULT_COOLDOWN_SECONDS
