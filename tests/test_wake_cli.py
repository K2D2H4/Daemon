"""`daemon wake calibrate` and `daemon wake test`.

No microphone, no OS speech service, no gate and no real `.env` - CONTRACTS forbids
all four, and the three seams in `wake_cli.Devices` exist so that none of them is
needed: audio, the recognizer and the stream of wake events are all faked at the
protocol boundary in `daemon/voice/base.py`.

Korean throughout, because the finding this feature is built on is Korean: the
on-device recognizer never returns a coined name, so what gets stored is what it
actually said, and two of the things that can go wrong with that - Unicode
normalisation and display width - only go wrong in Korean.
"""

from __future__ import annotations

import asyncio
import io
import logging
import unicodedata
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from daemon import cli, wake_cli
from daemon.config import Settings
from daemon.voice.base import WakeEvent
from daemon.voice.wake import WakeCounters

# --- fakes -------------------------------------------------------------------

AUDIO = b"WAKEAUDIO"
"""The stand-in for microphone PCM.

Deliberately printable. Real PCM is arbitrary bytes, so a leak of it into a log or
onto the terminal would be hard to search for; this way "the owner's audio must not
be printed anywhere" is one `in` check per surface."""

BLOCK = AUDIO * 100
"""900 bytes, roughly the 480-frame block `SoundDeviceAudio` yields."""

HEARD = "헤이 대문"
"""What `say -v Yuna` saying 헤이 데몬 actually came back as, three runs out of
three (daemon/wake_cli.py's table). The wrong words and the right test data."""


@dataclass
class FakeAudio:
    """`AudioIO`, with a fixed number of blocks and no hardware."""

    blocks: list[bytes] = field(default_factory=lambda: [BLOCK])
    sample_rate: int = 16_000
    playback_sample_rate: int = 24_000
    opened: int = 0
    closed: int = 0

    async def record(self) -> AsyncIterator[bytes]:
        self.opened += 1
        for block in self.blocks:
            yield block

    async def play(self, chunk: bytes) -> None:  # pragma: no cover - never played
        raise AssertionError("calibration must not play anything")

    async def stop_playback(self) -> None:  # pragma: no cover
        raise AssertionError("calibration must not play anything")

    async def close(self) -> None:
        self.closed += 1

    async def wait_for_input_release(self, within: float | None = None) -> bool:
        # No detached release to wait for without a real PortAudio stream, so the
        # device is already back. Present rather than absent on purpose: the gate's
        # closer calls this unguarded, so a fake missing it fails the round loudly
        # instead of quietly reporting a microphone it never checked.
        return True


@dataclass
class FakeRecognizer:
    """`SpeechRecognizer`, scripted per take."""

    heard: Sequence[str] = (HEARD,)
    available: bool = True
    calls: list[int] = field(default_factory=list)
    """Bytes handed to each `transcribe` call - i.e. how much audio was recorded."""

    async def transcribe(self, pcm: bytes) -> str:
        self.calls.append(len(pcm))
        index = min(len(self.calls), len(self.heard)) - 1
        return self.heard[index]


@dataclass
class FakeGate:
    """A `WakeSource`: the events it fires, and the tally it kept.

    `WakeCounters` is the real one from `daemon/voice/wake.py`, deliberately. It is
    what makes `wake_cli.GateCounters` more than a hopeful shape - the module states
    the fields it prints and cannot import them, so a rename there has to fail here.
    """

    fires: Sequence[WakeEvent] = ()
    ends: bool = False
    """True means the gate's own iterator finishes, which is the failure
    `daemon wake test` exists to catch: a process left alive and permanently deaf.
    False waits forever after the last event, which is what a healthy gate does -
    only the caller's timeout ends listening."""
    counters: WakeCounters = field(default_factory=WakeCounters)
    closed: int = 0

    async def listen(self) -> AsyncIterator[WakeEvent]:
        for event in self.fires:
            self.counters.fired += 1
            yield event
        if not self.ends:
            await asyncio.Event().wait()

    async def close(self) -> None:
        self.closed += 1


def build_gate(gate: FakeGate | None = None, *, raises: Exception | None = None) -> Any:
    """A `Devices.gate` factory: hand back this gate, or fail the way a machine
    without a VAD model or without macOS fails."""

    async def build(settings: Settings) -> tuple[FakeGate, Any]:
        if raises is not None:
            raise raises
        built = gate if gate is not None else FakeGate(ends=True)
        return built, built.close

    return build


def devices_for(
    audio: FakeAudio | None = None,
    recognizer: FakeRecognizer | None = None,
    gate: Any = None,
) -> wake_cli.Devices:
    kit = audio if audio is not None else FakeAudio()
    ears = recognizer if recognizer is not None else FakeRecognizer()
    return wake_cli.Devices(
        recognizer=lambda: ears,
        audio=lambda: kit,
        gate=gate if gate is not None else build_gate(),
    )


# --- harness -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Run:
    code: int
    out: str
    env_path: Path

    @property
    def written(self) -> str:
        return self.env_path.read_text(encoding="utf-8") if self.env_path.exists() else ""


def drive(
    tmp_path: Path,
    answers: Sequence[str],
    *,
    devices: wake_cli.Devices,
    existing: str | None = None,
    takes: int | None = None,
) -> Run:
    """Run a whole calibration against `answers`, one per prompt.

    `stdout` is a plain `StringIO`, which is not a terminal - so the `Theme` built
    from it cannot emit colour and every assertion below reads text rather than
    escape sequences.
    """
    env_path = tmp_path / ".env"
    if existing is not None:
        env_path.write_text(existing, encoding="utf-8")
    out = io.StringIO()
    code = wake_cli.calibrate(
        env_path=env_path,
        takes=takes,
        stdin=io.StringIO("".join(f"{answer}\n" for answer in answers)),
        stdout=out,
        devices=devices,
    )
    return Run(code, out.getvalue(), env_path)


def answers_for(takes: int = 3, *, enable: str | None = None, write: str = "y") -> list[str]:
    """Every prompt one calibration asks, in order: the phrase, one Enter per take,
    and the write itself.

    `enable` is the switch question, and it is only asked when voice is already on -
    `Settings` refuses the gate on with voice off, so there is nothing to offer. A
    run that answered it anyway would walk one prompt out of step, which is why it
    is `None` here rather than a harmless extra line.
    """
    return ["", *[""] * takes, *([] if enable is None else [enable]), write]


VOICE_OFF = "DAEMON_PROVIDER=ollama\nDAEMON_OLLAMA_MODEL=gemma3:4b\n"
"""A `.env` that loads and has voice off - the state a first calibration is in."""

VOICE_ON = (
    "DAEMON_PROVIDER=gemini\n"
    "GEMINI_API_KEY=AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKE0000\n"
    "DAEMON_GEMINI_MODEL=gemini-2.5-flash\n"
    "DAEMON_GEMINI_LIVE_MODEL=gemini-live-2.5-flash-preview\n"
    "DAEMON_VOICE_ENABLED=true\n"
)
"""One that loads with voice on, which is the only state in which the gate may be
switched on at all. The key is a fake of the right shape and is never sent
anywhere - nothing in this module makes a network call."""


def loads(env_path: Path) -> Settings:
    """The file, through the same validation `daemon run` does at startup.

    The point of asserting this at all: `DAEMON_WAKE_ENABLED` is refused alongside
    voice off and alongside an empty alias list, so a calibration that wrote the
    switch at the wrong moment would leave a `.env` that no longer starts - and the
    command would have looked like it succeeded.
    """
    return Settings(_env_file=env_path)


def flat(rendered: str) -> str:
    """Output with the layout taken back out.

    Same helper and same reason as `tests/test_setup.py`: everything here goes
    through `tui`, so it comes back wrapped to the width - and an assertion about
    wording must not secretly be an assertion about column 80.
    """
    return " ".join(rendered.split())


def value_of(written: str, key: str) -> str:
    from daemon.setup import parse_env

    return parse_env(written).get(key, "")


def settings_for(tmp_path: Path, *, aliases: str = "헤이 대문,루시") -> Settings:
    """A configuration that loads without a key, for the one command that needs one.

    Aliases by default, because `daemon wake test` with an empty list can only ever
    end in "nothing fired" - so a suite that left them out was testing the one state
    the command now refuses to enter, and would have gone on passing if the gate had
    stopped comparing against the list at all.
    """
    return Settings(
        _env_file=None,
        DAEMON_PROVIDER="ollama",
        DAEMON_OLLAMA_MODEL="gemma3:4b",
        DAEMON_DATA_DIR=str(tmp_path / "data"),
        DAEMON_WAKE_ALIASES=aliases,
    )


@pytest.fixture(autouse=True)
def _quick_takes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three seconds per take is right for a person and absurd in a suite.

    Both constants are read at call time precisely so this is possible without
    reaching inside a coroutine. The byte bound is what actually ends a take here -
    every fake runs out of blocks first - so the clock only matters to the one test
    that is about a device delivering nothing.
    """
    monkeypatch.setattr(wake_cli, "TAKE_SECONDS", 0.05)
    monkeypatch.setattr(wake_cli, "RECORD_GRACE", 0.05)


# --- how many takes, and what they produce ------------------------------------


def test_calibration_records_the_requested_number_of_takes(tmp_path: Path) -> None:
    audio, ears = FakeAudio(), FakeRecognizer(heard=[HEARD] * 3)
    result = drive(tmp_path, answers_for(3), devices=devices_for(audio, ears))

    assert result.code == 0
    assert len(ears.calls) == 3, "one transcription per take"
    assert audio.opened == 3
    assert audio.closed == 3, "the microphone is released after every take, not at the end"
    assert "Take 3 of 3" in result.out


def test_takes_is_configurable_and_one_take_is_called_out_as_no_evidence(
    tmp_path: Path,
) -> None:
    ears = FakeRecognizer(heard=[HEARD])
    result = drive(tmp_path, answers_for(1), devices=devices_for(recognizer=ears), takes=1)

    assert len(ears.calls) == 1
    assert "one take cannot show stability" in flat(result.out)


def test_distinct_transcriptions_become_distinct_aliases(tmp_path: Path) -> None:
    """The point of the whole command: what it hears is not one string.

    Both spellings are real - a recognizer that returns 헤이 대문 twice and 헤이
    데문 once will do it again, so both have to be in the list or the gate misses
    one time in three.
    """
    ears = FakeRecognizer(heard=["헤이 대문", "헤이 데문", "헤이 대문"])
    result = drive(tmp_path, answers_for(3), devices=devices_for(recognizer=ears))

    assert result.code == 0
    assert value_of(result.written, "DAEMON_WAKE_ALIASES") == "헤이 대문,헤이 데문"


def test_identical_transcriptions_collapse_to_one_alias(tmp_path: Path) -> None:
    ears = FakeRecognizer(heard=[HEARD] * 3)
    result = drive(tmp_path, answers_for(3), devices=devices_for(recognizer=ears))

    assert value_of(result.written, "DAEMON_WAKE_ALIASES") == HEARD
    assert "stable" in flat(result.out)


def test_two_normal_forms_of_the_same_korean_word_are_one_alias(tmp_path: Path) -> None:
    """The failure this would be without normalisation is silent and Korean-only.

    macOS hands back decomposed Hangul from some APIs and composed from others. The
    two strings look identical on screen, compare unequal, and would be stored as
    two aliases - one of which nobody can type and the gate can only match by luck.
    """
    decomposed = unicodedata.normalize("NFD", HEARD)
    assert decomposed != HEARD, "the fixture only means something if the two differ"

    ears = FakeRecognizer(heard=[HEARD, decomposed, HEARD])
    result = drive(tmp_path, answers_for(3), devices=devices_for(recognizer=ears))

    assert value_of(result.written, "DAEMON_WAKE_ALIASES") == HEARD
    assert "stable" in flat(result.out)


# --- when the phrase is a bad one --------------------------------------------


def test_an_unstable_phrase_is_reported_and_not_written_by_pressing_enter(
    tmp_path: Path,
) -> None:
    """Warned, and not offered at all.

    Three takes, three different transcriptions: there is no string the gate could
    match twice. This asserted a write prompt defaulting to No until a real run
    showed why that is not enough - the tool printed "This phrase is a bad wake
    phrase / pick a phrase from the first kind and run this again" and then, four
    lines later, a Review block naming those three strings and "Write it (y/N)".
    The run ended in Ctrl-C there, which is the only sensible answer to a question
    the tool has just argued against. So there is no question now.

    Driven with voice already on, which is the state where the switch *would* be
    offered, so this pins that it is not offered either.
    """
    ears = FakeRecognizer(heard=["질문", "헤이씨", "루씨"])
    result = drive(
        tmp_path, answers_for(3, write=""), devices=devices_for(recognizer=ears), existing=VOICE_ON
    )

    out = flat(result.out)
    assert "unstable" in out
    assert "bad wake phrase" in out
    assert "ordinary Korean words transcribe reliably" in out, "it suggests what to try"
    assert "Write it" not in out, (
        "it offered to save strings it had just called unusable - the contradiction "
        "a real run stopped at"
    )
    assert "Review" not in out, "and it showed them in a Review block as if they were a plan"
    assert "Turn DAEMON_WAKE_ENABLED on" not in out
    assert "DAEMON_WAKE_ALIASES" not in result.written
    # Non-zero now, and the change is deliberate. It used to exit 0 because
    # *declining* an offer is a legitimate answer - but there is no offer any more, and
    # what happened is that the owner ran a command and got no usable phrase out of it.
    # `daemon wake test` refusing an empty alias list exits 1 for the same reason, so a
    # script that chains the two stops at the first one that achieved nothing.
    assert result.code == 1, "nothing usable came out of the run, so it is not a success"


def test_a_partly_stable_phrase_is_a_warning_and_still_saves_every_spelling(
    tmp_path: Path,
) -> None:
    ears = FakeRecognizer(heard=["헤이 대문", "헤이 대문", "헤이 데문"])
    result = drive(tmp_path, answers_for(3), devices=devices_for(recognizer=ears))

    out = flat(result.out)
    assert "mostly stable" in out
    assert "Write it (Y/n)" in out
    assert value_of(result.written, "DAEMON_WAKE_ALIASES") == "헤이 대문,헤이 데문"


def test_declining_to_save_writes_nothing(tmp_path: Path) -> None:
    before = "TELEGRAM_BOT_TOKEN=8012345678:AAH-realtokenABCD\n"
    result = drive(
        tmp_path,
        answers_for(3, write="n"),
        devices=devices_for(recognizer=FakeRecognizer(heard=[HEARD] * 3)),
        existing=before,
    )

    assert result.written == before, "not one byte of their file"
    assert "Nothing was written." in result.out


def test_an_alias_carrying_the_separator_is_refused_with_the_reason(tmp_path: Path) -> None:
    """A comma in a transcription is not a phrase problem, it is a format problem.

    `DAEMON_WAKE_ALIASES` is comma-separated, so storing 헤이 대문, 지금 would
    silently become two aliases, one of which was never said.
    """
    ears = FakeRecognizer(heard=["헤이 대문, 지금"] * 3)
    result = drive(tmp_path, ["", "", "", ""], devices=devices_for(recognizer=ears))

    out = flat(result.out)
    assert "contains a comma" in out
    assert "nothing to save" in out
    assert result.code == 1
    assert result.written == ""


def test_a_refused_alias_does_not_cost_the_ones_that_were_fine(tmp_path: Path) -> None:
    ears = FakeRecognizer(heard=[HEARD, "헤이 대문, 지금", HEARD])
    result = drive(tmp_path, answers_for(3), devices=devices_for(recognizer=ears))

    assert "contains a comma" in flat(result.out)
    assert value_of(result.written, "DAEMON_WAKE_ALIASES") == HEARD


# --- when the machine cannot answer ------------------------------------------


def test_an_unavailable_recognizer_fails_before_the_microphone_is_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absence is normal and has to be reportable, which is why `available` exists
    separately from `transcribe` - a missing locale and a failed transcription are
    indistinguishable after the fact."""
    monkeypatch.setattr(wake_cli.sys, "platform", "linux")
    audio = FakeAudio()
    ears = FakeRecognizer(available=False)
    result = drive(tmp_path, [], devices=devices_for(audio, ears))

    out = flat(result.out)
    assert result.code == 1
    assert "fail:" in out
    assert "macOS" in out and "linux" in out, "it names the reason it can determine"
    assert "the locale is not installed" in out, "and the ones it cannot"
    assert audio.opened == 0, "nothing was recorded"
    assert ears.calls == []
    assert result.written == ""


def test_an_unavailable_recognizer_on_macos_still_names_the_three_causes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wake_cli.sys, "platform", "darwin")
    result = drive(tmp_path, [], devices=devices_for(recognizer=FakeRecognizer(available=False)))

    out = flat(result.out)
    assert "cannot answer on this machine" in out
    assert "Dictation" in out
    # First in the list because it was the answer on the machine this was written
    # on: `available` is False with the voice extra absent, and pyobjc is where the
    # recognizer lives.
    assert "pip install -e '.[voice]'" in out
    assert result.code == 1


def test_a_microphone_that_delivers_nothing_is_named_as_a_device_problem(
    tmp_path: Path,
) -> None:
    """Distinct from "the recognizer heard nothing": no audio arrived at all, so
    nothing is yet known about the phrase, and the things to check are different."""
    audio, ears = FakeAudio(blocks=[]), FakeRecognizer()
    result = drive(tmp_path, ["", "", "", ""], devices=devices_for(audio, ears))

    out = flat(result.out)
    assert "no audio was captured" in out
    assert "input device is muted" in out
    assert ears.calls == [], "there was nothing to transcribe"
    assert audio.closed == 3, "and the device is still released"
    assert result.code == 1
    assert result.written == ""


def test_audio_that_arrives_but_says_nothing_gets_the_other_message(
    tmp_path: Path,
) -> None:
    ears = FakeRecognizer(heard=["", "", ""])
    result = drive(tmp_path, ["", "", "", ""], devices=devices_for(recognizer=ears))

    out = flat(result.out)
    assert "heard nothing it could put into words" in out
    assert "no audio was captured" not in out
    assert len(ears.calls) == 3, "it did ask, three times"
    assert result.code == 1


def test_a_take_that_hangs_is_bounded_by_the_clock(tmp_path: Path) -> None:
    """A device that opens and then delivers nothing must not own the terminal.

    The byte bound cannot end this take - no bytes arrive - so `RECORD_GRACE` is the
    only thing between the owner and a command that has to be killed.
    """

    class Wedged(FakeAudio):
        async def record(self) -> AsyncIterator[bytes]:
            self.opened += 1
            await asyncio.Event().wait()
            yield b""  # pragma: no cover - unreachable, and required to be a generator

    audio = Wedged()
    result = drive(tmp_path, ["", "", "", ""], devices=devices_for(audio, FakeRecognizer()))

    assert audio.opened == 3
    assert "no audio was captured" in flat(result.out)
    assert result.code == 1


async def test_record_gives_up_on_a_device_that_delivers_nothing() -> None:
    """The same bound as the test above, asserted instead of survived.

    Driving the whole command with a wedged device proves the command comes back -
    but if `RECORD_GRACE` were ever removed, that test would *hang* rather than fail,
    and a hanging suite reports nothing. An outer `wait_for` turns the same
    regression into a failure with a name.
    """

    class Wedged(FakeAudio):
        async def record(self) -> AsyncIterator[bytes]:
            self.opened += 1
            await asyncio.Event().wait()
            yield b""  # pragma: no cover - unreachable, and required to be a generator

    pcm = await asyncio.wait_for(wake_cli.record(Wedged(), seconds=0.05), timeout=5)
    assert pcm == b""


def test_a_recognizer_that_raises_stops_the_run_and_says_what_raised(
    tmp_path: Path,
) -> None:
    """`transcribe` is contracted not to raise for ordinary failure, so one that
    does is a defect worth naming rather than one worth retrying twice more."""

    class Broken(FakeRecognizer):
        async def transcribe(self, pcm: bytes) -> str:
            raise RuntimeError("no speech asset for ko-KR")

    result = drive(tmp_path, ["", ""], devices=devices_for(recognizer=Broken()))

    out = flat(result.out)
    assert "take 1 failed" in out
    assert "RuntimeError: no speech asset for ko-KR" in out
    assert result.code == 1
    assert result.written == ""


# --- the file it writes -------------------------------------------------------


def test_an_existing_list_is_shown_and_replaced(tmp_path: Path) -> None:
    before = "DAEMON_WAKE_ALIASES=구식\nTELEGRAM_BOT_TOKEN=8012345678:AAH-x\n"
    ears = FakeRecognizer(heard=[HEARD] * 3)
    result = drive(
        tmp_path, answers_for(3), devices=devices_for(recognizer=ears), existing=before
    )

    out = flat(result.out)
    assert "already holds: 구식" in out
    assert "was 구식" in out, "the review says what it replaces"
    assert value_of(result.written, "DAEMON_WAKE_ALIASES") == HEARD
    assert value_of(result.written, "TELEGRAM_BOT_TOKEN") == "8012345678:AAH-x"


def test_the_switch_is_offered_with_voice_on_and_the_file_still_starts(
    tmp_path: Path,
) -> None:
    """The whole write, read back through startup's own validation.

    Both halves matter: the switch is only legal alongside voice, and the aliases
    have to survive the comma-separated round trip in Korean - which is where
    `NoDecode` and `_split_aliases` in `daemon/config.py` earn their keep.
    """
    result = drive(
        tmp_path,
        answers_for(3, enable="y"),
        devices=devices_for(
            recognizer=FakeRecognizer(heard=["헤이 대문", "헤이 데문", "헤이 대문"])
        ),
        existing=VOICE_ON,
    )

    assert value_of(result.written, "DAEMON_WAKE_ENABLED") == "true"
    assert "written by `daemon wake calibrate`" in result.written, (
        "the block says which command wrote it, not which one owns the writer"
    )
    settings = loads(result.env_path)
    assert settings.wake_enabled is True
    assert settings.wake_aliases == ("헤이 대문", "헤이 데문")


def test_the_switch_is_not_offered_when_voice_is_off_and_the_reason_is_said(
    tmp_path: Path,
) -> None:
    """The trap: `Settings` refuses the gate on with voice off, so writing the switch
    here would leave a `.env` that `daemon run` cannot load - and calibration would
    have reported success while breaking the install."""
    result = drive(
        tmp_path,
        answers_for(3),
        devices=devices_for(recognizer=FakeRecognizer(heard=[HEARD] * 3)),
        existing=VOICE_OFF,
    )

    out = flat(result.out)
    assert "DAEMON_VOICE_ENABLED is off, so the gate cannot be switched on" in out
    assert "daemon setup" in out, "and it names where voice gets turned on"
    assert "DAEMON_WAKE_ENABLED" not in result.written
    assert value_of(result.written, "DAEMON_WAKE_ALIASES") == HEARD, "the measurement keeps"
    assert loads(result.env_path).wake_enabled is False


def test_the_switch_is_not_offered_when_it_is_already_on(tmp_path: Path) -> None:
    result = drive(
        tmp_path,
        answers_for(3),
        devices=devices_for(recognizer=FakeRecognizer(heard=[HEARD] * 3)),
        existing=VOICE_ON + "DAEMON_WAKE_ENABLED=true\n",
    )

    assert "Turn DAEMON_WAKE_ENABLED on" not in flat(result.out)
    assert value_of(result.written, "DAEMON_WAKE_ALIASES") == HEARD
    assert result.code == 0


def test_declining_the_switch_still_saves_the_aliases(tmp_path: Path) -> None:
    result = drive(
        tmp_path,
        answers_for(3, enable="n"),
        devices=devices_for(recognizer=FakeRecognizer(heard=[HEARD] * 3)),
        existing=VOICE_ON,
    )

    assert value_of(result.written, "DAEMON_WAKE_ALIASES") == HEARD
    assert "DAEMON_WAKE_ENABLED" not in result.written


def test_a_closed_stdin_stops_without_touching_the_file(tmp_path: Path) -> None:
    before = "DAEMON_WAKE_ALIASES=구식\n"
    result = drive(
        tmp_path, [], devices=devices_for(recognizer=FakeRecognizer()), existing=before
    )

    assert result.written == before
    assert "was not touched" in result.out
    assert result.code == 1


# --- the owner's room ---------------------------------------------------------


def test_the_owner_is_told_before_the_microphone_opens(tmp_path: Path) -> None:
    """The warning has to be above the first take, not in the summary."""
    result = drive(
        tmp_path, answers_for(3), devices=devices_for(recognizer=FakeRecognizer(heard=[HEARD] * 3))
    )

    warned = result.out.index(wake_cli.MIC_WARNING)
    first_take = result.out.index("Take 1 of 3")
    assert warned < first_take
    assert "never written anywhere and is never logged" in flat(result.out)


def test_the_owners_audio_reaches_neither_the_terminal_nor_a_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the text is allowed out, and only what the owner approved.

    Falsifiable because the marker really is in the audio that was recorded: the
    recognizer was handed 900 bytes of it per take, and the assertion below is the
    only thing standing between that and a debug log.
    """
    caplog.set_level(logging.DEBUG)
    ears = FakeRecognizer(heard=[HEARD] * 3)
    result = drive(tmp_path, answers_for(3), devices=devices_for(recognizer=ears))

    assert ears.calls == [len(BLOCK)] * 3, "audio really did reach the recognizer"
    assert HEARD in result.out, "and its text really was printed"
    assert "WAKEAUDIO" not in result.out
    assert "WAKEAUDIO" not in caplog.text
    assert "WAKEAUDIO" not in result.written


# --- daemon wake test --------------------------------------------------------

WINDOW = 0.1
"""Listening window for the tests that need one to expire. Short because nothing
here has any work to do - the fake yields everything it has immediately and then
waits, so this is dead time either way."""


def wake(heard: str, matched: str, confidence: float = 0.9) -> WakeEvent:
    return WakeEvent(heard=heard, matched=matched, confidence=confidence)


def watch(tmp_path: Path, gate: Any, *, seconds: float = WINDOW) -> tuple[int, str]:
    out = io.StringIO()
    code = wake_cli.listen(
        settings_for(tmp_path), seconds=seconds, stdout=out, devices=devices_for(gate=gate)
    )
    return code, out.getvalue()


def test_wake_test_prints_every_event_with_both_strings(tmp_path: Path) -> None:
    """`heard` and `matched` are different strings on purpose, and printing only one
    of them is what leaves an owner guessing why a phrase does or does not fire."""
    gate = FakeGate(fires=[wake("헤이 대문", HEARD, 0.93), wake("루시", "루시", 0.71)])
    code, out = watch(tmp_path, build_gate(gate))

    flattened = flat(out)
    assert "heard 헤이 대문 · matched 헤이 대문" in flattened
    assert "heard 루시 · matched 루시" in flattened
    assert "speech 93%" in flattened
    assert "2 wake event(s)" in flattened
    assert "matched an alias 2" in flattened, "and the tally agrees with the lines above"
    assert gate.closed == 1, "the microphone is released"
    assert code == 0


def test_a_quiet_room_is_reported_as_a_quiet_room(tmp_path: Path) -> None:
    """Audio arriving with no speech in it. The gate working exactly as designed -
    and the one silence whose fix is the input level rather than the aliases, so it
    must not be answered with advice about calibrating."""
    gate = FakeGate(counters=WakeCounters(frames_seen=1_200, segments_closed=0))
    code, out = watch(tmp_path, build_gate(gate))

    flattened = flat(out)
    assert "nothing fired." in flattened
    assert "frames seen 1200" in flattened, "and what the gate actually saw"
    assert "the VAD found no speech in it" in flattened
    assert "input level is low" in flattened, "which is what to do about it"
    assert code == 0, "silence is not a failure"


def test_the_general_advice_is_the_fallback_for_a_tally_that_fits_nothing(
    tmp_path: Path,
) -> None:
    """Every branch of the diagnosis is a shape someone anticipated, so there has to
    be something to print for a shape nobody did - otherwise a tally this file has
    not met yet produces "nothing fired." and no next step at all."""
    gate = FakeGate(counters=WakeCounters(frames_seen=900, segments_closed=2))
    _code, out = watch(tmp_path, build_gate(gate))

    flattened = flat(out)
    assert "right answer if nobody said the phrase" in flattened
    assert "daemon wake calibrate" in flattened, "it names what to do about it"


def test_nothing_firing_while_speech_was_transcribed_points_at_calibration(
    tmp_path: Path,
) -> None:
    """The case this whole feature exists for: it heard you and matched nothing.

    Indistinguishable from an empty room without the tally, and the fix is not the
    one anyone guesses - the alias has to be what the recognizer returns, which is
    never the name the owner chose.
    """
    gate = FakeGate(
        counters=WakeCounters(frames_seen=900, segments_closed=4, transcribed=4, fired=0)
    )
    _code, out = watch(tmp_path, build_gate(gate))

    flattened = flat(out)
    assert "transcribed 4 time(s) and no alias matched" in flattened
    assert "uncalibrated DAEMON_WAKE_ALIASES looks like" in flattened
    assert "the VAD found no speech" not in flattened, "it says the one thing that fits"


def test_a_deaf_recognizer_is_not_reported_as_a_quiet_room(tmp_path: Path) -> None:
    """`skipped_unavailable` is the counter that separates "nobody spoke" from "we
    have been deaf since Tuesday", and that is a different sentence and a different
    fix."""
    gate = FakeGate(
        counters=WakeCounters(frames_seen=900, segments_closed=3, skipped_unavailable=3)
    )
    _code, out = watch(tmp_path, build_gate(gate))

    flattened = flat(out)
    assert "3 segment(s) of speech went untranscribed" in flattened
    assert "This is the deaf case, not a quiet room" in flattened
    assert "the locale is not installed" in flattened


def test_no_audio_at_all_is_reported_as_a_device_problem(tmp_path: Path) -> None:
    _code, out = watch(tmp_path, build_gate(FakeGate()))

    flattened = flat(out)
    assert "no audio arrived at all" in flattened
    assert "input device is muted" in flattened


def test_a_gate_that_stops_listening_is_a_failure_and_says_why(tmp_path: Path) -> None:
    """The worst failure a companion has: alive, healthy-looking, permanently deaf.
    A gate whose iterator ends has done exactly that, and nothing else would say so.
    """
    gate = FakeGate(fires=[wake(HEARD, HEARD)], ends=True)
    code, out = watch(tmp_path, build_gate(gate))

    flattened = flat(out)
    assert "stopped listening on its own, after 1 event(s)" in flattened
    assert "permanently deaf" in flattened
    assert gate.closed == 1, "and it still lets go of the microphone"
    assert code == 1


def test_a_gate_that_cannot_be_built_is_a_sentence_and_not_a_traceback(
    tmp_path: Path,
) -> None:
    """Which is the state of a machine with no VAD model file, or no macOS."""
    code, out = watch(
        tmp_path, build_gate(raises=ModuleNotFoundError("No module named 'onnxruntime'"))
    )

    flattened = flat(out)
    assert "the gate could not run" in flattened
    assert "onnxruntime" in flattened
    assert "daemon/voice/wake.py" in flattened, "it names where the gate comes from"
    assert code == 1


def test_wake_test_refuses_before_recording_when_there_are_no_aliases(
    tmp_path: Path,
) -> None:
    """An empty list cannot match anything, so listening is a minute wasted.

    A real run spent 60 seconds talking to the microphone and was then told the
    aliases looked uncalibrated - which is what a *wrong* alias looks like, not a
    missing one. Same symptom, different state, and only one of them is worth
    opening a microphone for.
    """
    built: list[bool] = []

    async def never(settings: Settings) -> tuple[FakeGate, Any]:  # pragma: no cover
        built.append(True)
        gate = FakeGate(ends=True)
        return gate, gate.close

    out = io.StringIO()
    code = wake_cli.listen(
        settings_for(tmp_path, aliases=""),
        seconds=WINDOW,
        stdout=out,
        devices=devices_for(gate=never),
    )

    flattened = flat(out.getvalue())
    assert "is empty, so nothing can match" in flattened
    assert "daemon wake calibrate" in flattened, "it names the command that fixes it"
    assert "this room is being listened to" not in flattened, (
        "it warned about a microphone it had already decided not to open"
    )
    assert built == [], "the gate was built, so the microphone was opened anyway"
    assert code == 1


def test_wake_test_warns_that_the_room_is_being_listened_to(tmp_path: Path) -> None:
    _code, out = watch(tmp_path, build_gate(FakeGate()))

    flattened = flat(out)
    assert "this room is being listened to" in flattened
    assert "Nothing is recorded to disk" in flattened


# --- reachable from the command line -----------------------------------------


def test_wake_calibrate_is_reachable_and_needs_no_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`daemon wake calibrate` runs before Settings is built, deliberately: it reads
    nothing out of the configuration, and an install whose `.env` does not load yet
    is exactly the one that has never calibrated."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DAEMON_PROVIDER=nonsense\n", encoding="utf-8")
    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(wake_cli, "calibrate", fake)
    assert cli.main(["wake", "calibrate", "--takes", "2"]) == 0
    assert seen == {"takes": 2}


def test_wake_test_is_reachable_and_is_given_the_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        f"DAEMON_PROVIDER=ollama\nDAEMON_OLLAMA_MODEL=gemma3:4b\n"
        f"DAEMON_DATA_DIR={tmp_path / 'data'}\n",
        encoding="utf-8",
    )
    seen: dict[str, Any] = {}

    def fake(settings: Settings, **kwargs: Any) -> int:
        seen["provider"] = settings.provider
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(wake_cli, "listen", fake)
    assert cli.main(["wake", "test", "--seconds", "5"]) == 0
    assert seen == {"provider": "ollama", "seconds": 5}


def test_the_app_seam_hands_the_gate_every_value_it_was_configured_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`app.build_wake_gate` has one job: turn `.env` into a gate. This is the only
    place `daemon wake test` gets one, so a setting dropped here is a documented
    no-op - the owner tunes a value, nothing reads it, and nothing anywhere says so.

    Asserted on the construction call rather than through behaviour, deliberately.
    Six of these are numbers whose effect needs its own crafted audio fixture, and a
    mutation run found `threshold` silently deleted while the behavioural test of
    `min_speech_ms` next door still passed. For a composition root the call *is* the
    behaviour.
    """
    from daemon import app
    from daemon.voice import apple_speech
    from daemon.voice import audio as audio_module
    from daemon.voice import vad as vad_module
    from daemon.voice import wake as wake_module

    seen: dict[str, Any] = {}

    class Recorder:
        def __init__(self, audio: Any, vad: Any, recognizer: Any, aliases: Any, **rest: Any):
            seen.update(rest, aliases=aliases)
            self.counters = WakeCounters()

    monkeypatch.setattr(wake_module, "WakeGate", Recorder)
    monkeypatch.setattr(vad_module, "SileroVad", lambda: object())
    monkeypatch.setattr(apple_speech, "AppleSpeechRecognizer", lambda: object())
    monkeypatch.setattr(audio_module, "SoundDeviceAudio", FakeAudio)

    settings = Settings(
        _env_file=None,
        DAEMON_PROVIDER="ollama",
        DAEMON_OLLAMA_MODEL="gemma3:4b",
        DAEMON_WAKE_ALIASES="헤이 대문, 루씨",
        DAEMON_WAKE_VAD_THRESHOLD="0.7",
        DAEMON_WAKE_HANGOVER_MS="800",
        DAEMON_WAKE_PRE_ROLL_MS="200",
        DAEMON_WAKE_MIN_SPEECH_MS="300",
        DAEMON_WAKE_MAX_SEGMENT_MS="4000",
        DAEMON_WAKE_COOLDOWN_SECONDS="9.0",
    )
    gate, close = asyncio.run(_built(app, settings))

    assert isinstance(gate, Recorder)
    assert seen == {
        "aliases": ("헤이 대문", "루씨"),
        "threshold": 0.7,
        "hangover_ms": 800,
        "pre_roll_ms": 200,
        "min_speech_ms": 300,
        "max_segment_ms": 4000,
        "cooldown_seconds": 9.0,
    }
    asyncio.run(close())


async def _built(app: Any, settings: Settings) -> tuple[Any, Any]:
    return await app.build_wake_gate(settings)


def test_wake_needs_a_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """A two-level command whose first level does something surprising on its own is
    worse than one that says what it wants - `daemon pairing` is the same shape."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["wake"])
    assert exit_info.value.code == 2
    assert "calibrate" in capsys.readouterr().err
