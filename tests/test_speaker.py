"""The local speaker, without making the machine talk.

Hard rule for this file, from tests/CLAUDE.md: **no test here runs a real
subprocess or produces sound.** A suite that speaks Korean at whoever is running it
is broken, and one that spawns `/usr/bin/say` asserts nothing on a Linux CI box.
`LocalSpeaker` takes its process spawner as a constructor argument for exactly that
reason, and `FakeProcess` below stands in for the process rather than for the
class's own logic - so the timeout, the kill-and-reap and the exit-code branches are
all really exercised.

What these pin, in rough order of how much they are worth:

**The text never reaches a shell, and never reaches argv either.** Two different
holes, and `-f -` closes both. The first is command injection, because the sentence
is written by a model; the second silently loses any utterance starting with a dash,
because `say` parses its own argv. Both were verified against the real command
before being pinned here - `"; touch /tmp/PWNED ; echo"` created no file, and
`say -v Yuna -- 그거...` exits 1.

**A failure returns `False` and never raises.** daemon/proactivity/base.py's reason:
a failed utterance must not take down the Telegram copy that went with it. So the
failure paths outnumber the happy one here, deliberately.

**Overlap is dropped.** macOS was measured happily running two `say` processes at
once with the audio mixed, so the exclusion exists only in our code and these are
what hold it.

One thing deliberately *not* asserted: that anything was audible. The real command
exits 0 with a wrong voice name while producing 16 ms of nothing, so no test can
tell "spoke" from "silently mispronounced" - which is why the module docstring
bounds what `True` means instead of the tests pretending otherwise.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from daemon.proactivity import speaker as speaker_module
from daemon.proactivity.base import Speaker
from daemon.proactivity.speaker import (
    DEFAULT_VOICE,
    MAX_CHARS,
    SAY,
    SPAWN_OVERHEAD_SECONDS,
    SPEECH_SECONDS_PER_CHAR,
    LocalSpeaker,
)

UTTERANCE = "어제 발표 있다고 했잖아. 어떻게 됐어?"
"""A real docs/PLAN.md 6.1 type-A line. 22 characters, measured at 3.73-3.78 s
through the real `say -v Yuna` on this machine."""

MEASURED_SECONDS = 3.78
"""The slowest of those three runs. The timeout's headroom is asserted against it."""

LOGGER = "daemon.proactivity.speaker"

BOUND_SECONDS = 5.0
"""Outer bound for the tests that depend on the *production* timeout firing.

tests/CLAUDE.md: nothing here may hang. Several tests below hand the class a
process that never finishes on its own, so if the timeout, the kill or the overlap
guard were removed, the test would wedge rather than fail - and a wedged suite
reports nothing while a broken speaker ships. Generous enough never to fire against
the shrunk constants those tests use, so this is a backstop and not the assertion.
"""


async def _bounded[T](awaitable: object) -> T:
    """Await something that must not be able to hang."""
    async with asyncio.timeout(BOUND_SECONDS):
        return await awaitable  # type: ignore[misc, no-any-return]


class FakeProcess:
    """Stands in for `asyncio.subprocess.Process`, recording what it was fed.

    `hang=True` makes `communicate` wait until the process is killed, which is what
    the real one does - and getting that right matters more than it looks: a fake
    whose `communicate` slept for a fixed time instead would make the overlap test
    pass without the kill ever mattering.
    """

    def __init__(
        self,
        *,
        returncode: int = 0,
        stderr: bytes = b"",
        hang: bool = False,
        speaking: asyncio.Event | None = None,
    ) -> None:
        self._returncode = returncode
        self._stderr = stderr
        self._hang = hang
        self._speaking = speaking
        self._signalled = asyncio.Event()
        self.returncode: int | None = None
        self.stdin_bytes: bytes | None = None
        self.killed = False
        self.waited = False

    async def communicate(self, data: bytes | None = None) -> tuple[bytes, bytes]:
        self.stdin_bytes = data
        if self._speaking is not None:
            self._speaking.set()
        if self._hang:
            # Returns when, and only when, somebody kills it. A real `say` behaves
            # the same way, and it is what lets the aclose-mid-utterance test
            # actually finish.
            await self._signalled.wait()
            self.returncode = -9
            return b"", self._stderr
        self.returncode = self._returncode
        return b"", self._stderr

    def kill(self) -> None:
        self.killed = True
        self._signalled.set()

    async def wait(self) -> int:
        self.waited = True
        if self.returncode is None:
            self.returncode = -9
        return self.returncode


class Spawner:
    """Records every spawn, and hands out prepared processes in order.

    The last one is reused once the queue runs down, so a test that only cares
    about the first call does not have to enumerate the rest.
    """

    def __init__(self, *processes: FakeProcess | Exception) -> None:
        self._queue: list[FakeProcess | Exception] = list(processes) or [FakeProcess()]
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def __call__(self, *argv: str, **kwargs: object) -> FakeProcess:
        self.calls.append((argv, kwargs))
        nxt = self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    @property
    def argv(self) -> tuple[str, ...]:
        return self.calls[0][0]


def _speaker(
    *processes: FakeProcess | Exception, voice: str | None = DEFAULT_VOICE
) -> tuple[LocalSpeaker, Spawner]:
    spawner = Spawner(*processes)
    return LocalSpeaker(platform="darwin", voice=voice, spawn=spawner), spawner


# --- the protocol, and the happy path ----------------------------------------


def test_it_satisfies_the_speaker_protocol() -> None:
    """`Speaker` is runtime-checkable and base.py is frozen: the point of building
    against it is that the judge can hold this without importing it."""
    speaker, _ = _speaker()
    assert isinstance(speaker, Speaker)


async def test_a_korean_utterance_is_spoken() -> None:
    speaker, spawner = _speaker()
    assert await speaker.say(UTTERANCE) is True
    assert spawner.argv == (SAY, "-v", DEFAULT_VOICE, "-f", "-")


async def test_the_default_voice_is_korean_and_unambiguous() -> None:
    """`say` has no language flag - the man page's options are `-v -r -o -f -a -i`
    and nothing else - so `-v` *is* the language selection.

    `Yuna` is the only one of the nine ko_KR voices whose bare name appears exactly
    once in `say -v ?`; the other eight exist in fourteen locales each and the bare
    name resolves to the wrong one. Measured on the same Korean text: `-v Yuna`
    produced 1.45 s of audio and `-v Eddy` produced 0.016 s, **both exiting 0**. So
    a default that drifted to a shared name would be silent and would look healthy.
    """
    speaker, spawner = _speaker()
    await speaker.say(UTTERANCE)
    assert "-v" in spawner.argv, "the voice must be pinned, not left to the system"
    assert spawner.argv[spawner.argv.index("-v") + 1] == "Yuna"


async def test_no_voice_flag_when_the_system_default_is_asked_for() -> None:
    speaker, spawner = _speaker(voice=None)
    assert await speaker.say(UTTERANCE) is True
    assert "-v" not in spawner.argv


# --- the text does not reach a shell, and does not reach argv -----------------


async def test_the_text_is_piped_and_never_interpolated_into_the_command() -> None:
    """The most important test in this file.

    The sentence is written by an LLM. If it reached a shell, `;` would end the
    command and the rest would run; if it reached argv, `say` itself would parse a
    leading dash as a flag. Three things are asserted together: the text arrives
    over stdin byte-for-byte, no element of argv contains any fragment of it, and
    the argv is a fixed list of plain strings.
    """
    hostile = "안녕; touch /tmp/PWNED\n`whoami`\n$(rm -rf ~)\n& echo 'q' | tee /tmp/x"
    process = FakeProcess()
    speaker, spawner = _speaker(process)

    assert await speaker.say(hostile) is True

    # Byte-identical, so nothing escaped, quoted or rewrote it on the way.
    assert process.stdin_bytes == hostile.encode("utf-8")

    argv, kwargs = spawner.calls[0]
    assert argv == (SAY, "-v", DEFAULT_VOICE, "-f", "-")
    # Not "the whole sentence is absent" - any *fragment* in argv would mean the
    # command was assembled by string surgery, which is the thing that must never
    # start being true.
    for element in argv:
        for dangerous in (";", "`", "$", "|", "&", "PWNED", "whoami", "rm "):
            assert dangerous not in element
    # `create_subprocess_exec` has no `shell` parameter, so a caller switching to
    # `create_subprocess_shell` would have to collapse argv into one string.
    assert "shell" not in kwargs
    assert all(isinstance(element, str) for element in argv)


async def test_an_utterance_starting_with_a_dash_still_speaks() -> None:
    """Why `-f -` and not the sentence in argv.

    Verified against the real command: `say -v Yuna -- 그거 어떻게 됐어?` exits 1
    with ``unrecognized option `--' ``. A model writing an em-dash as two hyphens is
    ordinary, and losing an utterance to punctuation is a silent product failure.
    Over stdin nothing parses it.
    """
    process = FakeProcess()
    speaker, _ = _speaker(process)
    assert await speaker.say("-- 그거 어떻게 됐어?") is True
    assert process.stdin_bytes == "-- 그거 어떻게 됐어?".encode()


async def test_newlines_survive_intact() -> None:
    """`say -f -` reads the whole stream and speaks newlines as pauses - measured,
    two Korean lines, exit 0. So interior newlines are passed through rather than
    collapsed, while the outer whitespace goes: a trailing newline is not speech."""
    process = FakeProcess()
    speaker, _ = _speaker(process)
    assert await speaker.say("\n첫 줄이야.\n두 번째 줄.\n") is True
    assert process.stdin_bytes == "첫 줄이야.\n두 번째 줄.".encode()


# --- nothing to say ----------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n", "\t\n  "])
async def test_empty_text_never_reaches_the_synthesiser(text: str) -> None:
    """Two reasons, and the second is the real one. The real `say` accepts empty
    input and exits 0 after ~0.36 s, so this saves a pointless spawn - but an empty
    `Utterance.text` is how base.py spells *declining to speak*, and reporting a
    decline as speech would corrupt the utterance log."""
    speaker, spawner = _speaker()
    assert await speaker.say(text) is False
    assert spawner.calls == []


# --- the failure paths, which are most of them -------------------------------


async def test_a_non_zero_exit_is_false_and_keeps_the_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The honest half of `say`'s reporting, and a missing audio device is the case
    it exists for: measured, `-a NoSuchDevice` exits 1 with
    ``Found no Audio Output Device matching `NoSuchDevice' ``. The return code alone
    does not say which failure it was, so stderr is kept verbatim."""
    stderr = b"Found no Audio Output Device matching `NoSuchDevice'\n"
    speaker, _ = _speaker(FakeProcess(returncode=1, stderr=stderr))

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert await speaker.say(UTTERANCE) is False

    assert "Found no Audio Output Device" in caplog.text


async def test_a_missing_say_binary_is_false_not_an_exception() -> None:
    """A macOS without `/usr/bin/say`, or a stripped image claiming to be darwin."""
    speaker, _ = _speaker(FileNotFoundError(2, "No such file or directory"))
    assert await speaker.say(UTTERANCE) is False


async def test_an_unexpected_exception_from_the_process_is_swallowed() -> None:
    """base.py's guarantee has to hold for the failures nobody anticipated too -
    the same reasoning as `presence.py`'s broad except on a scheduler tick."""

    class Exploding(FakeProcess):
        async def communicate(self, data: bytes | None = None) -> tuple[bytes, bytes]:
            raise RuntimeError("coreaudiod went away")

    speaker, _ = _speaker(Exploding())
    assert await speaker.say(UTTERANCE) is False


async def test_cancellation_kills_the_utterance_and_propagates() -> None:
    """Shutdown mid-sentence. `CancelledError` must not be swallowed - that breaks
    shutdown - but the process has to die first, or a cancelled await detaches this
    coroutine and leaves `say` talking into the room."""
    started = asyncio.Event()
    process = FakeProcess(hang=True, speaking=started)
    speaker, _ = _speaker(process)

    talking = asyncio.create_task(speaker.say(UTTERANCE))
    await started.wait()
    talking.cancel()

    with pytest.raises(asyncio.CancelledError):
        await _bounded(talking)
    assert process.killed, "cancelling the await must not leave the speaker talking"


async def test_a_wedged_say_is_killed_and_reaped() -> None:
    """Not merely killed. An abandoned process still holds its pipes, so skipping
    the `wait` leaks a file descriptor per utterance on a process meant to run for
    weeks - `presence.py`'s reasoning, which is why both halves are asserted."""
    hung = FakeProcess(hang=True)
    speaker, _ = _speaker(hung)

    # Shrunk so the test does not wait out the real 16 s ceiling. What the ceiling
    # is worth is asserted separately, against the measured duration.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(speaker_module, "SPAWN_OVERHEAD_SECONDS", 0.01)
        patch.setattr(speaker_module, "SPEECH_SECONDS_PER_CHAR", 0.0)
        assert await _bounded(speaker.say(UTTERANCE)) is False

    assert hung.killed, "a synthesiser that never returns must be killed"
    assert hung.waited, "and reaped - killing without waiting leaks the pipes"


async def test_the_timeout_scales_with_the_length_of_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derived from the text, not flat.

    A flat ceiling large enough for `MAX_CHARS` would be ~2 minutes and would mean
    nothing when it fired. Scaled off the measured 0.145 s/char instead, so a
    timeout is evidence the synthesiser is wedged rather than evidence the sentence
    was long. The constants are shrunk here so the test is quick; what is asserted
    is that the ceiling *moves with the length*.
    """
    monkeypatch.setattr(speaker_module, "SPAWN_OVERHEAD_SECONDS", 0.02)
    monkeypatch.setattr(speaker_module, "SPEECH_SECONDS_PER_CHAR", 0.005)

    elapsed: list[float] = []
    for text in ("짧아.", "짧아." * 30):
        hung = FakeProcess(hang=True)
        speaker, _ = _speaker(hung)
        started = time.perf_counter()
        assert await _bounded(speaker.say(text)) is False
        elapsed.append(time.perf_counter() - started)
        assert hung.killed and hung.waited

    assert elapsed[1] > elapsed[0] * 2, (
        f"the timeout did not scale with the text: {elapsed}. A flat timeout passes "
        "every other test in this file."
    )


def test_the_real_timeout_leaves_headroom_over_the_measured_duration() -> None:
    """The constants shipped, against the number they were sized from.

    Both directions matter. Too tight and an ordinary utterance is killed mid-word;
    too loose and a wedged `say` mutes proactive speech for minutes, because an
    overlap is dropped rather than queued.
    """
    ceiling = SPAWN_OVERHEAD_SECONDS + SPEECH_SECONDS_PER_CHAR * len(UTTERANCE)
    assert ceiling > MEASURED_SECONDS * 3, "not enough headroom over a real utterance"
    assert ceiling < 30, "a ceiling this loose stops being a timeout"


async def test_a_signal_exit_reports_false_without_calling_it_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """How `aclose` ends an utterance: SIGTERM leaves -15, SIGKILL -9. Not a fault -
    cutting the voice off because a meeting started is the most ordinary use there
    is (docs/PLAN.md 6.4) - but not speech either, so `False`."""
    speaker, _ = _speaker(FakeProcess(returncode=-15))

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert await speaker.say(UTTERANCE) is False

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], "an interrupted utterance is not an error"


# --- platform ----------------------------------------------------------------


@pytest.mark.parametrize("platform", ["linux", "win32", "freebsd"])
async def test_no_local_synthesiser_off_macos(platform: str) -> None:
    """`False` rather than an attempt: docs/PLAN.md 6.3 routes by presence and 6.4's
    asymmetry says falling back to Telegram costs nothing. Injected because this
    branch will always be written on a Mac and otherwise never run."""
    spawner = Spawner()
    speaker = LocalSpeaker(platform=platform, spawn=spawner)
    assert await speaker.say(UTTERANCE) is False
    assert spawner.calls == [], "nothing may be spawned where there is no synthesiser"


# --- length ------------------------------------------------------------------


async def test_an_essay_is_truncated_rather_than_spoken_in_full() -> None:
    """Speech duration is linear in character count - measured 0.145 s/char, holding
    from 9 to 165 characters - so an unbounded model output is unbounded speech into
    a room. Truncated rather than refused because the Telegram copy carries the
    whole text, and part of it out loud beats silence."""
    process = FakeProcess()
    speaker, _ = _speaker(process)
    essay = "오늘 하루도 정말 고생 많았어. " * 40
    assert len(essay) > MAX_CHARS

    assert await speaker.say(essay) is True

    assert process.stdin_bytes is not None
    assert process.stdin_bytes.decode() == essay.strip()[:MAX_CHARS]


# --- overlap -----------------------------------------------------------------


async def test_a_second_utterance_is_dropped_while_the_first_is_speaking() -> None:
    """macOS will not arbitrate this: two concurrent `say` processes were measured
    both running to completion in ~2.2 s with the audio mixed, which is noise rather
    than two messages.

    Dropped rather than queued - a late utterance is the one thing docs/PLAN.md 6.2
    is built to avoid - and rather than interrupting, since the first already passed
    the gate and its Telegram copy has gone out. With a three-a-day budget an
    overlap means something upstream is broken, so it is reported as `False`.
    """
    started = asyncio.Event()
    first = FakeProcess(hang=True, speaking=started)
    speaker, spawner = _speaker(first, FakeProcess())

    talking = asyncio.create_task(speaker.say(UTTERANCE))
    await started.wait()

    assert await _bounded(speaker.say("두 번째 문장이야.")) is False
    assert len(spawner.calls) == 1, "the second utterance must not spawn a process"
    assert not first.killed, "and must not interrupt the first one either"

    await speaker.aclose()
    assert await _bounded(talking) is False


async def test_the_speaker_is_usable_again_after_every_failure_path() -> None:
    """The other direction, and it guards the failure tests/CLAUDE.md names: a queue
    behind a dead speaker. A lock leaked on any exit path would leave the daemon
    permanently mute while every call kept returning a plausible `False`."""
    speaker, spawner = _speaker(
        FakeProcess(returncode=1, stderr=b"boom"),
        FakeProcess(hang=True),
        FakeProcess(returncode=-15),
        FakeProcess(),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(speaker_module, "SPAWN_OVERHEAD_SECONDS", 0.01)
        patch.setattr(speaker_module, "SPEECH_SECONDS_PER_CHAR", 0.0)
        assert await _bounded(speaker.say(UTTERANCE)) is False  # non-zero exit
        assert await _bounded(speaker.say(UTTERANCE)) is False  # wedged, killed
        assert await _bounded(speaker.say(UTTERANCE)) is False  # stopped by a signal
        assert await _bounded(speaker.say(UTTERANCE)) is True  # and still works
    assert len(spawner.calls) == 4


# --- aclose ------------------------------------------------------------------


async def test_aclose_on_an_idle_speaker_does_nothing() -> None:
    speaker, _ = _speaker()
    await speaker.aclose()  # must not raise, and must not need a process


async def test_aclose_does_not_kill_a_process_that_already_finished() -> None:
    """A second kill on a reaped pid is a `ProcessLookupError` at best and somebody
    else's process at worst."""
    process = FakeProcess()
    speaker, _ = _speaker(process)
    assert await speaker.say(UTTERANCE) is True
    await speaker.aclose()
    assert not process.killed


async def test_a_process_that_vanishes_before_the_kill_is_not_an_error() -> None:
    """`say` finishing between the timeout firing and the signal landing. Real, and
    the reason `presence.py` catches this too."""

    class Vanishing(FakeProcess):
        def kill(self) -> None:
            raise ProcessLookupError

    vanished = Vanishing(hang=True)
    speaker, _ = _speaker(vanished)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(speaker_module, "SPAWN_OVERHEAD_SECONDS", 0.01)
        patch.setattr(speaker_module, "SPEECH_SECONDS_PER_CHAR", 0.0)
        assert await _bounded(speaker.say(UTTERANCE)) is False
    assert not vanished.waited, "there is nothing left to reap"


# --- the Linux branch, carried over from the class this one replaced ----------
# Ported when two `LocalSpeaker` classes collided and the older one - which had a
# working espeak-ng/spd-say path - was removed. Dropping that path to resolve a
# name collision would have been a silent regression for self-hosters, and a
# mutation check showed the ported code had no coverage at all.


def test_linux_uses_espeak_ng_when_it_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        speaker_module.shutil,
        "which",
        lambda name: "/usr/bin/espeak-ng" if name == "espeak-ng" else None,
    )

    assert LocalSpeaker(platform="linux").command() == ["/usr/bin/espeak-ng", "--stdin"]


def test_linux_falls_back_to_spd_say(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two spellings a Linux desktop is likely to already have, tried in order."""
    monkeypatch.setattr(
        speaker_module.shutil,
        "which",
        lambda name: "/usr/bin/spd-say" if name == "spd-say" else None,
    )

    assert LocalSpeaker(platform="linux").command() == [
        "/usr/bin/spd-say",
        "--wait",
        "--pipe-mode",
    ]


def test_linux_with_no_synthesiser_has_no_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(speaker_module.shutil, "which", lambda _name: None)

    assert LocalSpeaker(platform="linux").command() is None


def test_the_linux_command_also_reads_from_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The property that matters on every platform: a model-authored sentence never
    reaches argv, so it can be neither shell-interpolated nor parsed as an option."""
    monkeypatch.setattr(
        speaker_module.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name == "espeak-ng" else None,
    )

    command = LocalSpeaker(platform="linux").command()

    assert command is not None
    assert "--stdin" in command
    assert not any(arg.startswith("안녕") for arg in command)


async def test_a_machine_with_nothing_to_speak_with_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Answered `False`, never raised: the gate falls back to Telegram, which loses
    nothing per the failure-cost asymmetry in docs/PLAN.md 6.4."""
    monkeypatch.setattr(speaker_module.shutil, "which", lambda _name: None)
    spoke: list[str] = []

    async def spawner(*argv: str, **_kw: object) -> object:  # pragma: no cover - never called
        spoke.append(argv[0])
        raise AssertionError("nothing should have been spawned")

    assert await LocalSpeaker(platform="linux", spawn=spawner).say("안녕") is False
    assert spoke == []


# --- finding 5, whole-branch review: refuse while the mic is held -----------
# docs/adr/0012 made `mic_busy` subtract our own hold, which stopped the gate
# from ever seeing a live voice session mid-call - so this file, not the gate,
# has to refuse. `mic_held` is injected the same way presence.py injects it,
# not exercised through the real `daemon.mic_hold` module, so these tests
# cannot leak state into any other test's process-wide counter.


async def test_speaking_is_refused_while_the_microphone_is_held() -> None:
    """The regression: a live voice session holds the microphone for its whole
    run, and `Gate._route` can still return `both` mid-session (nothing in a
    `Reading` says a call is live once our own hold is subtracted). Without this
    check `say` would speak over the session and into the open microphone."""
    spawner = Spawner()
    speaker = LocalSpeaker(platform="darwin", spawn=spawner, mic_held=lambda: True)

    assert await speaker.say(UTTERANCE) is False
    assert spawner.calls == []


async def test_speaking_proceeds_when_the_microphone_is_free() -> None:
    """The control for the test above: the same speaker, the same utterance,
    only `mic_held` differs - proves the refusal is about the hold and not
    about something else in the setup."""
    speaker, spawner = _speaker()

    assert await speaker.say(UTTERANCE) is True
    assert spawner.calls != []


async def test_a_refused_utterance_logs_why(caplog: pytest.LogCaptureFixture) -> None:
    speaker = LocalSpeaker(platform="darwin", spawn=Spawner(), mic_held=lambda: True)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await speaker.say(UTTERANCE)

    assert "microphone" in caplog.text
