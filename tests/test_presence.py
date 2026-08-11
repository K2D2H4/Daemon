"""Presence probes, without a probe.

Hard rule for this file: **no test here runs a real subprocess, and none of them
depends on how long ago somebody touched this keyboard.** A test that reads the
machine's actual idle time passes on the author's Mac and asserts nothing
anywhere else, and `MachinePresence` has three seams for that reason - the
command runner, the audio call, and whether we hold the microphone ourselves are
all injectable.

What is being pinned is the direction the failures resolve in. docs/PLAN.md 6.4:
an ignored notification costs nothing, a voice out of the speaker during a
meeting is an accident. So "the probe could not answer" must arrive as `None`
with a reason, never as `False` and never as `0.0` - the last one being the
subtle case, because idle 0.0 reads as `at_keyboard is True`, which routes to the
speaker.

The `lsappinfo` cases carry more weight than they look like they should, because
that command **exits 0 for every failure it has** - an unrecognised subcommand, an
unresolvable ASN, garbage input. Nothing in the return code says anything, so
parsing is the only detection there is, and these are the tests holding it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from daemon.proactivity import presence as presence_module
from daemon.proactivity.base import AT_KEYBOARD_SECONDS, Presence, Reading
from daemon.proactivity.presence import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    FRONTMOST,
    IOREG,
    IS_RUNNING_SOMEWHERE,
    LSAPPINFO,
    MUTED,
    OSASCRIPT,
    SYSTEM_OBJECT,
    SYSTEM_PROFILER,
    UNKNOWN_OBJECT,
    VOLUME,
    MachinePresence,
    ProbeError,
    audio_running,
)

PINNED = datetime(2026, 8, 3, 21, 30, tzinfo=UTC)
"""The suite's convention (tests/CLAUDE.md). Pinned so `Reading.at` is assertable."""

# One real `ioreg -c IOHIDSystem -d 1 -r` dump, trimmed to the shape that
# matters: the indentation and the `|` gutter are what a positional parser would
# trip over, so they are kept verbatim.
IOREG_DUMP = """+-o IOHIDSystem  <class IOHIDSystem, id 0x1000008e0, registered, matched>
  | {
  |   "IOClass" = "IOHIDSystem"
  |   "HIDIdleTime" = 6654789416
  |   "IOProviderClass" = "IOResources"
  | }
"""

# And the two real `lsappinfo` outputs, verbatim including the quoting - which is
# the whole defence against `[ NULL ]`, so it is not cosmetic here.
LS_FRONT = "ASN:0x0-0x761761:\n"
LS_INFO = '"LSDisplayName"="Google Chrome"\n'
LS_NULL_INFO = '"LSDisplayName"=[ NULL ]\n'

# Two `system_profiler SPAudioDataType` dumps, trimmed to the shape that matters:
# one block per device, and the `Default Output Device: Yes` marker on the block
# `_default_output_transport` has to find - it is not always the first block, and
# real output never guarantees an order.
#
# NOTE for whoever verifies this against the real machine (2026-08-11, this
# machine): a live `system_profiler SPAudioDataType` capture here shows the
# default output device - `MacBook Pro Speakers (eqMac)` - answering
# `Transport: USB`, not `Transport: Virtual`. That is *inside*
# `HEADPHONE_TRANSPORTS` as written, so `_headphones()` on this real machine
# currently reads True for built-in speakers behind eqMac's proxy - the opposite
# of what `HEADPHONE_TRANSPORTS`'s own docstring claims a virtual device does.
# `SP_AUDIO_BUILTIN` below uses `Transport: Virtual` instead, to isolate the
# claim this test suite is actually pinning (a transport outside the list is not
# headphones) from that unresolved real-machine discrepancy. See the task report.
SP_AUDIO_HEADPHONES = """Audio:

    Devices:

        MacBook Pro Microphone:

          Input Channels: 1
          Manufacturer: Apple Inc.
          Current SampleRate: 48000
          Transport: Built-in
          Input Source: MacBook Pro Microphone

        AirPods Pro:

          Default Output Device: Yes
          Default System Output Device: Yes
          Manufacturer: Apple Inc.
          Output Channels: 2
          Current SampleRate: 48000
          Transport: Bluetooth
          Output Source: AirPods Pro
"""

SP_AUDIO_BUILTIN = """Audio:

    Devices:

        MacBook Pro Microphone:

          Default Input Device: Yes
          Input Channels: 1
          Manufacturer: Apple Inc.
          Current SampleRate: 48000
          Transport: Built-in
          Input Source: MacBook Pro Microphone

        MacBook Pro Speakers (eqMac):

          Default Output Device: Yes
          Default System Output Device: Yes
          Manufacturer: Bitgapp Ltd
          Output Channels: 2
          Current SampleRate: 44100
          Transport: Virtual
          Output Source: MacBook Pro Speakers (eqMac)
"""


class FakeRunner:
    """Stands in for the command runner.

    Keyed by command, and `lsappinfo`'s two calls are separate keys, because the
    chain fails in two different places and each one has to be reachable from a
    test. `osascript` is keyed by *script* (`argv[2]`, the text after `-e`), not
    just by command: the foreground probe, the mute probe and the volume probe
    are all `osascript`, and one canned answer for all three is how a mute-state
    call would silently receive the foreground app's name instead.
    """

    def __init__(
        self,
        *,
        ioreg: str | Exception = IOREG_DUMP,
        ls_front: str | Exception = LS_FRONT,
        ls_info: str | Exception = LS_INFO,
        osascript: str | Exception = "Google Chrome\n",
        muted: str | Exception = "false\n",
        volume: str | Exception = "50\n",
        sp_audio: str | Exception = SP_AUDIO_BUILTIN,
    ) -> None:
        self.ioreg = ioreg
        self.ls_front = ls_front
        self.ls_info = ls_info
        self.osascript = osascript
        self.muted = muted
        self.volume = volume
        self.sp_audio = sp_audio
        self.calls: list[list[str]] = []

    async def __call__(self, argv: Sequence[str]) -> str:
        self.calls.append(list(argv))
        if argv[0] == IOREG:
            reply = self.ioreg
        elif argv[0] == LSAPPINFO:
            reply = self.ls_front if argv[1] == "front" else self.ls_info
        elif argv[0] == OSASCRIPT:
            script = argv[2]
            if script == MUTED:
                reply = self.muted
            elif script == VOLUME:
                reply = self.volume
            elif script == FRONTMOST:
                reply = self.osascript
            else:  # pragma: no cover - a new script must be added deliberately
                raise AssertionError(f"unexpected osascript {script!r}")
        elif argv[0] == SYSTEM_PROFILER:
            reply = self.sp_audio
        else:  # pragma: no cover - a new command must be added deliberately
            raise AssertionError(f"unexpected command {argv[0]!r}")
        if isinstance(reply, Exception):
            raise reply
        return reply

    def ran(self, command: str) -> list[list[str]]:
        return [argv for argv in self.calls if argv[0] == command]


RUNNER_KEYS = ("ioreg", "ls_front", "ls_info", "osascript", "muted", "volume", "sp_audio")


def build(**kwargs: object) -> tuple[MachinePresence, FakeRunner]:
    """A macOS presence whose hardware is entirely fake."""
    runner = FakeRunner(**{k: v for k, v in kwargs.items() if k in RUNNER_KEYS})  # type: ignore[arg-type]
    rest = {k: v for k, v in kwargs.items() if k not in RUNNER_KEYS}
    rest.setdefault("audio", lambda selector: False)
    # An unlocked session with no lock key, same as the real machine unlocked
    # (verified 2026-08-11) - so a test that does not care about screen_locked
    # does not have to fake one, and none of them touches the real Quartz call.
    rest.setdefault("session", lambda: {})
    return (
        MachinePresence(platform="darwin", run=runner, now=PINNED, **rest),  # type: ignore[arg-type]
        runner,
    )


def reason_for(reading: Reading, field: str) -> str:
    """The `unknown` entry for one field, or "" if that field answered."""
    for entry in reading.unknown:
        if entry.startswith(f"{field}: "):
            return entry
    return ""


# --- the happy path ----------------------------------------------------------


async def test_a_complete_reading_answers_every_probe() -> None:
    reader, _ = build(audio=lambda selector: True)
    reading = await reader.read()

    assert reading.at == PINNED
    assert reading.idle_seconds == pytest.approx(6.654789416)
    assert reading.foreground_app == "Google Chrome"
    assert reading.mic_busy is True
    assert reading.output_busy is True
    assert reading.unknown == ()


async def test_machine_presence_satisfies_the_protocol() -> None:
    reader, _ = build()
    assert isinstance(reader, Presence)


async def test_hid_idle_time_is_read_as_nanoseconds() -> None:
    """The one unit conversion in the module, and the one worth a test of its own.

    Verified against the wall clock rather than looked up (see the module
    docstring): sampling `HIDIdleTime` around a 2.0 s sleep gave 2.017/2.024/2.032
    seconds after dividing by 1e9. Wrong by 1e9 the other way, every reading would
    say idle ~= 0, which is `at_keyboard is True`, which is the speaker.
    """
    dump = IOREG_DUMP.replace("6654789416", "540000000000")  # nine minutes
    reader, _ = build(ioreg=dump)

    reading = await reader.read()

    assert reading.idle_seconds == pytest.approx(540.0)
    assert reading.at_keyboard is False


async def test_the_smallest_idle_time_wins_when_several_nodes_report_one() -> None:
    """Idle means "since the last event on any device", so the freshest one is the
    answer. Averaging or taking the first would let an untouched device claim the
    user is away while they are typing."""
    dump = IOREG_DUMP + '  |   "HIDIdleTime" = 900000000000\n'
    reader, _ = build(ioreg=dump)

    reading = await reader.read()

    assert reading.idle_seconds == pytest.approx(6.654789416)


# --- the foreground app: two probes, cheapest first --------------------------


async def test_lsappinfo_answers_and_osascript_is_never_asked() -> None:
    """The reason `lsappinfo` is primary: 8 ms against 233 ms, and no Automation
    grant. If this ever regresses, every reading silently costs 30x more and
    acquires a permission dependency nobody asked for.

    `osascript` is not asserted absent outright: the mute and headphone probes
    are unconditional `osascript`/`system_profiler` users with nothing to do with
    the foreground-app fallback this test pins. What must never happen is that
    fallback - the `FRONTMOST` script - running while `lsappinfo` still answers.
    """
    reader, runner = build()

    reading = await reader.read()

    assert reading.foreground_app == "Google Chrome"
    assert all(argv[2] != FRONTMOST for argv in runner.ran(OSASCRIPT))
    assert [argv[1] for argv in runner.ran(LSAPPINFO)] == ["front", "info"]


async def test_the_asn_from_the_first_call_is_passed_to_the_second() -> None:
    reader, runner = build()
    await reader.read()
    assert runner.ran(LSAPPINFO)[1] == [LSAPPINFO, "info", "-only", "name", "ASN:0x0-0x761761:"]


async def test_a_null_display_name_does_not_become_the_application(
) -> None:
    """The sharp edge. `lsappinfo` prints `"LSDisplayName"=[ NULL ]` - unquoted -
    for an ASN that stopped resolving, and **exits 0 doing it**. Treating that as a
    name would give the gate an app called `[ NULL ]` to match focus rules against,
    forever, with nothing anywhere reporting a failure."""
    reader, runner = build(ls_info=LS_NULL_INFO, osascript="Slack\n")

    reading = await reader.read()

    assert reading.foreground_app == "Slack"
    assert reading.foreground_app != "[ NULL ]"
    assert runner.ran(OSASCRIPT), "the fallback should have been asked"


async def test_lsappinfo_exiting_zero_on_a_refusal_is_still_a_failure() -> None:
    """Verified on this machine: `lsappinfo -only name front` prints
    `Unrecognized command` and exits 0. So the return-code check in `_spawn`
    protects nothing here and parsing is the entire defence."""
    reader, _ = build(ls_front="Unrecognized command: -only\n", osascript="Warp\n")

    reading = await reader.read()

    assert reading.foreground_app == "Warp"


async def test_a_first_call_without_an_asn_does_not_make_a_second() -> None:
    """No point spending a second call on an ASN that is not one, and the reason
    should name the call that actually broke."""
    reader, runner = build(ls_front="\n", osascript="Warp\n")

    await reader.read()

    assert [argv[1] for argv in runner.ran(LSAPPINFO)] == ["front"]


async def test_the_osascript_fallback_carries_the_reading_when_lsappinfo_is_gone() -> None:
    reader, runner = build(
        ls_front=ProbeError("/usr/bin/lsappinfo could not start: no such file"),
        osascript="System Settings\n",
    )

    reading = await reader.read()

    assert reading.foreground_app == "System Settings"
    assert reading.unknown == ()  # a fallback that worked is not a degradation
    assert runner.ran(OSASCRIPT)


async def test_both_foreground_probes_failing_keeps_both_reasons() -> None:
    """One entry, both reasons. An operator needs to know whether the fast path
    broke or whether the whole idea of asking broke - different fixes."""
    reader, _ = build(
        ls_front=ProbeError("lsappinfo could not start"),
        osascript=ProbeError("osascript exited 1: Not authorized (-1743)"),
    )

    reading = await reader.read()

    assert reading.foreground_app is None
    assert reason_for(reading, "foreground_app") == (
        "foreground_app: lsappinfo could not start; "
        "osascript exited 1: Not authorized (-1743)"
    )


def test_the_two_probes_are_asked_the_same_question() -> None:
    """A regression guard for a real defect, not a style check.

    docs/PLAN.md 6.3's spelling is `name of ... application process`, which returns
    the *executable* name. Running the probes against each other on one frontmost
    window caught it: `lsappinfo` said `Warp`, `osascript` said `stable`, because
    Warp's binary is `Contents/MacOS/stable`. The gate matches this field against
    app names a person typed into configuration, so two vocabularies means a focus
    rule that fires or not depending on which probe answered.
    """
    assert "displayed name" in FRONTMOST


@pytest.mark.parametrize("via", ["lsappinfo", "osascript"])
async def test_a_korean_application_name_survives_either_probe(via: str) -> None:
    if via == "lsappinfo":
        reader, _ = build(ls_info='"LSDisplayName"="카카오톡"\n')
    else:
        reader, _ = build(ls_front=ProbeError("gone"), osascript="카카오톡\n")

    reading = await reader.read()

    assert reading.foreground_app == "카카오톡"
    # And it has to survive being JSON for `proactive_utterances.gate_snapshot`.
    assert reading.as_snapshot()["foreground_app"] == "카카오톡"


@pytest.mark.parametrize("via", ["lsappinfo", "osascript"])
async def test_a_hostile_application_name_is_bounded_and_single_line(via: str) -> None:
    """An application names itself, so this is untrusted text on its way into a
    JSON column and an LLM prompt. Bounded, and the first line only - on both
    paths, because a bound only one probe applies is not a bound."""
    hostile = "가" * 500 + "\\nsecond line"
    if via == "lsappinfo":
        reader, _ = build(ls_info=f'"LSDisplayName"="{hostile}"\n')
    else:
        reader, _ = build(ls_front=ProbeError("gone"), osascript="가" * 500 + "\nsecond line\n")

    reading = await reader.read()

    assert reading.foreground_app is not None
    assert len(reading.foreground_app) == 128


async def test_an_empty_answer_from_both_probes_is_unknown() -> None:
    reader, _ = build(ls_info='"LSDisplayName"=""\n', osascript="\n  \n")
    reading = await reader.read()
    assert reading.foreground_app is None
    assert "no application name" in reason_for(reading, "foreground_app")


# --- a probe that cannot answer ---------------------------------------------


async def test_an_ioreg_dump_without_hid_idle_time_is_unknown_not_zero() -> None:
    """`ioreg` exits 0 with empty output when its class does not match, so the
    failure arrives looking exactly like success. Reporting 0.0 here would be the
    most expensive wrong answer the module can give: it reads as the user sitting
    at the machine, and routes an utterance to the speaker."""
    reader, _ = build(ioreg="+-o Root  <class IORegistryEntry>\n")

    reading = await reader.read()

    assert reading.idle_seconds is None
    assert reading.at_keyboard is None
    assert "HIDIdleTime" in reason_for(reading, "idle_seconds")
    # The other two still answered - one dead probe is not a dead reading.
    assert reading.foreground_app == "Google Chrome"
    assert reading.mic_busy is False
    assert reading.output_busy is False


async def test_an_audio_probe_that_fails_is_unknown_not_idle() -> None:
    """The `False` this must not invent is the one PLAN 6.4 is written about:
    "nothing is using the audio device" is the reading that clears a voice to come
    out of the speaker, and a probe that failed has not established it."""

    def broken(selector: int) -> bool:
        raise ProbeError("CoreAudio returned OSStatus 1852797029")

    reader, _ = build(audio=broken)
    reading = await reader.read()

    assert reading.mic_busy is None
    assert "OSStatus" in reason_for(reading, "mic_busy")
    assert reading.as_snapshot()["mic_busy"] is None


async def test_every_probe_failing_still_produces_a_reading() -> None:
    reader, _ = build(
        ioreg=ProbeError("ioreg timed out"),
        ls_front=ProbeError("lsappinfo timed out"),
        osascript=ProbeError("osascript timed out"),
        audio=lambda selector: (_ for _ in ()).throw(ProbeError("no default audio device")),
    )

    reading = await reader.read()

    fields = (reading.idle_seconds, reading.foreground_app, reading.mic_busy, reading.output_busy)
    assert fields == (None, None, None, None)
    assert reading.at == PINNED
    assert len(reading.unknown) == 4  # one entry per field, not per command


async def test_read_never_raises_on_an_unanticipated_failure() -> None:
    """A raising scheduler job is logged once and the schedule then reads healthy
    forever while presence never answers again (daemon/CLAUDE.md). So an exception
    nobody predicted has to come back as a reason, not as a traceback."""

    def catastrophe(selector: int) -> bool:
        raise MemoryError("ctypes went sideways")

    reader, _ = build(ioreg=ValueError("something nobody planned for"), audio=catastrophe)

    reading = await reader.read()

    assert reading.idle_seconds is None
    assert reading.mic_busy is None
    assert "unexpected ValueError" in reason_for(reading, "idle_seconds")
    assert "unexpected MemoryError" in reason_for(reading, "mic_busy")


async def test_an_unexpected_failure_in_the_foreground_chain_is_still_a_reason() -> None:
    """The chain catches `ProbeError` to try the fallback; anything else has to
    reach `_probe`'s net rather than escaping `read()`."""
    reader, _ = build(ls_front=RuntimeError("lsappinfo did something novel"))

    reading = await reader.read()

    assert reading.foreground_app is None
    assert "unexpected RuntimeError" in reason_for(reading, "foreground_app")


# --- not a Mac ---------------------------------------------------------------


@pytest.mark.parametrize("platform", ["linux", "win32", "freebsd14"])
async def test_a_non_macos_platform_answers_unknown_without_probing(platform: str) -> None:
    """Not a crash and not a pretence. Everything unknown, once, with the platform
    named - and nothing spawned, because these commands do not exist there."""
    runner = FakeRunner()

    def must_not_run(selector: int) -> bool:
        raise AssertionError("the audio probe ran on a platform that has no CoreAudio")

    reading = await MachinePresence(
        platform=platform, run=runner, audio=must_not_run, now=PINNED
    ).read()

    fields = (reading.idle_seconds, reading.foreground_app, reading.mic_busy, reading.output_busy)
    assert fields == (None, None, None, None)
    assert reading.at_keyboard is None
    assert reading.unknown == (f"platform {platform!r}: no presence probes implemented",)
    assert runner.calls == []


# --- at_keyboard, the three-way answer --------------------------------------


async def test_at_keyboard_is_none_when_idle_is_unknown() -> None:
    """`base.py`'s third state, end to end. Python truthiness would flatten
    "cannot be known" into "away", and the gate needs to tell them apart."""
    reader, _ = build(ioreg="nothing useful here")
    reading = await reader.read()
    assert reading.at_keyboard is None
    assert reading.at_keyboard is not False


@pytest.mark.parametrize(
    ("nanoseconds", "expected"),
    [
        (0, True),
        (int(AT_KEYBOARD_SECONDS * 1e9) - 1, True),
        (int(AT_KEYBOARD_SECONDS * 1e9), False),
        (int(3600 * 1e9), False),
    ],
)
async def test_at_keyboard_follows_the_measured_idle_time(
    nanoseconds: int, expected: bool
) -> None:
    reader, _ = build(ioreg=f'  |   "HIDIdleTime" = {nanoseconds}\n')
    reading = await reader.read()
    assert reading.at_keyboard is expected


# --- the command runner, with no command ------------------------------------


class FakeProcess:
    """An `asyncio.subprocess.Process` as far as `_spawn` uses one."""

    def __init__(
        self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0, hang: bool = False
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self._hang = hang
        self.killed = False
        self.reaped = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.Event().wait()  # forever, like osascript behind a TCC dialog
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        self.reaped = True
        return self.returncode


@pytest.fixture
def spawned(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Intercept process creation. Nothing in this file starts a real one."""
    seen: dict[str, object] = {"argvs": []}

    async def fake_exec(*argv: str, **kwargs: object) -> object:
        argvs = seen["argvs"]
        assert isinstance(argvs, list)
        argvs.append(list(argv))
        seen["kwargs"] = kwargs
        return seen["process"]

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return seen


async def test_a_hung_command_is_killed_reaped_and_reported(spawned: dict[str, object]) -> None:
    """The failure the timeout exists for: `osascript` behind an Automation consent
    dialog does not fail, it waits for a click that is never coming. Unbounded that
    is one wedged tick every five minutes, forever."""
    process = FakeProcess(hang=True)
    spawned["process"] = process
    reader = MachinePresence(platform="darwin", timeout=0.05, audio=lambda selector: False)

    with pytest.raises(ProbeError, match="did not answer in 0.05s"):
        await reader._spawn([OSASCRIPT, "-e", "anything"])

    # Killed *and* reaped: an orphan still holding the pipes leaks a pair of file
    # descriptors per tick.
    assert process.killed and process.reaped


async def test_a_timed_out_probe_becomes_an_unknown_rather_than_an_exception(
    spawned: dict[str, object],
) -> None:
    spawned["process"] = FakeProcess(hang=True)
    reader = MachinePresence(
        platform="darwin", timeout=0.05, audio=lambda selector: False, now=PINNED
    )

    reading = await reader.read()

    assert reading.idle_seconds is None and reading.foreground_app is None
    assert "did not answer" in reason_for(reading, "idle_seconds")


async def test_a_missing_binary_is_a_reason_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def missing(*argv: str, **kwargs: object) -> object:
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing)
    reader = MachinePresence(platform="darwin", audio=lambda selector: False, now=PINNED)

    reading = await reader.read()

    assert reading.idle_seconds is None
    assert "could not start" in reason_for(reading, "idle_seconds")


async def test_a_nonzero_exit_carries_what_the_command_said(spawned: dict[str, object]) -> None:
    """osascript's real refusal - "Not authorized to send Apple events" - arrives
    on stderr with rc 1, and that text is the only thing that tells an operator
    the fix is a permission rather than a bug."""
    spawned["process"] = FakeProcess(
        returncode=1, stderr=b"execution error: Not authorized to send Apple events. (-1743)"
    )
    reader = MachinePresence(platform="darwin", audio=lambda selector: False, now=PINNED)

    reading = await reader.read()

    assert reading.foreground_app is None
    assert "-1743" in reason_for(reading, "foreground_app")
    assert "exited 1" in reason_for(reading, "foreground_app")


async def test_a_silent_failure_still_says_something(spawned: dict[str, object]) -> None:
    spawned["process"] = FakeProcess(returncode=127)
    reader = MachinePresence(platform="darwin", audio=lambda selector: False)
    with pytest.raises(ProbeError, match="exited 127: no output"):
        await reader._spawn([IOREG])


async def test_every_command_is_a_list_with_no_shell_and_no_stdin(
    spawned: dict[str, object],
) -> None:
    """`shell=True` is how "the probe reads a value" becomes "the probe runs a
    string", and a closed stdin is why a command that decides to read one cannot
    block the tick."""
    spawned["process"] = FakeProcess(stdout=IOREG_DUMP.encode())
    reader = MachinePresence(platform="darwin", audio=lambda selector: False, now=PINNED)

    await reader.read()

    kwargs = spawned["kwargs"]
    assert isinstance(kwargs, dict)
    assert "shell" not in kwargs
    assert kwargs["stdin"] is asyncio.subprocess.DEVNULL
    argvs = spawned["argvs"]
    assert isinstance(argvs, list) and argvs
    for argv in argvs:
        assert isinstance(argv, list)
        # Absolute paths, because PATH under a LaunchAgent is not PATH in a terminal.
        assert argv[0].startswith("/"), argv


async def test_undecodable_output_is_parsed_rather_than_raising(
    spawned: dict[str, object],
) -> None:
    """`ioreg` prints device names, and a peripheral's name is whatever its
    firmware says it is. A byte sequence that is not UTF-8 must not cost the tick
    a reading it otherwise had."""
    spawned["process"] = FakeProcess(stdout=b'  \xff\xfe "HIDIdleTime" = 1500000000\n')
    reader = MachinePresence(platform="darwin", audio=lambda selector: False, now=PINNED)

    reading = await reader.read()

    assert reading.idle_seconds == pytest.approx(1.5)


# --- the CoreAudio probe itself ---------------------------------------------


@pytest.mark.parametrize(
    ("selector", "device_id", "running", "expected"),
    [
        (DEFAULT_OUTPUT, 84, 0, False),
        (DEFAULT_OUTPUT, 84, 1, True),  # music, or our own LocalSpeaker
        (DEFAULT_INPUT, 100, 0, False),
        (DEFAULT_INPUT, 100, 1, True),  # the microphone is held - a call
    ],
)
def test_audio_running_reads_one_device_by_selector(
    monkeypatch: pytest.MonkeyPatch, selector: int, device_id: int, running: int, expected: bool
) -> None:
    """One device per call, not the OR of both - the merge is what let the wake
    listener's own hold on the input device present as the audio hardware being
    busy generally, which is the bug this split exists to fix."""

    def fake_property(obj: int, prop_selector: int) -> int:
        if obj == SYSTEM_OBJECT:
            return device_id
        assert prop_selector == IS_RUNNING_SOMEWHERE
        return running

    monkeypatch.setattr(presence_module, "_uint32_property", fake_property)
    assert audio_running(selector) is expected


def test_a_missing_default_device_is_a_probe_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A machine with no microphone has no default input device. Nothing was
    measured, so "not busy" would be invented rather than read - and each
    selector now answers for itself instead of falling back to the other."""
    monkeypatch.setattr(
        presence_module, "_uint32_property", lambda obj, selector: UNKNOWN_OBJECT
    )
    with pytest.raises(ProbeError, match="no such default audio device"):
        audio_running(DEFAULT_INPUT)


async def test_a_stalled_coreaudio_call_does_not_stall_the_reading() -> None:
    """It is 0.1 ms warm, so this branch is `coreaudiod` wedged. Bounded anyway,
    because the tick has to finish either way."""

    def wedged(selector: int) -> bool:
        time.sleep(0.5)
        return True

    reader = MachinePresence(
        platform="darwin", run=FakeRunner(), audio=wedged, timeout=0.05, now=PINNED
    )

    started = time.perf_counter()
    reading = await reader.read()
    elapsed = time.perf_counter() - started

    assert reading.mic_busy is None
    assert "did not answer" in reason_for(reading, "mic_busy")
    assert elapsed < 0.4, "the audio probe blocked the reading past its own timeout"
    # The rest of the reading survived it.
    assert reading.idle_seconds == pytest.approx(6.654789416)


# --- the snapshot the gate stores -------------------------------------------


async def test_the_snapshot_carries_the_reasons_a_bad_call_would_need() -> None:
    """`proactive_utterances.gate_snapshot` is where a wrong decision gets read
    back off, so which probes were blind has to be in it, not only in a log."""
    reader, _ = build(ls_front=ProbeError("timed out"), osascript=ProbeError("timed out"))

    snapshot = (await reader.read()).as_snapshot()

    assert snapshot["idle_seconds"] == pytest.approx(6.654789416)
    assert snapshot["foreground_app"] is None
    assert snapshot["unknown"] == ["foreground_app: timed out; timed out"]


# --- Reading: the merged audio signal split in two --------------------------


def test_reading_separates_microphone_from_output() -> None:
    """The merged `audio_busy` is what made enabling voice disable the speaker:
    the wake listener holds the input device, and the gate could not tell that
    apart from a call. See the spec, section 1.1 cause 3."""
    reading = Reading(at=PINNED, mic_busy=False, output_busy=True)
    assert reading.mic_busy is False
    assert reading.output_busy is True


def test_reading_snapshot_carries_every_new_field() -> None:
    """gate_snapshot is how a bad call is diagnosed months later. A field the
    gate reads but the snapshot drops is a decision nobody can reconstruct."""
    reading = Reading(
        at=PINNED,
        idle_seconds=1.0,
        foreground_app="Warp",
        mic_busy=False,
        output_busy=False,
        output_muted=True,
        screen_locked=False,
        headphones=True,
    )
    snapshot = reading.as_snapshot()
    for key in (
        "idle_seconds", "foreground_app", "mic_busy", "output_busy",
        "output_muted", "screen_locked", "headphones", "unknown",
    ):
        assert key in snapshot, f"{key} is missing from the gate snapshot"
    assert "audio_busy" not in snapshot, "the merged field must be gone, not kept"


# --- the microphone and the output device, read apart ------------------------


def audio_probe(*, mic: bool, out: bool):
    """A stand-in for the CoreAudio probe, answering per device selector."""

    def probe(selector: int) -> bool:
        return mic if selector == DEFAULT_INPUT else out

    return probe


async def test_our_own_microphone_hold_is_not_a_call() -> None:
    """The whole point. With the wake listener running, the raw probe says the
    input device is busy; the gate must not read that as somebody on a call."""
    reader, _ = build(audio=audio_probe(mic=True, out=False), mic_held=lambda: True)
    reading = await reader.read()
    assert reading.mic_busy is False


async def test_somebody_elses_microphone_hold_is_a_call() -> None:
    reader, _ = build(audio=audio_probe(mic=True, out=False), mic_held=lambda: False)
    reading = await reader.read()
    assert reading.mic_busy is True


async def test_output_is_read_independently_of_the_microphone() -> None:
    reader, _ = build(audio=audio_probe(mic=False, out=True), mic_held=lambda: False)
    reading = await reader.read()
    assert reading.mic_busy is False
    assert reading.output_busy is True


async def test_we_do_not_probe_a_device_we_already_hold() -> None:
    """Not an optimisation. If we hold it the device is busy by definition, so
    the probe can only return the answer we must not act on."""

    def must_not_run(selector: int) -> bool:
        raise AssertionError("the microphone probe ran while we held the device")

    reader, _ = build(audio=must_not_run, mic_held=lambda: True)
    reading = await reader.read()
    assert reading.mic_busy is False


# --- mute, screen lock, headphones -------------------------------------------


async def test_muted_output_is_read_as_muted() -> None:
    reader, _ = build(muted="true\n")
    reading = await reader.read()
    assert reading.output_muted is True


async def test_zero_volume_counts_as_muted() -> None:
    """Nobody hears 0% either, and `say` still exits 0. The two states differ in
    the Settings pane and not in the room."""
    reader, _ = build(muted="false\n", volume="0\n")
    reading = await reader.read()
    assert reading.output_muted is True


async def test_an_audible_machine_is_not_muted() -> None:
    reader, _ = build(muted="false\n", volume="50\n")
    reading = await reader.read()
    assert reading.output_muted is False


async def test_an_unreadable_mute_state_is_unknown_not_audible() -> None:
    """`None` and not `False`: False is what lets a line be spoken aloud, and an
    osascript that answered nonsense is not evidence anybody would hear it."""
    reader, _ = build(muted="Google Chrome\n")
    reading = await reader.read()
    assert reading.output_muted is None
    assert reason_for(reading, "output_muted") != ""


async def test_a_locked_screen_is_recorded() -> None:
    reader, _ = build(session=lambda: {"CGSSessionScreenIsLocked": 1})
    reading = await reader.read()
    assert reading.screen_locked is True


async def test_an_absent_lock_key_means_unlocked_not_unknown() -> None:
    """macOS omits the key entirely when unlocked - it does not set it to 0
    (verified 2026-08-11). A probe that read the absence as "could not answer"
    would route every utterance to Telegram for the life of the process."""
    reader, _ = build(session=lambda: {"kCGSSessionOnConsoleKey": True})
    reading = await reader.read()
    assert reading.screen_locked is False


async def test_no_quartz_is_unknown_rather_than_unlocked() -> None:
    reader, _ = build(session=lambda: None)
    reading = await reader.read()
    assert reading.screen_locked is None


async def test_headphones_are_read_from_the_default_output_transport() -> None:
    reader, _ = build(sp_audio=SP_AUDIO_HEADPHONES)
    reading = await reader.read()
    assert reading.headphones is True


async def test_built_in_speakers_are_not_headphones() -> None:
    """The default output on the development machine is
    `MacBook Pro Speakers (eqMac)` - a virtual device in front of the built-in
    speakers. The name describes the proxy; the transport is what is read."""
    reader, _ = build(sp_audio=SP_AUDIO_BUILTIN)
    reading = await reader.read()
    assert reading.headphones is False
