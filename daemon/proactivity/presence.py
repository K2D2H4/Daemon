"""Reading the machine, so the gate can decide where an utterance may go.

The `Presence` implementation behind docs/PLAN.md 6.3's routing decision. Three
probes, and the reason the shape is this careful is docs/PLAN.md 6.4: an ignored
notification costs nothing, and **a voice coming out of the speaker during a
meeting is an accident.** The asymmetry runs one way, so every failure here
resolves towards `None` rather than towards a cheerful `False`, because `False`
for `mic_busy` or `output_busy` is what routes to the speaker.

## What each probe is, and what it actually cost

Measured on this machine (M4 Max, macOS 26.5.2 / Darwin 25.5.0), which is the
machine docs/PLAN.md 6.3 recorded the probes working on:

| probe | mechanism | cost |
|---|---|---|
| `idle_seconds` | `ioreg -c IOHIDSystem` | 11-16 ms |
| `foreground_app` | `lsappinfo`, two calls | 6.9-12 ms, median 7.7 ms |
| `foreground_app` fallback | `osascript` + System Events | 168-233 ms warm, 464 ms cold |
| `mic_busy` / `output_busy` | CoreAudio, via ctypes | 0.08-0.14 ms warm, 77 ms first call |
| `screen_locked` | Quartz, in-process | 0.09-0.15 ms warm, ~18 ms cold |
| `output_muted` | `osascript`: `MUTED`, `VOLUME` if not muted | 134-178 ms, +145-247 ms if not |

A reading costs ~20 ms when `lsappinfo` answers, plus `output_muted`'s cost:
**~155-200 ms when the machine is muted** (one `osascript` call) and **~300-445 ms
when it is not** (two). Add ~200 ms more on top of either if `foreground_app`
falls through to its own `osascript` call. All of this is fine on a five-minute
tick and none of it belongs on the voice latency path; nothing calls this from
there.

## Why `foreground_app` has two probes and `lsappinfo` goes first

Same reasoning as reading both audio directions: two mechanisms that fail
differently beat one that fails silently.

`lsappinfo` is 30x faster and, being a LaunchServices query rather than an Apple
event, depends on no TCC grant at all - see the section below for why that
matters more than the speed. It costs two calls because there is no one-shot
spelling: `lsappinfo info -only name front` and `lsappinfo -only name front` both
decline (`Unrecognized command`), so the ASN comes from `lsappinfo front` and the
name from a second call.

**`lsappinfo` never exits non-zero, and that is the trap.** Verified: an
unrecognised command, an unresolvable ASN and pure garbage as the target all exit
0. So `_spawn`'s return-code check catches nothing here and every failure has to
be caught while parsing. The two shapes are an empty stdout, and this:

    "LSDisplayName"=[ NULL ]

which is what an ASN that no longer resolves prints. Note it is *unquoted* - so
`_LS_DISPLAY_NAME` requiring a quote after the `=` is what keeps `[ NULL ]` from
becoming the application's name. That is load-bearing, not incidental, and
`tests/test_presence.py` pins it.

`osascript` stays as the fallback, and the fallback is not decoration: it is the
probe docs/PLAN.md 6.3 recorded as working, it goes through an entirely different
system service, and if `lsappinfo`'s output shape ever changes it is what keeps
this field answerable.

**`HIDIdleTime` is in nanoseconds, and this was verified rather than looked up.**
Sampling it twice around a 2.0 s sleep gave deltas of 2.017, 2.024 and 2.032
seconds after dividing by 1e9 - three decimal places of agreement with the wall
clock, so the unit is not a guess. It matters more than it looks: a unit that is
wrong by 1e9 in the *other* direction reports every reading as idle_seconds ~= 0,
which is `at_keyboard is True`, which is the speaker. So the one number this file
converts is the one worth having measured.

## Why the audio probe is ctypes and not a command

Every documented CLI answer for "is something using the audio device" is dead on
Apple Silicon. Tried, and each returned nothing at all while `afplay` was
provably running: `ioreg -c IOAudioEngine`, `ioreg -c AppleHDAEngineOutput`,
`ioreg -c IOUserAudioDevice`, and `ioreg -l | grep IOAudioEngineState` - those
classes do not exist in this IORegistry. `pmset -g assertions` shows no
`coreaudiod` entry during playback either.

What does work is the public C API, and it works in both directions:
`kAudioDevicePropertyDeviceIsRunningSomewhere` on the default output device read
0 while silent and 1 while `afplay` played, and on the default *input* device it
read 0 then 1 while ffmpeg held the microphone. The input half is the one PLAN
6.4 is actually about - a mic in use is a call in progress - so both are read and
either one busy means busy.

Two consequences worth knowing. The first call in a process pays 77 ms setting up
the connection to `coreaudiod`; every later call is ~0.1 ms, so a resident daemon
pays it once. And this cannot distinguish *our own* speaker from someone else's
call, so while `LocalSpeaker` is talking this reads busy - which is the answer
that stops two utterances overlapping, so it is left as is.

## The `osascript` permission question, unresolved - and why it is now cheap

docs/PLAN.md 6.3 records `osascript` foreground lookup as measured working. It is,
from a terminal. The concern is that the measurement was taken from a process that
had already been granted Automation access, and M3's precondition is residency -
the daemon running under launchd, which is a *different* TCC client from the
terminal a developer tests in.

**Verified on this machine:**

  * `osascript` + System Events works from an interactive shell: 168-233 ms,
    returns the name.
  * The LaunchAgent already exists and is running: `ai.daemon.default`, state
    `running`, domain `gui/501`, program `.venv/bin/daemon`. So production is a
    launchd job today, not hypothetically.
  * The user TCC database cannot be read to check who holds the grant -
    `authorization denied`, SIP - and `log show` on the `com.apple.TCC`
    subsystem has no AppleEvents decision in the last six hours. So the grant
    cannot be enumerated from here at all.
  * `lsappinfo` needs no Automation grant, and it never prompted.

**Reasoned, not verified** (and one earlier claim of mine was wrong):

  * I previously wrote that a LaunchAgent "cannot prompt". That is incorrect. The
    agent runs in `gui/501`, which *has* a GUI session and therefore can display a
    consent dialog. What remains is that until somebody clicks it, `osascript`
    does not fail - it waits - and that the resulting grant binds to the
    `.venv/bin/daemon` path, so rebuilding the venv plausibly invalidates it.
  * Whether it prompts, denies with -1743, or hangs, on a *first* launchd run is
    untested. Settling it needs either a job submitted into launchd or a consent
    dialog answered, and answering one writes a new TCC grant - a change to the
    machine's security settings that is not this file's to make. Left open
    deliberately.

**Why the open question does not block anything.** Two independent reasons, which
is the actual point of the change that made `lsappinfo` primary. First, correctness
no longer depends on the answer: `lsappinfo` carries this field with no grant, and
`osascript` is a fallback whose absence costs 0. Second, all three outcomes are
already handled and tested - a denial is a non-zero exit with its stderr kept, and
a hang is what `PROBE_TIMEOUT_SECONDS` exists for. There is no outcome here that
produces a wrong reading rather than an honest `None`.

## Never raising is a requirement, not politeness

This runs on the scheduler. A raising job is logged once and the schedule then
reads as healthy forever while nothing works (see `daemon/CLAUDE.md`, which makes
that the rule for every background job here). So a probe that cannot answer
appends *why* to `Reading.unknown` and yields `None`, and the caller gets a
reading with holes in it rather than an exception. The reasons are not decoration:
they land in `proactive_utterances.gate_snapshot`, which is the only way a bad
call is diagnosable after the fact instead of guessed at.
"""

from __future__ import annotations

import asyncio
import ctypes
import functools
import logging
import re
import struct
import sys
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime

from daemon import mic_hold
from daemon.clock import now as clock_now
from daemon.proactivity.base import Reading

logger = logging.getLogger(__name__)

IOREG = "/usr/sbin/ioreg"
LSAPPINFO = "/usr/bin/lsappinfo"
OSASCRIPT = "/usr/bin/osascript"
CORE_AUDIO = "/System/Library/Frameworks/CoreAudio.framework/CoreAudio"

PROBE_TIMEOUT_SECONDS = 2.0
"""Wall-clock ceiling per probe. Ten times the slowest measured warm call.

Sized for the failure it exists for rather than for the numbers above. If macOS
has not granted this process Automation access, `osascript` may not fail - it can
wait behind a consent dialog, for as long as nobody clicks it. Unbounded, that is
one wedged scheduler tick per five minutes forever. See the docstring section on
what is verified about that and what is not.
"""

NANOSECONDS = 1_000_000_000
"""`HIDIdleTime`'s unit, verified against the wall clock - see the module docstring."""

MAX_APP_NAME = 128
"""An application chooses its own name, so it is untrusted text on its way into a
JSON snapshot and possibly a prompt. Real names are short; this is a bound, not a
judgement about what a name should be."""

FRONTMOST = (
    'tell application "System Events" to '
    "get displayed name of first application process whose frontmost is true"
)
"""`displayed name`, not `name`, and this was a real disagreement rather than a
preference.

docs/PLAN.md 6.3's spelling is `name`, which returns the *executable* name. Caught
by running the two probes against each other on the same frontmost window:
`lsappinfo` said `Warp` and `osascript` said `stable`, because Warp's binary is
`/Applications/Warp.app/Contents/MacOS/stable`. Electron apps do the same thing in
the other direction.

That matters because the gate matches this field against app names a *person*
wrote in configuration - "Zoom", "Keynote". With two probes returning two
vocabularies, such a rule would fire or not depending on which probe happened to
answer, which is the worst version of a focus rule: one that works until it
quietly does not. `displayed name` agrees with `LSDisplayName` and costs the same
169-194 ms.
"""

_LS_DISPLAY_NAME = re.compile(r'"LSDisplayName"\s*=\s*"(.*)"')
"""`lsappinfo info -only name` prints `"LSDisplayName"="Slack"`.

The required quote after the `=` is the whole defence: an ASN that no longer
resolves prints `"LSDisplayName"=[ NULL ]` *unquoted*, and since `lsappinfo` exits
0 either way, this pattern not matching is the only thing standing between that
and an application called `[ NULL ]`.
"""

_ASN = re.compile(r"ASN:\S+")
"""`lsappinfo front` prints one token, `ASN:0x0-0x761761:`. Matched rather than
trusted because it is about to become argv for the second call, and because a
first call that answered with something else should fail here with a clear reason
instead of one call later with a confusing one."""

_HID_IDLE = re.compile(r'"HIDIdleTime"\s*=\s*(\d+)')
"""`ioreg`'s output is a debugging dump, not an API: the line arrives as
`  |   "HIDIdleTime" = 6654789416` under an indentation that depends on where the
node sits in the tree. So the value is found by pattern rather than by position,
and a dump that does not contain it at all is a failed probe (below) rather than
a zero."""

MUTED = 'output muted of (get volume settings)'
VOLUME = 'output volume of (get volume settings)'
"""Two calls rather than one AppleScript returning a record: parsing
`{output volume:50, output muted:true}` means owning a record parser, and the
second call costs ~120 ms on a five-minute tick.

`get volume settings` is Standard Additions, not System Events, so it needs no
Automation grant - unlike the `osascript` foreground fallback this file already
warns about. Verified answering on this machine (2026-08-11).
"""


class ProbeError(Exception):
    """One probe could not answer, with the reason that goes into `Reading.unknown`."""


def _excerpt(text: str) -> str:
    """Enough of an unexpected output to diagnose it, bounded because it is going
    into a JSON column rather than a log line."""
    return repr(text.strip()[:60])


def _app_name(raw: str, source: str) -> str:
    """One application name, cleaned the same way whichever probe found it.

    First line only, then bounded: an application chooses its own name, so this is
    untrusted text on its way into `proactive_utterances.gate_snapshot` and into an
    LLM prompt.
    """
    stripped = raw.strip()
    name = stripped.splitlines()[0].strip() if stripped else ""
    if not name:
        raise ProbeError(f"{source} returned no application name")
    return name[:MAX_APP_NAME]


# --- audio occupancy, through CoreAudio -------------------------------------
#
# The C API this reaches is `AudioObjectGetPropertyData`, stable since 10.x and
# not deprecated. Everything below it is the four-character-code plumbing that
# call needs; see the module docstring for why no command-line probe replaces it.


def _fourcc(code: str) -> int:
    """A CoreAudio selector: four ASCII bytes read as a big-endian uint32."""
    return struct.unpack(">I", code.encode("ascii"))[0]


SYSTEM_OBJECT = 1  # kAudioObjectSystemObject
UNKNOWN_OBJECT = 0  # kAudioObjectUnknown - "there is no such device"
SCOPE_GLOBAL = _fourcc("glob")
DEFAULT_OUTPUT = _fourcc("dOut")
DEFAULT_INPUT = _fourcc("dIn ")  # the trailing space is part of the code
IS_RUNNING_SOMEWHERE = _fourcc("gone")


class _PropertyAddress(ctypes.Structure):
    """`AudioObjectPropertyAddress`: three UInt32s, in this order."""

    _fields_ = (
        ("selector", ctypes.c_uint32),
        ("scope", ctypes.c_uint32),
        ("element", ctypes.c_uint32),
    )


@functools.cache
def _core_audio() -> ctypes.CDLL:
    """The framework, loaded once. Raises `OSError` if it is not there at all.

    `argtypes` is declared rather than left to ctypes' default int promotion,
    because two of these arguments are pointers and guessing about pointer width
    is how a working call becomes a silent memory bug on some future arch.
    """
    library = ctypes.CDLL(CORE_AUDIO)
    library.AudioObjectGetPropertyData.argtypes = [
        ctypes.c_uint32,  # AudioObjectID
        ctypes.POINTER(_PropertyAddress),
        ctypes.c_uint32,  # qualifier size
        ctypes.c_void_p,  # qualifier data
        ctypes.POINTER(ctypes.c_uint32),  # in/out data size
        ctypes.c_void_p,  # out data
    ]
    library.AudioObjectGetPropertyData.restype = ctypes.c_int32  # OSStatus
    return library


def _uint32_property(obj: int, selector: int) -> int:
    """Read one UInt32 property off one audio object."""
    address = _PropertyAddress(selector, SCOPE_GLOBAL, 0)
    value = ctypes.c_uint32(0)
    size = ctypes.c_uint32(ctypes.sizeof(value))
    status = _core_audio().AudioObjectGetPropertyData(
        obj, ctypes.byref(address), 0, None, ctypes.byref(size), ctypes.byref(value)
    )
    if status != 0:
        raise ProbeError(f"CoreAudio returned OSStatus {status}")
    return value.value


def audio_running(selector: int) -> bool:
    """Whether the default device named by `selector` is running for anybody.

    One device per call, where this used to OR the two together. The merge is
    what let the wake listener's own hold on the input device present as "the
    audio hardware is busy", which the gate spent as "on a call" - so voice being
    on was what kept the speaker route unreachable. The two directions mean
    different things and the caller needs them apart.

    Blocking, and called through a thread: sub-millisecond warm, but it talks to
    another process to do it and `coreaudiod` restarting is a real thing.
    """
    device = _uint32_property(SYSTEM_OBJECT, selector)
    if device == UNKNOWN_OBJECT:
        # A machine with no microphone has no default input device. Nothing was
        # measured, so saying False would be a guess - and False on the input
        # side is what routes to the speaker.
        raise ProbeError("no such default audio device")
    return bool(_uint32_property(device, IS_RUNNING_SOMEWHERE))


# --- the session dictionary, through Quartz ---------------------------------


def _window_server_session() -> dict[str, object] | None:
    """The window server's session dictionary, or None if Quartz is unavailable.

    Imported lazily: pyobjc is present in this install, but presence must keep
    answering on a machine where it is not, and a module-scope import would make
    that a crash at import time rather than one `None` field.
    """
    try:
        import Quartz
    except ImportError:
        return None
    return Quartz.CGSessionCopyCurrentDictionary()


# --- the reading ------------------------------------------------------------


class MachinePresence:
    """`Presence` over this machine's own signals; unknown everywhere else.

    One class rather than a macOS one plus a null one, because the fallback has
    to answer the same three-way question and a second class would only differ by
    returning it earlier.
    """

    def __init__(
        self,
        *,
        platform: str | None = None,
        timeout: float = PROBE_TIMEOUT_SECONDS,
        run: Callable[[Sequence[str]], Awaitable[str]] | None = None,
        audio: Callable[[int], bool] | None = None,
        mic_held: Callable[[], bool] | None = None,
        session: Callable[[], dict[str, object] | None] | None = None,
        now: datetime | None = None,
    ) -> None:
        # sys.platform, not platform.system(): it is a constant fixed at compile
        # time so reading it costs nothing, and it is what daemon/service.py and
        # daemon/voice/audio.py already branch on - three spellings of "is this a
        # Mac" in one codebase is how they drift apart. Injectable because the
        # non-macOS path has to be testable from a Mac, which is where it will
        # always be written and never be exercised.
        self._platform = platform if platform is not None else sys.platform
        self._timeout = timeout
        # The two hardware seams, both injectable, because docs/CONTRACTS.md's
        # testing rules mean no test may spawn a process or touch the audio
        # device - and a probe tested only against this Mac's actual idle time is
        # a probe with no failure-path coverage at all.
        self._run = run if run is not None else self._spawn
        self._audio = audio if audio is not None else audio_running
        self._mic_held = mic_held if mic_held is not None else mic_hold.held
        self._session = session if session is not None else _window_server_session
        self._now = now

    async def read(self) -> Reading:
        """The four probes, in one reading. Never raises."""
        at = self._now or clock_now()
        if self._platform != "darwin":
            # One entry, not four: the fact is about the platform, and repeating
            # it per field would pad the gate snapshot without adding anything.
            # The fields are `None`, so the gate already sees nothing is known.
            return Reading(
                at=at,
                unknown=(f"platform {self._platform!r}: no presence probes implemented",),
            )

        unknown: list[str] = []
        # Sequential, cheapest first: `screen_locked` is an in-process Quartz
        # call that costs a fraction of a millisecond warm, so it goes ahead of
        # `output_muted`, which spawns one or two `osascript` processes (~155-445
        # ms - see the module docstring). Gathering them would save little of that
        # - `osascript` dominates either way - and cost the property that a
        # failure is attributed to exactly one probe.
        idle = await self._probe("idle_seconds", self._idle_seconds, unknown)
        app = await self._probe("foreground_app", self._foreground_app, unknown)
        mic = await self._probe("mic_busy", self._mic_busy, unknown)
        output = await self._probe("output_busy", self._output_busy, unknown)
        locked = await self._probe("screen_locked", self._screen_locked, unknown)
        muted = await self._probe("output_muted", self._output_muted, unknown)
        return Reading(
            at=at,
            idle_seconds=idle,
            foreground_app=app,
            mic_busy=mic,
            output_busy=output,
            screen_locked=locked,
            output_muted=muted,
            unknown=tuple(unknown),
        )

    async def _probe[T](
        self,
        label: str,
        probe: Callable[[], Awaitable[T]],
        unknown: list[str],
    ) -> T | None:
        """Run one probe, and turn any failure into a recorded `None`.

        The broad `except` is the requirement, not a shortcut. `ProbeError` covers
        what the probes know how to fail at; the rest exists because this is on a
        scheduler tick, where one unanticipated exception is logged once and then
        the schedule reads healthy forever while presence never answers again.
        `CancelledError` is a `BaseException` and passes through, so shutdown is
        still shutdown.
        """
        try:
            return await probe()
        except ProbeError as exc:
            unknown.append(f"{label}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a scheduler tick must not raise
            logger.exception("presence: %s probe failed unexpectedly", label)
            unknown.append(f"{label}: unexpected {type(exc).__name__}: {exc}")
        return None

    async def _idle_seconds(self) -> float:
        """Seconds since the last human input device event.

        **Do not tune `AT_KEYBOARD_SECONDS` against this machine.** It runs a
        mouse-jiggler (`Jiggler`, visible in `pmset -g assertions` holding a
        `UserIsActive` assertion), and it resets `HIDIdleTime` on its own: sampling
        here showed the counter climbing normally and then dropping to 0.26 s and
        0.12 s with nobody touching anything. Anything calibrated against idle
        readings taken here is calibrated against a machine that is never idle.
        """
        # -d 1 rather than the deeper walk: the property sits on IOHIDSystem
        # itself, and one level measured the same 11-16 ms with less to parse.
        dump = await self._run([IOREG, "-c", "IOHIDSystem", "-d", "1", "-r"])
        found = _HID_IDLE.findall(dump)
        if not found:
            # ioreg exits 0 with empty output when the class does not match, so
            # "no HIDIdleTime" is the shape of a broken probe rather than of an
            # error - and it is indistinguishable from a machine at idle 0 unless
            # this branch exists. Verified: `ioreg -c NoSuchClassXYZ` -> rc 0, no
            # output.
            raise ProbeError("ioreg reported no HIDIdleTime")
        # The smallest, because idle time means "since the last event on *any*
        # device" and a second HID node would carry its own, larger, count.
        return min(int(value) for value in found) / NANOSECONDS

    async def _foreground_app(self) -> str:
        """The frontmost application's name. `lsappinfo` first, `osascript` after.

        Both reasons are kept when both fail, joined into the one `unknown` entry
        this field owns - an operator needs to know whether the fast path broke or
        whether the whole idea of asking is broken, and those have different fixes.
        """
        reasons: list[str] = []
        for probe in (self._frontmost_via_lsappinfo, self._frontmost_via_osascript):
            try:
                return await probe()
            except ProbeError as exc:
                reasons.append(str(exc))
        raise ProbeError("; ".join(reasons))

    async def _frontmost_via_lsappinfo(self) -> str:
        """The LaunchServices route: two calls, no Automation grant, ~8 ms."""
        answer = await self._run([LSAPPINFO, "front"])
        found = _ASN.search(answer)
        if found is None:
            raise ProbeError(f"lsappinfo front gave no ASN ({_excerpt(answer)})")
        # The ASN can be stale by the time the second call runs - the two are ~4 ms
        # apart and the user can alt-tab between them. Accepted, and it is not a
        # real cost: the caller is a five-minute tick, so a name up to four
        # milliseconds behind is far fresher than the reading's own granularity.
        # The case that actually matters is the app *quitting*, and that does not
        # produce a wrong name - the ASN stops resolving, `[ NULL ]` comes back,
        # and this falls through to `osascript`.
        info = await self._run([LSAPPINFO, "info", "-only", "name", found.group(0)])
        name = _LS_DISPLAY_NAME.search(info)
        if name is None:
            # Includes the `[ NULL ]` case; see `_LS_DISPLAY_NAME`. `lsappinfo`
            # exits 0 for every one of these, so this branch is the only detection.
            raise ProbeError(f"lsappinfo gave no LSDisplayName ({_excerpt(info)})")
        return _app_name(name.group(1), "lsappinfo")

    async def _frontmost_via_osascript(self) -> str:
        """The Apple-event route. Slower, and may need a permission this process
        has no way to request non-interactively - see the module docstring."""
        return _app_name(await self._run([OSASCRIPT, "-e", FRONTMOST]), "osascript")

    async def _mic_busy(self) -> bool:
        """Whether somebody *other than us* holds the microphone.

        Our own hold is subtracted rather than probed around, because CoreAudio
        has no per-process answer: `kAudioDevicePropertyDeviceIsRunningSomewhere`
        is exactly as wide as its name. The daemon does know its own state, so it
        asks itself (`daemon/mic_hold.py`) instead of guessing.

        Checked *before* the probe: if we hold it, the device is busy by
        definition and the answer cannot be anything else.
        """
        if self._mic_held():
            return False
        return await self._device_running(DEFAULT_INPUT)

    async def _output_busy(self) -> bool:
        return await self._device_running(DEFAULT_OUTPUT)

    async def _device_running(self, selector: int) -> bool:
        try:
            async with asyncio.timeout(self._timeout):
                # In a thread because this is a blocking IPC to coreaudiod. A
                # timeout cannot cancel the thread, so a wedged coreaudiod leaks
                # one worker per tick - accepted, because 0.1 ms warm makes this
                # the improbable branch and the alternative is blocking the loop.
                return await asyncio.to_thread(self._audio, selector)
        except TimeoutError:
            raise ProbeError(f"CoreAudio did not answer in {self._timeout:g}s") from None
        except OSError as exc:
            raise ProbeError(f"CoreAudio unavailable: {exc}") from exc

    async def _screen_locked(self) -> bool:
        """Whether the session is locked.

        macOS *omits* `CGSSessionScreenIsLocked` when unlocked rather than
        setting it to 0 (verified 2026-08-11), so an absent key is the answer
        "unlocked" and not the answer "unknown". Reading it as unknown would send
        every utterance to Telegram for the rest of the process's life, which is
        this project's signature defect wearing a probe's clothes.
        """
        session = await asyncio.to_thread(self._session)
        if session is None:
            raise ProbeError("no window-server session dictionary")
        return bool(session.get("CGSSessionScreenIsLocked", False))

    async def _output_muted(self) -> bool:
        """Muted, or turned all the way down.

        Both are the same fact for this file's purpose - `say` exits 0 and the
        room stays quiet either way (`daemon/proactivity/speaker.py` measured
        that a misconfigured voice is silent with a clean exit). Treating only
        the mute switch as mute would leave the zero-volume case recorded as a
        line spoken aloud.
        """
        muted = (await self._run([OSASCRIPT, "-e", MUTED])).strip().casefold()
        if muted == "true":
            return True
        if muted != "false":
            raise ProbeError(f"osascript gave no mute state ({_excerpt(muted)})")
        level = (await self._run([OSASCRIPT, "-e", VOLUME])).strip()
        try:
            return int(level) == 0
        except ValueError:
            raise ProbeError(f"osascript gave no volume ({_excerpt(level)})") from None

    # `Reading.headphones` has no probe and stays `None` forever. There was one:
    # `system_profiler SPAudioDataType`, reading the `Transport:` line of whichever
    # device claims `Default Output Device: Yes`. It measured wrong on the machine
    # this file is developed on, and not by a small margin. That machine's default
    # output is always `MacBook Pro Speakers (eqMac)` - a virtual driver in front
    # of the real hardware - and it answers `Transport: USB`, same as real USB
    # headphones would. The other devices on that machine, for reference:
    #
    #   MacBook Pro Speakers (eqMac)  -> USB       (the default output, always)
    #   MacBook Pro Speakers          -> Built-in
    #   LG FULL HD                    -> HDMI
    #   Microsoft Teams Audio         -> Virtual
    #   eqMac Export                  -> USB
    #
    # So the probe read "headphones" for the laptop's own built-in speakers, every
    # time, regardless of what was actually plugged in - not a miscalibration to
    # tune, a signal this mechanism cannot see past the virtual device to obtain.
    # And `headphones` is the one field that only ever *widens* what the gate
    # allows (PLAN 6.4's asymmetry: an ignored notification costs nothing, a voice
    # in a meeting is an accident) - so a signal this file cannot verify must stay
    # unmeasured rather than measured wrong in the direction that costs more.
    # `system_profiler` was also the single most expensive probe in this file
    # (0.21-0.30 s) for a signal it could not actually deliver. Removed entirely
    # rather than left disabled, so a later reader does not find dead code that
    # looks load-bearing.

    async def _spawn(self, argv: Sequence[str]) -> str:
        """Run a probe command and return its stdout. Bounded, and never a shell.

        `create_subprocess_exec`, so argv stays a list and no shell parses it -
        nothing here interpolates user text into a command, and the way that stops
        being true is somebody reaching for `shell=True` later. Absolute paths for
        the same reason: `PATH` under a LaunchAgent is not the `PATH` in a
        terminal.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise ProbeError(f"{argv[0]} could not start: {exc}") from exc

        try:
            async with asyncio.timeout(self._timeout):
                stdout, stderr = await process.communicate()
        except TimeoutError:
            # Killed and reaped, not abandoned. An orphan still holding the pipes
            # is a file-descriptor leak measured in one per tick, and SIGKILL
            # rather than SIGTERM because osascript blocked on a consent dialog
            # is the case this is for and it does not have a graceful exit.
            try:
                process.kill()
            except ProcessLookupError:
                pass  # finished between the timeout and the signal
            else:
                await process.wait()
            raise ProbeError(f"{argv[0]} did not answer in {self._timeout:g}s") from None

        if process.returncode:
            detail = stderr.decode("utf-8", "replace").strip() or "no output"
            raise ProbeError(f"{argv[0]} exited {process.returncode}: {detail[:MAX_APP_NAME]}")
        return stdout.decode("utf-8", "replace")
