"""`daemon wake` - measure the wake phrase on this voice, then hear the gate work.

The wake gate exists because a `VoiceSession` is billed per minute, so something
free has to decide when to open one (docs/PLAN.md 6.5, daemon/voice/base.py). These
two commands are how a person finds out whether that decision is being made about
*them*:

  * `calibrate` records the owner saying their phrase a few times, prints what the
    on-device recognizer actually returned, and writes those strings to
    `DAEMON_WAKE_ALIASES`.
  * `test` runs the gate here and prints every `WakeEvent`, which is the only thing
    that answers "does it fire for me, and does it stay quiet for the television".

**Why calibration is a command and not a table in this file.** Measured on this
project's target machine, three runs each, the output identical every time:

    said                    the recognizer returned
    헤이 데몬                헤이 대문
    데몬                     질문
    데몬아 안녕              질문 아 안녕
    루시                     루씨
    루시야                   루시
    헤이 루시                헤이씨
    오늘 날씨가 참 좋네요     exact

macOS on-device Korean recognition never emits a coined name, and `contextualStrings`
was measured to make **zero** difference. So the gate cannot match the ideal
spelling; it has to match what the recognizer returns for this speaker. Those
numbers came from `say -v Yuna`, not from a person, which is the whole reason this
is a measurement the owner takes rather than a constant: their voice has its own
stable set, and `.env` is where it belongs.

Two things this module is careful about, both because a microphone is involved. The
owner is told the room is about to be recorded, before it is, and only the fixed
number of short takes is captured - the audio is transcribed in this process,
never written anywhere and never logged, and only the text is shown. And the text
that comes back is treated as untrusted: it arrived from a room, it is on its way
to a terminal, and it ends up as a `KEY=value` line in a line-oriented file.

Synchronous for the same reason `daemon/setup.py` is: the I/O here is a human
pressing Enter. Each take is one `asyncio.run`, so the microphone is open for the
length of a take and not for the length of the conversation.
"""

from __future__ import annotations

import asyncio
import sys
import unicodedata
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TextIO

from daemon.config import ENV_FILE, Settings
from daemon.setup import Cancelled, Prompt, merge_env, parse_env, write_private_file
from daemon.tui import Row, heading, status, table, wrap

if TYPE_CHECKING:  # protocols only - nothing here constructs an implementation
    from daemon.voice.base import AudioIO, SpeechRecognizer, WakeEvent

Closer = Callable[[], Awaitable[None]]


class GateCounters(Protocol):
    """What a gate has done, as `daemon wake test` reports it.

    Written down as a shape rather than imported, because `WakeCounters` lives with
    the gate (daemon/voice/wake.py) and this module may not reach in there. Every
    field below is printed, and a renamed one has to break loudly here rather than
    quietly stop being reported.
    """

    frames_seen: int
    segments_closed: int
    transcribed: int
    fired: int
    skipped_short: int
    skipped_cooldown: int
    skipped_unavailable: int
    errors: int


class WakeSource(Protocol):
    """The gate, as this command uses it: a stream of events, and a tally.

    Both halves are needed and they answer different questions - the stream says
    "it fired", the tally says why it did not. A gate that is deaf because its
    recognizer cannot answer produces exactly the same silence as an empty room,
    and only the tally tells them apart."""

    counters: GateCounters

    def listen(self) -> AsyncIterator[WakeEvent]: ...


OK = 0
PROBLEM = 1

ALIASES_KEY = "DAEMON_WAKE_ALIASES"
ENABLED_KEY = "DAEMON_WAKE_ENABLED"
VOICE_KEY = "DAEMON_VOICE_ENABLED"
"""Read but never written here. `Settings` refuses the gate switched on with voice
off - the gate exists only to open a voice session - so offering the one without
the other would write a `.env` that no longer starts."""

SEPARATOR = ","
"""What splits the aliases in `.env`. Also the one character an alias may not
contain, which is why a transcription carrying one is reported instead of saved."""

TRUTHY = frozenset({"1", "true", "yes", "on"})
"""The same set `daemon/setup.py` accepts, so a value written by one command reads
the same to the other."""

ENV_NOTE = "written by `daemon wake calibrate`"

TAKES = 3
"""Takes per calibration.

Three is the smallest number that distinguishes a stable transcription from a
lucky one: two that agree could be a coincidence, and one says nothing at all.
A fourth take mostly spends the owner's patience. `--takes` moves it."""

MIN_STABLE_TAKES = 2
"""Below this, "every take agreed" is not a finding worth saving on."""

TAKE_SECONDS = 3.0
"""Recorded per take. A wake phrase is one or two words; a longer window records
more of the room for no gain in what the recognizer has to work with."""

RECORD_GRACE = 1.5
"""Wall-clock slack over `TAKE_SECONDS` before a take gives up.

Not decoration: a device that opens and then delivers nothing would otherwise
leave this awaiting a microphone forever, with the light on and no way out but
Ctrl-C. A take that hits this reports no audio at all, which is a different
problem from a phrase nobody can hear."""

LISTEN_SECONDS = 60
"""Default window for `daemon wake test`. Long enough to say the phrase a few
times and to leave a television talking; `--seconds 0` waits for Ctrl-C, which is
what the television test actually wants."""

ALIAS_LIMIT = 64
"""Longest alias this command will store, and the bound on any recognizer output
it prints. The text came out of a room and is on its way to both a terminal and a
`KEY=value` line, so its length is not its own decision."""

BYTES_PER_SAMPLE = 2
"""16-bit PCM, the format `VoiceActivityDetector.probability` and `AudioIO` both
name."""

DEFAULT_PHRASE = "헤이 데몬"
"""Offered as the default because it is the phrase the table above was measured
with - so an owner who just presses Enter gets the case this command has evidence
about, rather than an invented one."""

WHY = (
    "The on-device recognizer never returns a coined name. Measured on this "
    "project's target machine, three runs each, identical every time: 헤이 데몬 "
    "came back as 헤이 대문, 데몬 as 질문, 루시야 as 루시. Ordinary Korean words "
    "came back exact.",
    "So the gate matches what the recognizer actually hears, not what you meant. "
    "This records you saying your phrase a few times and offers to save those "
    "strings as the aliases the gate matches on.",
)

RECORDING_NOTE = (
    "{takes} take(s), about {seconds:.0f} seconds each, and each one starts when "
    "you press Enter. The audio is transcribed in this process, is never written "
    "anywhere and is never logged - only the text below, and only what you "
    "approve, leaves this screen."
)

MIC_WARNING = "The microphone is about to record this room."

ALTERNATIVES = (
    "  A phrase the recognizer hears differently every time cannot be an alias:",
    "  there is nothing stable to match. What the same measurement found:",
    "    ordinary Korean words transcribe reliably - 헤이 자기 came back exact, and",
    "      so did a whole sentence of ordinary Korean.",
    "    a coined name never does. 데몬 came back as 질문 and 헤이 루시 as 헤이씨.",
    "    a stable *mistake* is fine. 헤이 데몬 -> 헤이 대문 every single time, and",
    "      that is a usable alias - stability is the property, not correctness.",
    "",
    "  Pick a phrase from the first kind and run this again.",
)

DEAF_NEXT = (
    "  The microphone produced no audio at all, so nothing was transcribed and",
    "  nothing about the phrase is known yet. Usually one of:",
    "    the input device is muted, or the wrong one is selected.",
    "    this process has no microphone permission (macOS asks once, per app -",
    "      the terminal you are in is the app it asks about).",
    "    no capture device exists on this machine.",
)

NOTHING_HEARD_NEXT = (
    "  Audio arrived and the recognizer could not make out a word in any take.",
    "  Either it started before you spoke - each take is short, so speak as soon",
    "  as it says listening - or the input level is very low.",
)

UNAVAILABLE_NEXT = (
    "  Four things produce that answer, and they are indistinguishable from a",
    "  failed transcription - which is why it is asked before anything is recorded:",
    "    the voice extra is not installed. The recognizer is pyobjc's Speech",
    "      framework, and it lives there:  pip install -e '.[voice]'",
    "      This is the one to check first - it was the answer on the machine this",
    "      command was written on.",
    "    the locale is not installed. macOS downloads Korean dictation the first",
    "      time it is switched on (System Settings > Keyboard > Dictation).",
    "    on-device recognition is unsupported for this locale on this hardware.",
    "    this process has not been granted speech recognition.",
    "",
    "  Without it the gate can hear that someone spoke and never know what was",
    "  said, so there is nothing to match and nothing to calibrate.",
)

GATE_UNBUILT = (
    "  The gate is `WakeGate` in daemon/voice/wake.py, built by `build_wake_gate` in",
    "  daemon/app.py out of a VAD, the on-device recognizer and your aliases. If",
    "  one of those is missing this is what it looks like: the message above names",
    "  which.",
)

LIVE_MIC = (
    "The microphone stays open until this command exits, so this room is being "
    "listened to. Nothing is recorded to disk and nothing is sent anywhere - the "
    "VAD and the recognizer are both local. Only the text of a phrase that "
    "matched is printed."
)

DEAD_GATE = (
    "A wake gate that ends leaves a process that is alive, healthy-looking and "
    "permanently deaf, which is the worst failure a companion has. Whatever ended "
    "it is the reason above, not the absence of a wake phrase."
)

NOTHING_FIRED = (
    "  That is the right answer if nobody said the phrase - silence is what this",
    "  gate is for. If you did say it:",
    "    `daemon wake calibrate` prints what the recognizer heard, which is the",
    "      only thing the gate can match. A coined name is never it.",
    f"    {ALIASES_KEY} in .env is the list it matches against.",
)


# --- what the machine provides ------------------------------------------------


def _machine_recognizer() -> SpeechRecognizer:
    from daemon.app import build_wake_recognizer

    return build_wake_recognizer()


def _machine_audio() -> AudioIO:
    from daemon.app import build_wake_audio

    return build_wake_audio()


async def _machine_gate(settings: Settings) -> tuple[WakeSource, Closer]:
    from daemon.app import build_wake_gate

    return await build_wake_gate(settings)


@dataclass(frozen=True, slots=True)
class Devices:
    """The three things these commands need from the machine, in one injectable
    bundle - the same shape and the same reason as `setup.Checks`: the tests need
    no microphone, no OS speech service and no gate.

    Every default goes through `daemon/app.py`, which is the only module allowed to
    import an implementation (docs/CONTRACTS.md 4)."""

    recognizer: Callable[[], SpeechRecognizer] = _machine_recognizer
    audio: Callable[[], AudioIO] = _machine_audio
    gate: Callable[[Settings], Awaitable[tuple[WakeSource, Closer]]] = _machine_gate


# --- presentation -------------------------------------------------------------

MIN_PROSE_WIDTH = 20


def _prose(prompt: Prompt, text: str, *, indent: int = 2) -> None:
    """A paragraph wrapped to this terminal, by display width.

    Same reason as `setup.Wizard._prose`: these are sentences, not lines someone
    broke by hand, and a Korean one measured with `len()` wraps in the wrong place.
    """
    for line in wrap(text, max(MIN_PROSE_WIDTH, prompt.theme.width - indent)):
        prompt.say(" " * indent + line)


def _event_line(prompt: Prompt, event: WakeEvent) -> str:
    """One wake event. `heard` and `matched` are printed as two different facts
    because they are two different strings - that asymmetry is the whole finding
    this feature is built on (daemon/voice/base.py's `WakeEvent`)."""
    return status(
        prompt.theme,
        "ok",
        f"wake · heard {clean(event.heard) or '(nothing)'} · matched "
        f"{clean(event.matched)} · speech {event.confidence:.0%}",
    )


# --- what a take produced -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class Take:
    """One recording, and what the recognizer made of it."""

    index: int
    heard: str
    """Cleaned recognizer output. `""` means it could not say."""
    recorded: int
    """Bytes of PCM the microphone produced. Zero is a device problem and not a
    phrase problem, and the two need different sentences."""

    @property
    def silent(self) -> bool:
        return self.recorded == 0


@dataclass(frozen=True, slots=True)
class Reading:
    """Every take together: what can be saved, and whether it is worth saving."""

    takes: tuple[Take, ...]
    aliases: tuple[str, ...]
    """Distinct transcriptions, in the order they were first heard."""
    rejected: tuple[str, ...]
    """Heard, but unstorable - it carries the separator."""

    @property
    def heard(self) -> tuple[str, ...]:
        return tuple(take.heard for take in self.takes if take.heard)

    @property
    def deaf(self) -> bool:
        """No take produced any audio at all."""
        return all(take.silent for take in self.takes)

    @property
    def stable(self) -> bool:
        return len(self.heard) == len(self.takes) and len(set(self.heard)) == 1

    @property
    def unstable(self) -> bool:
        """Every take that was heard came back differently, so there is no string
        the gate could match twice."""
        return len(self.heard) > 1 and len(set(self.heard)) == len(self.heard)


def clean(text: str) -> str:
    """One recognizer result, made safe to compare, to print and to store.

    NFC first, and it is the part that matters. Korean that arrives decomposed
    compares unequal to the identical composed string, so two takes that said the
    same thing would be saved as two aliases - and the gate would then be matching
    on a spelling nobody can type or read back.

    Then: unprintable characters go, because this text came out of a room and is
    about to be printed next to a prompt; whitespace runs collapse; and the result
    is bounded, because it also becomes a `KEY=value` line.
    """
    composed = unicodedata.normalize("NFC", text)
    kept = "".join(char for char in composed if char.isprintable() or char.isspace())
    return " ".join(kept.split())[:ALIAS_LIMIT].strip()


def reading_of(takes: Sequence[Take]) -> Reading:
    """Fold the takes into what would be written, keeping first-heard order."""
    aliases: list[str] = []
    rejected: list[str] = []
    for take in takes:
        if not take.heard:
            continue
        target = rejected if SEPARATOR in take.heard else aliases
        if take.heard not in target:
            target.append(take.heard)
    return Reading(tuple(takes), tuple(aliases), tuple(rejected))


def split_aliases(value: str) -> tuple[str, ...]:
    """What `.env` already holds, read the way the config key is documented."""
    return tuple(part.strip() for part in value.split(SEPARATOR) if part.strip())


def _on(value: str) -> bool:
    """A dotenv boolean, read the way `daemon/setup.py` reads one."""
    return value.strip().lower() in TRUTHY


# --- the microphone -----------------------------------------------------------


async def record(audio: AudioIO, *, seconds: float) -> bytes:
    """`seconds` of microphone PCM, bounded twice.

    By bytes, because that is what "three seconds of audio" means, and because it
    keeps a test tied to the audio it feeds in rather than to a wall clock. And by
    the clock as well, because a device that opens and then delivers nothing would
    otherwise leave this awaiting a microphone forever.

    The stream is closed on every path: `AudioIO.record` is a generator that stops
    the input stream in its own `finally`, and abandoning it leaves the microphone
    light on until the garbage collector gets there.
    """
    wanted = int(seconds * audio.sample_rate) * BYTES_PER_SAMPLE
    blocks: list[bytes] = []
    got = 0
    stream = audio.record()
    try:
        async with asyncio.timeout(seconds + RECORD_GRACE):
            async for block in stream:
                blocks.append(block)
                got += len(block)
                if got >= wanted:
                    break
    except TimeoutError:
        # Reported by the caller as no audio, or as short audio. Not raised: the
        # answer to "is this microphone working" is what this command is for.
        pass
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()
    return b"".join(blocks)


async def one_take(
    devices: Devices, recognizer: SpeechRecognizer, *, seconds: float
) -> tuple[str, int]:
    """Open the microphone, record one take, close it, and transcribe.

    The device is built and closed inside the take rather than held across the
    conversation, so the microphone is open for three seconds at a time and not for
    as long as someone takes to read the screen.
    """
    audio = devices.audio()
    try:
        pcm = await record(audio, seconds=seconds)
    finally:
        with suppress(Exception):
            await audio.close()
    if not pcm:
        return "", 0
    return clean(await recognizer.transcribe(pcm)), len(pcm)


# --- calibrate ----------------------------------------------------------------


def calibrate(
    *,
    env_path: Path | None = None,
    takes: int | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    devices: Devices | None = None,
) -> int:
    """`daemon wake calibrate`. Writes nothing but `.env`, and nothing until asked."""
    path = Path(env_path) if env_path is not None else Path.cwd() / ENV_FILE
    prompt = Prompt(stdin, stdout)
    wanted = TAKES if takes is None else max(1, takes)
    try:
        return _calibrate(prompt, path, wanted, devices if devices is not None else Devices())
    except Cancelled as exc:
        prompt.say()
        prompt.say(f"Stopped ({exc}). {path} was not touched.")
        return PROBLEM
    except KeyboardInterrupt:
        prompt.say()
        prompt.say(f"Stopped. {path} was not touched.")
        return PROBLEM


def _calibrate(prompt: Prompt, path: Path, takes: int, devices: Devices) -> int:
    theme = prompt.theme
    say = prompt.say

    say(heading(theme, "daemon wake calibrate"))
    say()
    for paragraph in WHY:
        _prose(prompt, paragraph)
        say()

    recognizer = devices.recognizer()
    if not recognizer.available:
        say(status(theme, "fail", _unavailable()))
        for line in UNAVAILABLE_NEXT:
            say(line)
        say()
        return PROBLEM

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    env = parse_env(existing)
    current = split_aliases(env.get(ALIASES_KEY, ""))
    if current:
        say(status(theme, "ok", f"{ALIASES_KEY} already holds: {', '.join(current)}"))
        _prose(prompt, "Calibrating replaces that list, and only after you have seen it.")
        say()

    phrase = clean(prompt.ask("  Phrase", default=DEFAULT_PHRASE)) or DEFAULT_PHRASE
    say()
    say(status(theme, "warn", MIC_WARNING))
    _prose(prompt, RECORDING_NOTE.format(takes=takes, seconds=TAKE_SECONDS))
    say()

    results: list[Take] = []
    for index in range(1, takes + 1):
        prompt.ask(f"  Take {index} of {takes} - Enter, then say {phrase}")
        say(theme.dim("  listening..."))
        try:
            # `TAKE_SECONDS` read here rather than bound as a default argument, so
            # a test can shorten a take without reaching inside the coroutine.
            heard, recorded = asyncio.run(one_take(devices, recognizer, seconds=TAKE_SECONDS))
        except Exception as exc:  # noqa: BLE001 - the reason is the answer, not a traceback
            # `SpeechRecognizer.transcribe` is contracted not to raise for ordinary
            # failure, so one that does is a defect worth naming rather than one
            # worth retrying twice more.
            say(status(theme, "fail", f"take {index} failed: {exc.__class__.__name__}: {exc}"))
            say("  Nothing was written.")
            return PROBLEM
        results.append(Take(index, heard, recorded))
        say(status(theme, "ok" if heard else "warn", f"heard: {heard or '(nothing)'}"))
        say()

    reading = reading_of(results)
    say(
        table(
            theme,
            [
                Row(
                    f"take {take.index}",
                    take.heard or "(nothing)",
                    "no audio at all" if take.silent else "",
                )
                for take in reading.takes
            ],
        )
    )
    say()

    if not _report(prompt, reading, phrase, takes):
        return PROBLEM
    return _save(prompt, path, existing, env, reading)


def _report(prompt: Prompt, reading: Reading, phrase: str, takes: int) -> bool:
    """Say what the takes mean. False when there is nothing worth saving."""
    theme = prompt.theme
    say = prompt.say

    if reading.deaf:
        say(status(theme, "fail", "no audio was captured in any take."))
        for line in DEAF_NEXT:
            say(line)
        say()
        return False
    if not reading.heard:
        say(status(theme, "fail", "the recognizer heard nothing it could put into words."))
        for line in NOTHING_HEARD_NEXT:
            say(line)
        say()
        return False

    if reading.stable:
        say(status(theme, "ok", f"stable: every take came back as {reading.heard[0]}"))
        if takes < MIN_STABLE_TAKES:
            say(
                status(
                    theme,
                    "warn",
                    "one take cannot show stability - run it again without --takes "
                    "to hear whether it says the same thing three times.",
                )
            )
    elif reading.unstable:
        say(
            status(
                theme,
                "fail",
                f"unstable: {len(reading.heard)} takes, {len(reading.heard)} different "
                "transcriptions. This phrase is a bad wake phrase.",
            )
        )
        for line in ALTERNATIVES:
            say(line)
    else:
        say(
            status(
                theme,
                "warn",
                f"mostly stable: {len(set(reading.heard))} transcription(s) across "
                f"{len(reading.heard)} take(s) that were heard.",
            )
        )
        _prose(
            prompt,
            "Saving all of them is the right answer - each one is what this "
            "recognizer hears some of the time, and the gate matches any alias.",
        )
    say()

    if phrase in reading.aliases:
        say(status(theme, "ok", f"{phrase} came back verbatim, which is the easy case."))
        say()
    for text in reading.rejected:
        say(
            status(
                theme,
                "warn",
                f"{text} contains a comma, which is what separates the aliases in "
                ".env, so it cannot be stored as one.",
            )
        )
    if not reading.aliases:
        say(status(theme, "fail", "nothing storable came back, so there is nothing to save."))
        say()
        return False
    return True


def _save(
    prompt: Prompt, path: Path, existing: str, env: dict[str, str], reading: Reading
) -> int:
    """Offer the write, once, with everything it would change on screen first."""
    theme = prompt.theme
    say = prompt.say

    updates = {ALIASES_KEY: SEPARATOR.join(reading.aliases)}
    # Nothing about the switch when the reading was unstable. Offering to turn the
    # gate on for a phrase just called a bad one reads as a contradiction, and the
    # honest next step - printed above - is to pick another phrase.
    if not reading.unstable and not _on(env.get(ENABLED_KEY, "")):
        if not _on(env.get(VOICE_KEY, "")):
            # Not offered, because writing it would break the install: Settings
            # refuses a configuration with the gate on and voice off - the gate
            # exists only to open a voice session - so `daemon run` would then fail
            # to start. Said out loud instead, with the command that fixes it.
            say(status(theme, "warn", f"{VOICE_KEY} is off, so the gate cannot be switched on."))
            _prose(
                prompt,
                "The gate exists only to open a voice session, and a configuration "
                "with one on and the other off is refused at startup. `daemon setup` "
                "is where voice gets turned on - it needs a key. The aliases below "
                "are saved either way, and are what the gate will use when it can run.",
            )
            say()
        else:
            # Asked rather than left as homework: hand-editing `.env` is the thing
            # `daemon setup` exists to remove, and a calibration that measured the
            # aliases and then told someone to go and set the switch by hand would
            # be putting it straight back.
            _prose(
                prompt,
                f"{ENABLED_KEY} is off, so nothing listens for these yet. `daemon "
                "wake test` runs the gate here either way; the switch is what makes "
                "the resident process do it.",
            )
            if prompt.ask_yes_no(f"  Turn {ENABLED_KEY} on as well", default=True):
                updates[ENABLED_KEY] = "true"
            say()

    rows = [
        Row(key, value, "" if env.get(key) is None else f"was {env[key] or '(empty)'}")
        for key, value in updates.items()
    ]
    say(heading(theme, f"Review - {path}"))
    say()
    say(table(theme, rows))
    say()
    say("Everything else in the file is left alone.")
    # Default No when the phrase is unstable. Saving strings the recognizer will not
    # produce again leaves a gate that looks configured and never fires - which is
    # the exact failure this command exists to make visible, so it must not be the
    # thing that happens to someone who presses Enter through it.
    if not prompt.ask_yes_no("Write it", default=not reading.unstable):
        say()
        say("Nothing was written.")
        say("Run this again with another phrase, or with --takes for more evidence.")
        # Not an error. Declining an unstable reading is the correct answer, and an
        # exit code calling it a failure would be wrong about the one case this
        # command most wants a person to take.
        return OK

    write_private_file(path, merge_env(existing, updates, note=ENV_NOTE))
    say()
    say(status(theme, "ok", f"wrote {path} (mode 0600)"))
    say()
    say("Next:")
    say("  daemon wake test  - run the gate here and see it fire on your voice")
    say("  daemon run        - the resident process reads .env at startup, so")
    say("                      restart it for these aliases to take effect")
    return OK


def _unavailable() -> str:
    """Why there is nothing to calibrate against, as specifically as we can say.

    `SpeechRecognizer.available` is a bool by design - an OS speech service can be
    missing, unauthorised or lack the locale, and all three look identical from a
    failed transcription - so the one thing this process can determine for itself
    is named first.
    """
    if sys.platform != "darwin":
        return (
            "on-device speech recognition here is macOS's, and this machine is "
            f"{sys.platform}, so there is no recognizer to calibrate against."
        )
    return "the on-device speech recognizer says it cannot answer on this machine."


# --- test ---------------------------------------------------------------------


def listen(
    settings: Settings,
    *,
    seconds: float | None = None,
    stdout: TextIO | None = None,
    devices: Devices | None = None,
) -> int:
    """`daemon wake test`. Runs the gate here and prints every event it fires."""
    prompt = Prompt(None, stdout)
    theme = prompt.theme
    say = prompt.say
    window = LISTEN_SECONDS if seconds is None else max(0, seconds)
    kit = devices if devices is not None else Devices()

    say(heading(theme, "daemon wake test"))
    say()
    _prose(
        prompt,
        "This is the gate the resident process runs: a VAD decides whether a frame "
        "is speech at all, the on-device recognizer says what a segment was, and an "
        f"alias from {ALIASES_KEY} is what turns it into a wake event.",
    )
    say()
    say(status(theme, "warn", LIVE_MIC))
    say()
    say(
        theme.dim(
            f"  Listening for {window}s. Ctrl-C stops."
            if window
            else "  Listening until Ctrl-C."
        )
    )
    say()

    fired: list[WakeEvent] = []
    tally: list[GateCounters] = []
    try:
        ended = asyncio.run(_run(kit, settings, fired, tally, window, prompt))
    except KeyboardInterrupt:
        say()
        say(theme.dim("  stopped."))
        ended = False
    except Exception as exc:  # noqa: BLE001 - an unbuilt gate is a sentence, not a traceback
        say()
        say(status(theme, "fail", f"the gate could not run: {exc.__class__.__name__}: {exc}"))
        for line in GATE_UNBUILT:
            say(line)
        say()
        return PROBLEM

    say()
    counters = tally[0] if tally else None
    if counters is not None:
        say(table(theme, _tally_rows(counters)))
        say()
    if ended:
        say(
            status(
                theme,
                "fail",
                f"the gate stopped listening on its own, after {len(fired)} event(s).",
            )
        )
        _prose(prompt, DEAD_GATE)
        say()
        return PROBLEM
    if fired:
        say(status(theme, "ok", f"{len(fired)} wake event(s)."))
        _prose(
            prompt,
            "Anything in that list you did not say is a false wake, and the phrase "
            "is what to change: a longer one has more for the recognizer to agree "
            "with. `daemon wake calibrate` is where that is measured.",
        )
        say()
        return OK

    say(status(theme, "warn", "nothing fired."))
    # The tally first, because it can say *which* silence this was. The general
    # advice is the fallback for a tally that fits none of the known shapes -
    # printed second it read as a repetition of the diagnosis above it.
    if counters is None or not _diagnose(prompt, counters):
        for line in NOTHING_FIRED:
            say(line)
        say()
    return OK


def _tally_rows(counters: GateCounters) -> list[Row]:
    """What the gate did, as the four numbers that answer different questions.

    Printed on every run, not only a silent one: the same table read after a fire
    is how someone sees that the recognizer is being asked ten times for every
    match, which is a phrase problem long before it is a bug.
    """
    return [
        Row("frames seen", str(counters.frames_seen), "audio arriving at all"),
        Row("speech segments", str(counters.segments_closed), "the VAD called these speech"),
        Row("transcribed", str(counters.transcribed), "segments that cost a recognizer call"),
        Row("matched an alias", str(counters.fired), "wake events"),
    ]


def _diagnose(prompt: Prompt, counters: GateCounters) -> bool:
    """Why nothing fired, from the tally rather than from a guess.

    True when it had something specific to say, which is what lets the caller keep
    the general advice for a tally that fits none of these shapes.

    The dangerous failures here all look like a quiet house from outside, and each
    of these branches is a different one of them (daemon/voice/wake.py's
    `WakeCounters`). Reporting the state instead of assuming it is the whole point:
    "nothing fired" is either the product working or the product deaf.
    """
    theme = prompt.theme
    say = prompt.say
    if not counters.frames_seen:
        say(status(theme, "fail", "no audio arrived at all, so nothing was even listened to."))
        for line in DEAF_NEXT:
            say(line)
        say()
        return True
    if counters.skipped_unavailable:
        say(
            status(
                theme,
                "fail",
                f"{counters.skipped_unavailable} segment(s) of speech went untranscribed "
                "because the recognizer could not answer. This is the deaf case, not a "
                "quiet room.",
            )
        )
        for line in UNAVAILABLE_NEXT:
            say(line)
        say()
        return True
    if counters.transcribed and not counters.fired:
        # The case this whole feature was built around, so it is named exactly.
        say(
            status(
                theme,
                "warn",
                f"speech was heard and transcribed {counters.transcribed} time(s) and no "
                f"alias matched. That is what an uncalibrated {ALIASES_KEY} looks like.",
            )
        )
        say("  `daemon wake calibrate` prints what it heard and saves those strings.")
        say()
        return True
    said = False
    if not counters.segments_closed:
        say(
            status(
                theme,
                "warn",
                "audio arrived and the VAD found no speech in it. Either nobody spoke, "
                "or the input level is low enough that speech does not read as speech.",
            )
        )
        say()
        said = True
    if counters.skipped_short:
        say(
            status(
                theme,
                "warn",
                f"{counters.skipped_short} segment(s) were too short to be worth "
                "transcribing. A one-syllable wake phrase lands here.",
            )
        )
        say()
        said = True
    if counters.errors:
        say(
            status(
                theme,
                "warn",
                f"{counters.errors} error(s) were stepped over rather than raised - the "
                "gate outlives them on purpose. They are in the log above.",
            )
        )
        say()
        said = True
    return said


async def _run(
    devices: Devices,
    settings: Settings,
    fired: list[WakeEvent],
    tally: list[GateCounters],
    seconds: float,
    prompt: Prompt,
) -> bool:
    """Build the gate, watch it, and release the microphone whatever happens.

    `fired` and `tally` are filled in place rather than returned, so a Ctrl-C in the
    middle still leaves the caller with everything that had already happened - which
    is the run a person is most likely to have: they stop it when they are satisfied.
    """
    gate, closing = await devices.gate(settings)
    tally.append(gate.counters)
    try:
        return await _watch(gate.listen(), fired, seconds, prompt)
    finally:
        await closing()


async def _watch(
    events: AsyncIterator[WakeEvent], fired: list[WakeEvent], seconds: float, prompt: Prompt
) -> bool:
    """Print events as they arrive. True if the gate ended on its own."""
    ended = False
    try:
        async with asyncio.timeout(seconds or None):
            async for event in events:
                fired.append(event)
                prompt.say(_event_line(prompt, event))
            # The iterator ran out. Not a timeout, and not nothing: the thing that
            # was supposed to listen forever stopped.
            ended = True
    except TimeoutError:
        pass
    finally:
        aclose = getattr(events, "aclose", None)
        if aclose is not None:
            with suppress(Exception):
                await aclose()
    return ended
