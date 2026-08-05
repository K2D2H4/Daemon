"""Saying one already-chosen sentence out loud, at this machine and nowhere else.

The `Speaker` implementation behind docs/PLAN.md 6.3's left-hand branch: when the
user is at the keyboard, a proactive utterance comes out of the local speaker and
**nothing leaves the device.** That is the one privacy claim in docs/PLAN.md 7 that
survives voice being switched on, so it is worth protecting in code rather than
only in prose - hence no client, no session, no socket in this file, and a single
`/usr/bin/say` whose text goes in over a pipe.

This is not a special case of `VoiceSession` and cannot be built on one.
docs/PLAN.md 6.5: **Live API has no verbatim TTS path.** `realtimeInput.text` is a
prompt, so the model *answers* the sentence instead of reading it. Speaking a
sentence the judge already picked is therefore a local job by elimination, not by
preference.

## What was measured here, and the one number that shapes the file

macOS 26.5.2 / Darwin 25.5.0, M4 Max. docs/PLAN.md 6.5 recorded "ko_KR voices 9";
confirmed exactly nine, and the list is the interesting part rather than the count.

Speaking `-v Yuna`, Korean, wall clock from spawn to exit:

| text | chars | duration |
|---|---|---|
| `물 한 잔 마셔.` | 9 | 1.87-2.01 s |
| `어제 발표 있다고 했잖아. 어떻게 됐어?` | 22 | 3.73-3.78 s |
| three sentences | 54 | 8.49-8.51 s |
| nine sentences | 165 | 24.36-24.38 s |

Linear, and tightly: **t ~= 0.145 s per Korean character + 0.5 s.** Three
consequences, and they are most of the design.

*It is slow enough to matter.* A realistic one-liner holds the speaker for about
four seconds, and 6 s needs only a 38-character sentence.

*But it does not stall the event loop, and the premise that it does is worth
correcting.* `create_subprocess_exec` + `await communicate()` was measured against
a 10 ms heartbeat task while a 6.82 s utterance played: 619 ticks, 91/s against an
ideal 100/s - i.e. ordinary `asyncio.sleep` overhead, not starvation. The inbound
poll is a separate task and keeps running. What *is* true is that the coroutine
which awaits `say` is suspended for the whole utterance, so a tick that awaits it
inline takes those four seconds. On a five-minute tick that is free. It would only
have been a real stall with a blocking `subprocess.run`, which is why the async
spelling is not a style choice here.

*And it makes the length of a model's output a physical quantity.* An unbounded
sentence is unbounded speech into a room, so `MAX_CHARS` exists.

## `say` exits 0 whether or not anybody heard anything

The finding that cost the most to establish, and the reason this file does not
trust its own return value very far. Synthesising the same Korean text to a file
and reading the duration back with `afinfo`:

| | audio produced | exit |
|---|---|---|
| `-v Yuna` | 1.45 s | 0 |
| `-v 'Eddy (Korean (South Korea))'` | 2.19 s | 0 |
| `-v Eddy` | **0.016 s** | 0 |
| `-v Nonexistent` | 0.90 s, byte-identical to `-v Samantha` | 0 |

So a misconfigured voice is **silent, or mumbled in English, with a clean exit and
an empty stderr.** `-v Nonexistent` falls back to Samantha (en_US) and produces
0.9 s of something that is not Korean; `-v Eddy` produces sixteen milliseconds of
nothing. Neither is distinguishable from success by any signal a caller has.

That is exactly the failure class `CLAUDE.md` names as the dangerous one here, and
it bounds what `say`'s `True` is allowed to mean: **the synthesiser accepted the
text and exited cleanly.** It does not mean the user heard it, and no available
mechanism makes it mean that. A caller deciding whether to also send the Telegram
copy must not read `True` as "they know".

Device-level failures *are* reported honestly, which is the useful half of the
asymmetry - a wrong audio device exits 1 with
`Found no Audio Output Device matching ...`, and an unreadable input file exits 1
with a readable reason. So the return code catches the environment being broken
and never catches the voice being wrong.

## Why the voice is pinned, and why it is `Yuna`

`say` has **no language flag.** The man page's options are `-v -r -o -f -a -i`
and `--progress`/`--quality`; there is no `-l`, no `--language`. So the voice name
is not a preference layered on top of a language selection - it *is* the language
selection, and the table above is what happens when it is wrong.

Eight of the nine ko_KR voices are locale variants of a shared name: `Eddy`,
`Flo`, `Grandma`, `Grandpa`, `Reed`, `Rocko`, `Sandy`, `Shelley` each exist in
**fourteen locales** on this machine, and the bare name resolves to the wrong one -
that is the 0.016 s row. Selecting them needs the full parenthesised spelling,
`-v 'Eddy (Korean (South Korea))'`.

`Yuna` is the only ko_KR voice whose bare name appears exactly once in
`say -v ?`, and it was also the fastest of the working Korean voices for the same
54-character text (8.50 s against Eddy-ko's 11.04 s). Unambiguous and fastest, so
it is the default. Leaving the system default instead was measured at 17.3 s for
that text - twice Yuna - because whatever answers is not a Korean voice.

`voice=None` is still accepted, and means "whatever the system is set to". It is
not the default for the reason above.

## Text goes in over stdin, and there is no shell anywhere

Two separate injection questions, and `-f -` answers both while argv answers only
one.

**No shell, ever.** The text is a *model-authored sentence*. `create_subprocess_exec`
takes an argv list that goes to `execve` with no interpreter in between; a shell
string or `shell=True` would hand a sentence containing `;` or backticks to `sh`,
which is a command-execution hole. Verified both ways round: a text of
`"; touch /tmp/PWNED ; echo"` through this path created no file, and neither did
`` "안녕; touch /tmp/PWNED\n`id`\n$(id)" `` over stdin. **Do not "simplify" this to a
shell string or an f-string command.** That single edit is the whole vulnerability.

**And no option injection either, which argv does not give you.** Passing the
sentence as `argv[-1]` is shell-safe but still parsed by `say` itself, and a
sentence beginning with a dash is then read as a flag: `say -v Yuna -- 그거...`
exits 1 with ``say: unrecognized option `--' ``. A model writing a leading dash or
an em-dash-as-two-hyphens is entirely ordinary, so the utterance would be lost to
punctuation. `-f -` takes the text off stdin where nothing parses it at all.

`-f -` is the correct spelling and `-f /dev/stdin` is not: the latter fails with
`/dev/stdin: Bad file descriptor`. Multi-line input is fine - newlines are read as
pauses, exit 0.

## Two of these must not overlap: the second one is dropped

macOS will not arbitrate this. Two concurrent `say` processes were measured both
running to completion in ~2.2 s, audio mixed - which is not two messages, it is
noise. So the exclusion has to be here, and the choice is between queue, interrupt
and drop.

**Drop, and report `False`.** docs/PLAN.md 6.2 sets the whole daily budget at three
utterances, so a collision is not a busy speaker - it is a symptom, and `False`
plus a warning makes it visible instead of smoothing it over.

*Not a queue*, because a queued utterance speaks late, and timing is the entire
product thesis - "말 걸 가치 있으면" is a judgement about *now*. A queue behind a
24-second utterance (measured, 165 chars) delivers something the moment for which
has passed, and it would make `say` block its caller for the sum of everything
ahead of it.

*Not an interrupt*, which is the other defensible answer and is what
`daemon/voice/audio.py`'s speaker does. Right there, wrong here: in a conversation
a newer reply supersedes an older one, whereas a proactive utterance already
passed the gate and its Telegram copy has already gone out. Cutting it off
mid-word means the room hears half a sentence that the user's phone shows in full.
`aclose` still kills an utterance outright, because docs/PLAN.md 6.4 makes stopping
one a requirement - that is a different act from starting a competing one.

## Can this collide with `daemon/voice/audio.py`?

Yes, at the device: `SoundDeviceAudio` plays PCM through PortAudio while `say` is
its own CoreAudio client, and nothing arbitrates between the two. The interlock is
upstream and already exists. `daemon/proactivity/presence.py` reads
`kAudioDevicePropertyDeviceIsRunningSomewhere`, and its docstring notes that this
"cannot distinguish *our own* speaker from someone else's call" - so a live voice
conversation reads `audio_busy=True`, and the gate declines before routing here.
That over-broad reading is load-bearing rather than a wart, and it is why this file
holds no lock against the voice path: the two cannot both be reached for the same
tick unless the gate is bypassed.

## Never raising is the contract, not politeness

`daemon/proactivity/base.py` gives the reason: a failed utterance must not lose the
Telegram copy that went with it. So every path here returns a bool, including the
ones nobody anticipated - same discipline as `presence.py`, and for the same reason
that a raising job on the scheduler is logged once while the schedule reads healthy
forever.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

SAY = "/usr/bin/say"
"""Absolute, like `presence.py`'s probes: `PATH` under a LaunchAgent is not the
`PATH` in the terminal this was written in."""

DEFAULT_VOICE = "Yuna"
"""The only ko_KR voice with an unambiguous bare name, and the fastest of the ones
that work. See the module docstring - this constant is the language selection,
because `say` has no language flag."""

MAX_CHARS = 240
"""A ceiling on how long the machine may talk uninterrupted, expressed in the unit
the text arrives in.

At the measured 0.145 s/char this is ~35 s, which is already far past anything the
judge should produce - docs/PLAN.md 6.1 asks it for "한 마디", one remark. It is
here because the text is model-authored and speech duration is linear in its
length, so a malfunctioning judge returning an essay would otherwise hold the
room's attention for minutes. Over-length text is truncated rather than refused:
the Telegram copy carries the whole thing, and saying the first part is better than
saying nothing.
"""

SPAWN_OVERHEAD_SECONDS = 5.0
SPEECH_SECONDS_PER_CHAR = 0.5
"""The timeout is derived from the text rather than flat, and both numbers come
from measurement plus deliberate slack.

0.5 s/char is ~3.5x Yuna's measured 0.145 and ~1.5x the *slowest* thing observed
(0.32 s/char, which is the system default voice mispronouncing Korean). The 5 s
floor covers process start plus the ~0.5 s fixed cost in the fit.

Derived, not flat, because a timeout should mean something when it fires. A
22-character sentence gets 16 s against a real 3.75 s, so hitting it says the
synthesiser is wedged; a flat ceiling large enough for `MAX_CHARS` would be ~2
minutes and would say nothing at all. What it guards is not slowness but a `say`
that never returns - the same hazard `presence.py` sizes `PROBE_TIMEOUT_SECONDS`
for, since both talk to system audio services that can hang.
"""


class LocalSpeaker:
    """`Speaker` over this machine's own synthesiser. Never raises, never networks.

    One class with a platform branch rather than a macOS one plus a null one, for
    the reason `MachinePresence` gives: the fallback has to answer the same
    question, and a second class would differ only in returning earlier.
    """

    def __init__(
        self,
        *,
        voice: str | None = DEFAULT_VOICE,
        platform: str | None = None,
        spawn: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self._voice = voice
        # sys.platform for the same reason presence.py gives: a compile-time
        # constant, and the spelling daemon/service.py and daemon/voice/audio.py
        # already branch on. Injectable because the non-macOS path will always be
        # written on a Mac and otherwise never exercised.
        self._platform = platform if platform is not None else sys.platform
        # The process seam, injected rather than monkeypatched, so that the
        # timeout-kill-reap logic is what gets tested instead of being faked away.
        # tests/CLAUDE.md forbids a test that makes the machine talk.
        self._spawn = spawn if spawn is not None else asyncio.create_subprocess_exec
        self._speaking = asyncio.Lock()
        self._process: Any = None

    def command(self) -> list[str] | None:
        """The argv for this platform, or `None` if nothing here can speak.

        macOS is the measured path. The Linux branch is carried over from an
        earlier implementation of this class that lived in `daemon/voice/audio.py`
        and was removed as a duplicate: dropping it to resolve the name collision
        would have been a silent regression for self-hosters, and `shutil.which`
        costs nothing. **It has never been run on a real Linux box** - if a
        self-hoster reports silence, this branch is the first thing to check.

        Both spellings read the text from stdin, which is the same property `-f -`
        gives on macOS and the reason no utterance is ever passed in argv.
        """
        if self._platform == "darwin":
            return [SAY, *(("-v", self._voice) if self._voice else ()), "-f", "-"]
        for name, args in (("espeak-ng", ["--stdin"]), ("spd-say", ["--wait", "--pipe-mode"])):
            found = shutil.which(name)
            if found:
                return [found, *args]
        return None

    async def say(self, text: str) -> bool:
        """Speak one line and wait for it to finish. True if `say` exited cleanly.

        Bounded by what the module docstring establishes: a clean exit means the
        synthesiser accepted the text, **not** that it was audible - a wrong voice
        name is silent and exits 0.
        """
        line = text.strip()
        if not line:
            # Guarded rather than passed through. `say` accepts empty and
            # whitespace-only input and exits 0 after ~0.36 s, so this is a
            # pointless process spawn and a pointless "spoke it" - and an empty
            # `Utterance.text` is how base.py spells *declining to speak*, which
            # must never be reported as speech.
            logger.debug("speaker: nothing to say")
            return False

        argv = self.command()
        if argv is None:
            # Nothing to speak with. Answered `False` rather than raised, so the
            # gate falls back to Telegram - which loses nothing, per the
            # failure-cost asymmetry in docs/PLAN.md 6.4.
            logger.warning(
                "speaker: no local synthesiser on platform %r; "
                "route proactive utterances to a channel instead",
                self._platform,
            )
            return False

        if len(line) > MAX_CHARS:
            logger.warning(
                "speaker: truncating a %d-character utterance to %d; at ~0.145 s "
                "per character the full text would hold the speaker for ~%.0f s",
                len(line),
                MAX_CHARS,
                len(line) * 0.145,
            )
            line = line[:MAX_CHARS]

        if self._speaking.locked():
            # Dropped, not queued and not interrupting - see the module docstring.
            # No `await` between this check and acquiring the lock below, so on a
            # single-threaded event loop the pair is atomic and two callers cannot
            # both pass.
            logger.warning(
                "speaker: already speaking, dropping this utterance; with a "
                "three-a-day budget an overlap means something upstream is wrong"
            )
            return False

        async with self._speaking:
            return await self._speak(argv, line)

    async def aclose(self) -> None:
        """Cut off anything still speaking.

        docs/PLAN.md 6.4: a voice coming out of the speaker during a meeting is an
        accident, so being able to stop one is a requirement rather than a
        courtesy - and a daemon shutting down while it talks into an empty room is
        the mild version of the same thing.
        """
        process, self._process = self._process, None
        if process is None or process.returncode is not None:
            return
        await self._reap(process)

    async def _speak(self, argv: list[str], line: str) -> bool:
        """One synthesiser process, start to finish. Holds `self._speaking`.

        `argv` is resolved by the caller so the platform check and the early return
        happen before the overlap lock is taken - a machine that cannot speak must
        not be able to hold the lock.
        """
        # A list going to execve, never a string going to a shell. The text is
        # model-authored, so a shell here would be a command-injection hole; it is
        # not in argv at all, because `say` parses a leading dash as an option (see
        # the module docstring). Both facts are why `-f -` and `create_subprocess_exec`
        # are load-bearing and must not be "simplified".

        try:
            process = await self._spawn(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            # `say` missing or not executable - a macOS without it, or a stripped
            # container image claiming to be darwin.
            logger.error("speaker: %s could not start: %s", SAY, exc)
            return False

        self._process = process
        timeout = SPAWN_OVERHEAD_SECONDS + SPEECH_SECONDS_PER_CHAR * len(line)
        try:
            async with asyncio.timeout(timeout):
                # The text crosses here, over a pipe, as bytes. Nothing parses it.
                _, stderr = await process.communicate(line.encode("utf-8"))
        except TimeoutError:
            await self._reap(process)
            logger.error(
                "speaker: %s did not finish %d characters in %.0fs and was killed",
                SAY,
                len(line),
                timeout,
            )
            return False
        except asyncio.CancelledError:
            # Shutdown mid-utterance. Kill first: a cancelled await detaches this
            # coroutine but leaves `say` talking into the room, and BaseException
            # skips the handler below.
            await self._reap(process)
            raise
        except Exception:  # noqa: BLE001 - base.py: a failure must not raise here
            logger.exception("speaker: %s failed unexpectedly", SAY)
            await self._reap(process)
            return False
        finally:
            if self._process is process:
                self._process = None

        if process.returncode is not None and process.returncode < 0:
            # Killed by a signal, which is how `aclose` ends an utterance. Not an
            # error, and reported as `False` anyway: the sentence was not said, and
            # base.py's guarantee is that `False` costs nothing because the
            # Telegram copy is independent of it.
            logger.info(
                "speaker: utterance stopped by signal %d", -process.returncode
            )
            return False
        if process.returncode:
            # The honest half of `say`'s reporting: a missing audio device exits 1
            # with `Found no Audio Output Device matching ...`, an unreadable input
            # file with a readable reason. Worth keeping verbatim - the return code
            # alone does not distinguish those.
            detail = stderr.decode("utf-8", "replace").strip() or "no output"
            logger.error("speaker: %s exited %d: %s", SAY, process.returncode, detail)
            return False
        return True

    async def _reap(self, process: Any) -> None:
        """Kill, and then actually wait for it.

        Both halves, following `presence.py`. An abandoned process still holds its
        pipes, so skipping the `wait` leaks a file descriptor per utterance; and
        SIGKILL rather than SIGTERM because a wedged synthesiser has no graceful
        exit to wait for. Measured cost: 6 ms to reap after SIGKILL, 3 ms after
        SIGTERM.
        """
        try:
            process.kill()
        except ProcessLookupError:
            return  # finished between the timeout firing and the signal
        try:
            await process.wait()
        except Exception:  # noqa: BLE001 - nothing above this may raise
            logger.exception("speaker: %s could not be reaped", SAY)
