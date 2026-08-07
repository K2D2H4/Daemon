"""The built-in tools: what Daemon can do to the machine it lives on.

Seven, against Hermes' ~81 and OpenClaw's ~26. The count is deliberate - those
two are platforms with a plugin surface to fill, and this is one companion on one
person's computer. Each of these earns its place by being something the owner
would otherwise have to go and do by hand mid-conversation.

Two rules the whole file follows.

**No shell.** `run_command` execs an argv vector; it never spawns `sh`. So there
is no quoting to get wrong, no `$(...)`, and a metacharacter is a parse error
rather than a second command. This costs pipes - `ls | grep x` cannot be
expressed - and buys the elimination of the entire injection class. Policy mode
`full` does not change it: `full` means "do not ask", not "add a shell".

**No data ever becomes code.** The one place that was tempting is `notify`, where
the obvious `osascript -e 'display notification "TEXT"'` lets a title containing a
quote run arbitrary AppleScript. Values are passed as `argv` to an `on run argv`
handler instead, so the script is a constant.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import re
import shlex
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any

from daemon import clock
from daemon.llm.base import ToolSpec
from daemon.proactivity.base import Presence
from daemon.tools.base import Risk, ToolError
from daemon.tools.extract import extract_document

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECS = 20.0
DEFAULT_MAX_OUTPUT = 4000
"""Characters of tool output handed back to the model. Every byte here is
re-sent on every subsequent round of the same turn, so a `cat` of something large
would cost more than the answer is worth."""

READ_MAX_BYTES = 200_000
"""Ceiling on a single `read_file`, before truncation to MAX_OUTPUT. Stops a
multi-gigabyte file being pulled into memory to then be thrown away."""

OUTPUT_READ_CAP = 1_000_000
"""Bytes of a command's output actually kept while it runs.

Measured, because the obvious `communicate()` does not do this: a command writing
200 MB to stdout grew the daemon's RSS by 651 MB before `_truncate` threw all but a
thousand characters of it away. One `cat` of the wrong file was enough to take the
process down. Beyond this the output is drained and discarded rather than the child
being killed - a command may print a lot and then do something useful, and the
existing timeout already bounds how long it has."""

SHELL_OPERATORS = (";", "&&", "||", "|", "`", "$(", ">", "<", "&", "\n")
"""Rejected in a command string. Not a security control - there is no shell to
inject into - but a correctness one: without it `ls; rm x` would exec a program
literally named `ls;` and the failure would read as "command not found".
"""

DENIED_NAMES = frozenset(
    {
        ".ssh",
        ".aws",
        ".gnupg",
        ".kube",
        ".docker",
        ".netrc",
        ".git-credentials",
        ".npmrc",
        ".pypirc",
        "Keychains",
    }
)
"""Path components that are refused outright, at any depth."""

DENIED_GLOBS = ("*.pem", "*.key", "id_rsa*", "id_ed25519*", ".env", ".env.*", "*.sqlite3*")
"""Filenames that are refused. `.env` is here because it holds this daemon's own
API keys, and a companion that can be talked into reading them out has handed over
the user's billing."""


class PathScope:
    """Where the file tools may look.

    Symlinks are resolved *before* the check, not after, so a link inside an
    allowed root pointing at `/etc` is refused as an escape. That ordering is the
    whole content of this class.
    """

    def __init__(self, roots: Sequence[str | Path]) -> None:
        resolved: list[Path] = []
        for root in roots:
            try:
                resolved.append(Path(root).expanduser().resolve())
            except (OSError, ValueError) as exc:
                # ValueError, not just OSError: a root containing a null byte raises
                # `ValueError: embedded null character` out of `lstat`, which escaped
                # the constructor and cost every *other* root as well - the opposite
                # of "skip the bad one".
                logger.warning("tool root %s cannot be resolved (%s); skipping it", root, exc)
        if not resolved:
            raise ValueError("a path scope needs at least one usable root")
        self.roots = tuple(resolved)

    def resolve(self, raw: object) -> Path:
        """A checked absolute path, or `ToolError` explaining the refusal."""
        if not isinstance(raw, str) or not raw.strip():
            raise ToolError("path must be a non-empty string")
        try:
            path = Path(raw).expanduser().resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            # A symlink loop does *not* land here on 3.13 - non-strict `resolve()`
            # hands back the path unchanged, and the loop surfaces as ELOOP when the
            # file is opened, which the callers turn into a ToolError. What does land
            # here is a null byte (ValueError) and a path too long (OSError).
            raise ToolError(f"{raw!r} cannot be resolved: {exc}") from exc

        if not any(path == root or root in path.parents for root in self.roots):
            allowed = ", ".join(str(root) for root in self.roots)
            raise ToolError(
                f"{path} is outside the paths I may touch ({allowed}). "
                "Ask the owner to widen DAEMON_TOOLS_ROOTS if that is wrong."
            )

        parts = set(path.parts)
        denied = parts & DENIED_NAMES
        if denied:
            raise ToolError(f"{path} is in {next(iter(denied))}, which is off limits")
        if any(PurePath(path.name).match(pattern) for pattern in DENIED_GLOBS):
            raise ToolError(f"{path.name} holds credentials, so it is off limits")
        return path


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit]}\n… [{dropped} more characters, not shown]"


def _looks_binary(raw: bytes) -> bool:
    """Whether a file should be refused as text rather than decoded to gibberish.

    A NUL byte is the classic signal - text files effectively never carry one,
    and images, archives, executables and PDF streams almost always do. The
    replacement-ratio is the backstop for a binary that happens to have no NUL in
    the read prefix. The 0.30 threshold sits above a real log with an odd byte or
    two (`test_read_file_survives_a_bad_byte` is ~11%) and below the 32% measured
    on the PDF that started all this.
    """
    if b"\x00" in raw:
        return True
    text = raw.decode("utf-8", errors="replace")
    if not text:
        return False
    return text.count("�") / len(text) > 0.30


async def _drain(process: asyncio.subprocess.Process, cap: int) -> bytes:
    """Read a child's output, keeping at most `cap` bytes of it.

    Reading with `communicate()` keeps everything, which is how 200 MB of stdout
    became 651 MB of RSS before any of it was truncated. Draining past the cap
    rather than stopping matters: a pipe nobody reads fills up and the child blocks
    on write forever, so the timeout would fire instead of the command finishing.
    """
    kept: list[bytes] = []
    held = 0
    stream = process.stdout
    if stream is not None:
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            if held < cap:
                room = cap - held
                kept.append(chunk[:room])
                held += min(room, len(chunk))
    await process.wait()
    return b"".join(kept)


# --- filesystem -------------------------------------------------------------


class ListDir:
    risk: Risk = "safe"
    spec = ToolSpec(
        name="list_dir",
        description=(
            "List the entries of a directory on this computer. Use this instead of "
            "running `ls`, so it works without asking the owner for approval."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path, or ~ relative."}
            },
            "required": ["path"],
        },
    )

    def __init__(self, scope: PathScope, *, max_entries: int = 200) -> None:
        self._scope = scope
        self._max_entries = max_entries

    def preview(self, arguments: Mapping[str, Any]) -> str:
        return f"list {arguments.get('path', '?')}"

    async def run(self, arguments: Mapping[str, Any]) -> str:
        path = self._scope.resolve(arguments.get("path"))
        try:
            entries = await asyncio.to_thread(lambda: sorted(path.iterdir()))
        except NotADirectoryError as exc:
            raise ToolError(f"{path} is a file, not a directory") from exc
        except FileNotFoundError as exc:
            raise ToolError(f"{path} does not exist") from exc
        except OSError as exc:
            raise ToolError(f"{path} could not be listed: {exc}") from exc

        lines = [f"{entry.name}{'/' if entry.is_dir() else ''}" for entry in entries]
        if not lines:
            return f"{path} is empty"
        shown = lines[: self._max_entries]
        if len(lines) > self._max_entries:
            shown.append(f"… and {len(lines) - self._max_entries} more")
        return "\n".join(shown)


class ReadFile:
    risk: Risk = "safe"
    spec = ToolSpec(
        name="read_file",
        description=(
            "Read a file's text on this computer. Handles plain text and also PDF, "
            "Word (.docx), Excel (.xlsx) and PowerPoint (.pptx) documents, pulling "
            "out their text. Use this instead of running `cat`, so it works without "
            "asking the owner for approval."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path, or ~ relative."}
            },
            "required": ["path"],
        },
    )

    def __init__(self, scope: PathScope, *, max_output: int = DEFAULT_MAX_OUTPUT) -> None:
        self._scope = scope
        self._max_output = max_output

    def preview(self, arguments: Mapping[str, Any]) -> str:
        return f"read {arguments.get('path', '?')}"

    async def run(self, arguments: Mapping[str, Any]) -> str:
        path = self._scope.resolve(arguments.get("path"))

        def _read() -> str:
            extracted = extract_document(path)
            if extracted is not None:
                text = extracted.strip()
                if not text:
                    # A scanned PDF is images, not text. Saying so beats returning
                    # nothing and letting the model claim it read the file.
                    return f"{path.name} has no extractable text (it may be a scanned image)."
                return text
            with path.open("rb") as handle:
                raw = handle.read(READ_MAX_BYTES)
            if _looks_binary(raw):
                raise ToolError(
                    f"{path.name} is not a text file, so I cannot read it as text. I can "
                    "read text files and PDF, Word, Excel and PowerPoint documents."
                )
            # `replace` rather than raising: a log file with one bad byte is still
            # worth reading, and a tool that refuses it teaches the model nothing.
            return raw.decode("utf-8", errors="replace")

        try:
            text = await asyncio.to_thread(_read)
        except FileNotFoundError as exc:
            raise ToolError(f"{path} does not exist") from exc
        except IsADirectoryError as exc:
            raise ToolError(f"{path} is a directory; use list_dir") from exc
        except OSError as exc:
            raise ToolError(f"{path} could not be read: {exc}") from exc
        return _truncate(text, self._max_output) if text else f"{path} is empty"


class WriteFile:
    risk: Risk = "guarded"
    spec = ToolSpec(
        name="write_file",
        description=(
            "Create or overwrite a text file on this computer. This replaces the "
            "whole file. The owner is asked before it runs."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path, or ~ relative."},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    )

    def __init__(self, scope: PathScope) -> None:
        self._scope = scope

    def preview(self, arguments: Mapping[str, Any]) -> str:
        content = arguments.get("content")
        size = len(content) if isinstance(content, str) else 0
        return f"write {size} characters to {arguments.get('path', '?')}"

    async def run(self, arguments: Mapping[str, Any]) -> str:
        path = self._scope.resolve(arguments.get("path"))
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        existed = path.exists()

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        try:
            await asyncio.to_thread(_write)
        except OSError as exc:
            raise ToolError(f"{path} could not be written: {exc}") from exc
        return f"{'overwrote' if existed else 'created'} {path} ({len(content)} characters)"


# --- execution --------------------------------------------------------------


class RunCommand:
    risk: Risk = "guarded"
    spec = ToolSpec(
        name="run_command",
        description=(
            "Run one program on this computer and return its output. There is no "
            "shell: pipes, redirection and `;` are not available, so run one command "
            "at a time. Unless the command is allowlisted the owner is asked first, "
            "so prefer read_file and list_dir where they will do."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command with its arguments, e.g. 'git status'.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Directory to run it in. Defaults to the owner's home.",
                },
            },
            "required": ["command"],
        },
    )

    def __init__(
        self,
        scope: PathScope,
        *,
        timeout_secs: float = DEFAULT_TIMEOUT_SECS,
        max_output: int = DEFAULT_MAX_OUTPUT,
    ) -> None:
        self._scope = scope
        self._timeout = timeout_secs
        self._max_output = max_output

    def preview(self, arguments: Mapping[str, Any]) -> str:
        command = arguments.get("command", "?")
        cwd = arguments.get("cwd")
        return f"run `{command}`" + (f" in {cwd}" if cwd else "")

    def argv(self, arguments: Mapping[str, Any]) -> list[str]:
        raw = arguments.get("command")
        if not isinstance(raw, str) or not raw.strip():
            raise ToolError("command must be a non-empty string")
        for operator in SHELL_OPERATORS:
            if operator in raw:
                raise ToolError(
                    f"{operator!r} is not available: I run one program directly rather "
                    "than through a shell. Run the steps as separate calls."
                )
        try:
            argv = shlex.split(raw)
        except ValueError as exc:  # unbalanced quotes
            raise ToolError(f"{raw!r} is not a valid command: {exc}") from exc
        if not argv or not argv[0]:
            # `shlex.split('""')` is `['']`, not `[]`. Without the second check that
            # reached `shutil.which('')`, which answers None, and the owner got
            # "'' is not installed, or not on PATH" instead of being told the command
            # was empty.
            raise ToolError("command must name a program")
        return argv

    async def run(self, arguments: Mapping[str, Any]) -> str:
        argv = self.argv(arguments)
        raw_cwd = arguments.get("cwd")
        cwd = self._scope.resolve(raw_cwd) if raw_cwd else self._scope.roots[0]
        if not cwd.is_dir():
            raise ToolError(f"{cwd} is not a directory")

        executable = shutil.which(argv[0], path=os.environ.get("PATH"))
        if executable is None:
            raise ToolError(f"{argv[0]} is not installed, or not on PATH")

        try:
            process = await asyncio.create_subprocess_exec(
                executable,
                *argv[1:],
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                # Merged, because a command's diagnosis is usually the useful half
                # and two streams would need interleaving to make sense of anyway.
                stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise ToolError(f"{argv[0]} could not be started: {exc}") from exc

        try:
            stdout = await asyncio.wait_for(
                _drain(process, OUTPUT_READ_CAP), timeout=self._timeout
            )
        except TimeoutError:
            # Killed, then reaped: without the second await the process is left a
            # zombie and asyncio complains at shutdown, on a path that by
            # definition only runs when something is already wrong.
            process.kill()
            await process.wait()
            raise ToolError(
                f"`{' '.join(argv)}` was still running after {self._timeout:.0f}s and was "
                "stopped. Nothing about the result is known."
            ) from None

        output = _truncate(stdout.decode("utf-8", errors="replace").rstrip(), self._max_output)
        code = process.returncode
        if code == 0:
            return output or "(no output)"
        # Not a ToolError: a non-zero exit is information, and the model usually
        # needs the output to act on it.
        return f"exited {code}\n{output}" if output else f"exited {code} with no output"


class OpenPath:
    risk: Risk = "guarded"
    spec = ToolSpec(
        name="open_path",
        description=(
            "Open a file, folder, application or URL with whatever this computer "
            "uses by default. The owner is asked first."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "A path, or an http(s) URL.",
                }
            },
            "required": ["target"],
        },
    )

    URL_RE = re.compile(r"^https?://", re.IGNORECASE)
    SCHEME_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.\-]*):")
    """Any scheme at all, not just the ones with `//`. `javascript:alert(1)` and
    `file:///etc/passwd` both have to be caught, and only the second has a slash -
    matching on `://` let the first fall through to the path check, where it was
    refused for the wrong reason and with a misleading message."""

    def __init__(self, scope: PathScope, *, timeout_secs: float = DEFAULT_TIMEOUT_SECS) -> None:
        self._scope = scope
        self._timeout = timeout_secs

    def preview(self, arguments: Mapping[str, Any]) -> str:
        return f"open {arguments.get('target', '?')}"

    def argv(self, arguments: Mapping[str, Any]) -> list[str]:
        target = arguments.get("target")
        if not isinstance(target, str) or not target.strip():
            raise ToolError("target must be a non-empty string")
        target = target.strip()
        opener = "open" if platform.system() == "Darwin" else "xdg-open"
        if self.URL_RE.match(target):
            # A URL is guarded rather than safe precisely because it leaves the
            # machine, so it is not scope-checked - it is approval-checked.
            return [opener, target]
        scheme = self.SCHEME_RE.match(target)
        if scheme is not None:
            raise ToolError(
                f"{scheme.group(1)}: links are not opened, only http(s) and local paths"
            )
        return [opener, str(self._scope.resolve(target))]

    async def run(self, arguments: Mapping[str, Any]) -> str:
        argv = self.argv(arguments)
        executable = shutil.which(argv[0])
        if executable is None:
            raise ToolError(f"{argv[0]} is not available on this machine")
        process = await asyncio.create_subprocess_exec(
            executable,
            *argv[1:],
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=self._timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise ToolError(f"opening {argv[-1]} did not finish in time") from None
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise ToolError(f"could not open {argv[-1]}: {detail or 'unknown failure'}")
        return f"opened {argv[-1]}"


# --- presence and attention -------------------------------------------------

NOTIFY_SCRIPT = (
    "on run argv",
    "display notification (item 2 of argv) with title (item 1 of argv)",
    "end run",
)
"""AppleScript as a constant, with the text supplied as `argv`.

Building the script by interpolation is the obvious version and it is a remote
code execution bug: a title containing a double quote closes the string and
everything after it is AppleScript. `on run argv` keeps data out of the program.
"""


class Notify:
    risk: Risk = "safe"
    spec = ToolSpec(
        name="notify",
        description=(
            "Show a desktop notification on this computer. Local: nothing is sent "
            "anywhere. Use it when the owner is at the machine and something is "
            "worth a glance rather than a message."
        ),
        parameters={
            "type": "object",
            "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
            "required": ["title", "body"],
        },
    )

    def __init__(self, *, timeout_secs: float = 10.0) -> None:
        self._timeout = timeout_secs

    def preview(self, arguments: Mapping[str, Any]) -> str:
        return f"notify: {arguments.get('title', '')}"

    async def run(self, arguments: Mapping[str, Any]) -> str:
        title = str(arguments.get("title", "")).strip()
        body = str(arguments.get("body", "")).strip()
        if not title and not body:
            raise ToolError("a notification needs a title or a body")

        if platform.system() == "Darwin":
            argv = ["osascript"]
            for line in NOTIFY_SCRIPT:
                argv += ["-e", line]
            argv += ["--", title, body]
        else:
            # `--` is not decoration: without it a title of `--help` (or worse) is
            # read as an option to notify-send rather than as the text to show.
            argv = ["notify-send", "--", title, body]

        executable = shutil.which(argv[0])
        if executable is None:
            raise ToolError(f"{argv[0]} is not available, so I cannot show notifications here")
        process = await asyncio.create_subprocess_exec(
            executable,
            *argv[1:],
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=self._timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise ToolError("the notification did not go out in time") from None
        if process.returncode != 0:
            raise ToolError(
                f"the notification was refused: {stderr.decode(errors='replace').strip()}"
            )
        return "shown"


class SystemState:
    """Read-only: is the owner here, and what are they doing?

    Delegates to `daemon/proactivity/presence.py`, which M3 needed first and which
    measured the probes this tool originally guessed at: `HIDIdleTime` is
    nanoseconds; `name of ... application process` returns the *executable* rather
    than the app (`stable`, not `Warp`), so `lsappinfo` is used instead; and no
    `ioreg` audio class answers on Apple Silicon, so audio comes from CoreAudio.
    Keeping a second implementation would have meant shipping the wrong readings
    next to the right ones.

    Takes the `Presence` protocol rather than the class - `daemon/app.py` injects the
    concrete one, which is the direction this repo's layering allows.
    """

    risk: Risk = "safe"
    spec = ToolSpec(
        name="system_state",
        description=(
            "Check whether the owner is at this computer right now: how long the "
            "keyboard and mouse have been idle, which app is in front, whether "
            "something is holding the audio device, and the local time. Read-only."
        ),
        parameters={"type": "object", "properties": {}},
    )

    def __init__(self, presence: Presence | None = None) -> None:
        self._presence = presence

    def preview(self, arguments: Mapping[str, Any]) -> str:
        return "check whether the owner is at the machine"

    async def run(self, arguments: Mapping[str, Any]) -> str:
        # Local time, not UTC: the question this answers is "is it the middle of
        # their night", and clock.now() is deliberately UTC everywhere else.
        lines = [
            f"platform: {platform.system()} {platform.release()}",
            f"local time: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
            f"utc: {clock.now_iso()}",
        ]
        if self._presence is None:
            lines.append("machine readings: unavailable in this configuration")
            return "\n".join(lines)

        reading = await self._presence.read()
        if reading.idle_seconds is None:
            lines.append("input idle: unknown")
        else:
            lines.append(f"input idle: {reading.idle_seconds:.0f}s")
            # The three-way answer, not the number's truthiness: `at_keyboard` is
            # None when it cannot be known, and that is not "away".
            at = reading.at_keyboard
            answer = "yes" if at else "no" if at is False else "unknown"
            lines.append(f"at the keyboard: {answer}")
        if reading.foreground_app:
            lines.append(f"frontmost app: {reading.foreground_app}")
        if reading.audio_busy is not None:
            lines.append(f"audio in use: {'yes' if reading.audio_busy else 'no'}")
        for note in reading.unknown:
            # Said rather than dropped: a probe that failed is a fact about the
            # answer, and omitting it silently reads as "nothing to report".
            lines.append(f"could not read: {note}")
        return "\n".join(lines)


def builtin_tools(
    *,
    roots: Sequence[str | Path],
    timeout_secs: float = DEFAULT_TIMEOUT_SECS,
    max_output: int = DEFAULT_MAX_OUTPUT,
    presence: Presence | None = None,
) -> list[Any]:
    """Every built-in, in the order the model will see them.

    Read-only tools first, deliberately: the descriptions tell the model to prefer
    `read_file` over `cat`, and the ordering says the same thing again.
    """
    scope = PathScope(roots)
    return [
        ListDir(scope),
        ReadFile(scope, max_output=max_output),
        SystemState(presence),
        Notify(),
        WriteFile(scope),
        RunCommand(scope, timeout_secs=timeout_secs, max_output=max_output),
        OpenPath(scope, timeout_secs=timeout_secs),
    ]
