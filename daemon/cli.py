"""The `daemon` command.

`run` is the resident process; everything else is an operator command that has to
work on a machine where the service is misconfigured, so each one prints what it
found instead of raising a traceback.

Exit codes: 0 fine, 1 something is wrong with the install, 2 the command or the
configuration could not even be read.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daemon import __version__
from daemon.config import ENV_FILE, OLLAMA, ConfigError, Settings
from daemon.fs import DIR_MODE
from daemon.service import ServiceAction, ServiceError, ServiceStatus, service_for

OK = 0
PROBLEM = 1
USAGE = 2

GITHUB_REPO = "K2D2H4/Daemon"
PACKAGE_NAME = "daemon-ai"
"""The repo and the distributable. `daemon update` and install.sh must agree on
both, because update re-installs exactly what the one-liner does."""
INSTALL_SPEC = f"{PACKAGE_NAME}[mcp]"
"""What `uv tool install` requests, extra included. MCP defaults on (config.py),
so a plain `daemon-ai` install shows the admin's MCP tab and then fails every
connect with "No module named 'mcp'". `install.sh` installs this same spec - the
docstring on `_update` promises the two do not drift, and a bare-name reinstall
here would silently drop the extra on the next `daemon update`. The version cap
lives in the extra (pyproject `mcp>=1.9,<2`), so `[mcp]` also keeps `daemon update`
off the incompatible mcp 2.0."""

_GROUP_ORDER = (
    "the resident",
    "every day",
    "when something looks wrong",
    "setup",
    "now and then",
)
"""The order the help prints its groups in, and the set `build_parser` will accept.
Ordered by how often a command is typed, so the top of the list is the part an
owner actually uses."""

_LOG_LINES = 50
"""How much history `daemon log` shows before it starts following."""

_TAIL_BLOCK = 8192
"""How much `daemon log` reads per step when walking a log backwards."""

_LOG_POLL_SECONDS = 0.25
"""How long `daemon log -f` sleeps when the file has nothing new. Short enough to
feel live, long enough that watching an idle daemon is not a spin loop."""

_LOG_NOISE = re.compile(
    # The admin console health-polls itself and Telegram is long-polled once a
    # second: together 56% of a measured 22,790-line log. Both patterns end in the
    # success status on purpose - a 409 from getUpdates is the exact line that
    # explains a bot token clash, and a 500 from the admin is never noise.
    r'uvicorn\.access\b.*"\s+[23]\d\d\s*$'
    r"|getUpdates\s+\"HTTP/[\d.]+\s+2\d\d[^\"]*\"\s*$"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daemon",
        description="Self-hosted AI companion. `daemon run` with no arguments is the default.",
        # The command list is laid out by hand below; argparse must not re-wrap it.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Handled in `main` rather than argparse's `action="version"`, which raises
    # SystemExit - every other command here returns an int, and a bug report's
    # first line should not depend on catching an exception.
    parser.add_argument(
        "--version", action="store_true", help="print the installed version and exit"
    )
    # `metavar` collapses argparse's brace blob into one token. Not SUPPRESS: that
    # would also drop `<command>` from the usage line. The commands themselves are
    # registered without a `help=`, which is what keeps argparse from listing them
    # a second time - the grouped epilog below is the only listing.
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    groups: dict[str, list[tuple[str, str]]] = {}

    def add(name: str, *, group: str, help: str, **kwargs: Any) -> argparse.ArgumentParser:
        """Register a command and file it under a group, in one call.

        Sixteen commands in one flat list gave no clue which are typed daily and
        which are typed once ever. Grouping them is only safe if the group cannot
        drift from the command, so `group` is required here and there is no other
        way in: a command added without one does not compile, rather than quietly
        vanishing from the help.
        """
        assert group in _GROUP_ORDER, f"unknown help group {group!r}"
        groups.setdefault(group, []).append((name, help))
        # `help` is deliberately not forwarded: argparse only lists a subcommand if
        # it was given one, and passing SUPPRESS prints the string "==SUPPRESS==".
        # Withholding it is what leaves the epilog as the single listing.
        return sub.add_parser(name, **kwargs)

    add(
        "run",
        group="the resident",
        help="run the daemon in the foreground (what the service supervises)",
    )
    setup = add(
        "setup",
        group="setup",
        help="first-run onboarding: pick a preset, verify keys, write .env",
    )
    setup.add_argument(
        "--check",
        action="store_true",
        help="report what is missing and exit; asks nothing, contacts nobody",
    )
    install = add(
        "install", group="setup", help="install the OS service so it survives a reboot"
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="replace an existing unit file after showing what changes",
    )
    add("uninstall", group="setup", help="stop the OS service and remove its unit file")
    add("status", group="every day", help="is the service installed and running")
    log = add(
        "log",
        group="every day",
        help="what the resident has been writing to its log (`-f` to keep streaming)",
    )
    log.add_argument(
        "-f", "--follow", action="store_true", help="keep streaming new lines until Ctrl-C"
    )
    log.add_argument(
        "-n",
        "--lines",
        type=int,
        default=_LOG_LINES,
        help=f"how many past lines to show ({_LOG_LINES} if omitted)",
    )
    log.add_argument(
        "--raw",
        action="store_true",
        help="do not filter anything out, including the polling that is over half "
        "the file",
    )
    help_cmd = add(
        "help",
        group="every day",
        help="this list, or `daemon help <command>` for one command's own help",
    )
    help_cmd.add_argument(
        "topic", nargs="?", help="a command name; omitted, print this list"
    )
    add(
        "doctor",
        group="when something looks wrong",
        help="check configuration, Ollama, data dir and schema",
    )
    add(
        "reindex",
        group="when something looks wrong",
        help="rebuild the sqlite mirror from the markdown log",
    )
    add(
        "update",
        group="setup",
        help="reinstall the latest release in place (needs uv; source installs use git pull)",
    )
    reflect = add(
        "reflect",
        group="now and then",
        help="consolidate a day of conversation into memory and observations",
    )
    reflect.add_argument(
        "--date",
        help="a local YYYY-MM-DD. Omitted: every unreflected day except today, "
        "oldest first - today is still being written to.",
    )
    reflect.add_argument(
        "--force", action="store_true", help="redo a day that already has an artifact"
    )
    proactive = add(
        "proactive",
        group="now and then",
        help="run one proactivity round now and print its verdicts (dry by default)",
    )
    proactive.add_argument(
        "--speak",
        action="store_true",
        help="actually decide what to say and deliver it. Without this the round "
        "stops at the gate and costs no model call.",
    )

    add("voice", group="now and then", help="hold one spoken conversation at this machine")

    add(
        "request-mic",
        group="setup",
        help="claim macOS microphone access (used by Daemon.app during install)",
    )

    wake = add(
        "wake",
        group="now and then",
        help="the always-on wake phrase: measure it on your voice, then hear it work",
    )
    wake_sub = wake.add_subparsers(dest="wake_command", required=True)
    calibrate = wake_sub.add_parser(
        "calibrate",
        help="say the phrase a few times and save what the recognizer actually heard",
    )
    calibrate.add_argument(
        "--takes",
        type=int,
        # Not a literal default, so the number and the reasoning behind it stay in
        # one place - `wake_cli.TAKES`, which is 3. Naming it here would need
        # wake_cli imported to build the parser, and the parser is built for every
        # command.
        help="how many times to say it (3 if omitted; fewer than 2 cannot show "
        "whether the transcription is stable)",
    )
    wake_test = wake_sub.add_parser(
        "test", help="run the gate here and print every wake event it fires"
    )
    wake_test.add_argument(
        "--seconds",
        type=int,
        help="how long to listen (60 if omitted). 0 waits for Ctrl-C, which is what "
        "leaving a television on and watching for a false wake needs.",
    )

    pairing = add(
        "pairing", group="every day", help="see and approve who may talk to Daemon"
    )
    pairing_sub = pairing.add_subparsers(dest="pairing_command", required=True)
    pairing_sub.add_parser("list", help="pending pairing requests and their codes")
    approve = pairing_sub.add_parser("approve", help="approve a pairing code")
    approve.add_argument("code", help="the 8-character code the bot replied with")

    persona = add(
        "persona",
        group="now and then",
        help="see active learned persona rules (M4); no subcommand just lists them",
    )
    persona_sub = persona.add_subparsers(dest="persona_command")
    evolve = persona_sub.add_parser(
        "evolve", help="run the weekly persona-evolution pass now"
    )
    evolve.add_argument(
        "--force",
        action="store_true",
        help="run even if this week's diary already exists",
    )
    # `persona_forget`, not `forget`: `daemon tools forget` below binds that name
    # too, and two parsers sharing one variable is a trap for whoever adds an
    # argument to the first of them.
    persona_forget = persona_sub.add_parser(
        "forget", help="retire a learned rule - a human's deletion request"
    )
    persona_forget.add_argument("id", type=int, help="the rule id, from `daemon persona`")
    persona_forget.add_argument("--why", required=True, help="why this rule should be retired")
    tools = add(
        "tools",
        group="every day",
        help="what Daemon may do to this machine, and what it did",
    )
    tools_sub = tools.add_subparsers(dest="tools_command", required=True)
    tools_sub.add_parser("list", help="the tools that are loaded, and the policy in force")
    tools_sub.add_parser("log", help="recent tool calls, including the refused ones")
    tools_sub.add_parser("pending", help="approvals waiting on an answer")
    forget = tools_sub.add_parser("forget", help="drop a standing approval granted with 'always'")
    forget.add_argument("pattern", help="the command pattern to stop trusting, e.g. 'git status'")

    parser.epilog = _grouped_commands(groups)
    # Read by `daemon help <command>` and by the test that checks no command
    # escaped a group. Set here because `add` above is the only registration path,
    # so these two cannot disagree with what the parser actually accepts.
    parser.command_groups = groups  # type: ignore[attr-defined]
    parser.command_parsers = sub.choices  # type: ignore[attr-defined]
    return parser


def _grouped_commands(groups: dict[str, list[tuple[str, str]]]) -> str:
    """The command list, grouped by when you would reach for them.

    This is the *only* rendering of the list - the subparsers are registered with
    `help=SUPPRESS` - so there is no second copy to keep in step.
    """
    width = max(len(name) for entries in groups.values() for name, _ in entries)
    columns = min(shutil.get_terminal_size(fallback=(80, 24)).columns, 100)
    lines: list[str] = []
    for title in _GROUP_ORDER:
        entries = groups.get(title)
        if not entries:
            continue
        lines.append(f"{title}:")
        for name, text in entries:
            lines.extend(
                textwrap.wrap(
                    text,
                    width=max(columns, width + 24),
                    initial_indent=f"  {name.ljust(width)}  ",
                    subsequent_indent=" " * (width + 4),
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        # `daemon` alone kept running the server through all of M1a. Breaking that
        # to make a point about explicitness would only break people's plists.
        args_list = ["run"]

    args = build_parser().parse_args(args_list)

    if args.version:
        print(f"daemon {__version__}")
        return OK

    command: str = args.command

    if command == "help":
        # Before Settings, like doctor and setup: the moment you most need to be
        # told what the commands are is the moment the configuration is broken.
        return _help(args.topic)
    if command == "doctor":
        # Doctor is the one command that must survive a configuration it cannot
        # load - explaining the breakage is its whole job.
        return _doctor()
    if command == "setup":
        # Same reason, more so: setup exists for the machine that has no usable
        # configuration yet, so it must not require one to start.
        return _setup(check_only=args.check)
    if command == "request-mic":
        # Also before Settings: this is the foreground grant Daemon.app execs
        # during `daemon install`, and it needs no config to pop the prompt.
        return _request_mic()
    if command == "wake" and args.wake_command == "calibrate":
        # Also before Settings, and for setup's reason: calibration reads nothing
        # out of the configuration and writes one key into `.env`, so it has to work
        # on an install whose configuration does not load yet. `wake test` does need
        # settings - it builds the gate - so it stays below.
        from daemon.wake_cli import calibrate

        return calibrate(takes=args.takes)
    if command == "update":
        # Before Settings, and for setup's reason: you may be updating precisely
        # because a version is broken, and the new code should land regardless of
        # whether this one's configuration loads.
        return _update()

    try:
        settings = Settings()
    except ConfigError as exc:
        print(f"daemon: bad configuration: {exc}", file=sys.stderr)
        print("daemon: run `daemon doctor` for the full picture.", file=sys.stderr)
        return USAGE

    if command == "run":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        # Before the loop starts, because this is the command that suffers for it and
        # `daemon doctor` is the command nobody runs first. An install lost hours to a
        # repeating Telegram 409 whose cause was `~/.zshrc` exporting
        # TELEGRAM_BOT_TOKEN for a different tool: the environment outranks the file,
        # so `.env` named the right bot and was never consulted. Doctor reports it,
        # but only if you already suspect something.
        override = _env_override_check(settings)
        if not override.ok:
            logging.getLogger(__name__).warning("%s", override.detail)
        _load_env_secrets()
        return _serve(settings)
    if command == "reindex":
        inserted = _reindex(settings)
        print(f"reindexed {inserted} message(s) the mirror was missing")
        return OK
    if command == "proactive":
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
        return asyncio.run(_proactive(settings, speak=args.speak))
    if command == "reflect":
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
        return asyncio.run(_reflect(settings, date=args.date, force=args.force))
    if command == "voice":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        from daemon.app import run_voice

        return asyncio.run(run_voice(settings))
    if command == "wake":
        # Only `test` reaches here; `calibrate` returned above.
        #
        # Logging on, at WARNING: the gate steps over a raising VAD or recognizer
        # rather than dying, and logs it - so without a handler the one command whose
        # job is to explain why nothing fired would be hiding the explanation. Not
        # INFO: the gate logs every fire there too, and this command prints those
        # itself.
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
        from daemon.wake_cli import listen

        return listen(settings, seconds=args.seconds)
    if command == "pairing":
        return _pairing(settings, args)
    if command == "persona":
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
        return asyncio.run(_persona(settings, args))
    if command == "tools":
        return _tools(settings, args)
    if command == "log":
        return _log(settings, follow=args.follow, lines=args.lines, raw=args.raw)

    try:
        if command == "install":
            return _install(settings, force=args.force)
        if command == "uninstall":
            return _uninstall(settings)
        if command == "status":
            return _print_status(service_for(settings).status())
    except ServiceError as exc:
        print(f"daemon: {exc}", file=sys.stderr)
        return PROBLEM

    raise AssertionError(f"unhandled command {command!r}")  # pragma: no cover


def _help(topic: str | None) -> int:
    """`daemon help`, and `daemon help <command>`.

    `daemon help` prints the same parser's `--help`, not a second document written
    beside it, so the two can never drift apart.
    """
    parser = build_parser()
    if topic is None:
        parser.print_help()
        return OK
    choices: dict[str, argparse.ArgumentParser] = parser.command_parsers  # type: ignore[attr-defined]
    target = choices.get(topic)
    if target is None:
        print(f"daemon: no such command: {topic}", file=sys.stderr)
        print(f"daemon: commands are: {', '.join(sorted(choices))}", file=sys.stderr)
        return USAGE
    target.print_help()
    return OK


def _log(settings: Settings, *, follow: bool, lines: int, raw: bool) -> int:
    """`daemon log`: what the resident has been writing.

    The resident's own stderr, which is where every logger in the process ends up.
    Not the tool audit trail - that is `daemon tools log`, a different question.
    """
    path = service_for(settings).err_log
    if not path.exists():
        print(f"daemon: no log at {path}", file=sys.stderr)
        print(
            "daemon: the service may never have run here - `daemon status` says.",
            file=sys.stderr,
        )
        return PROBLEM

    keep = _keeper(raw)
    for line in _tail_lines(path, lines, keep):
        print(line)

    if not follow:
        return OK
    try:
        for line in _stream_lines(path):
            if keep(line):
                print(line, flush=True)
    except KeyboardInterrupt:
        # Ctrl-C is how you stop following. It is not an error, and a traceback
        # here would be the last thing printed after a good session.
        print()
    return OK


def _keeper(raw: bool) -> Callable[[str], bool]:
    if raw:
        return lambda _: True
    return lambda line: not _LOG_NOISE.search(line)


def _tail_lines(path: Path, count: int, keep: Callable[[str], bool]) -> list[str]:
    """The last `count` lines this would actually print.

    Filtering after taking the last N would be the wrong way round: over half the
    file is polling, so `daemon log -n 12` on an idle daemon showed twelve lines of
    noise reduced to an empty screen, which reads as "the log is broken". `-n` is
    how much you want to *see*, so the filter runs first and the count is of what
    survives.

    Reads backwards a block at a time and stops as soon as it has enough - the file
    this points at was 2.4MB after five days.
    """
    if count <= 0:
        return []
    kept: list[str] = []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        held = b""
        at_start = position == 0
        while not at_start and len(kept) < count:
            step = min(_TAIL_BLOCK, position)
            position -= step
            at_start = position == 0
            handle.seek(position)
            held = handle.read(step) + held
            pieces = held.split(b"\n")
            # Unless this block starts the file, its first piece is the tail of a
            # line whose head is in the block before it: hold it back.
            held = b"" if at_start else pieces.pop(0)
            lines = (piece.decode("utf-8", errors="replace") for piece in pieces)
            kept = [line for line in lines if line and keep(line)] + kept
    return kept[-count:]


def _stream_lines(path: Path) -> Iterator[str]:
    """New lines as they are appended, forever.

    Opens and seeks to the end here rather than inside the generator: a generator
    body does not run until the first `next()`, so a lazy version would decide
    where "the end" is at some arbitrary later moment and silently skip whatever
    was written in between. That is also what makes the follow path testable - a
    test can append a line after this returns and be sure it will be seen.
    """
    handle = path.open("r", encoding="utf-8", errors="replace")
    handle.seek(0, os.SEEK_END)
    return _emit_appended(handle, path)


def _emit_appended(handle: Any, path: Path) -> Iterator[str]:
    with handle:
        while True:
            line = handle.readline()
            if line:
                yield line.rstrip("\n")
                continue
            try:
                # Truncated or replaced under us - `daemon install` re-prepares the
                # log files. Start again from the top of the new one.
                if path.stat().st_size < handle.tell():
                    handle.seek(0)
            except OSError:
                return
            time.sleep(_LOG_POLL_SECONDS)


# --- seams the tests replace -------------------------------------------------


def _install(settings: Settings, *, force: bool) -> int:
    # macOS builds Daemon.app and points the LaunchAgent at its launcher so the
    # resident has the microphone-grantable identity (spec §1); elsewhere this is the
    # plain console-script service. Shared with `daemon setup`'s residency finish
    # through daemon.macapp.build_resident_service, so the two never drift.
    from daemon.macapp import build_resident_service, grant_after_install

    try:
        service = build_resident_service(settings)
    except (RuntimeError, OSError) as exc:
        # build_bundle raises a bare RuntimeError on a codesign failure and a
        # FileNotFoundError (an OSError) when codesign is not on PATH (an Xcode-less
        # machine). ServiceError is a RuntimeError subclass, so `except ServiceError`
        # in main() would NOT catch either - it would escape as a raw traceback,
        # which an operator command must not do (module docstring).
        print(f"daemon: could not build the app bundle: {exc}", file=sys.stderr)
        return PROBLEM
    rc = _print_action(service.install(force=force))
    grant_after_install(service)
    return rc


def _uninstall(settings: Settings) -> int:
    rc = _print_action(service_for(settings).uninstall(), verb="removed")
    if sys.platform == "darwin":
        from daemon.macapp import APP_DIR

        if APP_DIR.exists():
            shutil.rmtree(APP_DIR, ignore_errors=True)
            print(f"removed {APP_DIR}")
            print("(the microphone grant is kept - harmless, and a reinstall skips the prompt)")
    return rc


def _setup(*, check_only: bool) -> int:
    from daemon.setup import run

    return run(check_only=check_only)


def _request_mic() -> int:
    """Pop the macOS microphone prompt (or report the cached decision) and exit.

    This is what Daemon.app's launcher execs during `daemon install`'s one-time
    foreground grant. It only claims the grant - it does not start a daemon - so it
    never collides with the resident LaunchAgent (spec §4.4, design decision 3).
    """
    from daemon.voice.mic_access import request_microphone_access

    status = request_microphone_access(timeout=60.0)  # a human has to click Allow
    print(f"microphone: {status}")
    return 0 if status == "authorized" else 1


def _admin_url(settings: Settings) -> str:
    """The admin console's *connect* URL. `DAEMON_HOST` is a bind address, so
    `0.0.0.0`/`::` is not something a browser connects to - fall back to loopback,
    the address that actually reaches a control plane that binds every interface."""
    host = settings.host if settings.host not in ("0.0.0.0", "::", "") else "127.0.0.1"
    return f"http://{host}:{settings.port}/admin/"


def _load_env_secrets() -> None:
    """Make `.env` values visible in `os.environ`, so secret indirection survives a
    restart.

    The admin writes an MCP key into `.env` under its variable name
    (daemon/admin/settings_io.py) and the engine reads it back with
    `os.environ.get(name)` at connect time (daemon/tools/mcp.py `_secret_value`).
    But pydantic pulls `.env` only into its own `Settings` fields, never into
    `os.environ`, so `TAVILY_API_KEY` and the like were invisible after a restart -
    a url MCP server connected once (the admin passed the key inline) and then lost
    its bearer on the next boot, failing with a taskgroup error. `setdefault`, not
    overwrite: a value the shell already exported still wins, matching how Settings
    resolves precedence."""
    path = Path(ENV_FILE)
    if not path.exists():
        return
    from daemon.setup import parse_env

    try:
        loaded = parse_env(path.read_text(encoding="utf-8"))
    except OSError as exc:
        logging.getLogger(__name__).warning("could not read %s for its secrets: %s", path, exc)
        return
    for key, value in loaded.items():
        os.environ.setdefault(key, value)


def _serve(settings: Settings) -> int:
    import uvicorn

    from daemon.app import create_app

    # Printed before uvicorn takes the terminal (its own logging is off,
    # log_config=None): the admin web has no other way to announce where it is, and
    # "how do I open the console" should not need reading the source.
    print(f"admin console: {_admin_url(settings)}")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)
    return OK


def _uv_present() -> bool:
    """A seam so tests do not probe the real PATH."""
    import shutil

    return shutil.which("uv") is not None


def _latest_ref() -> str:
    """The tag of the latest published GitHub release, or `main` if there is none
    (or the lookup fails). Mirrors install.sh, so `daemon update` and the one-liner
    resolve the same thing."""
    import httpx

    try:
        response = httpx.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
            timeout=15.0,
            follow_redirects=True,
        )
    except httpx.HTTPError:
        return "main"
    if response.status_code != 200:
        return "main"
    return response.json().get("tag_name") or "main"


def _run(cmd: list[str]) -> int:
    """Run an external command, letting its output through. A seam so tests never
    shell out."""
    import subprocess

    return subprocess.run(cmd, check=False).returncode


def _update_install_command(source: str) -> list[str]:
    """The `uv tool install` argv `daemon update` runs. Kept in step with install.sh.

    On macOS it forces a uv-managed (non-`.app`) CPython. The python.org framework
    build is itself a `Python.app` bundle, and when Daemon.app's launcher execs into
    it, macOS attributes the microphone (TCC) grant to Python.app instead of
    Daemon.app - so the headless wake gate is silently denied the mic even though
    the grant was given (measured on the owner's Mac). A managed standalone python is
    a plain binary and keeps the Daemon.app identity. `--python 3.13` alone would let
    uv pick the framework build if it is present, which is exactly what broke.

    One requirement string, not `--from <url> <name>`: uv rejects extras on the
    positional when `--from` is also given, so the extra rides a PEP 508 direct
    reference - `daemon-ai[mcp] @ <url>`.
    """
    command = ["uv", "tool", "install", "--force", "--python", "3.13"]
    if sys.platform == "darwin":
        command += ["--python-preference", "only-managed"]
    command.append(f"{INSTALL_SPEC} @ {source}")
    return command


def _update() -> int:
    """Reinstall the latest release through uv - the same tarball the one-liner
    installs, so the two cannot drift. Not available to a source/pip install, which
    has no uv and updates with `git pull` instead.

    A source tarball, not a git ref: a bare machine (a fresh Mac before the Command
    Line Tools) has no git, which is the same reason install.sh stopped using git+.
    """
    if not _uv_present():
        print(
            "daemon update needs uv, which this install does not have - you are "
            "probably running from source. Update with `git pull`, or reinstall "
            "with the one-liner in the README."
        )
        return PROBLEM
    ref = os.environ.get("DAEMON_VERSION") or _latest_ref()
    source = f"https://github.com/{GITHUB_REPO}/archive/{ref}.tar.gz"
    print(f"updating to {ref} ...")
    install = _update_install_command(source)
    if _run(install) != 0:
        print("update failed - the install output above says why.")
        return PROBLEM
    print(f"updated to {ref}.")
    # uv replaced the code in place, but a running supervisor holds the old code
    # until it re-execs - so an update that stops here is one the resident never
    # sees, which is exactly the "did it even update?" confusion this avoids.
    _restart_after_update()
    return OK


def _restart_after_update() -> None:
    """Restart the resident service so it runs the code just installed.

    Best-effort and after the fact: the reinstall already succeeded, so a config
    that will not load or a machine with no service installed must not turn into a
    failure - it turns into a line telling the user what to do by hand. Only the
    installed OS service is restarted; a `daemon run` in a terminal is a foreground
    process this command cannot reach, so it is told, not touched.
    """
    try:
        settings = Settings()
    except ConfigError:
        print(
            "the code is updated; restart the service yourself to pick it up "
            "(config did not load here, so I could not do it automatically)."
        )
        return
    try:
        service = service_for(settings)
        status = service.status()
    except ServiceError as exc:
        print(f"the code is updated; I could not check the service ({exc}) - "
              "restart it yourself to pick it up.")
        return
    if not status.installed:
        print(
            "the code is updated. It is not installed as a background service, so "
            "there is nothing to restart - run `daemon run`, or `daemon install` to "
            "keep it resident."
        )
        return
    try:
        service.restart()
    except ServiceError as exc:
        # `installed` only means the unit file exists; launchd may not have it
        # loaded (a common state when the daemon is actually being run by hand with
        # `daemon run`). Name both remedies rather than echoing a bare launchctl code.
        print(f"the code is updated, but I could not restart it automatically ({exc}).")
        print(
            "  If you run it with `daemon run`, restart that. If it should be a "
            "background service, `daemon install` loads it (then update restarts it)."
        )
        return
    print("restarted the resident service on the new version.")


def _pairing(settings: Settings, args: Any) -> int:
    """See and approve pairing requests.

    This is the other half of onboarding: the bot hands a stranger a code, and the
    only place that code can be approved is a terminal on the machine that holds
    the data. Nothing about approval is reachable from the channel itself, which
    is what keeps a stranger from approving themselves.
    """
    from daemon.app import DB_FILENAME
    from daemon.channels.pairing import Pairing, PairingError
    from daemon.channels.telegram import TelegramChannel
    from daemon.memory.store import Store

    store = Store.open(settings.data_dir / DB_FILENAME)
    try:
        pairing = Pairing(store, TelegramChannel.name)
        if args.pairing_command == "list":
            requests = pairing.pending()
            if not requests:
                print("no pending requests. Message your bot and one appears here.")
                return OK
            for request in requests:
                print(
                    f"  {request.code}  from id={request.sender_id}  "
                    f"expires {request.expires_at:%H:%M}"
                )
            print(f"\napprove one with: daemon pairing approve {requests[0].code}")
            return OK

        try:
            approval = pairing.approve(args.code)
        except PairingError as exc:
            print(f"daemon: {exc}", file=sys.stderr)
            return USAGE
        who = "owner" if approval.is_owner else "guest"
        print(f"approved id={approval.sender_id} as {who}. They can talk to Daemon now.")
        return OK
    finally:
        store.close()


def _tools(settings: Settings, args: Any) -> int:
    """What Daemon may do to this machine, and what it has done.

    The `log` subcommand is the one that earns this command's existence. Tool calls
    are audited into sqlite rather than into the markdown log (schema.sql), so
    without a way to read them back the audit trail is real but unreachable, which
    is the same as not having one.
    """
    from daemon.app import DB_FILENAME
    from daemon.memory.store import Store
    from daemon.tools.builtin import builtin_tools

    if args.tools_command == "list":
        print(f"tools:      {'on' if settings.tools_enabled else 'off (DAEMON_TOOLS_ENABLED)'}")
        print(f"mode:       {settings.tools_mode}")
        print(f"roots:      {', '.join(settings.tools_roots) or '(none)'}")
        print(f"allowlist:  {', '.join(settings.tools_allowlist) or '(none)'}")
        browser = f"on ({settings.browser_app})" if settings.browser_enabled else "off"
        print(f"browser:    {browser}")
        print(f"screen:     {'on' if settings.screen_enabled else 'off'}")
        print(f"mcp:        {'on' if settings.mcp_enabled else 'off'}")
        print()
        try:
            available = builtin_tools(
                roots=settings.tools_roots,
                timeout_secs=settings.tools_timeout_secs,
                max_output=settings.tools_max_output,
            )
            if settings.browser_enabled:
                # Listed here too, or this command answers "what may it do" with
                # three of the answers missing.
                from daemon.tools.browser import browser_tools

                available += browser_tools(
                    app=settings.browser_app,
                    timeout_secs=settings.tools_timeout_secs,
                    max_output=settings.tools_max_output,
                )
            if settings.screen_enabled:
                from daemon.tools.screen import screen_tools

                available += screen_tools(
                    max_px=settings.screen_max_px,
                    timeout_secs=settings.tools_timeout_secs,
                )
        except Exception as exc:
            print(f"daemon: the built-in tools cannot be built: {exc}", file=sys.stderr)
            return PROBLEM
        for tool in available:
            gate = "asks first" if tool.risk == "guarded" else "runs freely"
            print(f"  {tool.spec.name:<14} {gate}")
        # MCP tools are not listed: finding out what a server offers means starting
        # it, and an operator command should not spawn subprocesses to print a list.
        if settings.mcp_enabled:
            print("\n  plus whatever mcp.json's servers offer, once the daemon is running")
        return OK

    store = Store.open(settings.data_dir / DB_FILENAME)
    try:
        if args.tools_command == "log":
            rows = store.recent_tool_calls(30)
            if not rows:
                print("nothing yet. No tool has been asked for on this install.")
                return OK
            for row in rows:
                ran = "ran" if row["ran"] else row["verdict"]
                mark = "" if row["ok"] in (1, None) else " FAILED"
                print(f"  {row['ts']}  {ran:<6}{mark}  {row['preview']}")
                if not row["ran"]:
                    print(f"                                    {row['reason']}")
            return OK

        if args.tools_command == "pending":
            from daemon import clock

            rows = store.pending_tool_approvals(now=clock.now())
            if not rows:
                print("no approvals are waiting.")
                return OK
            for row in rows:
                print(f"  {row['code']}  {row['preview']}  (lapses {row['expires_at']})")
            print("\nApprovals are answered in the conversation, not here: reply /approve <code>.")
            return OK

        removed = store.remove_tool_allowlist_entry(args.pattern)
        if not removed:
            standing = store.all_tool_allowlist()
            print(f"daemon: nothing standing matches {args.pattern!r}", file=sys.stderr)
            if standing:
                print("granted:", file=sys.stderr)
                for row in standing:
                    print(f"  {row['pattern']}  ({row['tool']})", file=sys.stderr)
            return USAGE
        print(f"forgotten. {args.pattern!r} will be asked about again.")
        return OK
    finally:
        store.close()


def _reindex(settings: Settings) -> int:
    from daemon.app import DB_FILENAME
    from daemon.memory.curated import rebuild as rebuild_curated
    from daemon.memory.entities import rebuild as rebuild_entities
    from daemon.memory.reindex import reindex
    from daemon.memory.store import Store
    from daemon.persona.rules import rebuild as rebuild_persona_rules

    store = Store.open(settings.data_dir / DB_FILENAME)
    try:
        inserted = reindex(settings.data_dir, store)
        # The other markdown tiers are mirrors too, and a rebuild that only
        # restored messages would silently drop every curated fact, entity note
        # and persona rule the daemon had concluded - which is the thing
        # non-negotiable 1 exists to make impossible.
        rebuild_curated(settings.data_dir, store)
        rebuild_entities(settings.data_dir, store)
        rebuild_persona_rules(settings.data_dir, store)
        return inserted
    finally:
        store.close()


async def _proactive(settings: Settings, *, speak: bool = False) -> int:
    """One tick, printed. This is how the deterministic half gets checked.

    Without `--speak` it cannot speak at all: the tick is assembled with no judge
    and no delivery, so there is no gateway, no channel and no speaker to reach.
    That is the order PLAN 6.4 asks for - a gate whose verdicts nobody has read is
    not a gate anyone should trust with a speaker - and it is also the cheapest way
    to see why the daemon has been quiet, since it costs no model call.

    What it prints either way: the reading the probes took, every candidate that was
    due, and which rule allowed or blocked each one.
    """
    from daemon.app import build_proactive_tick

    tick, closing = await build_proactive_tick(settings, speak=speak)
    try:
        result = await tick.run()
    finally:
        await closing()

    reading = result.reading
    idle = "unknown" if reading.idle_seconds is None else f"{reading.idle_seconds:.0f}s"
    tristate = {True: "busy", False: "free", None: "unknown"}
    mic = tristate[reading.mic_busy]
    output = tristate[reading.output_busy]
    print(f"presence: idle {idle} · app {reading.foreground_app or 'unknown'}")
    print(f"          mic {mic} · output {output}")
    for reason in reading.unknown:
        print(f"  ! {reason}")

    if result.disabled:
        print("\nproactivity is off (DAEMON_PROACTIVE_ENABLED=true to turn it on).")
        print("Nothing was generated, so switching it on later starts from today.")
        return OK

    print(f"\ngenerated {result.generated} new candidate(s), expired {result.expired}")
    if not result.considered:
        print("nothing is due. That is the default and usually the right answer.")
        return OK

    for item in result.considered:
        mark = "SPEAK" if item.verdict.allowed else "  -  "
        where = f" -> {item.verdict.delivery}" if item.verdict.allowed else ""
        print(f"\n[{mark}] {item.candidate.kind}{where}")
        print(f"        why : {item.verdict.why}")
        print(f"        said: {item.candidate.reason}")
    blocked = result.blocked_by
    if blocked:
        print("\nblocked by: " + " · ".join(f"{rule} x{n}" for rule, n in sorted(blocked.items())))
    if result.declined:
        print(f"\n{result.declined} allowed, and there was nothing worth saying.")
        print("That is the default answer and usually the right one.")
    if result.spoke:
        said = next(item for item in result.considered if item.delivered)
        print(f"\nspoke via {said.delivered.route}: {said.utterance.text}")
        print(f"  label it with the buttons on the message (id {said.delivered.utterance_id})")
    elif result.allowed and not speak:
        print(f"\n{len(result.allowed)} would have been spoken. Add --speak to let it.")
    return OK


async def _reflect(settings: Settings, *, date: str | None, force: bool) -> int:
    """Run the reflection pass now. This is also how the pass is verified at all:
    the scheduler runs it at 04:00 and nobody is awake to read the log."""
    from daemon.app import build_reflection

    reflection, closing = await build_reflection(settings)
    try:
        results = (
            [await reflection.run(date, force=force)]
            if date
            else await reflection.catch_up()
        )
    finally:
        await closing()

    if not results:
        print("nothing to reflect on: no day has a log without a reflection already.")
        return OK
    for result in results:
        print(
            f"{result.date}: {result.status}"
            + (f" - {result.detail}" if result.detail else "")
            + (
                f" ({result.messages_read} message(s) -> {result.facts} fact(s), "
                f"{result.entities} entity(ies), {result.observations} observation(s))"
                if result.status == "written"
                else ""
            )
        )
        for problem in result.problems:
            print(f"  ! {problem}")
    return OK if all(result.ok for result in results) else PROBLEM


async def _persona(settings: Settings, args: Any) -> int:
    """See and manage learned persona rules (M4).

    Plain `daemon persona` costs no model call and no provider: it just reads
    the mirror, the same way `daemon persona` is meant to be safe to run on an
    install with no hosted key configured yet. `evolve` is the only subcommand
    that reaches a model - it is also the only way anyone can verify the weekly
    pass without waiting for Monday 05:00, the same reason `daemon reflect` and
    `daemon proactive` exist.
    """
    from daemon.app import DB_FILENAME, build_persona_evolution
    from daemon.memory.store import Store
    from daemon.persona.evolve import DIARY_SUBDIR
    from daemon.persona.rules import LearnedFileDiverged, LearnedRules

    sub = getattr(args, "persona_command", None)

    if sub == "evolve":
        evolution, closing = await build_persona_evolution(settings)
        try:
            result = await evolution.run(force=args.force)
        finally:
            await closing()
        print(
            f"{result.date}: {result.skipped or 'ran'} "
            f"({result.observations_read} observation(s) read -> {result.proposed} proposed, "
            f"{result.added} added, {result.retired} retired)"
        )
        for problem in result.problems:
            print(f"  ! {problem}")
        return OK

    store = Store.open(settings.data_dir / DB_FILENAME)
    try:
        if sub == "forget":
            try:
                retired = await LearnedRules(settings.data_dir, store).retire(
                    args.id, why=args.why
                )
            except LearnedFileDiverged as diverged:
                # Refused rather than done: the rewrite would have taken the
                # orphaned bullets with it, so a request to forget one rule would
                # have cost the user rules they never named.
                print(f"daemon: {diverged}", file=sys.stderr)
                return USAGE
            if not retired:
                print(f"daemon: no active rule with id {args.id}", file=sys.stderr)
                return USAGE
            print(f"retired rule {args.id}: {args.why}")
            return OK

        # No subcommand: what is active right now, for a person checking whether
        # the weekly pass has done anything.
        rows = store.active_persona_rules()
        if not rows:
            print("no active persona rules yet.")
        else:
            for row in rows:
                evidence = json.loads(row["evidence"])
                print(f"[{row['id']}] {row['body']}")
                print(f"     created {row['created_at']} - {len(evidence)} observation(s)")
        last = store.last_persona_rule_created_at()
        print(f"\nlast rule created: {last or 'never'}")

        diary_dir = settings.data_dir / DIARY_SUBDIR
        diaries = sorted(diary_dir.glob("*.md")) if diary_dir.exists() else []
        print(f"last diary: {diaries[-1].stem if diaries else 'none yet'}")
        return OK
    finally:
        store.close()


async def probe_ollama(settings: Settings) -> tuple[bool, str]:
    """Is Ollama reachable? Every preset routes embeddings there, so recall is
    dead without it even under a fully hosted setup."""
    from daemon.llm.providers.ollama import OllamaProvider

    provider = OllamaProvider(settings.ollama_base_url)
    try:
        reachable = await provider.health()
    finally:
        await provider.aclose()
    return reachable, settings.ollama_base_url


# --- doctor ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


def _doctor() -> int:
    try:
        settings = Settings()
    except ConfigError as exc:
        checks = [Check("config", False, str(exc))]
        admin = None
    else:
        table = ", ".join(
            f"{task.value}->{route.provider}" for task, route in settings.routing_table().items()
        )
        checks = [
            Check(
                "config",
                True,
                f"preset={settings.preset} voice={settings.voice_enabled} [{table}]",
            ),
            _env_override_check(settings),
            _data_dir_check(settings),
            _schema_check(settings),
            _memory_check(settings),
            _persona_check(settings),
            _proactivity_check(settings),
            _tools_check(settings),
            _screen_check(settings),
            *_ollama_checks(settings),
        ]
        admin = _admin_url(settings)

    failed = 0
    for check in checks:
        print(f"[{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
        failed += not check.ok
    if admin:
        # Where the operator opens the web console - the one capability doctor knows
        # about that has a URL rather than a yes/no.
        print(f"\nadmin console: {admin}  (while the daemon is running)")
    if failed:
        # Flushed first, or the summary lands above the checks it summarises when
        # stdout is a pipe and stderr is not.
        sys.stdout.flush()
        print(f"\n{failed} check(s) failed.", file=sys.stderr)
    return PROBLEM if failed else OK


def _env_override_check(settings: Settings) -> Check:
    """Does the shell's environment silently outrank `.env`?

    pydantic-settings reads the process environment *before* the file, by design, and
    that is normally what you want. It stops being what you want when a credential is
    involved and nobody said so: an install spent hours on a repeating Telegram 409
    because `~/.zshrc` exported `TELEGRAM_BOT_TOKEN` for a different tool, so every
    terminal quietly pointed `daemon run` at that tool's bot - which the tool was
    already polling. `.env` was correct the whole time and never used.

    Nothing is changed here. The environment winning is a legitimate thing to want,
    and the fix depends on which of the two the owner meant. Values are compared but
    never printed: only the fact that they differ, and for a bot token the numeric id,
    which is not a secret.
    """
    from daemon.setup import parse_env

    path = Path(".env")
    if not path.exists():
        return Check("environment", True, "no .env, so nothing to be overridden")
    try:
        on_disk = parse_env(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return Check("environment", False, f"could not read {path}: {exc}")

    clashes: list[str] = []
    for key, filed in on_disk.items():
        live = os.environ.get(key)
        if live is None or live == filed:
            continue
        if key == "TELEGRAM_BOT_TOKEN":
            # The id half identifies the bot and is not a secret - it is what the
            # 409 message prints, and naming it here is the whole point.
            clashes.append(
                f"{key} (env names bot {live.split(':')[0]}, .env names {filed.split(':')[0]})"
            )
        else:
            clashes.append(f"{key} (env value differs)")
    if not clashes:
        return Check("environment", True, "nothing in the environment overrides .env")
    return Check(
        "environment",
        False,
        f"the shell environment overrides .env for {', '.join(clashes)}"
        " - the environment wins, so .env is being ignored for these."
        " `unset` them, or remove them from .env, whichever you meant",
    )


def _tools_check(settings: Settings) -> Check:
    """What Daemon may do to this machine, said out loud.

    Here because tools are on by default: a capability nobody was asked about and
    which is reported nowhere is the silent state this project keeps being bitten by
    (CLAUDE.md, "report state, do not assume it"). `daemon setup` does not ask about
    tools, so this line and the startup log are the only places the answer appears.
    """
    if not settings.tools_enabled:
        return Check("tools", True, "off (DAEMON_TOOLS_ENABLED=true to turn it on)")

    extras = []
    if settings.browser_enabled:
        extras.append(f"browser={settings.browser_app}")
    if settings.mcp_enabled:
        extras.append("mcp=on")
    detail = f"on mode={settings.tools_mode} roots={', '.join(settings.tools_roots)}"
    if settings.tools_mode != "full":
        # The allowlist is what runs without a prompt under `ask`/`allowlist`. Under
        # `full` *everything* does, so naming the (usually empty) allowlist there
        # reads as the exact opposite of the truth.
        allowed = ", ".join(settings.tools_allowlist) or "nothing"
        detail += f" runs-without-asking={allowed}"
    if extras:
        detail += " " + " ".join(extras)
    # `full` is the default (config.py) and the one setting where nothing but the
    # origin gate is left, so it passes - marking the default as a failed check
    # would make every fresh install's doctor read as broken - but the detail names
    # the posture plainly rather than letting it be skimmed past. Use `ask` or
    # `allowlist` for a prompt before anything changes.
    if settings.tools_mode == "full":
        return Check(
            "tools", True, detail + " - runs everything without asking; only the origin gate holds"
        )
    return Check("tools", True, detail)


def _screen_check(settings: Settings) -> Check:
    """Whether Daemon may capture the screen and, on macOS, whether the OS has
    actually granted it.

    Screen capture is TCC-gated the same way microphone/camera access is, and a
    denial does not surface as an ordinary error - `screencapture` either exits
    non-zero or silently writes a zero-byte file (see `daemon/tools/screen.py`'s
    TCC_HINT). Since this is the one place that finds out *before* the owner asks
    for a screenshot and gets a confusing failure, it is worth a real probe - a
    throwaway capture to a temp file, discarded either way. It must never be the
    reason `doctor` itself fails to run, so any problem running the probe is
    reported as a failed check rather than raised.
    """
    if not settings.screen_enabled:
        return Check("screen", True, "off (DAEMON_SCREEN_ENABLED=true to turn it on)")
    if platform.system() != "Darwin":
        return Check("screen", True, "on (screen capture is macOS-only)")

    from daemon.tools.screen import SCREENCAPTURE_ARGS

    screencapture = shutil.which("screencapture")
    if screencapture is None:
        return Check("screen", False, "on, but screencapture is not on PATH")

    try:
        with tempfile.TemporaryDirectory() as scratch:
            path = str(Path(scratch) / "probe.jpg")
            argv = SCREENCAPTURE_ARGS(path, None)
            argv[0] = screencapture
            subprocess.run(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            probe = Path(path)
            size = probe.stat().st_size if probe.exists() else 0
    except Exception as exc:  # best-effort: a probe failure is not a doctor crash
        return Check("screen", False, f"on, but the Screen Recording probe failed: {exc}")

    if size == 0:
        return Check(
            "screen",
            False,
            "on, but Screen Recording not yet granted - allow it in System "
            "Settings > Privacy & Security > Screen Recording, then restart me",
        )
    return Check("screen", True, "on, Screen Recording granted")


def _proactivity_check(settings: Settings) -> Check:
    """Whether it is on, and what the labels say.

    The label counts are the M3 gate itself - *"오답률·방해도가 허용 범위. 스토커도
    죽은봇도 아니다"* is not answerable from a log line, and docs/PLAN.md 8.1 says
    the label clock cannot be compressed: precision needs dozens of real presses
    over weeks. Printing the tally here is what makes the weeks visible.
    """
    from daemon.app import DB_FILENAME
    from daemon.memory.store import Store

    if not settings.proactive_enabled:
        return Check("proactivity", True, "off (DAEMON_PROACTIVE_ENABLED=true to turn it on)")

    # A missing seed is a *blocker*, not a cosmetic gap: the judge declines every
    # candidate without one rather than speaking as a generic assistant, so
    # proactivity would look switched on and never say a word. Reported by the
    # agent that wrote the judge, which could not reach this file.
    seed = settings.data_dir / "persona" / "seed.md"
    if not seed.exists() or not seed.read_text(encoding="utf-8").strip():
        return Check(
            "proactivity",
            False,
            f"on, but {seed} is empty or missing. Every candidate will be declined "
            "rather than spoken in a generic voice - run `daemon setup` to write a "
            "persona seed.",
        )

    speaker = "speaker on" if settings.voice_enabled else "telegram only"
    quiet = settings.proactive_quiet_hours or "no quiet window"
    kinds = ", ".join(
        f"{kind} {cap}" for kind, cap in settings.proactive_kind_budgets.items()
    )
    detail = (
        f"on, {speaker} · budget {settings.proactive_daily_budget}/day "
        f"({kinds}) · quiet {quiet}"
    )

    path = settings.data_dir / DB_FILENAME
    if not path.exists():
        return Check("proactivity", True, f"{detail}\n         nothing spoken yet")
    try:
        store = Store.open(path)
    except Exception as exc:  # noqa: BLE001 - doctor reports state, it does not die
        return Check("proactivity", True, f"{detail}\n         labels unreadable ({exc})")
    try:
        counts = store.label_counts()
        total = sum(n for verdict, n in counts.items() if verdict != "responded")
        if not total:
            return Check("proactivity", True, f"{detail}\n         nothing spoken yet")
        good, bad = counts.get("good", 0), counts.get("bad", 0)
        judged = good + bad
        line = (
            f"{total} spoken · {good} good, {bad} bad, "
            f"{counts.get('unlabeled', 0)} unlabeled · {counts.get('responded', 0)} replied to"
        )
        if judged:
            line += f" · precision {good / judged:.0%}"
        else:
            # Said plainly: an unlabelled history is not a good result, it is no
            # result, and the tuning PLAN 6.2 defers to labels cannot start.
            line += " · no labels yet, so precision is unknown"
        return Check("proactivity", True, f"{detail}\n         {line}")
    finally:
        store.close()


def _memory_check(settings: Settings) -> Check:
    """What reflection has actually built.

    Reported rather than logged because an empty graph and a working one look
    identical from the outside, and the M2 gate is "an entity graph I did not fix
    by hand is worth reading" - so it has to be readable without opening sqlite.
    The backlog is in here for the same reason: a reflection loop that has run
    zero times leaves no trace anywhere else.
    """
    from daemon import clock
    from daemon.app import DB_FILENAME
    from daemon.memory import log
    from daemon.memory.entities import EntityNotes
    from daemon.memory.store import Store
    from daemon.reflection import pending_days

    # The backlog comes off the filesystem, so it is known even when the mirror is
    # not. That ordering is the fix for a real hole: with the sqlite file deleted -
    # which non-negotiable 1 calls a legitimate state - this reported "nothing
    # recorded yet" and hid a month of unreflected log behind it.
    #
    # Today is dropped, because `Reflection.catch_up` drops it: the day is still
    # being written to. Counting it made doctor report "1 day(s) not reflected on
    # yet - run `daemon reflect`" every single day, and the command it named
    # answered "nothing to reflect on: no day has a log without a reflection
    # already". Two commands disagreeing about one day is worse than either being
    # wrong, because it teaches you to stop reading both.
    today = log.local_date(clock.now())
    backlog = [day for day in pending_days(settings.data_dir) if day != today]
    behind = (
        f"{len(backlog)} day(s) not reflected on yet (oldest {backlog[0]})"
        " - run `daemon reflect`"
        if backlog
        else ""
    )

    path = settings.data_dir / DB_FILENAME
    if not path.exists():
        return Check("memory", True, behind or "nothing recorded yet")

    try:
        store = Store.open(path)
    except Exception as exc:  # noqa: BLE001 - doctor reports state, it does not die
        # A database from a newer build refuses to open, and the schema check above
        # already says so with the version numbers. Letting this one raise turned
        # the whole of `daemon doctor` into a traceback - the command whose entire
        # job is to explain what is wrong.
        return Check("memory", True, f"not readable ({exc.__class__.__name__}); see schema above")

    try:
        notes = EntityNotes(settings.data_dir, store)
        graph = notes.graph(limit=10_000)
        detail = (
            f"{store.count_entries()} curated fact(s), {len(graph)} entity(ies), "
            f"{store.count_observations()} observation(s)"
        )
        if behind:
            detail += f"; {behind}"
        for name, mentions, linked in graph[:5]:
            arrow = f" -> {', '.join(linked)}" if linked else ""
            detail += f"\n         {name} ({mentions}){arrow}"
        return Check("memory", True, detail)
    finally:
        store.close()


def _persona_check(settings: Settings) -> Check:
    """What the weekly persona-evolution pass has actually built (M4).

    Same reasoning as `_memory_check`: an empty rule set and a working weekly
    pass look identical from the outside, and "it stayed silent" has to be
    diagnosable without opening sqlite or waiting for a Monday. So this reports
    not just what exists but whether *the next run* would even attempt
    anything - each of `PersonaEvolution.run`'s three zero-model-call gates,
    checked here the same way it checks them.
    """
    from daemon.app import DB_FILENAME
    from daemon.clock import now as clock_now
    from daemon.memory.store import Store
    from daemon.persona.evolve import DIARY_SUBDIR, _week_start
    from daemon.persona.loader import learned_path, rule_bodies
    from daemon.persona.rules import diverged_bodies

    path = settings.data_dir / DB_FILENAME

    # Read before touching the mirror at all, and before the "no database"
    # shortcut below - `rm daemon.sqlite3` (non-negotiable 1 calls that
    # legitimate) means `path.exists()` is False with the file still full of
    # rules the (absent) mirror cannot confirm. Reporting "nothing recorded
    # yet" in that state would hide exactly what `daemon reindex` exists to
    # repair - the same hole `_memory_check` already had to close for its own
    # backlog.
    learned_file = learned_path(settings.data_dir)
    learned_text = learned_file.read_text(encoding="utf-8") if learned_file.exists() else ""
    file_bodies = rule_bodies(learned_text)

    if not path.exists():
        if not file_bodies:
            return Check("persona", True, "nothing recorded yet")
        return Check(
            "persona",
            False,
            f"learned.md has {len(file_bodies)} rule(s) but there is no database yet "
            "- run `daemon reindex` to rebuild the mirror before the next evolve",
        )

    try:
        store = Store.open(path)
    except Exception as exc:  # noqa: BLE001 - doctor reports state, it does not die
        return Check("persona", True, f"not readable ({exc.__class__.__name__}); see schema above")

    try:
        active_rows = store.active_persona_rules()
        active = len(active_rows)
        unconsumed = len(store.unconsumed_observations(limit=10_000))
        last = store.last_persona_rule_created_at()

        diverged = diverged_bodies(file_bodies, (row["body"] for row in active_rows))

        week = _week_start(clock_now())
        diary = settings.data_dir / DIARY_SUBDIR / f"{week}.md"
        if diary.exists():
            next_run = "already ran this week"
        elif unconsumed < settings.persona_min_observations:
            next_run = (
                f"not enough observations ({unconsumed}<{settings.persona_min_observations})"
            )
        elif active >= settings.persona_max_active_rules:
            next_run = f"rule budget full ({active}/{settings.persona_max_active_rules})"
        else:
            next_run = ""

        detail = (
            f"{active} active rule(s), {unconsumed} unconsumed observation(s), "
            f"last evolved {last or 'never'}"
        )
        if next_run:
            detail += f"; next run would skip: {next_run}"

        if diverged:
            detail += (
                f"; learned.md has {len(diverged)} rule(s) the mirror does not know "
                "about - run `daemon reindex` to repair"
            )
            return Check("persona", False, detail)

        return Check("persona", True, detail)
    finally:
        store.close()


def _data_dir_check(settings: Settings) -> Check:
    path = settings.data_dir
    if not path.exists():
        return Check(
            "data dir",
            False,
            f"{path} does not exist; it holds the markdown source of truth "
            "(it is created on first run, so this only matters if you expected history)",
        )
    mode = path.stat().st_mode & 0o777
    if mode != DIR_MODE:
        return Check(
            "data dir",
            False,
            f"{path} is mode {mode:04o}, expected {DIR_MODE:04o} - verbatim "
            "conversations are readable by other local accounts",
        )
    return Check("data dir", True, f"{path} exists, mode {mode:04o}")


def _schema_check(settings: Settings) -> Check:
    from daemon.app import DB_FILENAME
    from daemon.memory.store import SCHEMA_VERSION, Store

    path = settings.data_dir / DB_FILENAME
    if not path.exists():
        return Check(
            "schema", True, f"no database yet at {path}; it is built on first run from the markdown"
        )
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return Check("schema", False, f"cannot open {path}: {exc}")
    try:
        found = Store(conn).schema_version()
    except sqlite3.DatabaseError as exc:
        return Check("schema", False, f"{path} is not a readable database: {exc}")
    finally:
        conn.close()

    if found is None:
        return Check("schema", False, f"{path} has no schema version recorded")
    if found > SCHEMA_VERSION:
        return Check(
            "schema",
            False,
            f"{path} is v{found}, this build understands v{SCHEMA_VERSION}; it will "
            "refuse to open. Downgrade Daemon or move the file aside - the markdown "
            "under memory/ can rebuild it",
        )
    if found < SCHEMA_VERSION:
        return Check("schema", True, f"v{found}, migrates to v{SCHEMA_VERSION} on next start")
    return Check("schema", True, f"v{found}")


def _ollama_checks(settings: Settings) -> list[Check]:
    providers = {route.provider for route in settings.routing_table().values()}
    if OLLAMA not in providers:
        return []
    reachable, url = asyncio.run(probe_ollama(settings))
    if not reachable:
        return [
            Check(
                "ollama",
                False,
                f"not reachable at {url}; local routing and recall embeddings "
                f"({settings.embed_model}) cannot work",
            )
        ]
    return [Check("ollama", True, f"reachable at {url}, embed model {settings.embed_model}")]


# --- printing ----------------------------------------------------------------


def _print_action(action: ServiceAction, *, verb: str = "installed") -> int:
    if action.changes:
        print(f"changes to {action.unit_path}:")
        for line in action.changes:
            print(f"  {line}")
    for note in action.notes:
        print(note)
    for command in action.commands:
        print(f"ran: {' '.join(command)}")
    if action.applied:
        # `applied` only says something happened. Printing "installed" for every
        # outcome meant a successful uninstall reported "ai.daemon.default
        # installed at ..." - the action was right and the sentence was wrong,
        # which is worse than a crash because nothing looks broken.
        print(f"{action.label} {verb}: {action.unit_path}")
        return OK
    # Not applied is not always an error, but "nothing happened" should never
    # look like success to a script.
    return OK if not action.changes else PROBLEM


def _print_status(status: ServiceStatus) -> int:
    print(f"label:     {status.label}")
    print(f"unit file: {status.unit_path} ({'present' if status.installed else 'missing'})")
    print(f"loaded:    {status.loaded}")
    print(f"running:   {status.running}")
    if status.detail:
        print(f"detail:    {status.detail}")
    for note in status.notes:
        print(note)
    return OK if status.installed and status.running else PROBLEM


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
