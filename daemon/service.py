"""OS-supervised residency: install Daemon as a per-user service.

docs/PLAN.md 3.1 calls this the precondition for proactivity, and that is the
whole point of the module. A process that only exists while a terminal is open
cannot decide to speak first at 9pm - the 5-minute tick of docs/PLAN.md 6.1 has
to survive a closed terminal and a reboot before anything in M3 means anything.

Two platforms, one shape:

  * macOS  - a LaunchAgent plist in `~/Library/LaunchAgents`, loaded with
    `launchctl bootstrap gui/<uid>`.
  * Linux  - a systemd **user** unit in `~/.config/systemd/user`, enabled with
    `systemctl --user`. A user unit, not a system one: this process reads the
    user's private data dir and needs no root.

**No secrets in the unit file.** It names a working directory and nothing else;
the process reads `.env` from there. The reasons are concrete rather than
decorative: `launchctl print` and `systemctl --user cat` echo the file back,
`~/Library` ends up in backups and sync clients, and a second copy of an API key
is a second copy to rotate when it leaks.

Every `launchctl`/`systemctl` invocation goes through an injected runner, so the
tests exercise this file without spawning a process or touching a real
`~/Library`.
"""

from __future__ import annotations

import difflib
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

from daemon.config import SERVICE_LABEL_RE, Settings
from daemon.fs import DIR_MODE, FILE_MODE, secure_dir, secure_file, write_private_replace

DARWIN = "darwin"
LINUX = "linux"
SUPPORTED = (DARWIN, LINUX)

LABEL_PREFIX = "ai.daemon."
THROTTLE_SECONDS = 30
"""Minimum seconds between restarts. launchd's default of 10 turns a bad API key
into a crash loop that spins all day; 30 still recovers from a real crash fast
enough that a proactive tick is not missed."""

BOOTOUT_SETTLE_ATTEMPTS = 25
BOOTOUT_SETTLE_INTERVAL = 0.2
"""How long `install --force` waits for a bootout to finish before it bootstraps.
`launchctl bootout` is asynchronous - the old process gets SIGTERM and lingers in
the domain for a moment. Bootstrapping into that window fails with EIO and loads
nothing, while `_is_loaded` sees the dying job and reports success (measured twice
on the owner's Mac: install printed "installed" and left the service unloaded). Up
to 5s of polling closes the race - see `_await_unloaded`."""

LINGER_NOTE = (
    "systemd user services stop at logout unless lingering is on. "
    "Run `loginctl enable-linger $USER` or this will not survive a reboot."
)


class ServiceError(RuntimeError):
    """The service cannot be managed here. Raised before anything is written."""


@dataclass(frozen=True, slots=True)
class RunResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[Sequence[str]], RunResult]


def subprocess_runner(command: Sequence[str]) -> RunResult:
    """The real runner. Bounded: `launchctl` can block on a wedged domain."""
    # Fixed argv, never a shell string: nothing here is interpolated by a shell.
    completed = subprocess.run(
        list(command), capture_output=True, text=True, timeout=30, check=False
    )
    return RunResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True, slots=True)
class ServiceAction:
    """Outcome of install/uninstall. `changes` is a unified diff of the unit file
    so an existing install is never silently rewritten."""

    label: str
    unit_path: Path
    applied: bool
    changes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    commands: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    label: str
    unit_path: Path
    installed: bool
    """The unit file is on disk."""
    loaded: bool
    """The init system knows about the job."""
    running: bool
    detail: str = ""
    notes: tuple[str, ...] = field(default=())


def default_program() -> tuple[str, ...]:
    """Argv the service supervises.

    Prefers the console script next to the interpreter (`.venv/bin/daemon`)
    because it does not depend on the working directory resolving an import.
    Falls back to the module for a checkout that was never installed.
    """
    console = Path(sys.executable).with_name("daemon")
    if console.exists():
        return (str(console), "run")
    return (sys.executable, "-m", "daemon.cli", "run")


class Service:
    """One installable service definition. Cheap to construct; nothing touches
    the filesystem until `install`/`uninstall`/`status` is called."""

    def __init__(
        self,
        *,
        label: str = "default",
        working_dir: Path,
        log_dir: Path,
        program: Sequence[str] | None = None,
        home: Path | None = None,
        platform: str = sys.platform,
        uid: int | None = None,
        runner: Runner = subprocess_runner,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not SERVICE_LABEL_RE.match(label):
            raise ServiceError(
                f"{label!r} is not a usable service label; expected letters, digits, "
                "dot, dash or underscore (it becomes a filename)"
            )
        self.label = f"{LABEL_PREFIX}{label}"
        self.working_dir = Path(working_dir).expanduser().resolve()
        # Absolute, always. DAEMON_DATA_DIR is relative by default, and neither
        # init system will accept that: systemd rejects `append:` with a relative
        # path outright, and launchd resolves it against a directory that is not
        # this one, so the logs would land somewhere nobody looks. Resolving
        # against the install-time cwd is right because that is `working_dir`.
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.program = tuple(program) if program is not None else default_program()
        self._home = Path(home) if home is not None else Path.home()
        self._platform = platform
        self._uid = os.getuid() if uid is None else uid
        self._run = runner
        self._sleep = sleep

    # --- paths --------------------------------------------------------------

    @property
    def unit_path(self) -> Path:
        if self._platform == DARWIN:
            return self._home / "Library" / "LaunchAgents" / f"{self.label}.plist"
        if self._platform == LINUX:
            return self._home / ".config" / "systemd" / "user" / f"{self.label}.service"
        raise self._unsupported()

    @property
    def out_log(self) -> Path:
        return self.log_dir / f"{self.label}.out.log"

    @property
    def err_log(self) -> Path:
        return self.log_dir / f"{self.label}.err.log"

    # --- rendering ----------------------------------------------------------

    def _require_renderable_paths(self) -> None:
        """Refuse paths a unit file cannot hold safely.

        The plist writer escapes its values; the systemd writer interpolates them
        into a line-oriented format where a newline starts a new directive, a
        space splits ExecStart's argv, and `%` is a specifier. A checkout inside a
        directory whose name contains a newline therefore turns `daemon install`
        into arbitrary code that runs at every login, and the far more ordinary
        case - a space in the path - silently produces an ExecStart pointing at
        the wrong binary. Only the label was validated; paths were not.

        Refusing is the whole fix. Quoting would also work for the space case but
        would change the rendered output of every existing install, which the
        diff in `install()` would then report as a pending change.
        """
        for value in (str(self.working_dir), str(self.log_dir), *self.program):
            bad = next((c for c in value if c in "\n\r\0" or ord(c) < 32), None)
            if bad is not None:
                raise ServiceError(
                    f"refusing to write a unit file for a path containing "
                    f"{bad!r}: {value!r}. A control character there can inject "
                    f"directives into the unit file."
                )
            if self._platform == LINUX and (" " in value or "%" in value):
                raise ServiceError(
                    f"this path cannot be used in a systemd unit: {value!r}. "
                    "systemd splits ExecStart on spaces and expands '%' as a "
                    "specifier, so the service would start the wrong thing. "
                    "Move the checkout somewhere without spaces."
                )

    def render(self) -> str:
        """The unit file, byte for byte. Contains no secrets - see module docs."""
        if self._platform == DARWIN:
            return self._render_plist()
        if self._platform == LINUX:
            return self._render_unit()
        raise self._unsupported()

    def _render_plist(self) -> str:
        arguments = "\n".join(f"        <string>{escape(arg)}</string>" for arg in self.program)
        # No EnvironmentVariables key on purpose: WorkingDirectory is how the
        # process finds .env, and that is the only place credentials live.
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{escape(self.label)}</string>
    <key>ProgramArguments</key>
    <array>
{arguments}
    </array>
    <key>WorkingDirectory</key>
    <string>{escape(str(self.working_dir))}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>{THROTTLE_SECONDS}</integer>
    <key>StandardOutPath</key>
    <string>{escape(str(self.out_log))}</string>
    <key>StandardErrorPath</key>
    <string>{escape(str(self.err_log))}</string>
</dict>
</plist>
"""

    def _render_unit(self) -> str:
        exec_start = " ".join(self.program)
        return f"""[Unit]
Description=Daemon - self-hosted AI companion ({self.label})
After=network-online.target

[Service]
Type=simple
WorkingDirectory={self.working_dir}
ExecStart={exec_start}
Restart=always
RestartSec={THROTTLE_SECONDS}
StandardOutput=append:{self.out_log}
StandardError=append:{self.err_log}

[Install]
WantedBy=default.target
"""

    # --- install / uninstall ------------------------------------------------

    def install(self, *, force: bool = False) -> ServiceAction:
        """Write the unit file and hand it to the init system.

        An existing, differing install is **not** overwritten: the diff comes back
        instead, so the caller can show what would change. Someone who hand-edited
        their plist should find out from a diff, not from it being gone.
        """
        self._require_supported()
        self._require_renderable_paths()
        desired = self.render()
        path = self.unit_path
        existing = path.read_text(encoding="utf-8") if path.exists() else None

        if existing == desired:
            return ServiceAction(
                label=self.label,
                unit_path=path,
                applied=False,
                # Not "reloaded it": on macOS the bootstrap below is refused when
                # launchd already has the job, and what `_load` then confirms is
                # that it is loaded - not that anything was restarted.
                notes=(
                    "already installed and unchanged; the job is loaded",
                    *self._platform_notes(),
                ),
                commands=self._load(),
            )

        changes = _diff(existing or "", desired, path)
        if existing is not None and not force:
            return ServiceAction(
                label=self.label,
                unit_path=path,
                applied=False,
                changes=changes,
                notes=(
                    f"{path} already exists and differs; nothing was written. "
                    "Re-run with --force to replace it.",
                ),
            )

        commands: tuple[tuple[str, ...], ...] = ()
        if existing is not None:
            # A changed definition has to be unloaded first; launchd keeps the old
            # one in memory otherwise and the edit appears to do nothing.
            commands += self._unload()
            if self._platform == DARWIN:
                # bootout is async - wait for the job to actually leave the domain
                # before the bootstrap below, or the two race and the service is left
                # unloaded while the install reports success (see `_await_unloaded`).
                self._await_unloaded()

        path.parent.mkdir(parents=True, exist_ok=True, mode=DIR_MODE)
        self._prepare_logs()
        # Atomic and fsynced, like the markdown tiers: a plist that was caught
        # half-written is a plist launchd will not parse, and the next login has
        # nothing to fall back to because the previous one was truncated in place.
        # `secure_parent=False` - see `fs.write_private_replace`; the directory
        # this lands in belongs to every agent the user has, not to us.
        write_private_replace(path, desired, secure_parent=False)
        commands += self._load()

        return ServiceAction(
            label=self.label,
            unit_path=path,
            applied=True,
            changes=changes,
            notes=self._platform_notes(),
            commands=commands,
        )

    def uninstall(self) -> ServiceAction:
        """Stop the job and remove the unit file. Logs are left alone - they are
        the user's, and a removal is not a request to delete history."""
        self._require_supported()
        path = self.unit_path
        if not path.exists():
            return ServiceAction(
                label=self.label,
                unit_path=path,
                applied=False,
                notes=(f"{path} does not exist; nothing to remove",),
            )
        commands = self._unload()
        path.unlink()
        return ServiceAction(
            label=self.label,
            unit_path=path,
            applied=True,
            notes=(f"removed {path}", f"logs kept in {self.log_dir}"),
            commands=commands,
        )

    def restart(self) -> ServiceAction:
        """Re-launch the job so it picks up a program binary that changed underneath
        it. `daemon update` reinstalls the code in place; the supervisor holds the
        old code until it re-execs, so update calls this to make the change take.

        Not a reload: the unit file itself is unchanged, only the code behind it
        moved. `launchctl kickstart -k` kills the current instance and starts a
        fresh one from the same plist; a job that is not loaded fails loudly here,
        which is why the caller checks `status().installed` first. On Linux the
        equivalent is `systemctl --user restart`.
        """
        self._require_supported()
        if self._platform == DARWIN:
            command: tuple[str, ...] = (
                "launchctl", "kickstart", "-k", f"gui/{self._uid}/{self.label}"
            )
        else:
            command = ("systemctl", "--user", "restart", self.unit_path.name)
        self._check(command)
        return ServiceAction(
            label=self.label,
            unit_path=self.unit_path,
            applied=True,
            commands=(command,),
        )

    def status(self) -> ServiceStatus:
        self._require_supported()
        path = self.unit_path
        installed = path.exists()
        if self._platform == DARWIN:
            result = self._run(self._print_command())
            loaded = result.returncode == 0
            running = "state = running" in result.stdout
            detail = _first_line(result.stdout if loaded else result.stderr)
            return ServiceStatus(self.label, path, installed, loaded, running, detail)

        unit = path.name
        enabled = self._run(("systemctl", "--user", "is-enabled", unit))
        active = self._run(("systemctl", "--user", "is-active", unit))
        return ServiceStatus(
            label=self.label,
            unit_path=path,
            installed=installed,
            loaded=enabled.stdout.strip() == "enabled",
            running=active.stdout.strip() == "active",
            detail=_first_line(active.stdout or active.stderr),
            notes=(LINGER_NOTE,),
        )

    # --- internals ----------------------------------------------------------

    def _load(self) -> tuple[tuple[str, ...], ...]:
        if self._platform == DARWIN:
            command = ("launchctl", "bootstrap", f"gui/{self._uid}", str(self.unit_path))
            result = self._run(command)
            # launchd will not say why. Bootstrapping a job it already has exits
            # **5 with `Input/output error`** - measured, and byte for byte the same
            # answer it gives for a plist path that does not exist. So there is no
            # string to allow here; the only reliable question is whether the job is
            # in the domain, which is the same query `status` already makes.
            if result.returncode != 0 and not self._is_loaded():
                raise self._failed(command, result)
            return (command,)
        reload_ = ("systemctl", "--user", "daemon-reload")
        enable = ("systemctl", "--user", "enable", "--now", self.unit_path.name)
        self._check(reload_)
        self._check(enable)
        return (reload_, enable)

    def _unload(self) -> tuple[tuple[str, ...], ...]:
        if self._platform == DARWIN:
            command = ("launchctl", "bootout", f"gui/{self._uid}/{self.label}")
            # Not loaded is the desired end state, so it is not a failure.
            self._check(command, allow="No such process")
            return (command,)
        disable = ("systemctl", "--user", "disable", "--now", self.unit_path.name)
        self._check(disable, allow="not loaded")
        return (disable,)

    def _print_command(self) -> tuple[str, ...]:
        """Asked by both `status` and `_load`, so the domain string is written once."""
        return ("launchctl", "print", f"gui/{self._uid}/{self.label}")

    def _is_loaded(self) -> bool:
        """Does launchd have this job? Exit 0 from `print` is the whole answer; the
        alternative is 113 and `Could not find service ...`."""
        return self._run(self._print_command()).returncode == 0

    def _await_unloaded(self) -> None:
        """Block until launchd has actually removed the job, after a bootout.

        `bootout` returns before the teardown finishes: the old process is still
        getting SIGTERM. A `bootstrap` fired into that window fails with EIO (exit 5,
        byte-for-byte what "already loaded" returns), and `_load`'s fallback then
        asks `_is_loaded`, which sees the *dying* job and answers True - so the
        install reports success while nothing ends up running. Measured twice on the
        owner's Mac: `daemon install --force` printed "installed" and left
        `launchctl print` returning 113. Waiting for the label to leave the domain
        before the bootstrap is the fix.

        If the budget runs out with the job still present, this returns anyway rather
        than raise: `_load`'s own check surfaces the real failure if the bootstrap
        then does not take."""
        for _ in range(BOOTOUT_SETTLE_ATTEMPTS):
            if not self._is_loaded():
                return
            self._sleep(BOOTOUT_SETTLE_INTERVAL)

    def _check(self, command: Sequence[str], *, allow: str | None = None) -> RunResult:
        result = self._run(command)
        if result.returncode == 0:
            return result
        output = f"{result.stdout}\n{result.stderr}"
        if allow is not None and allow.lower() in output.lower():
            return result
        raise self._failed(command, result)

    def _failed(self, command: Sequence[str], result: RunResult) -> ServiceError:
        return ServiceError(
            f"`{' '.join(command)}` failed with exit {result.returncode}: "
            f"{_first_line(result.stderr or result.stdout)}"
        )

    def _prepare_logs(self) -> None:
        """Create the log files owner-only before the init system does.

        launchd and systemd create these with the ambient umask, which typically
        means world-readable. A traceback in the error log can quote a message the
        user sent, so the mode has to be ours from the start.
        """
        secure_dir(self.log_dir)
        for path in (self.out_log, self.err_log):
            if not path.exists():
                os.close(os.open(path, os.O_WRONLY | os.O_CREAT, FILE_MODE))
            secure_file(path)

    def _platform_notes(self) -> tuple[str, ...]:
        return (LINGER_NOTE,) if self._platform == LINUX else ()

    def _require_supported(self) -> None:
        if self._platform not in SUPPORTED:
            raise self._unsupported()

    def _unsupported(self) -> ServiceError:
        return ServiceError(
            f"installing a service is not supported on {self._platform!r}. "
            "Daemon supervises itself through launchd (macOS) or a systemd user "
            "unit (Linux); Windows has neither, and a Scheduled Task cannot keep a "
            "process alive the way residency needs (docs/PLAN.md 3.1). Run "
            "`daemon run` yourself, or use WSL2 where the systemd path applies."
        )


def service_for(settings: Settings) -> Service:
    """Build the service definition from settings.

    The working directory is where `.env` lives - the current directory, which is
    the directory the user is standing in when they install. The unit file carries
    that path and nothing else, so the secrets stay in one file.

    Here rather than in `cli.py` so that `daemon setup`'s guided finish and the
    `daemon install` command build the resident the same way from one place.
    """
    return Service(
        label=settings.service_label,
        working_dir=Path.cwd(),
        log_dir=settings.data_dir / "logs",
    )


def _diff(before: str, after: str, path: Path) -> tuple[str, ...]:
    return tuple(
        line.rstrip("\n")
        for line in difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{path} (installed)",
            tofile=f"{path} (new)",
        )
    )


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""
