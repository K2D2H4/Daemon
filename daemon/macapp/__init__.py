"""Build the thin native-launcher `Daemon.app` (spec §3, §4.1).

The .app exists only to be the grantable microphone (TCC) identity. It has zero
per-install content — Info.plist and entitlements are static, and the daemon's
path travels as a plist ProgramArgument (see daemon/cli.py), not baked in — so the
same bundle serves every install and the grant survives `daemon update` and even a
reinstall to a different daemon path.

One `codesign` subprocess, through an injected runner (mirroring daemon/service.py)
so tests never sign anything.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from daemon.fs import DIR_MODE
from daemon.service import Runner, Service, default_program, service_for, subprocess_runner

if TYPE_CHECKING:
    from daemon.config import Settings

BUNDLE_ID = "ai.daemon.app"
"""Stable and fixed — the TCC identity. Never vary it per install or the grant is
not found (spec §4.1)."""

_LAUNCHER_SRC = Path(__file__).resolve().parent / "launcher"

# LSUIElement=true: a background agent, no Dock icon (set in the spike, did not
# block the prompt). The two usage strings are what TCC shows and requires: mic
# for the wake gate, speech for the on-device recognizer.
_INFO_PLIST = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>{BUNDLE_ID}</string>
    <key>CFBundleName</key><string>Daemon</string>
    <key>CFBundleExecutable</key><string>launcher</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSUIElement</key><true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Daemon listens for its wake word.</string>
    <key>NSSpeechRecognitionUsageDescription</key>
    <string>Daemon recognizes its wake word on-device.</string>
</dict>
</plist>
"""

# Included per the verified recipe (spec Appendix A) but only consulted under
# hardened runtime — which we deliberately do NOT enable. Harmless otherwise.
_ENTITLEMENTS = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.device.audio-input</key><true/>
</dict>
</plist>
"""


def build_bundle(app_path: Path, *, runner: Runner = subprocess_runner) -> Path:
    """Assemble Daemon.app at `app_path`, ad-hoc sign it, return the launcher path.

    Idempotent: a reinstall or `daemon update` re-runs this over an existing
    bundle. The launcher binary is byte-identical every time, so re-signing does
    not change the code identity and the grant is kept.
    """
    contents = app_path / "Contents"
    macos = contents / "MacOS"
    macos.mkdir(parents=True, exist_ok=True, mode=DIR_MODE)

    (contents / "Info.plist").write_text(_INFO_PLIST, encoding="utf-8")
    entitlements = contents / "entitlements.plist"
    entitlements.write_text(_ENTITLEMENTS, encoding="utf-8")

    launcher = macos / "launcher"
    shutil.copyfile(_LAUNCHER_SRC, launcher)
    launcher.chmod(0o755)

    # Ad-hoc (`--sign -`), no `--options runtime`. `--deep` is a no-op for a
    # single-executable bundle but matches the verified recipe (spec Appendix A).
    result = runner(
        (
            "codesign",
            "--force",
            "--deep",
            "--sign",
            "-",
            "--entitlements",
            str(entitlements),
            str(app_path),
        )
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"codesign failed with exit {result.returncode}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return launcher


# --- residency: the .app-aware install both `daemon install` and setup share -----

APP_DIR = Path.home() / "Applications" / "Daemon.app"
"""Where the TCC-identity bundle lives (spec Q2). Holds no secrets."""

_LAUNCHER_REL = ("Contents", "MacOS", "launcher")


def macos_program(launcher: Path, daemon_argv: tuple[str, ...]) -> tuple[str, ...]:
    """The plist ProgramArguments: the launcher, then the daemon argv it execs.

    Order is load-bearing - the launcher is argv[0] (what launchd starts native,
    giving the .app's TCC identity) and it execs argv[1] (the daemon path) with the
    rest (the subcommand). service.py renders this tuple verbatim.
    """
    return (str(launcher), *daemon_argv)


def build_resident_service(settings: Settings) -> Service:
    """The Service to install for residency, macOS-aware - the single builder both
    `daemon install` and `daemon setup`'s residency finish use, so the two can never
    drift.

    On macOS it builds Daemon.app and points the LaunchAgent's ProgramArguments at
    the bundle's launcher: a launchd-spawned bare Python is silently denied the
    microphone (spec §1), and the launcher execs the real `daemon run` under the
    .app's grantable TCC identity. Everywhere else it is the plain console-script
    service. It raises whatever `build_bundle` raises on a codesign failure, so
    callers wrap it rather than let a traceback escape (an operator command prints
    what it found; it does not dump a stack).
    """
    if sys.platform != "darwin":
        return service_for(settings)
    launcher = build_bundle(APP_DIR)
    return Service(
        label=settings.service_label,
        working_dir=Path.cwd(),
        log_dir=settings.data_dir / "logs",
        program=macos_program(launcher, default_program()),
    )


def grant_after_install(service: Service) -> None:
    """Pop the one-time mic prompt under the .app identity once the resident is
    installed - but only for a real macOS launcher-Service.

    A plain (Linux) or fake (test) Service has no Daemon.app launcher in its
    ProgramArguments, so nothing runs. That gate is load-bearing for the test suite:
    `daemon setup`'s residency tests inject a `FakeService` (no `program`) and never
    pin the platform, so on a developer's Mac this is what keeps them off
    AVFoundation - the same "no test touches a microphone" rule Task 5 restored.
    """
    if sys.platform != "darwin":
        return
    program = tuple(getattr(service, "program", ()) or ())
    if not program or program[0] != str(APP_DIR.joinpath(*_LAUNCHER_REL)):
        return
    grant_microphone_once(Path(program[0]), program[1:])


def grant_open_argv(app: Path, daemon_argv: tuple[str, ...]) -> list[str]:
    """The `open` argv for the one-time foreground grant.

    `daemon_argv` is `default_program()`'s result - either the 2-tuple
    `(daemon, "run")` or the 4-tuple `(python, "-m", "daemon.cli", "run")` (a
    checkout with no console script installed). Either way the trailing element is
    the `run` subcommand, which must become `request-mic`; everything before it
    (the interpreter and, in the checkout case, `-m daemon.cli`) has to be kept, or
    the launcher execs `python request-mic` and silently fails to pop the prompt.
    """
    return ["open", str(app), "--args", *daemon_argv[:-1], "request-mic"]


def grant_microphone_once(launcher: Path, daemon_argv: tuple[str, ...]) -> None:
    """Launch the .app foreground so the mic prompt appears under its TCC identity.

    Runs `daemon request-mic` (via the launcher), which claims the grant and exits -
    NOT a second `daemon run`, so it never collides with the LaunchAgent (design
    decision 3). Fire-and-forget: the grant persists once the user clicks Allow.
    """
    app = launcher.parents[2]  # …/Daemon.app
    print("\nA microphone permission dialog will appear - click Allow.")
    print("(Daemon listens for its wake word; the grant persists across reboots and updates.)")
    # Fixed argv vector, no shell (CONTRACTS 13): open the bundle and pass the
    # launcher the daemon path (+ any `-m daemon.cli` prefix) + the request-mic
    # subcommand.
    subprocess.run(grant_open_argv(app, daemon_argv), check=False)
