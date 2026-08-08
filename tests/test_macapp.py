"""The thin native-launcher .app (spec §3, §4.1). No real codesign/open/launchctl:
the runner is injected, mirroring tests/test_service.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from daemon import macapp
from daemon.config import Settings
from daemon.macapp import (
    APP_DIR,
    BUNDLE_ID,
    build_bundle,
    build_resident_service,
    grant_after_install,
)
from daemon.service import RunResult, Service, default_program

LAUNCHER = Path(__file__).resolve().parent.parent / "daemon" / "macapp" / "launcher"


@pytest.mark.skipif(sys.platform != "darwin", reason="lipo is macOS-only")
def test_committed_launcher_is_universal2_macho() -> None:
    assert LAUNCHER.exists(), "the prebuilt launcher must be committed (spec §7)"
    archs = subprocess.run(
        ["lipo", "-archs", str(LAUNCHER)], capture_output=True, text=True, check=True
    ).stdout.split()
    assert set(archs) == {"arm64", "x86_64"}, f"launcher must be universal2, got {archs}"


class FakeRunner:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command) -> RunResult:
        self.commands.append(tuple(command))
        return RunResult(0)


def test_build_bundle_lays_out_the_app(tmp_path) -> None:
    app = tmp_path / "Daemon.app"
    launcher = build_bundle(app, runner=FakeRunner())

    assert launcher == app / "Contents" / "MacOS" / "launcher"
    assert launcher.exists()
    info = (app / "Contents" / "Info.plist").read_text(encoding="utf-8")
    assert f"<string>{BUNDLE_ID}</string>" in info
    assert "<key>CFBundleExecutable</key>" in info and "<string>launcher</string>" in info
    assert "<key>LSUIElement</key>" in info and "<true/>" in info
    assert "NSMicrophoneUsageDescription" in info
    assert "NSSpeechRecognitionUsageDescription" in info
    # No hardened runtime opt-in anywhere (it can suppress the prompt — spec §6).
    assert "runtime" not in info.lower()


def test_build_bundle_signs_ad_hoc_without_hardened_runtime(tmp_path) -> None:
    app = tmp_path / "Daemon.app"
    runner = FakeRunner()
    build_bundle(app, runner=runner)

    assert len(runner.commands) == 1
    cmd = runner.commands[0]
    assert cmd[0] == "codesign"
    assert "--force" in cmd and "--sign" in cmd
    # Ad-hoc identity is the literal "-" right after --sign.
    assert cmd[cmd.index("--sign") + 1] == "-"
    assert "--options" not in cmd, "hardened runtime must not be enabled (spec §6)"
    assert cmd[-1] == str(app)


def test_build_bundle_is_idempotent(tmp_path) -> None:
    app = tmp_path / "Daemon.app"
    build_bundle(app, runner=FakeRunner())
    # A re-run (reinstall / daemon update) must not raise on existing files.
    launcher = build_bundle(app, runner=FakeRunner())
    assert launcher.exists()


def test_build_bundle_raises_when_codesign_fails(tmp_path) -> None:
    class Failing(FakeRunner):
        def __call__(self, command):
            super().__call__(command)
            return RunResult(1, stderr="errSecInternalComponent")

    with pytest.raises(RuntimeError, match="codesign"):
        build_bundle(tmp_path / "Daemon.app", runner=Failing())


# --- the shared residency builder + grant gate (daemon install AND setup use it) --


def test_build_resident_service_is_the_plain_service_off_darwin(monkeypatch) -> None:
    """Off macOS, residency is the plain console-script service - no .app, and no
    launcher in its ProgramArguments."""
    monkeypatch.setattr(macapp.sys, "platform", "linux")
    service = build_resident_service(Settings(_env_file=None, preset="offline"))
    assert service.program == default_program()


def test_build_resident_service_points_the_launchagent_at_the_launcher_on_darwin(
    monkeypatch, tmp_path
) -> None:
    """On macOS the LaunchAgent must start Daemon.app's launcher (argv[0]), which
    then execs the real daemon argv. build_bundle is stubbed so no codesign runs."""
    monkeypatch.setattr(macapp.sys, "platform", "darwin")
    fake_launcher = tmp_path / "Daemon.app" / "Contents" / "MacOS" / "launcher"
    monkeypatch.setattr(macapp, "build_bundle", lambda app_dir: fake_launcher)

    service = build_resident_service(Settings(_env_file=None, preset="offline"))

    assert service.program == (str(fake_launcher), *default_program())


def test_grant_after_install_pops_the_grant_for_a_real_launcher_service(
    monkeypatch, tmp_path
) -> None:
    """The one place the mic prompt fires: a resident whose ProgramArguments point at
    Daemon.app's launcher. grant_microphone_once is stubbed so no real `open` runs."""
    monkeypatch.setattr(macapp.sys, "platform", "darwin")
    launcher = str(APP_DIR / "Contents" / "MacOS" / "launcher")
    calls: list[tuple[Path, tuple[str, ...]]] = []
    monkeypatch.setattr(
        macapp, "grant_microphone_once", lambda lp, argv: calls.append((lp, argv))
    )
    service = Service(
        label="default",
        working_dir=tmp_path,
        log_dir=tmp_path,
        program=(launcher, "/x/daemon", "run"),
        home=tmp_path,
        platform="darwin",
        runner=lambda command: RunResult(0),
    )

    grant_after_install(service)

    assert calls == [(Path(launcher), ("/x/daemon", "run"))]


def test_grant_after_install_is_a_noop_for_a_plain_or_fake_service(monkeypatch) -> None:
    """A plain (Linux) service or a FakeService (setup's residency tests inject one
    with no `program`) has no Daemon.app launcher, so the grant never runs. That gate
    is what keeps setup's tests off AVFoundation on a developer's Mac."""
    monkeypatch.setattr(macapp.sys, "platform", "darwin")
    calls: list[int] = []
    monkeypatch.setattr(macapp, "grant_microphone_once", lambda lp, argv: calls.append(1))

    grant_after_install(SimpleNamespace())  # no `program` at all, like FakeService
    grant_after_install(SimpleNamespace(program=("/usr/local/bin/daemon", "run")))

    assert calls == []


def test_grant_after_install_is_a_noop_off_darwin(monkeypatch) -> None:
    """Even a launcher-shaped program does not prompt off macOS - there is no TCC."""
    monkeypatch.setattr(macapp.sys, "platform", "linux")
    calls: list[int] = []
    monkeypatch.setattr(macapp, "grant_microphone_once", lambda lp, argv: calls.append(1))
    launcher = str(APP_DIR / "Contents" / "MacOS" / "launcher")

    grant_after_install(SimpleNamespace(program=(launcher, "/x/daemon", "run")))

    assert calls == []
