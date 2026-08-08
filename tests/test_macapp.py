"""The thin native-launcher .app (spec §3, §4.1). No real codesign/open/launchctl:
the runner is injected, mirroring tests/test_service.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from daemon.macapp import BUNDLE_ID, build_bundle
from daemon.service import RunResult

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS bundle tooling")

LAUNCHER = Path(__file__).resolve().parent.parent / "daemon" / "macapp" / "launcher"


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
