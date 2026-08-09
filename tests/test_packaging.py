"""macOS must never ship without the voice stack again (spec §4.6).

The recognizer lived behind the `voice` extra, and both the installer and
`daemon update` install bare `daemon-ai` — so an update silently stripped voice
and the wake gate went deaf while looking healthy. Guard: on macOS the voice
deps are core dependencies (sys_platform == 'darwin'), not optional.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

VOICE_PACKAGES = (
    "websockets",
    "sounddevice",
    "certifi",
    "onnxruntime",
    "pyobjc-framework-Speech",
    "pyobjc-framework-AVFoundation",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
INSTALL_SH = REPO_ROOT / "install.sh"


def _core_deps() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["dependencies"]


def test_voice_stack_is_a_default_macos_dependency() -> None:
    core = _core_deps()
    for pkg in VOICE_PACKAGES:
        matches = [d for d in core if d.split(">=")[0].split(";")[0].strip() == pkg]
        assert matches, f"{pkg} is not a core dependency; a macOS update would strip voice"
        assert all("sys_platform == 'darwin'" in d for d in matches), (
            f"{pkg} must be gated to darwin so Linux core installs stay lean: {matches}"
        )


def test_voice_extra_still_exists_for_linux() -> None:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    assert "voice" in data["project"]["optional-dependencies"]


def test_installer_forces_a_managed_python_on_macos() -> None:
    """The python.org framework build is a `Python.app` bundle; the .app launcher
    exec'ing into it makes macOS attribute the mic grant to Python.app, not
    Daemon.app, so the headless wake gate is denied the mic (measured live). install.sh
    must force a uv-managed (non-.app) python on macOS - and `daemon update` matches it
    (tested in test_cli.py::test_update_install_command_forces_managed_python_on_macos)."""
    lines = INSTALL_SH.read_text(encoding="utf-8").splitlines()
    guard = next(
        (i for i, ln in enumerate(lines) if "uname -s" in ln and "Darwin" in ln), None
    )
    assert guard is not None, "the managed-python forcing must be gated on macOS (uname Darwin)"
    end = next((j for j in range(guard + 1, len(lines)) if lines[j].strip() == "fi"), None)
    assert end is not None, "the Darwin guard must be a closed if/fi block"
    # Inside the guard, not global: forcing only-managed on Linux would download a
    # managed python needlessly and is not the fix.
    assert "only-managed" in "\n".join(lines[guard : end + 1]), (
        "--python-preference only-managed must live inside the `uname = Darwin` guard"
    )
