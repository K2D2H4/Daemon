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

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


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
