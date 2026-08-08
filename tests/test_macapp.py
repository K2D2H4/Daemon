"""The thin native-launcher .app (spec §3, §4.1). No real codesign/open/launchctl:
the runner is injected, mirroring tests/test_service.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS bundle tooling")

LAUNCHER = Path(__file__).resolve().parent.parent / "daemon" / "macapp" / "launcher"


def test_committed_launcher_is_universal2_macho() -> None:
    assert LAUNCHER.exists(), "the prebuilt launcher must be committed (spec §7)"
    archs = subprocess.run(
        ["lipo", "-archs", str(LAUNCHER)], capture_output=True, text=True, check=True
    ).stdout.split()
    assert set(archs) == {"arm64", "x86_64"}, f"launcher must be universal2, got {archs}"
