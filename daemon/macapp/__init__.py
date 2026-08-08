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
from pathlib import Path

from daemon.fs import DIR_MODE
from daemon.service import Runner, subprocess_runner

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
