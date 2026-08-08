# macOS headless voice-wake via a thin native-launcher `.app` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the resident macOS daemon obtain the microphone TCC grant and wake on "벨라" headless under launchd, by wrapping `daemon run` in a minimal ad-hoc-signed `Daemon.app` whose code-signing identity *is* the grant.

**Architecture:** A prebuilt universal2 Mach-O `launcher` (committed as package data) is copied into `~/Applications/Daemon.app`; the LaunchAgent plist points its `ProgramArguments` at `[launcher, <daemon-path>, run]`, so launchd starts a native binary that `execv`'s the real `daemon run` under the app's TCC identity. The daemon claims the mic once via `AVCaptureDevice.requestAccess`; `daemon install` launches the app foreground once so the prompt appears while the user is present. Because the launcher is separate from the daemon code, `daemon update` (which replaces the uv-tool env) never changes the launcher's code identity, so the grant survives updates.

**Tech Stack:** Python 3.13, pyobjc (AVFoundation/Speech), C (clang universal2), launchd, `codesign` (ad-hoc), hatchling package data.

## Global Constraints

Every task's requirements implicitly include this section. Values are verbatim from the spec (`docs/superpowers/specs/2026-08-08-macos-voice-app-bundle-design.md`, commit `d39e04e`) and `docs/CONTRACTS.md`.

- **Python `>=3.13`**, `from __future__ import annotations`, modern generics (`list[str]`). Async on I/O paths; no blocking calls inside async functions (use `asyncio.to_thread`).
- **Code, comments, commit messages, identifiers in English.** Design docs in `docs/` stay Korean.
- **Layering (CONTRACTS 4):** only `daemon/app.py` imports concrete voice implementations. New voice helpers are imported function-locally from `app.py`, never at module scope elsewhere.
- **macOS-only changes; Linux must stay working.** Every new OS-touching branch is guarded by `sys.platform == "darwin"`. The Linux path (systemd user unit + PortAudio) is untouched.
- **Bundle identity is fixed and stable:** `CFBundleIdentifier = ai.daemon.app`. Never vary it per install or label — it is the TCC identity and the grant is found by it.
- **`.app` location:** `~/Applications/Daemon.app` (open question Q2 → `~/Applications`; it holds no secrets).
- **The launcher MUST be a native universal2 Mach-O.** Never fall back to a shell-script launcher (spike fact 2: it triggers Rosetta/x86_64 and a weak TCC identity).
- **Ad-hoc signing only** (`codesign --force --deep --sign -`). **Do NOT enable hardened runtime** (`--options runtime`) — the spike showed it can suppress the prompt, and it is unnecessary without notarization. No Apple Developer account, no notarization (out of scope §9).
- **PortAudio never prompts** (spike fact 3): the prompt only comes from an explicit `AVCaptureDevice.requestAccessForMediaType_`. **Never call `SFSpeechRecognizer.requestAuthorization_`** — it SIGABRTs a bare process (`apple_speech.py` docstring); leave that untouched.
- **No test may touch the network, a key, a microphone, a speaker, real `launchctl`, `codesign`, or `open`.** Inject the runner (mirror `daemon/service.py`) and the frameworks (mirror `daemon/voice/apple_speech.py`'s `frameworks` param). Real hardware verification lives in Task 9, run by hand.
- **`data/persona/seed.md` untouched; no schema, no frozen-file changes** are required by this plan.

### Design decisions & deviations (read before starting — these differ from a literal reading of the spec)

1. **§4.6 fix = `pyproject.toml` only (Q1: voice default-on for macOS).** The spec's alternative — "fix `daemon update` to carry `[voice]`" — targets code that is **not in `main`** (this branch). `install.sh` and the `daemon update` command live only on the separate `v0.1.x` release line, which diverged from `main` and never merged back; both install bare `daemon-ai` with no `[voice]` extra. So the only self-contained, correct fix here is to make the voice deps part of core `dependencies`, gated `; sys_platform == 'darwin'`. This also fixes the release line automatically whenever it is next cut from `main`. (Task 1.)
2. **Launcher takes the daemon path as an argv, not baked or discovered.** The plist `ProgramArguments` is `[launcher, <daemon-path>, run]`; the launcher is a 10-line `execv(argv[1], &argv[1])`. This keeps the committed binary **generic and identical for every install** (grant survives even a reinstall to a different daemon path) and needs **no change to `service.py`'s renderer** — `_render_plist` already renders `self.program` as-is. This honors the spec's stated preference in §4.2 ("keep `service.py` about the unit file only; build in the install command and pass the launcher path in") over its earlier suggestion to change `default_program()`. **`default_program()` is left unchanged.**
3. **The foreground grant uses a dedicated `daemon request-mic` subcommand, not `open Daemon.app` alone.** Launching the app plainly would `execv` the *full* `daemon run`, starting a second daemon that collides with the LaunchAgent (same data dir, same port). `daemon request-mic` only calls `requestAccess` and exits, so the grant is claimed with no collision. The install runs `open <app> --args <daemon-path> request-mic`.
4. **The grant step lives in `daemon install`, not `daemon setup`.** `daemon install` is where OS residency already lives; `daemon setup` currently ends by telling the user to run `daemon install` (it does not install the service itself). Duplicating install into setup would be two install paths. This satisfies §4.4's "and/or `daemon install`". Setup's closing guidance is updated to mention the one-time prompt.
5. **`/health` gains a `mic` field (Q3 → include now).** Read via `AVCaptureDevice.authorizationStatusForMediaType_` at request time (a status read, never a prompt).
6. **New module home:** `daemon/macapp/` (a package: `__init__.py` + `launcher.c` + prebuilt `launcher`), and `daemon/voice/mic_access.py`. Package data is picked up by hatchling with the existing `packages = ["daemon"]` (same as `daemon/voice/models/silero_vad.onnx`). No hatch config change.

---

### Task 1: Voice as a default dependency on macOS (§4.6 blocker — do first)

Without the `voice` extra there is no recognizer, so every `.app` change below is inert. Fix: the voice deps become core `dependencies` on macOS, so any install/update of `daemon-ai` on macOS carries them regardless of extras. The `[voice]` extra stays for Linux and explicit installs.

**Files:**
- Modify: `pyproject.toml` (the `[project].dependencies` array and a comment)
- Test: `tests/test_packaging.py` (create)

**Interfaces:**
- Produces: nothing importable; a packaging invariant asserted by test.

- [ ] **Step 1: Write the failing test**

Create `tests/test_packaging.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_packaging.py -v`
Expected: FAIL on `test_voice_stack_is_a_default_macos_dependency` (packages are only in the extra, not core).

- [ ] **Step 3: Add the darwin-gated voice deps to core `dependencies`**

In `pyproject.toml`, append to the `[project].dependencies` array (after `"numpy>=2.0",`), with a comment explaining why:

```toml
    # Voice is default-on for macOS (spec §4.6). The recognizer used to live only
    # behind the `voice` extra, and both the one-liner installer and `daemon
    # update` install bare `daemon-ai` — so an update silently dropped voice and
    # the wake gate went deaf while /health still said running. Making these core
    # on darwin removes the extra that kept getting stripped. Linux keeps the lean
    # core and reaches voice through the `[voice]` extra below.
    "websockets>=14.0; sys_platform == 'darwin'",
    "sounddevice>=0.5; sys_platform == 'darwin'",
    "certifi>=2024.0; sys_platform == 'darwin'",
    "onnxruntime>=1.20; sys_platform == 'darwin'",
    "pyobjc-framework-Speech>=10.0; sys_platform == 'darwin'",
    "pyobjc-framework-AVFoundation>=10.0; sys_platform == 'darwin'",
```

Leave the `[project.optional-dependencies].voice` block exactly as it is — Linux and explicit `pip install '.[voice]'` still use it, and its per-line comments (the certifi/onnxruntime rationale) remain the reference.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_packaging.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_packaging.py
git commit -m "packaging: voice is a default dependency on macOS (spec §4.6)"
```

---

### Task 2: The native launcher — C source + prebuilt universal2 binary

The launcher is the app's executable. It must be a native universal2 Mach-O (spec §6, fact 2) and generic across installs. It receives the real daemon path and subcommand as argv (from the plist), and `execv`'s it under the app's TCC identity.

**Files:**
- Create: `daemon/macapp/__init__.py` (empty for now — Task 3 fills it; makes `daemon/macapp` a package so its data files ship)
- Create: `daemon/macapp/launcher.c`
- Create (build artifact, committed): `daemon/macapp/launcher` (universal2 Mach-O)
- Test: `tests/test_macapp.py` (create; the binary-shape check)

**Interfaces:**
- Produces: a committed file at `daemon/macapp/launcher`, an arm64+x86_64 Mach-O. Task 3's `build_bundle` copies it; Task 7 points the plist at the copy.

- [ ] **Step 1: Write the launcher source**

Create `daemon/macapp/launcher.c`:

```c
/*
 * Daemon.app's launcher.
 *
 * The .app exists only to be the grantable microphone (TCC) identity: a
 * launchd-spawned bare Python cannot obtain the grant, but a process started
 * from a signed bundle can, and macOS keys the grant to the bundle's code
 * identity — which survives `daemon update` because that only replaces the
 * daemon's Python env, never this binary.
 *
 * This MUST be a native universal2 Mach-O, never a shell script: a script main
 * executable makes LaunchServices launch the app as x86_64 (a Rosetta prompt)
 * and yields a weak TCC identity. A compiled launcher fixed both (spike fact 2).
 *
 * It is deliberately generic — identical for every install. The real daemon's
 * absolute path and subcommand arrive as argv from the LaunchAgent plist
 * (ProgramArguments = [this, <daemon-path>, run]) or from `open --args`
 * (<daemon-path> request-mic). So it just execs argv[1] with argv[1:]; the
 * exec'd process inherits this bundle's TCC identity (spike facts 4, 5).
 */
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "launcher: expected the daemon path as argv[1]\n");
        return 2;
    }
    execv(argv[1], &argv[1]);
    /* Only reached if execv failed. */
    perror("launcher: execv");
    return 1;
}
```

- [ ] **Step 2: Compile the universal2 binary and verify its two architectures**

Run:

```bash
clang -arch arm64 -arch x86_64 -O2 -o daemon/macapp/launcher daemon/macapp/launcher.c
file daemon/macapp/launcher
lipo -archs daemon/macapp/launcher
```

Expected: `file` reports `Mach-O universal binary with 2 architectures: [arm64 ... x86_64 ...]`; `lipo -archs` prints `arm64 x86_64`. If `clang` cannot cross-compile x86_64 here, the plan cannot proceed on this host — stop and report (the committed binary is required by §7 so no compiler is needed at install).

- [ ] **Step 3: Create the empty package init**

Create `daemon/macapp/__init__.py` as an empty file (Task 3 replaces its contents). This makes `daemon/macapp/launcher` ship as package data under the existing `packages = ["daemon"]`, exactly like `daemon/voice/models/silero_vad.onnx`.

- [ ] **Step 4: Write the failing binary-shape test**

Create `tests/test_macapp.py` (the bundle tests come in Task 3; start with the committed-binary invariant):

```python
"""The thin native-launcher .app (spec §3, §4.1). No real codesign/open/launchctl:
the runner is injected, mirroring tests/test_service.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parent.parent / "daemon" / "macapp" / "launcher"


def test_committed_launcher_is_universal2_macho() -> None:
    assert LAUNCHER.exists(), "the prebuilt launcher must be committed (spec §7)"
    archs = subprocess.run(
        ["lipo", "-archs", str(LAUNCHER)], capture_output=True, text=True, check=True
    ).stdout.split()
    assert set(archs) == {"arm64", "x86_64"}, f"launcher must be universal2, got {archs}"
```

Note: this test shells out to `lipo`, so it is macOS-only. Guard it if CI runs on Linux — add at module top:

```python
import sys
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS bundle tooling")
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_macapp.py -v`
Expected: PASS on macOS (binary committed and universal2).

- [ ] **Step 6: Commit**

```bash
git add daemon/macapp/__init__.py daemon/macapp/launcher.c daemon/macapp/launcher tests/test_macapp.py
git commit -m "macapp: committed universal2 launcher and its source (spec §2, §7)"
```

---

### Task 3: `daemon/macapp` — build the `.app` bundle

Assemble `Daemon.app` from committed assets: render `Info.plist` and `entitlements.plist` (both static — no per-install values), copy the prebuilt launcher, and ad-hoc codesign the bundle. Pure filesystem plus one injected `codesign` subprocess (mirror `service.py`).

**Files:**
- Modify: `daemon/macapp/__init__.py`
- Test: `tests/test_macapp.py` (extend)

**Interfaces:**
- Consumes: `daemon.service.RunResult`, `daemon.service.Runner`, `daemon.service.subprocess_runner` (reused, not redefined); the committed `daemon/macapp/launcher`.
- Produces:
  - `BUNDLE_ID: str = "ai.daemon.app"`
  - `build_bundle(app_path: Path, *, runner: Runner = subprocess_runner) -> Path` — writes the bundle at `app_path`, signs it, returns the launcher path `app_path / "Contents" / "MacOS" / "launcher"`.

- [ ] **Step 1: Write the failing tests**

Extend `tests/test_macapp.py`:

```python
from daemon.macapp import BUNDLE_ID, build_bundle
from daemon.service import RunResult


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

    import pytest

    with pytest.raises(RuntimeError, match="codesign"):
        build_bundle(tmp_path / "Daemon.app", runner=Failing())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_macapp.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_bundle'`.

- [ ] **Step 3: Implement `build_bundle`**

Replace `daemon/macapp/__init__.py` with:

```python
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
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>{BUNDLE_ID}</string>
    <key>CFBundleName</key><string>Daemon</string>
    <key>CFBundleExecutable</key><string>launcher</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>LSUIElement</key><true/>
    <key>NSMicrophoneUsageDescription</key><string>Daemon listens for its wake word.</string>
    <key>NSSpeechRecognitionUsageDescription</key><string>Daemon recognizes its wake word on-device.</string>
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_macapp.py -v`
Expected: PASS (all five tests).

- [ ] **Step 5: Commit**

```bash
git add daemon/macapp/__init__.py tests/test_macapp.py
git commit -m "macapp: build the ad-hoc-signed Daemon.app bundle (spec §4.1)"
```

---

### Task 4: `daemon/voice/mic_access.py` — request access + read status

The two Apple calls PortAudio cannot make: pop the mic prompt (`requestAccess`) and read the current authorization (for `/health`). Apple-guarded and injectable, mirroring `apple_speech.py`.

**Files:**
- Create: `daemon/voice/mic_access.py`
- Test: `tests/test_mic_access.py` (create)

**Interfaces:**
- Produces:
  - `Frameworks` dataclass: `av` (AVFoundation), `foundation` (Foundation).
  - `microphone_authorization_status(*, frameworks: Frameworks | None = None) -> str` — one of `"authorized" | "denied" | "restricted" | "not_determined" | "unavailable"`. Never prompts, never raises.
  - `request_microphone_access(*, timeout: float = 12.0, frameworks: Frameworks | None = None) -> str` — same return set; pops the prompt only when status is `not_determined` **and** a GUI session can show it. Never raises.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mic_access.py`:

```python
"""macOS mic TCC access, tested with injected fake frameworks — no AVFoundation,
no prompt, no microphone (tests/CLAUDE.md rule).
"""

from __future__ import annotations

from daemon.voice.mic_access import Frameworks, microphone_authorization_status, request_microphone_access


class FakeCaptureDevice:
    def __init__(self, status: int, granted: bool | None = None) -> None:
        self._status = status
        self._granted = granted
        self.requested = False

    def authorizationStatusForMediaType_(self, _media) -> int:
        return self._status

    def requestAccessForMediaType_completionHandler_(self, _media, handler) -> None:
        self.requested = True
        # The real API delivers asynchronously; the fake fires synchronously, which
        # is enough for the pump loop to see "done" on its first check.
        if self._granted is not None:
            handler(self._granted)


class FakeAV:
    AVMediaTypeAudio = "audio"

    def __init__(self, device: FakeCaptureDevice) -> None:
        self.AVCaptureDevice = device


class FakeDate:
    @staticmethod
    def dateWithTimeIntervalSinceNow_(_secs):
        return object()


class FakeLoop:
    def runMode_beforeDate_(self, _mode, _date) -> None:  # never needs to pump
        raise AssertionError("pumped the runloop though the handler already fired")


class FakeRunLoop:
    @staticmethod
    def currentRunLoop() -> FakeLoop:
        return FakeLoop()


class FakeFoundation:
    NSDate = FakeDate
    NSRunLoop = FakeRunLoop


def _fw(status: int, granted: bool | None = None) -> tuple[Frameworks, FakeCaptureDevice]:
    device = FakeCaptureDevice(status, granted)
    return Frameworks(av=FakeAV(device), foundation=FakeFoundation()), device


def test_status_maps_the_avauthorization_ints() -> None:
    for code, expected in {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}.items():
        fw, _ = _fw(code)
        assert microphone_authorization_status(frameworks=fw) == expected


def test_status_never_raises_on_absent_frameworks() -> None:
    # frameworks=None on a machine without AVFoundation → the real import fails and
    # is caught. Simulate by passing a Frameworks whose av lacks the method.
    class Broken:
        pass

    fw = Frameworks(av=Broken(), foundation=Broken())
    assert microphone_authorization_status(frameworks=fw) == "unavailable"


def test_request_returns_authorized_when_already_authorized_without_prompting() -> None:
    fw, device = _fw(3)
    assert request_microphone_access(frameworks=fw) == "authorized"
    assert device.requested is False, "must not re-prompt when already decided"


def test_request_returns_denied_when_already_denied_without_prompting() -> None:
    fw, device = _fw(2)
    assert request_microphone_access(frameworks=fw) == "denied"
    assert device.requested is False


def test_request_prompts_when_not_determined_and_grant_is_given() -> None:
    fw, device = _fw(0, granted=True)
    assert request_microphone_access(frameworks=fw) == "authorized"
    assert device.requested is True


def test_request_prompts_when_not_determined_and_grant_is_refused() -> None:
    fw, device = _fw(0, granted=False)
    assert request_microphone_access(frameworks=fw) == "denied"
    assert device.requested is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_mic_access.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'daemon.voice.mic_access'`.

- [ ] **Step 3: Implement the module**

Create `daemon/voice/mic_access.py`:

```python
"""macOS microphone (TCC) access: the one prompt, and the status read.

In the Apple-only corner next to apple_speech.py, guarded the same way: AVFoundation
is pyobjc, exists only on macOS, and a text-only or Linux install must still import
the package. Everything is caught — an absent or broken framework reads as
"unavailable", never an exception into a caller that has to outlive it.

Why this exists (spec §1, §4.3): a launchd-spawned bare Python cannot obtain the
mic grant (no TCC prompt is possible headless). The daemon is wrapped in a thin
Daemon.app whose code-signing identity *is* the grant. PortAudio's HAL access never
pops the prompt — it just returns silence when ungranted — so the prompt has to come
from an explicit AVCaptureDevice.requestAccess call. That call is a real prompt under
the .app foreground (`daemon request-mic`) and a cached no-op headless.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# AVAuthorizationStatus (AVFoundation): the integer the OS returns.
_STATUS = {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}
_PUMP_SLICE_SECONDS = 0.2


@dataclass(frozen=True, slots=True)
class Frameworks:
    """Injected in tests so no OS media service is touched (mirrors apple_speech)."""

    av: Any  # AVFoundation
    foundation: Any  # Foundation


def _load(frameworks: Frameworks | None) -> Frameworks:
    if frameworks is not None:
        return frameworks
    import AVFoundation
    import Foundation

    return Frameworks(av=AVFoundation, foundation=Foundation)


def microphone_authorization_status(*, frameworks: Frameworks | None = None) -> str:
    """The current TCC decision, asking nothing of the user. Never prompts."""
    try:
        fw = _load(frameworks)
        status = fw.av.AVCaptureDevice.authorizationStatusForMediaType_(fw.av.AVMediaTypeAudio)
        return _STATUS.get(int(status), "unavailable")
    except Exception:
        logger.debug("mic access: no AVFoundation here", exc_info=True)
        return "unavailable"


def request_microphone_access(*, timeout: float = 12.0, frameworks: Frameworks | None = None) -> str:
    """Claim the mic grant. Prompts only when the decision is still open and a GUI
    session can show it; otherwise returns the cached decision immediately.

    Pumps the runloop in short slices until the completion handler fires or the
    deadline passes — the handler is what the prompt (or the cached decision)
    resolves to. Headless-and-ungranted it returns "not_determined" without hanging
    past `timeout`.
    """
    try:
        fw = _load(frameworks)
        av = fw.av
        current = _STATUS.get(
            int(av.AVCaptureDevice.authorizationStatusForMediaType_(av.AVMediaTypeAudio)),
            "unavailable",
        )
        # Only a genuinely-undecided state should pump for a prompt; anything else
        # (already decided, or an unmapped/unknown code) is returned as-is.
        if current != "not_determined":
            return current

        done: dict[str, bool] = {}
        av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            av.AVMediaTypeAudio, lambda granted: done.__setitem__("g", bool(granted))
        )
        loop = fw.foundation.NSRunLoop.currentRunLoop()
        deadline = time.monotonic() + timeout
        while "g" not in done and time.monotonic() < deadline:
            loop.runMode_beforeDate_(
                "kCFRunLoopDefaultMode",
                fw.foundation.NSDate.dateWithTimeIntervalSinceNow_(_PUMP_SLICE_SECONDS),
            )
        if "g" not in done:
            return "not_determined"
        return "authorized" if done["g"] else "denied"
    except Exception:
        logger.debug("mic access: request failed", exc_info=True)
        return "unavailable"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_mic_access.py -v`
Expected: PASS (all six tests).

- [ ] **Step 5: Commit**

```bash
git add daemon/voice/mic_access.py tests/test_mic_access.py
git commit -m "voice: mic TCC request + status helpers, Apple-guarded (spec §4.3)"
```

---

### Task 5: `daemon request-mic` subcommand + claim the mic at wake-gate startup

The foreground grant path (`daemon request-mic`, run under the .app) and the headless claim (`daemon run` at wake-gate startup, D1). Both call `request_microphone_access`.

**Files:**
- Modify: `daemon/cli.py` (add the subparser, an early dispatch branch, `_request_mic`)
- Modify: `daemon/app.py` (claim the mic in the wake-gate start branch)
- Test: `tests/test_cli.py` (extend, or create if absent — check first)

**Interfaces:**
- Consumes: `daemon.voice.mic_access.request_microphone_access` (Task 4).
- Produces: CLI command `daemon request-mic` (exit 0 iff authorized); `app._claim_microphone(settings)` coroutine helper.

- [ ] **Step 1: Write the failing test**

First check for an existing CLI test file: `ls tests/test_cli.py` (if absent, create it). Add:

```python
def test_request_mic_reports_status_and_exit_code(monkeypatch, capsys) -> None:
    import daemon.cli as cli

    monkeypatch.setattr(
        "daemon.voice.mic_access.request_microphone_access", lambda **_: "authorized"
    )
    assert cli.main(["request-mic"]) == 0
    assert "authorized" in capsys.readouterr().out

    monkeypatch.setattr(
        "daemon.voice.mic_access.request_microphone_access", lambda **_: "denied"
    )
    assert cli.main(["request-mic"]) == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_cli.py -k request_mic -v`
Expected: FAIL (unknown command `request-mic` → argparse error / nonzero, or AttributeError).

- [ ] **Step 3: Add the subparser and dispatch**

In `daemon/cli.py` `build_parser()`, next to the other `sub.add_parser(...)` calls (e.g. after the `voice` parser):

```python
    sub.add_parser(
        "request-mic",
        help="claim macOS microphone access (used by Daemon.app during install)",
    )
```

In `main()`, add an early branch — before `Settings` is loaded, alongside `doctor`/`setup` (the mic grant needs no config):

```python
    if command == "request-mic":
        return _request_mic()
```

And add the handler near `_setup`:

```python
def _request_mic() -> int:
    """Pop the macOS microphone prompt (or report the cached decision) and exit.

    This is what Daemon.app's launcher execs during `daemon install`'s one-time
    foreground grant. It only claims the grant — it does not start a daemon — so it
    never collides with the resident LaunchAgent (spec §4.4, design decision 3).
    """
    from daemon.voice.mic_access import request_microphone_access

    status = request_microphone_access(timeout=60.0)  # a human has to click Allow
    print(f"microphone: {status}")
    return 0 if status == "authorized" else 1
```

- [ ] **Step 4: Claim the mic at wake-gate startup in `app.py`**

In `daemon/app.py`, in the wake-gate start branch (`elif settings.wake_enabled:`, ~line 298), claim the mic before building the recognizer. Add as the first line inside the `try:`:

```python
            await _claim_microphone(settings)
```

And add the helper near the other wake helpers (e.g. after `_wake_forever`, respecting the layering rule with a function-local import):

```python
async def _claim_microphone(settings: Settings) -> None:
    """macOS: claim the mic grant under the .app identity before PortAudio opens a
    stream that would otherwise return silence (spec D1). Headless-and-granted this
    is instant; ungranted it is the harmless no-op that only `daemon request-mic`
    (foreground, under Daemon.app) can turn into a prompt. Off the event loop
    because the runloop pump is blocking.
    """
    if sys.platform != "darwin":
        return
    from daemon.voice.mic_access import request_microphone_access

    status = await asyncio.to_thread(request_microphone_access, timeout=2.0)
    if status != "authorized":
        logger.warning(
            "wake gate: microphone not granted (%s); run `daemon install` and click "
            "Allow so it can hear the wake word",
            status,
        )
```

(`sys`, `asyncio`, `logger` are already imported in `app.py`.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_cli.py -k request_mic -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add daemon/cli.py daemon/app.py tests/test_cli.py
git commit -m "voice: daemon request-mic + claim the mic at wake-gate startup (spec §4.3, D1)"
```

---

### Task 6: Surface mic authorization in `/health` (Q3)

So "wake gate running but mic not granted" stops being invisible (spec §6).

**Files:**
- Modify: `daemon/app.py` (health dict + `_mic_health` helper)
- Test: `tests/test_app_health.py` or the existing health test (check first)

**Interfaces:**
- Consumes: `daemon.voice.mic_access.microphone_authorization_status` (Task 4).
- Produces: `/health` gains a `"mic"` key (`"n/a"` off-darwin, else the status string).

- [ ] **Step 1: Write the failing test**

Find the existing health test: `grep -rln '"/health"\|health()' tests/` and extend it; otherwise create `tests/test_app_health.py`. Add a test that drives the assembled app and asserts `mic` is present (mirror how the existing acceptance/e2e tests build the app; if a TestClient is already used for `/health`, reuse it):

```python
def test_health_reports_microphone_status(monkeypatch) -> None:
    import daemon.app as app_module

    monkeypatch.setattr(app_module, "_mic_health", lambda: "authorized")
    # ... assemble the app the way the existing /health test does, GET /health ...
    # assert body["mic"] == "authorized"
```

If no `/health` test harness exists yet, assert on `_mic_health` directly instead:

```python
def test_mic_health_is_na_off_darwin(monkeypatch) -> None:
    import daemon.app as app_module

    monkeypatch.setattr(app_module.sys, "platform", "linux")
    assert app_module._mic_health() == "n/a"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_app_health.py -v` (or the file you extended)
Expected: FAIL (`_mic_health` undefined / `mic` key absent).

- [ ] **Step 3: Add the field and helper**

In `daemon/app.py`, add to the `/health` return dict (after `"wake_gate": _wake_health(app.state),`):

```python
            # macOS: a wake gate can be "running" while the mic is denied, which
            # reads as a quiet room. Naming the grant here is the difference between
            # a diagnosable and an invisible failure (spec §6).
            "mic": _mic_health(),
```

And the helper near `_wake_health`:

```python
def _mic_health() -> str:
    """The microphone TCC decision, read (never prompted) at request time. `n/a`
    off macOS, where there is no TCC gate."""
    if sys.platform != "darwin":
        return "n/a"
    from daemon.voice.mic_access import microphone_authorization_status

    return microphone_authorization_status()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_app_health.py -v` (or the file you extended)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add daemon/app.py tests/test_app_health.py
git commit -m "health: surface microphone authorization on macOS (spec §6, Q3)"
```

---

### Task 7: Wire `daemon install` / `daemon uninstall` for the `.app` (macOS)

On macOS, `daemon install` builds `~/Applications/Daemon.app`, points the LaunchAgent at `[launcher, <daemon-path>, run]`, and launches the app foreground once for the grant. `daemon uninstall` removes the `.app` (keeping the harmless TCC grant). Linux is unchanged.

**Files:**
- Modify: `daemon/cli.py` (`APP_DIR`, `_macos_program`, `_install`, `_uninstall`, `_grant_microphone_once`, imports; setup guidance string)
- Modify: `daemon/setup.py` (one closing-guidance line — optional, see Step 5)
- Test: `tests/test_cli.py` (extend — the pure helper only)

**Interfaces:**
- Consumes: `daemon.macapp.build_bundle` (Task 3); `daemon.service.default_program`, `Service`; `daemon.voice`… (none directly).
- Produces: `daemon install` writes a plist whose `ProgramArguments` is `[<app>/Contents/MacOS/launcher, <daemon-path>, run]` on macOS.

- [ ] **Step 1: Write the failing test (pure helper only)**

The build/open/launchctl orchestration is verified by running the real thing (Task 9) — mocking all three would only re-assert the mocks (tests/CLAUDE.md: "a test that passes for the wrong reason is worse than none"). The one load-bearing pure fact is the plist argv order: the launcher must be argv[0] and the daemon path the argv it execs. Extract and test that.

In `tests/test_cli.py`:

```python
def test_macos_program_puts_launcher_first_then_daemon_argv() -> None:
    import daemon.cli as cli
    from pathlib import Path

    program = cli._macos_program(Path("/Users/x/Applications/Daemon.app/Contents/MacOS/launcher"),
                                 ("/Users/x/.local/bin/daemon", "run"))
    assert program == (
        "/Users/x/Applications/Daemon.app/Contents/MacOS/launcher",
        "/Users/x/.local/bin/daemon",
        "run",
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_cli.py -k macos_program -v`
Expected: FAIL (`_macos_program` undefined).

- [ ] **Step 3: Implement the install/uninstall wiring**

In `daemon/cli.py` add imports at the top (with the existing stdlib imports):

```python
import shutil
import subprocess
```

and extend the service import:

```python
from daemon.service import Service, ServiceAction, ServiceError, ServiceStatus, default_program
```

Add near `service_for`:

```python
APP_DIR = Path.home() / "Applications" / "Daemon.app"
"""Where the TCC-identity bundle lives (spec Q2). Holds no secrets."""


def _macos_program(launcher: Path, daemon_argv: tuple[str, ...]) -> tuple[str, ...]:
    """The plist ProgramArguments: the launcher, then the daemon argv it execs.

    Order is load-bearing — the launcher is argv[0] (what launchd starts native,
    giving the .app's TCC identity) and it execs argv[1] (the daemon path) with the
    rest (the subcommand). service.py renders this tuple verbatim.
    """
    return (str(launcher), *daemon_argv)
```

Replace the `install`/`uninstall` dispatch in `main()`:

```python
        if command == "install":
            return _install(settings, force=args.force)
        if command == "uninstall":
            return _uninstall(settings)
```

Add the handlers:

```python
def _install(settings: Settings, *, force: bool) -> int:
    if sys.platform != "darwin":
        return _print_action(service_for(settings).install(force=force))

    # macOS: the LaunchAgent must start Daemon.app's launcher, not the bare console
    # script — a launchd-spawned bare Python is silently denied the microphone
    # (spec §1). The launcher execs the real `daemon run` under the .app identity.
    from daemon.macapp import build_bundle

    launcher = build_bundle(APP_DIR)
    daemon_argv = default_program()  # (…/daemon, "run") — what the launcher execs
    service = Service(
        label=settings.service_label,
        working_dir=Path.cwd(),
        log_dir=settings.data_dir / "logs",
        program=_macos_program(launcher, daemon_argv),
    )
    rc = _print_action(service.install(force=force))
    _grant_microphone_once(launcher, daemon_argv)
    return rc


def _uninstall(settings: Settings) -> int:
    rc = _print_action(service_for(settings).uninstall(), verb="removed")
    if sys.platform == "darwin" and APP_DIR.exists():
        shutil.rmtree(APP_DIR, ignore_errors=True)
        print(f"removed {APP_DIR}")
        print("(the microphone grant is kept — harmless, and a reinstall skips the prompt)")
    return rc


def _grant_microphone_once(launcher: Path, daemon_argv: tuple[str, ...]) -> None:
    """Launch the .app foreground so the mic prompt appears under its TCC identity.

    Runs `daemon request-mic` (via the launcher), which claims the grant and exits —
    NOT a second `daemon run`, so it never collides with the LaunchAgent (design
    decision 3). Fire-and-forget: the grant persists once the user clicks Allow.
    """
    app = launcher.parents[2]  # …/Daemon.app
    print("\nA microphone permission dialog will appear — click Allow.")
    print("(Daemon listens for its wake word; the grant persists across reboots and updates.)")
    # Fixed argv vector, no shell (CONTRACTS 13): open the bundle and pass the
    # launcher the daemon argv with its trailing subcommand ("run") replaced by
    # "request-mic". Uses `daemon_argv[:-1]`, not `daemon_argv[0]`, so the
    # `python -m daemon.cli` checkout shape still execs `... -m daemon.cli
    # request-mic` rather than `python request-mic` (final-review fix).
    subprocess.run(
        ["open", str(app), "--args", *daemon_argv[:-1], "request-mic"], check=False
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_cli.py -k macos_program -v`
Expected: PASS.

- [ ] **Step 5: Point setup's closing guidance at the grant (optional, macOS)**

In `daemon/setup.py`, the closing guidance already prints `daemon install - keeps it running after you close the terminal or reboot` (~line 2086). If the wizard is running on macOS, append a one-line note so the user expects the prompt. Locate that block and add, guarded by `sys.platform == "darwin"`:

```python
            say("                    on macOS it also asks once for microphone access — click Allow")
```

Keep the change to a single added line; do not restructure the guidance.

- [ ] **Step 6: Run the full suite + lint**

Run:
```bash
python3 -m pytest
python3 -m ruff check .
```
Expected: all green. Fix anything red before committing.

- [ ] **Step 7: Commit**

```bash
git add daemon/cli.py daemon/setup.py tests/test_cli.py
git commit -m "cli: install/uninstall build and point at Daemon.app on macOS (spec §4.2, §4.4)"
```

---

### Task 8: Reachability gate + docs touch-up

Confirm the new symbols are reachable (not just built) and update the two doc surfaces that enumerate commands/modules, so `scripts/check_docs.py` and the project's own map stay honest.

**Files:**
- Verify: `tests/test_reachable.py` (likely no code change — see below)
- Modify: `daemon/CLAUDE.md` (add `macapp` and `mic_access` to the layout tables; add `request-mic` to the cli list)
- Verify: `scripts/check_docs.py` passes

**Interfaces:** none.

- [ ] **Step 1: Run the reachability gate**

Run: `python3 -m pytest tests/test_reachable.py -v`
Expected: PASS. The new symbols are functions called by `cli.py`/`app.py` (`build_bundle`, `request_microphone_access`, `microphone_authorization_status`, `_request_mic`, `_mic_health`), reachable through their callers — `test_reachable` tracks `Task`/provider/channel/protocol *classes*, not free functions, so no `PENDING_*` entry is needed. If it unexpectedly fails, read the failure: it will name what to add and the milestone to tag (do **not** silence the assertion).

- [ ] **Step 2: Update `daemon/CLAUDE.md`**

Add a row to the layout table for the new module family (near `voice/` / `service.py`):

```markdown
| `macapp/` | the thin native-launcher `Daemon.app` (macOS): `build_bundle` assembles and ad-hoc-signs the bundle whose code identity is the microphone grant; `launcher.c`/`launcher` is the committed universal2 Mach-O it copies in |
```

Add `mic_access.py` to the `voice/` row (mic TCC request + status, Apple-guarded), and add `request-mic` to the `cli.py` command list.

- [ ] **Step 3: Verify docs check**

Run: `python3 scripts/check_docs.py`
Expected: PASS (documented paths exist).

- [ ] **Step 4: Commit**

```bash
git add daemon/CLAUDE.md
git commit -m "docs: map macapp and mic_access; note daemon request-mic"
```

---

### Task 9: Real acceptance on the target Mac (manual — the actual definition of done)

Green units are **not** done (project rule; CONTRACTS "Testing"). This is the spike, formalized (spec §8). It requires the owner to speak into the microphone, so it is a hand-off step — **pause here and ask the owner to run it and to say the wake word.**

**Files:** none (verification only).

- [ ] **Step 1: Fresh install with voice, on the target Mac**

From a checkout of this branch:

```bash
uv tool install --force --python 3.13 --from . daemon-ai
```

(Or the release one-liner once cut from `main` — with Task 1, `daemon-ai` now carries voice on macOS with no `[voice]` suffix.) Confirm the recognizer is present:

```bash
daemon doctor
```

- [ ] **Step 2: Configure and install the service (this builds the .app and prompts)**

```bash
daemon setup      # if not already configured (preset, keys, wake aliases)
daemon install    # builds ~/Applications/Daemon.app, installs the LaunchAgent, opens the app once
```

Expect a **microphone permission dialog** — click **Allow**. Confirm the grant:

```bash
curl -s localhost:8000/health | python3 -m json.tool | grep -E 'wake_gate|mic'
```

Expected: `"mic": "authorized"` and `"wake_gate"` not `unavailable`.

- [ ] **Step 3: Prove headless wake with no terminal attached**

Restart the resident agent through launchd (no terminal in the loop) and speak:

```bash
launchctl kickstart -k gui/$(id -u)/ai.daemon.default
# then say "벨라" out loud
tail -f data/logs/ai.daemon.default.err.log
```

**Acceptance:** the log shows a line like
`wake: heard '벨라' matching '벨라'; opening a voice session`
with no terminal holding a mic grant. That line is the definition of done.

- [ ] **Step 4: Prove the grant survives an update**

```bash
daemon update           # or: uv tool install --force --from . daemon-ai
launchctl kickstart -k gui/$(id -u)/ai.daemon.default
# say "벨라" again — it must still wake, with no new prompt
```

Expected: wakes again, no re-authorization (the launcher's code identity is unchanged).

- [ ] **Step 5: Note (owner's call): trim the wake aliases if 연락 false-wakes**

Spec §11: `DAEMON_WAKE_ALIASES` was calibrated to `벨라`, but `연락` was also captured and is a false-wake risk. If Step 3/4 show spurious wakes, trim to `벨라` in `.env` and re-run `daemon wake test`. Config-only, owner's choice — not a code change.

---

## Self-review (completed against the spec)

- **§1/§2 (problem & evidence):** encoded as Global Constraints and the launcher rationale (Tasks 2, 3). ✓
- **§3 (approach) + D1/D2/D3:** thin native-launcher `.app` (Task 3); requestAccess at wake-gate startup + foreground grant in install (Tasks 5, 7); prebuilt committed universal2 launcher (Task 2); ad-hoc sign, no hardened runtime (Task 3). ✓
- **§4.1 macapp:** Task 3. **§4.2 service/plist:** achieved via the plist-argv design without changing `default_program()` (decision 2, Task 7). **§4.3 requestAccess helper:** Task 4. **§4.4 first-run grant:** Task 7 (in `install`, decision 4). **§4.5 wiring:** Task 5 (`_claim_microphone`, layering-respected). **§4.6 blocker:** Task 1. ✓
- **§5 flow, §6 edge cases:** native-only launcher, no hardened runtime, grant-reset-on-rebuild documented, `mic` in `/health` (Task 6). ✓
- **§7 shipping the binary:** committed under `daemon/macapp/`, shipped by existing hatch config (Task 2, verified against `silero_vad.onnx` precedent). ✓
- **§8 testing/acceptance:** unit (Tasks 1,3,4,5,6,7), reachability (Task 8), real manual acceptance (Task 9). ✓
- **§9 non-goals:** no Developer ID / notarization / hardened runtime; Linux untouched; no menu-bar UI. ✓
- **§10 open questions:** Q1 default-on macOS (Task 1); Q2 `~/Applications` (Global Constraints, Task 7); Q3 `/health` mic (Task 6). ✓
- **Placeholder scan:** every code step carries real code; no TBD/TODO. ✓
- **Type consistency:** `build_bundle(app_path, *, runner) -> Path`, `request_microphone_access(*, timeout, frameworks) -> str`, `_macos_program(launcher, daemon_argv) -> tuple[str, ...]` used consistently across tasks. ✓

**One risk to watch during execution (verify in Task 9, not assumable from units):** that `open <app> --args <daemon-path> request-mic` delivers argv to the launcher and pops the prompt under the bundle identity. This is the exact shape the spike verified for the foreground grant (`open Daemon.app`, execv into Python, granted). If `open --args` does not forward argv in practice, fall back to invoking the launcher binary directly: `"$APP/Contents/MacOS/launcher" "<daemon-path>" request-mic` (same identity, same requestAccess) — try this before concluding anything is wrong with the bundle.
