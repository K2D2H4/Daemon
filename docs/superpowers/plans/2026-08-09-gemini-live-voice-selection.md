# Gemini Live Voice Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the owner pick the Gemini Live voice from the admin web, persist it to `.env`, and hear a bundled preview clip of each voice before saving.

**Architecture:** A new `gemini_live_voice` setting flows `config → admin settings_io → app.run_voice → GeminiLiveSession(voice_name=…)`. The admin gains a dropdown of the 30 native-audio voices and a `▶` preview button backed by a new `GET /admin/api/voice-sample/{voice}` route that serves pre-generated MP3s. The clips are produced once by an offline generator (`evals/gen_voice_samples.py`) that hits the live API and are committed as static assets, keeping the admin offline and preview-time free.

**Tech Stack:** Python 3.13, pydantic-settings, FastAPI, `fastapi.testclient.TestClient`, vanilla JS in one self-contained `index.html`, `ffmpeg` (generator only), `wave` (stdlib).

## Global Constraints

- **Layering:** `config.py` must not import `daemon/voice/*`; the voice allowlist is a `config.py` constant (same rule that duplicates `SENSITIVITIES`). Only `daemon/app.py` imports concrete providers/sessions.
- **Fail early, loud:** a bad voice name raises `ConfigError` at `Settings` construction, never on the wire (an unknown name is a server `1007` permanent close that ends voice mode).
- **Validate-before-write:** the admin PATCH path is unchanged — a candidate `Settings` must construct before any `.env` byte is written.
- **No test may touch the network, a key, a microphone, or a speaker** (tests/CLAUDE.md). The generator is not a test; it lives in `evals/` and is run by hand.
- **Admin stays offline/self-contained:** no CDN; preview plays committed local files.
- **Korean where the product is Korean:** add a Korean case where text is involved (here the admin tests already use Korean; keep that).
- **The 30 voices (verbatim, from `ai.google.dev/gemini-api/docs/speech-generation`, 2026-08-09):** Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, Callirrhoe, Autonoe, Enceladus, Iapetus, Umbriel, Algieba, Despina, Erinome, Algenib, Rasalgethi, Laomedeia, Achernar, Alnilam, Schedar, Gacrux, Pulcherrima, Achird, Zubenelgenubi, Vindemiatrix, Sadachbia, Sadaltager, Sulafat.
- **Preview phrase:** `Hi, I'm Daemon. This is what I sound like.`

---

### Task 1: Config field, allowlist constant, and validation

**Files:**
- Modify: `daemon/config.py` (constant near line 63 `SENSITIVITIES`; field near line 287 `gemini_live_model`; validation in `_check` near line 749)
- Modify: `.env.example` (add the blank key with a comment)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `GEMINI_LIVE_VOICES: frozenset[str]` in `daemon.config`; `Settings.gemini_live_voice: str` (env alias `DAEMON_GEMINI_LIVE_VOICE`, default `""`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py`, beside the existing voice cases, using the module's `make_settings` helper:

```python
def test_an_unknown_gemini_live_voice_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_GEMINI_LIVE_VOICE"):
        make_settings(gemini_live_voice="NotAVoice")


def test_a_known_gemini_live_voice_and_empty_both_construct() -> None:
    assert make_settings(gemini_live_voice="Kore").gemini_live_voice == "Kore"
    assert make_settings(gemini_live_voice="").gemini_live_voice == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_config.py -k gemini_live_voice -v`
Expected: FAIL — `Settings` has no `gemini_live_voice` (unexpected keyword / attribute error).

- [ ] **Step 3: Add the constant, the field, and the validation**

In `daemon/config.py`, add the constant near `SENSITIVITIES` (line ~63):

```python
GEMINI_LIVE_VOICES = frozenset({
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
})
"""Prebuilt Gemini Live voices. Native-audio models accept the full TTS voice set
(ai.google.dev/gemini-api/docs/speech-generation). Kept here rather than imported
from daemon/voice/*: importing the voice layer into config inverts the layering,
the same reason SENSITIVITIES is duplicated."""
```

Add the field beside `gemini_live_model` (line ~287):

```python
gemini_live_voice: str = Field(default="", alias="DAEMON_GEMINI_LIVE_VOICE")
"""Which prebuilt voice the Gemini Live session speaks in: one of
GEMINI_LIVE_VOICES, or empty to leave it to the server. Checked at construction,
not on the wire: an unknown name comes back as a 1007 close the session treats as
permanent, so a typo would end voice mode rather than fail the setting."""
```

Add the check in `_check`, right after the sensitivity loop (line ~757):

```python
if self.gemini_live_voice and self.gemini_live_voice not in GEMINI_LIVE_VOICES:
    problems.append(
        f"DAEMON_GEMINI_LIVE_VOICE is {self.gemini_live_voice!r}; expected one of "
        "the Gemini Live voices, or empty to leave it to the server"
    )
```

(Validated unconditionally, independent of `voice_enabled`: a typo is a typo, and the default `""` is harmless when voice is off.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_config.py -k gemini_live_voice -v`
Expected: PASS (both tests).

- [ ] **Step 5: Add the discoverability line to `.env.example`**

Add, near the other `DAEMON_VOICE_*`/`DAEMON_GEMINI_LIVE_*` lines in `.env.example`:

```bash
# Which prebuilt voice Gemini Live speaks in (empty = server default).
# One of: Zephyr, Puck, Charon, Kore, Fenrir, Leda, Orus, Aoede, … (see the admin dropdown).
DAEMON_GEMINI_LIVE_VOICE=
```

- [ ] **Step 6: Commit**

```bash
git add daemon/config.py tests/test_config.py .env.example
git commit -m "config: DAEMON_GEMINI_LIVE_VOICE with a validated 30-voice allowlist"
```

---

### Task 2: Surface the field in the admin settings API

**Files:**
- Modify: `daemon/admin/settings_io.py` (`STR_FIELDS` line ~41; `current_settings_payload` `options` line ~96; import from `daemon.config`)
- Test: `tests/test_admin.py`

**Interfaces:**
- Consumes: `GEMINI_LIVE_VOICES` from `daemon.config` (Task 1); `Settings.gemini_live_voice`.
- Produces: GET `/admin/api/settings` `editable.gemini_live_voice` and `options.gemini_live_voices: list[str]`; PATCH accepts `gemini_live_voice`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_admin.py`:

```python
def test_patch_sets_the_gemini_live_voice(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=offline\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"gemini_live_voice": "Kore"})
    assert resp.status_code == 200
    assert "DAEMON_GEMINI_LIVE_VOICE=Kore" in env.read_text(encoding="utf-8")

    got = client.get("/admin/api/settings").json()
    assert got["editable"]["gemini_live_voice"] == "Kore"
    assert "Kore" in got["options"]["gemini_live_voices"]
    assert got["options"]["gemini_live_voices"][0] == "", "empty (server default) must be offered first"


def test_patch_rejects_an_unknown_gemini_live_voice(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    original = "DAEMON_PRESET=offline\n"
    env.write_text(original, encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={"gemini_live_voice": "Nope"})
    assert resp.status_code == 400
    assert env.read_text(encoding="utf-8") == original, "a rejected voice still wrote"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_admin.py -k gemini_live_voice -v`
Expected: FAIL — `gemini_live_voice` is not editable (PATCH 400 "not editable"), and `options` lacks `gemini_live_voices`.

- [ ] **Step 3: Wire it in `settings_io.py`**

Extend the import from `daemon.config`:

```python
from daemon.config import (
    GEMINI_LIVE_VOICES,
    HOSTED_PROVIDERS,
    PRESETS,
    Settings,
)
```

Add to `STR_FIELDS`:

```python
STR_FIELDS: dict[str, str] = {
    "preset": "DAEMON_PRESET",
    "hosted_provider": "DAEMON_HOSTED_PROVIDER",
    "tools_mode": "DAEMON_TOOLS_MODE",
    "gemini_live_voice": "DAEMON_GEMINI_LIVE_VOICE",
}
```

Add to the `options` block in `current_settings_payload`:

```python
"options": {
    "presets": sorted(PRESETS),
    "hosted_providers": list(HOSTED_PROVIDERS),
    "tool_modes": list(TOOL_MODES),
    "gemini_live_voices": ["", *sorted(GEMINI_LIVE_VOICES)],
},
```

(`editable.gemini_live_voice` needs no extra line: `current_settings_payload` already iterates `STR_FIELDS`, and `EDITABLE` already includes `STR_FIELDS`, so PATCH accepts it with the validate-before-write guarantee intact.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_admin.py -k gemini_live_voice -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add daemon/admin/settings_io.py tests/test_admin.py
git commit -m "admin: offer and persist gemini_live_voice in the settings API"
```

---

### Task 3: Serve preview clips — `GET /admin/api/voice-sample/{voice}`

**Files:**
- Modify: `daemon/admin/routes.py` (imports; a `VOICE_SAMPLES` constant near `SHELL` line ~63; new route)
- Test: `tests/test_admin.py`

**Interfaces:**
- Consumes: `GEMINI_LIVE_VOICES` from `daemon.config`.
- Produces: `VOICE_SAMPLES: Path` (module constant, monkeypatchable in tests); route returning `audio/mpeg` bytes or 404.

- [ ] **Step 1: Write the failing tests**

In `tests/test_admin.py`:

```python
def test_voice_sample_serves_a_present_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = tmp_path / "voice-samples"
    samples.mkdir()
    (samples / "Kore.mp3").write_bytes(b"ID3-fake-mp3-bytes")
    monkeypatch.setattr("daemon.admin.routes.VOICE_SAMPLES", samples)

    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)
    resp = client.get("/admin/api/voice-sample/Kore")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content == b"ID3-fake-mp3-bytes"


def test_voice_sample_404_for_missing_or_unknown_and_never_reads_a_bad_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = tmp_path / "voice-samples"
    samples.mkdir()
    monkeypatch.setattr("daemon.admin.routes.VOICE_SAMPLES", samples)

    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)

    # Known voice, but no file generated yet -> 404, not 500.
    assert client.get("/admin/api/voice-sample/Kore").status_code == 404
    # A name outside the allowlist -> 404, and the allowlist check runs before any
    # filesystem touch, so a traversal attempt never resolves a path.
    assert client.get("/admin/api/voice-sample/Nope").status_code == 404
    assert client.get("/admin/api/voice-sample/..%2f..%2fetc%2fpasswd").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_admin.py -k voice_sample -v`
Expected: FAIL — the route does not exist (404 for the present clip too, and no `VOICE_SAMPLES` attribute to monkeypatch → `AttributeError`).

- [ ] **Step 3: Add the constant, import, and route**

In `daemon/admin/routes.py`, extend the response import:

```python
from fastapi.responses import HTMLResponse, JSONResponse, Response
```

Add the config import:

```python
from daemon.config import GEMINI_LIVE_VOICES
```

Beside `SHELL` (line ~63):

```python
VOICE_SAMPLES = Path(__file__).parent / "static" / "voice-samples"
"""Committed MP3 previews, one per Gemini Live voice, produced offline by
evals/gen_voice_samples.py. Served locally so preview needs no key and no network -
the admin stays offline (design decision 1)."""
```

Add the route (after `patch_settings`, before the MCP section):

```python
@router.get("/api/voice-sample/{voice}")
async def voice_sample(voice: str) -> Response:
    """A short preview clip for one Gemini Live voice, as audio/mpeg.

    The name is checked against the fixed voice set BEFORE any path is built, so an
    unknown or traversal name can never resolve a file. A known voice whose clip was
    not generated is a 404, not a 500 - a missing asset degrades to 'no preview'."""
    if voice not in GEMINI_LIVE_VOICES:
        return JSONResponse({"detail": f"no such voice {voice!r}"}, status_code=404)
    path = VOICE_SAMPLES / f"{voice}.mp3"
    if not path.is_file():
        return JSONResponse(
            {"detail": f"no preview generated for {voice!r}"}, status_code=404
        )
    return Response(path.read_bytes(), media_type="audio/mpeg")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_admin.py -k voice_sample -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add daemon/admin/routes.py tests/test_admin.py
git commit -m "admin: serve committed voice-preview clips, allowlist-gated"
```

---

### Task 4: Pass the chosen voice into the live session

**Files:**
- Modify: `daemon/app.py` (the `GeminiLiveSession(...)` construction in `run_voice`, line ~955)

**Interfaces:**
- Consumes: `Settings.gemini_live_voice` (Task 1); `GeminiLiveSession(voice_name=…)` (existing parameter).

- [ ] **Step 1: Add the argument**

In `daemon/app.py`, in the `GeminiLiveSession(...)` call inside `run_voice`, add one keyword argument alongside the existing sensitivity arguments (after `silence_duration_ms=…`):

```python
                silence_duration_ms=settings.voice_silence_duration_ms,
                # Empty passes straight through as "leave it to the server", exactly
                # as before this setting existed (daemon/voice/gemini_live.py).
                voice_name=settings.gemini_live_voice,
```

- [ ] **Step 2: Verify reachability (this wire has no isolated unit test)**

`run_voice` is the network path — the repo forbids unit tests from opening it (tests/CLAUDE.md), and `GeminiLiveSession` is already in `WIRED_CLASSES`. The `voice_name` argument is a `_constructed` blind spot (test_reachable.py's own note), but it is wired here immediately, so no `PENDING_WIRING` entry is needed. Confirm nothing regressed:

Run: `python3 -m pytest tests/test_reachable.py -v`
Expected: PASS. (The end-to-end proof is the live check in Task 7.)

- [ ] **Step 3: Commit**

```bash
git add daemon/app.py
git commit -m "voice: speak in the configured DAEMON_GEMINI_LIVE_VOICE"
```

---

### Task 5: The offline sample generator

**Files:**
- Create: `evals/gen_voice_samples.py`
- Create (directory, via first run): `daemon/admin/static/voice-samples/`

**Interfaces:**
- Consumes: `GEMINI_LIVE_VOICES` from `daemon.config`; `GeminiLiveSession` from `daemon.voice.gemini_live`; env `GEMINI_API_KEY`, `DAEMON_GEMINI_LIVE_MODEL`; `ffmpeg` on PATH.
- Produces: `daemon/admin/static/voice-samples/<Voice>.mp3` for each voice.

This task has no unit test (evals hit the live API and are never in CI — tests/CLAUDE.md). Its verification is a real run in Task 7.

- [ ] **Step 1: Write the generator**

Create `evals/gen_voice_samples.py`:

```python
"""Generate the admin's Gemini Live voice-preview clips. Manual; hits the live API.

    GEMINI_API_KEY=... DAEMON_GEMINI_LIVE_MODEL=... python -m evals.gen_voice_samples

Writes daemon/admin/static/voice-samples/<Voice>.mp3, one per GEMINI_LIVE_VOICES.
Needs ffmpeg on PATH for PCM->MP3. Not a test: the suite never uses a key or network
(tests/CLAUDE.md), so this lives in evals/, which may import product code and does.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from daemon.config import GEMINI_LIVE_VOICES
from daemon.voice.gemini_live import GeminiLiveSession

PHRASE = "Hi, I'm Daemon. This is what I sound like."
# send_text is a prompt, not verbatim TTS, so instruct the model to read the line.
READ_VERBATIM = "Repeat the user's message back word for word, and say nothing else."
OUT = Path(__file__).resolve().parents[1] / "daemon" / "admin" / "static" / "voice-samples"
PLAYBACK_RATE = 24_000  # Gemini Live returns 24 kHz mono 16-bit PCM.


async def _capture(api_key: str, model: str, voice: str) -> bytes:
    pcm = bytearray()
    async with GeminiLiveSession(
        api_key=api_key,
        model=model,
        system_instruction=READ_VERBATIM,
        voice_name=voice,
    ) as session:
        await session.send_text(PHRASE)
        async for item in session.receive():
            if isinstance(item, bytes):
                pcm += item
    return bytes(pcm)


def _to_mp3(pcm: bytes, dest: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        wav = Path(handle.name)
    try:
        with wave.open(str(wav), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(PLAYBACK_RATE)
            writer.writeframes(pcm)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav), "-b:a", "64k", str(dest)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        wav.unlink(missing_ok=True)


async def main() -> int:
    key = os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("DAEMON_GEMINI_LIVE_MODEL", "")
    if not key or not model:
        print("set GEMINI_API_KEY and DAEMON_GEMINI_LIVE_MODEL", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for voice in sorted(GEMINI_LIVE_VOICES):
        try:
            pcm = await _capture(key, model, voice)
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            failures += 1
            print(f"! {voice}: {exc}", file=sys.stderr)
            continue
        if not pcm:
            failures += 1
            print(f"! {voice}: no audio returned", file=sys.stderr)
            continue
        _to_mp3(pcm, OUT / f"{voice}.mp3")
        print(f"OK {voice} ({len(pcm)} bytes pcm)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 2: Sanity-check it imports and reports the missing-key path (no network)**

Run: `python3 -m evals.gen_voice_samples`
Expected: exits non-zero printing `set GEMINI_API_KEY and DAEMON_GEMINI_LIVE_MODEL` (no key in the environment). This confirms the module imports cleanly.

- [ ] **Step 3: Commit the generator (assets come in Task 7 after a real run)**

```bash
git add evals/gen_voice_samples.py
git commit -m "evals: offline generator for admin voice-preview clips"
```

---

### Task 6: Admin UI — voice dropdown and preview button

**Files:**
- Modify: `daemon/admin/static/index.html` (`loadSettings` render ~line 277; a small render helper near `fieldBool` ~line 268; one delegated click handler near the `settings-save` binding ~line 305)

**Interfaces:**
- Consumes: GET `/admin/api/settings` `editable.gemini_live_voice` + `options.gemini_live_voices` (Task 2); `GET /admin/api/voice-sample/{voice}` (Task 3).

No JS unit harness exists here; verification is the shell rendering (TestClient) plus a manual browser preview in Task 7.

- [ ] **Step 1: Render the dropdown and preview control**

In `daemon/admin/static/index.html`, add a helper next to `fieldBool`:

```js
function fieldVoicePreview(){return `<div class="field"><span class="n">preview</span><button type="button" class="btn" id="voice-play">▶</button><audio id="voice-sample" hidden></audio></div>`;}
```

In `loadSettings`, after the `voice_enabled` line (line ~277):

```js
    h+=fieldBool('voice_enabled',e.voice_enabled);
    h+=fieldStr('gemini_live_voice',e.gemini_live_voice,o.gemini_live_voices);
    h+=fieldVoicePreview();
```

(`collectPatch`'s generic string branch already picks up the `gemini_live_voice` select — no change there. The empty first option is the "(default)" choice, exactly as `hosted_provider` renders `['',…]`.)

- [ ] **Step 2: Wire the preview click (delegated, bound once)**

Near the `$('#settings-save').onclick = …` binding (line ~305), add a delegated handler on the static `#settings-form` container (its innerHTML is replaced each `loadSettings`, so bind on the container, not the button):

```js
$('#settings-form').addEventListener('click',ev=>{
  if(ev.target.id!=='voice-play')return;
  const sel=document.querySelector('[data-f="gemini_live_voice"]');
  if(!sel||!sel.value)return; // the empty "(default)" option has no clip
  const a=$('#voice-sample');
  a.src='/admin/api/voice-sample/'+encodeURIComponent(sel.value);
  a.play();
});
```

- [ ] **Step 3: Verify the shell still renders offline**

Run: `python3 -m pytest tests/test_admin.py -k shell -v`
Expected: PASS (`test_shell_page_renders_offline` — still self-contained, no CDN).

- [ ] **Step 4: Commit**

```bash
git add daemon/admin/static/index.html
git commit -m "admin ui: gemini voice dropdown with a preview button"
```

---

### Task 7: Full verification — suite, lint, real samples, live check

**Files:**
- Create (assets): `daemon/admin/static/voice-samples/*.mp3`

- [ ] **Step 1: Whole suite + lint + docs check**

Run:
```bash
python3 -m pytest
python3 -m ruff check .
python3 scripts/check_docs.py
```
Expected: all pass. (`check_docs.py` verifies documented paths exist — the plan referenced `daemon/admin/static/voice-samples/` and `evals/gen_voice_samples.py`, both present after Tasks 5–7.)

- [ ] **Step 2: Generate the real preview clips (needs a key + ffmpeg)**

Run:
```bash
GEMINI_API_KEY=<key> DAEMON_GEMINI_LIVE_MODEL=<live-model-id> python3 -m evals.gen_voice_samples
```
Expected: `OK <Voice>` for each of the 30, and 30 files in `daemon/admin/static/voice-samples/`. Spot-listen to two or three clips (e.g. `Kore.mp3`, `Puck.mp3`) and confirm they say the phrase in clearly different voices. This run is the plan's real-API check (green units are not proof — tests/CLAUDE.md).

- [ ] **Step 3: Live end-to-end — the voice actually changes**

With voice enabled, set `DAEMON_GEMINI_LIVE_VOICE` to one voice via the admin (or `.env`), run `daemon voice`, speak, and confirm the spoken timbre matches; change it to a different voice and confirm it changes. This exercises the `config → app.run_voice → GeminiLiveSession` wire that has no unit test (Task 4).

- [ ] **Step 4: Manual admin preview**

Open `/admin/`, go to Settings, pick a voice, click `▶`, and confirm the clip plays; confirm the empty "(default)" option plays nothing and the dropdown lists all 30 voices.

- [ ] **Step 5: Commit the assets**

```bash
git add daemon/admin/static/voice-samples/
git commit -m "admin assets: bundled Gemini Live voice-preview clips (30 voices)"
```

---

## Self-Review

**Spec coverage:**
- Config field + strict allowlist + validation → Task 1. ✓
- `settings_io` STR_FIELDS + options → Task 2. ✓
- `app.run_voice` voice_name pass-through → Task 4. ✓
- `index.html` dropdown → Task 6. ✓
- Bundled preview (generator, endpoint, UI button, committed assets) → Tasks 3, 5, 6, 7. ✓
- Tests (admin PATCH valid/invalid, sample serving, config validation) → Tasks 1–3. ✓
- Real-API verification + live `daemon voice` check → Task 7. ✓
- English preview phrase → Global Constraints + Task 5. ✓
- 30 voices exposed → Global Constraints + Tasks 1–2. ✓
- Phase B (OpenAI Realtime, `DAEMON_VOICE_PROVIDER`) explicitly out of scope → not in any task. ✓

**Placeholder scan:** No TBD/TODO; every code and test step carries real content. ✓

**Type/name consistency:** `GEMINI_LIVE_VOICES` (frozenset) and `gemini_live_voice` (str, alias `DAEMON_GEMINI_LIVE_VOICE`) used identically across Tasks 1, 2, 3, 4, 5; `VOICE_SAMPLES` (Path) defined in Task 3 and monkeypatched by the same name in its tests; `options.gemini_live_voices` key matches between Task 2 (produce) and Task 6 (consume); `voice-sample/{voice}` route path matches between Task 3 (define) and Task 6 (fetch). ✓
