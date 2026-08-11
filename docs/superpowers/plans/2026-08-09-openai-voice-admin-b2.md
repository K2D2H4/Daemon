# OpenAI Voice in the Admin (Phase B2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the owner choose the voice provider (gemini | openai) in the admin, pick that provider's voice with a `▶` preview, and set the OpenAI realtime model — all from the web, without breaking the working Gemini voice path.

**Architecture:** Surface `voice_provider` / `openai_realtime_voice` / `openai_realtime_model` through the existing admin settings API (`settings_io.STR_FIELDS` + `options`). Make the admin's voice picker provider-aware in the rewritten v0.1.37 `index.html` (a `voice_provider` select drives which voice select + preview render, in a replaceable `#voice-field`). Namespace the preview route `/{provider}/{voice}` and move the 30 committed Gemini clips under `voice-samples/gemini/`; add `voice-samples/openai/`. Extend the offline generator to two provider passes.

**Tech Stack:** Python 3.13, FastAPI, `fastapi.testclient`, vanilla JS in one self-contained `index.html`, `ffmpeg` (generator only), the B1 `OpenAIRealtimeSession`.

## Global Constraints

- **DO NOT REGRESS GEMINI LIVE.** The Gemini voice path (Phase A, shipped v0.1.20) must behave exactly as before: `voice_provider` defaults to `gemini`; the 30-voice dropdown still lists/selects/PATCHes `gemini_live_voice`; its `▶` preview still plays — now via `/voice-sample/gemini/<voice>` from `voice-samples/gemini/`. A route change and the moved clips land in the SAME commit. Task 2 includes an explicit Gemini-serving regression test; Task 5 drives it live.
- **Layering / validate-before-write unchanged:** new fields go through `apply_patch`'s construct-a-candidate-`Settings`-first path; `config.py` is not imported into the admin JS. The B1 config already validates `voice_provider`, `openai_realtime_voice`, and the provider-aware model/key requirements.
- **Admin stays offline / self-contained:** no CDN; preview plays only local `/admin/api/voice-sample/...` clips; the graceful "no preview available yet" fallback (v0.1.20) stays.
- **Route safety:** `{provider}` and `{voice}` are validated against fixed sets BEFORE any filesystem path is built (closes traversal), returning 404 for anything not served.
- **Voices:** Gemini `GEMINI_LIVE_VOICES` (30), OpenAI `OPENAI_REALTIME_VOICES` (10) — both from `daemon.config`. `VOICE_PROVIDERS = ("gemini","openai")`.
- **No unit test touches network/key/mic/speaker.** The generator + live clips are the key-owner's manual `evals/` run (OPENAI_API_KEY is available for the Task 5 live step).
- **Preview phrase:** `Hi, I'm Daemon. This is what I sound like.` (English), same as Gemini's.

---

### Task 1: settings_io — surface `voice_provider` / `openai_realtime_voice` / `openai_realtime_model`

**Files:**
- Modify: `daemon/admin/settings_io.py` (import ~line 28-38; `STR_FIELDS` line 43-63; `options` block line 159-166)
- Test: `tests/test_admin.py`

**Interfaces:**
- Consumes: `daemon.config.VOICE_PROVIDERS`, `OPENAI_REALTIME_VOICES` (both exist from B1).
- Produces: GET `/admin/api/settings` `editable.{voice_provider,openai_realtime_voice,openai_realtime_model}` and `options.{voice_providers, openai_realtime_voices}`; PATCH accepts all three.

- [ ] **Step 1: Write the failing tests**

In `tests/test_admin.py`, beside the existing `test_patch_sets_the_gemini_live_voice` (line 388):

```python
def test_patch_sets_the_openai_voice_provider_and_voice(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=offline\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    resp = client.patch("/admin/api/settings", json={
        "voice_provider": "openai",
        "openai_realtime_voice": "alloy",
        "openai_realtime_model": "gpt-realtime",
    })
    assert resp.status_code == 200
    text = env.read_text(encoding="utf-8")
    assert "DAEMON_VOICE_PROVIDER=openai" in text
    assert "DAEMON_OPENAI_REALTIME_VOICE=alloy" in text
    assert "DAEMON_OPENAI_REALTIME_MODEL=gpt-realtime" in text

    got = client.get("/admin/api/settings").json()
    assert "openai" in got["options"]["voice_providers"] and "gemini" in got["options"]["voice_providers"]
    assert got["options"]["openai_realtime_voices"][0] == ""  # server-default offered first
    assert "alloy" in got["options"]["openai_realtime_voices"]


def test_patch_rejects_an_unknown_voice_provider_and_openai_voice(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    original = "DAEMON_PRESET=offline\n"
    env.write_text(original, encoding="utf-8")
    app = create_app(_settings(tmp_path))
    app.state.env_path = env
    client = TestClient(app, base_url=LOOPBACK)

    assert client.patch("/admin/api/settings", json={"voice_provider": "anthropic"}).status_code == 400
    assert client.patch("/admin/api/settings", json={"openai_realtime_voice": "nope"}).status_code == 400
    assert env.read_text(encoding="utf-8") == original, "a rejected patch still wrote"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_admin.py -k "openai_voice_provider or unknown_voice_provider" -v`
Expected: FAIL — the fields are not editable (PATCH 400 "not editable"), and `options` lacks the new lists.

- [ ] **Step 3: Wire settings_io**

Extend the `from daemon.config import (...)` block (near line 28) to also import `OPENAI_REALTIME_VOICES` and `VOICE_PROVIDERS` (it already imports `GEMINI_LIVE_VOICES`, `HOSTED_PROVIDERS`, `PRESETS`, `SENSITIVITIES`, `Settings`).

Add to `STR_FIELDS` (after `gemini_live_model`, line 57):

```python
    # Voice provider is its own axis (config.py): gemini or openai. Its own realtime
    # model + voice, exactly like the gemini pair, so voice is fully web-configurable.
    "voice_provider": "DAEMON_VOICE_PROVIDER",
    "openai_realtime_model": "DAEMON_OPENAI_REALTIME_MODEL",
    "openai_realtime_voice": "DAEMON_OPENAI_REALTIME_VOICE",
```

Add to the `options` block (after `gemini_live_voices`, line 163):

```python
            "voice_providers": list(VOICE_PROVIDERS),
            "openai_realtime_voices": ["", *sorted(OPENAI_REALTIME_VOICES)],
```

(No other change: `current_settings_payload` iterates `STR_FIELDS` for `editable`, and `EDITABLE` includes `STR_FIELDS`, so PATCH accepts the three fields with validate-before-write intact.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_admin.py -q` then `python3 -m pytest -q`
Expected: PASS (the whole suite — `voice_provider` defaults to gemini, so nothing else changes).

- [ ] **Step 5: Commit**

```bash
git add daemon/admin/settings_io.py tests/test_admin.py
git commit -m "admin: offer voice_provider + openai realtime model/voice in the settings API"
```

---

### Task 2: routes — provider-namespaced preview route (+ move the Gemini clips)

**Files:**
- Modify: `daemon/admin/routes.py` (the `voice_sample` handler at line 258; the docstring endpoint line 11; import of `GEMINI_LIVE_VOICES` line 64)
- Move: `daemon/admin/static/voice-samples/*.mp3` → `daemon/admin/static/voice-samples/gemini/`
- Test: `tests/test_admin.py` (the two `voice_sample` tests at line 446, 465)

**Interfaces:**
- Consumes: `daemon.config.GEMINI_LIVE_VOICES`, `OPENAI_REALTIME_VOICES`, `VOICE_PROVIDERS`.
- Produces: `GET /admin/api/voice-sample/{provider}/{voice}` → `audio/mpeg` or 404; clips live at `VOICE_SAMPLES/<provider>/<voice>.mp3`.

- [ ] **Step 1: Rewrite the tests for the namespaced route (incl. the Gemini no-regression case)**

Replace the two existing `voice_sample` tests (lines 446-476) with:

```python
def _voice_allowlists():
    from daemon.config import GEMINI_LIVE_VOICES, OPENAI_REALTIME_VOICES
    return GEMINI_LIVE_VOICES, OPENAI_REALTIME_VOICES


def test_voice_sample_serves_both_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = tmp_path / "voice-samples"
    (samples / "gemini").mkdir(parents=True)
    (samples / "openai").mkdir(parents=True)
    (samples / "gemini" / "Kore.mp3").write_bytes(b"ID3-gemini")
    (samples / "openai" / "alloy.mp3").write_bytes(b"ID3-openai")
    monkeypatch.setattr("daemon.admin.routes.VOICE_SAMPLES", samples)

    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)

    g = client.get("/admin/api/voice-sample/gemini/Kore")   # Gemini no-regression
    assert g.status_code == 200 and g.headers["content-type"] == "audio/mpeg"
    assert g.content == b"ID3-gemini"
    o = client.get("/admin/api/voice-sample/openai/alloy")
    assert o.status_code == 200 and o.content == b"ID3-openai"


def test_voice_sample_404_for_unknown_provider_voice_or_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples = tmp_path / "voice-samples"
    (samples / "gemini").mkdir(parents=True)
    monkeypatch.setattr("daemon.admin.routes.VOICE_SAMPLES", samples)

    app = create_app(_settings(tmp_path))
    client = TestClient(app, base_url=LOOPBACK)

    assert client.get("/admin/api/voice-sample/gemini/Kore").status_code == 404   # known, no file
    assert client.get("/admin/api/voice-sample/gemini/Nope").status_code == 404   # unknown voice
    assert client.get("/admin/api/voice-sample/openai/Kore").status_code == 404   # wrong provider's voice
    assert client.get("/admin/api/voice-sample/nope/Kore").status_code == 404     # unknown provider
    assert client.get("/admin/api/voice-sample/gemini/..%2f..%2fetc%2fpasswd").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_admin.py -k voice_sample -v`
Expected: FAIL — the route is still `/{voice}` (the new `/{provider}/{voice}` paths 404 as unrouted, and the moved-clip layout does not exist yet).

- [ ] **Step 3: Move the Gemini clips**

```bash
mkdir -p daemon/admin/static/voice-samples/gemini
git mv daemon/admin/static/voice-samples/*.mp3 daemon/admin/static/voice-samples/gemini/
```

- [ ] **Step 4: Rewrite the route**

In `daemon/admin/routes.py`, extend the config import (line 64) to `from daemon.config import GEMINI_LIVE_VOICES, OPENAI_REALTIME_VOICES, VOICE_PROVIDERS`. Replace the `voice_sample` handler (line 258) with:

```python
_VOICE_ALLOWLISTS = {"gemini": GEMINI_LIVE_VOICES, "openai": OPENAI_REALTIME_VOICES}


@router.get("/api/voice-sample/{provider}/{voice}")
async def voice_sample(provider: str, voice: str) -> Response:
    """A short preview clip for one voice, as audio/mpeg.

    Both `provider` and `voice` are checked against fixed sets BEFORE any path is
    built, so an unknown provider/voice or a traversal name can never resolve a file.
    A known voice whose clip was not generated is a 404, not a 500 - a missing asset
    degrades to 'no preview'. Clips live under voice-samples/<provider>/<voice>.mp3."""
    allow = _VOICE_ALLOWLISTS.get(provider)
    if allow is None or voice not in allow:
        return JSONResponse({"detail": f"no such voice {provider}/{voice}"}, status_code=404)
    path = VOICE_SAMPLES / provider / f"{voice}.mp3"
    if not path.is_file():
        return JSONResponse(
            {"detail": f"no preview generated for {provider}/{voice}"}, status_code=404
        )
    return Response(path.read_bytes(), media_type="audio/mpeg")
```

Update the docstring endpoint inventory line (line 11) to:
`    GET   /admin/api/voice-sample/{provider}/{voice}  a voice preview clip (audio/mpeg), allowlist-gated`

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_admin.py -k voice_sample -v` then `python3 -m pytest -q` and `python3 -m ruff check .`
Expected: PASS. (The Gemini-serving test proves the moved clips + new route keep Gemini preview working.)

- [ ] **Step 6: Commit**

```bash
git add daemon/admin/routes.py tests/test_admin.py daemon/admin/static/voice-samples/
git commit -m "admin: namespace the voice-sample route /{provider}/{voice}; move Gemini clips under gemini/"
```

---

### Task 3: index.html — provider-aware voice UI + `openai_realtime_model`

**Files:**
- Modify: `daemon/admin/static/index.html` (`fieldVoice` at line 1039-1044; `renderSettings` MODEL card ~1054-1060 and VOICE card ~1063-1069; the `change`/`click` handlers ~1119-1131)
- Test: `tests/test_admin.py` (the shell-renders test)

**Interfaces:**
- Consumes: GET `/settings` `editable.{voice_provider,openai_realtime_voice,openai_realtime_model}` + `options.{voice_providers,openai_realtime_voices}` (Task 1); `GET /admin/api/voice-sample/{provider}/{voice}` (Task 2).

- [ ] **Step 1: Replace `fieldVoice` with a provider-aware `voiceField`**

In `daemon/admin/static/index.html`, replace the `fieldVoice(running,opts)` function (line 1039-1044) with:

```js
// Provider-aware voice picker. Renders the SELECTED provider's voice select + preview
// inside a replaceable #voice-field, so changing the provider re-renders only this.
function voiceField(provider,e,o){
  const isO=provider==='openai';
  const name=isO?'openai_realtime_voice':'gemini_live_voice';
  const opts=(isO?o.openai_realtime_voices:o.gemini_live_voices)||[''];
  const v=val(name,e[name]);
  const os=opts.map(x=>`<option ${x===v?'selected':''}>${esc(x)}</option>`).join('');
  return `<div class="field" id="voice-field">${nameCell(name)}<select data-f="${name}">${os}</select>
    <div class="voice-preview"><button type="button" class="btn" id="voice-play">▶ preview</button>
    <span class="cap" id="voice-preview-msg"></span><audio id="voice-sample" preload="none"></audio></div></div>`;
}
```

- [ ] **Step 2: Render the provider select + voice field, and add `openai_realtime_model`**

In `renderSettings`, in the MODEL card (line 1059), add `openai_realtime_model` beside `gemini_live_model`:

```js
    +fieldStr('gemini_live_model',e.gemini_live_model)
    +fieldStr('openai_realtime_model',e.openai_realtime_model)+'</div>'
```

In the VOICE card (line 1065), replace `+fieldVoice(e.gemini_live_voice,o.gemini_live_voices)` with the provider select + the provider-aware voice field:

```js
    +fieldStr('voice_provider',e.voice_provider,o.voice_providers)
    +voiceField(val('voice_provider',e.voice_provider),e,o)
```

- [ ] **Step 3: Re-render the voice field on provider change; fix the preview handler**

Add a `change` listener (near the existing `$('#settings-form').addEventListener('change',updateDirty);`, line 1119) that re-renders only the voice field when the provider changes, then recomputes dirty:

```js
$('#settings-form').addEventListener('change',ev=>{
  if(ev.target.dataset && ev.target.dataset.f==='voice_provider'){
    const vf=$('#voice-field');
    if(vf){vf.outerHTML=voiceField(ev.target.value,SETTINGS.editable,SETTINGS.options);updateDirty();}
  }
});
```

Update the `voice-play` click handler (lines 1123-1130) to read the current provider + the active voice select and hit the namespaced route:

```js
  if(ev.target.id!=='voice-play')return;
  const prov=$('[data-f="voice_provider"]');
  const sel=$('#voice-field select');
  const msg=$('#voice-preview-msg');
  if(!prov||!sel||!sel.value)return;   // empty "(default)" option has no clip
  const a=$('#voice-sample');
  if(msg)msg.textContent='';
  a.src='/admin/api/voice-sample/'+encodeURIComponent(prov.value)+'/'+encodeURIComponent(sel.value);
  a.play().catch(()=>{if(msg)msg.textContent='no preview available yet';});
```

(`collectPatch` is unchanged: it scans `[data-f]`, so it picks up `voice_provider` and whichever voice select is currently rendered, leaving the other provider's saved voice untouched.)

- [ ] **Step 4: Verify the shell still renders offline**

Run: `python3 -m pytest tests/test_admin.py -k shell -v`
Expected: PASS (still self-contained, no CDN).

- [ ] **Step 5: Commit**

```bash
git add daemon/admin/static/index.html
git commit -m "admin ui: voice_provider dropdown drives the voice picker + preview; openai_realtime_model field"
```

---

### Task 4: generator — two provider passes

**Files:**
- Modify: `evals/gen_voice_samples.py`

**Interfaces:**
- Consumes: `GEMINI_LIVE_VOICES`, `OPENAI_REALTIME_VOICES` (config), `GeminiLiveSession`, `OpenAIRealtimeSession`; env `GEMINI_API_KEY`+`DAEMON_GEMINI_LIVE_MODEL`, `OPENAI_API_KEY`+`DAEMON_OPENAI_REALTIME_MODEL`; `ffmpeg`.
- Produces: `daemon/admin/static/voice-samples/{gemini,openai}/<voice>.mp3`.

No unit test (evals hit the live API, never in CI — tests/CLAUDE.md). Verified by the live run in Task 5.

- [ ] **Step 1: Extend the generator to two passes**

In `evals/gen_voice_samples.py`: import `OPENAI_REALTIME_VOICES` from `daemon.config` and `OpenAIRealtimeSession` from `daemon.voice.openai_realtime`. Change `OUT` to a base dir and write per provider. Add an OpenAI capture mirroring the Gemini one, and make `main()` run both passes, each skipped (with a log) when its env is absent:

```python
OUT = Path(__file__).resolve().parents[1] / "daemon" / "admin" / "static" / "voice-samples"

async def _capture_openai(api_key: str, model: str, voice: str) -> bytes:
    pcm = bytearray()
    async with OpenAIRealtimeSession(
        api_key=api_key, model=model, system_instruction=READ_VERBATIM, voice_name=voice,
    ) as session:
        await session.send_text(PHRASE)
        async for item in session.receive():
            if isinstance(item, bytes):
                pcm += item
    return bytes(pcm)

async def _run_pass(provider, voices, capture, key, model) -> int:
    if not key or not model:
        print(f"skip {provider}: set its API key + realtime model", file=sys.stderr); return 0
    dest_dir = OUT / provider; dest_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for voice in sorted(voices):
        try:
            pcm = await capture(key, model, voice)
        except Exception as exc:  # noqa: BLE001 - report and continue the batch
            failures += 1; print(f"! {provider}/{voice}: {exc}", file=sys.stderr); continue
        if not pcm:
            failures += 1; print(f"! {provider}/{voice}: no audio", file=sys.stderr); continue
        _to_mp3(pcm, dest_dir / f"{voice}.mp3")
        print(f"OK {provider}/{voice} ({len(pcm)} bytes pcm)")
    return failures
```

Rewrite `main()` to call `_run_pass("gemini", GEMINI_LIVE_VOICES, _capture, os.environ.get("GEMINI_API_KEY",""), os.environ.get("DAEMON_GEMINI_LIVE_MODEL",""))` and `_run_pass("openai", OPENAI_REALTIME_VOICES, _capture_openai, os.environ.get("OPENAI_API_KEY",""), os.environ.get("DAEMON_OPENAI_REALTIME_MODEL",""))`, summing failures for the exit code. Keep the existing Gemini `_capture` and `_to_mp3` and the `READ_VERBATIM`/`PHRASE` constants. Update the module docstring to say it writes `<provider>/<voice>.mp3` for both providers.

- [ ] **Step 2: Sanity-check imports + the no-env path (no network)**

Run: `env -u GEMINI_API_KEY -u DAEMON_GEMINI_LIVE_MODEL -u OPENAI_API_KEY -u DAEMON_OPENAI_REALTIME_MODEL python3 -m evals.gen_voice_samples`
Expected: prints "skip gemini…" and "skip openai…", exits 0, no network. Confirms the module imports and both passes guard correctly.

- [ ] **Step 3: Lint + commit (assets come in Task 5 after the real run)**

Run: `python3 -m ruff check .`
```bash
git add evals/gen_voice_samples.py
git commit -m "evals: gen_voice_samples generates both providers into voice-samples/<provider>/"
```

---

### Task 5: live generation + full verification

**Files:**
- Create (assets): `daemon/admin/static/voice-samples/gemini/*.mp3` (regenerated), `daemon/admin/static/voice-samples/openai/*.mp3` (new, 10)

- [ ] **Step 1: Full keyless gates**

Run:
```bash
python3 -m pytest
python3 -m ruff check .
python3 scripts/check_docs.py
python3 -m pytest tests/test_reachable.py tests/test_acceptance.py -q
```
Expected: all pass.

- [ ] **Step 2: Generate the real clips (needs the keys + ffmpeg)**

Run (reads the keys from the environment / `.env`):
```bash
OPENAI_API_KEY=… DAEMON_OPENAI_REALTIME_MODEL=gpt-realtime \
GEMINI_API_KEY=… DAEMON_GEMINI_LIVE_MODEL=… \
python3 -m evals.gen_voice_samples
```
Expected: `OK gemini/<Voice>` for the 30 and `OK openai/<voice>` for the 10; `voice-samples/gemini/` (30) and `voice-samples/openai/` (10) populated. Spot-listen one of each.

- [ ] **Step 3: Live browser QA — Gemini no-regression FIRST, then OpenAI**

Boot the admin (uvicorn against `create_app` with an offline settings + the OpenAI/Gemini keys in `.env`) and drive it in a browser:
- **Gemini no-regression (load-bearing):** provider=gemini → the voice dropdown lists the 30, selecting + Save keeps the value, `▶` plays a clip from `/voice-sample/gemini/<voice>`.
- OpenAI: switch provider to openai → the voice field re-renders to the 10 OpenAI voices; `▶` plays from `/voice-sample/openai/<voice>`; `openai_realtime_model` is editable and persists to `.env`.
- Switching provider back and forth re-renders the voice row correctly; no console errors; the save-stays-on-screen behavior (v0.1.20) still holds.

- [ ] **Step 4: Commit the assets**

```bash
git add daemon/admin/static/voice-samples/
git commit -m "admin assets: bundled voice-preview clips — gemini/ (30) + openai/ (10)"
```

---

## Self-Review

**Spec coverage:**
- `voice_provider` + `openai_realtime_voice` + `openai_realtime_model` in the settings API → Task 1. ✓
- Namespaced `/{provider}/{voice}` route + moved Gemini clips (no-regression) → Task 2. ✓
- Provider-aware voice UI + `openai_realtime_model` field → Task 3. ✓
- Generator two-pass → Task 4; live clips (gemini regenerated + openai 10) → Task 5. ✓
- Gemini no-regression: default provider gemini (Task 1), Gemini-serving route test (Task 2), load-bearing live check (Task 5). ✓
- Tests keyless with the fake TestClient; live generation + browser QA deferred to Task 5 (key available). ✓
- Out of scope (turn_detection knobs, transcription-model choice) → not in any task. ✓

**Placeholder scan:** No TBD/TODO; every code/test step carries real content. The Task 5 live steps are genuine manual verification (the key is available), not skipped work.

**Type/name consistency:** `voiceField(provider,e,o)` (Task 3) matches the `data-f="voice_provider"` select rendered by `fieldStr('voice_provider',…)` (Task 3) and the `voice_providers`/`openai_realtime_voices` options (Task 1); the route `/{provider}/{voice}` (Task 2) matches the UI fetch (Task 3) and `VOICE_SAMPLES/<provider>/<voice>.mp3` (Tasks 2, 4, 5); `_VOICE_ALLOWLISTS` keys (`gemini`,`openai`) match `VOICE_PROVIDERS`. STR_FIELDS keys match the `editable.*` the UI reads.
