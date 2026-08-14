# Provider-first everywhere — collapse presets into an axis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the preset concept from the codebase, replacing it with two orthogonal axes (`DAEMON_PROVIDER` + `DAEMON_PROACTIVE_JUDGE_LOCAL`), and make both the admin and the wizard ask "provider first, then that provider's model(s)".

**Architecture:** `Settings.routing` becomes computed from the two axes instead of looked up in a `PRESETS` table. The wizard and admin, which read the old preset/hosted-provider surface, are reshaped in lockstep — the rename atomically breaks both, so this is one PR staged config → wizard → admin → docs. No migration (single-owner install); a leftover `DAEMON_PRESET` raises loudly.

**Tech Stack:** Python 3.13, pydantic-settings, FastAPI, pytest, ruff. Front-end is one hand-written `daemon/admin/static/index.html` (vanilla JS, no build).

**Spec:** `docs/design/2026-08-13-provider-first-collapse-presets-design.md` — read it first; this plan implements its decisions D1–D10.

## Global Constraints

- `ruff` line-length **100**, target `py313`, lint rules `E,F,I,UP,B,ASYNC`.
- Provider option values are env names: `anthropic` / `openai` / `gemini` / `openai_compatible` / `ollama` — never brand names. The admin's posture is "the screen says what `.env` says".
- The two axes reconstruct the old presets with **zero routing change**: `ollama`→all-local(=offline); hosted+`proactive_judge_local=true`→balanced; hosted+`false`→quality. `EMBED` is always `ollama`. `CHAT_VOICE` stays keyed off `voice_enabled`→`voice_provider` (ADR 0012, untouched).
- The toggle controls **`PROACTIVE_JUDGE` only** — reflection does not move.
- No migration code. `DAEMON_HOSTED_PROVIDER` is renamed to `DAEMON_PROVIDER` and the old key is **not** read. A raw `DAEMON_PRESET` in the env **raises `ConfigError`** at construction.
- The admin must never call the sync `check_*` probes directly inside an `async def` handler — always `asyncio.to_thread`. `HTTP_TIMEOUT` is 15s; a blocking call in the loop freezes the whole daemon.
- Admin UI is English-only (Silkscreen has no Hangul). The one Korean caption in the spec mock is illustrative — write the shipped captions in English.
- Full gate before each commit: `python3 -m pytest`, `python3 -m ruff check .`, `python3 scripts/check_docs.py`.
- This repo's rule: nothing is "working" until the real path runs. Final verification drives `daemon setup`, `daemon doctor`, and the live admin page — not only the suite.

## File Structure

| File | Responsibility in this change |
|---|---|
| `daemon/config.py` | delete `PRESETS`/`HOSTED`/`preset_providers`/`preset` field; add `provider` + `proactive_judge_local`; rename `hosted_provider`→`provider`; compute `routing`; stale-preset guard; `MODEL_SUGGESTIONS`; reshape `providers_for` |
| `daemon/setup.py` | drop `PRESET_CHOICES`/`_choose_preset`; add provider-first `_choose_provider`; thread the two axes through `needs_for`; keep `_choose_compatible_endpoint` |
| `daemon/admin/settings_io.py` | `preset` out, `hosted_provider`→`provider` (accepts `ollama`), add `ollama_model`, `options.model_suggestions`, compute the D9 note |
| `daemon/admin/routes.py` | `get_settings` awaits `_live_model_lists` (threadpool probe) → `options.model_lists` |
| `daemon/admin/static/index.html` | `brainField()` (provider-first MODEL card), `fieldModel()` (datalist, prefers live list), compat two-field, `brainPatch()`, D9 note |
| `docs/adr/0014-*.md`, `docs/PLAN.md`, `docs/ARCHITECTURE.md`, `.env.example`, `daemon/CLAUDE.md` | ADR; the preset sweep |
| `tests/test_config.py`, `tests/test_setup.py`, `tests/test_admin.py`, `tests/test_acceptance.py`, `tests/test_wake.py` + any other `preset` user | update to the two axes |

---

### Task 1: config core — the two axes replace PRESETS

**Files:**
- Modify: `daemon/config.py` (fields ~299–330; `_check` ~880–905; `routing` ~1164–1180; `preset_providers`/`providers_for` ~219–271; `PRESETS`/`HOSTED` ~41–176; `provider_model`, `active_tasks`, `route_for`)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Settings.provider: str` (alias `DAEMON_PROVIDER`, default `""`), `Settings.proactive_judge_local: bool` (alias `DAEMON_PROACTIVE_JUDGE_LOCAL`, default `True`).
  - `Settings.routing -> dict[Task, str]` computed (table below).
  - `providers_for(*, provider: str, proactive_judge_local: bool, voice_enabled: bool, voice_provider: str) -> list[str]` — **no `preset`, no `hosted`**.
  - `config.MODEL_SUGGESTIONS: dict[str, tuple[str, ...]]` keyed by provider name.
  - `PRESETS`, `HOSTED`, `preset_providers`, `Settings.preset`, `Settings.hosted_provider`, `DEFAULT_PRESET`, `DEFAULT_HOSTED_PROVIDER` **no longer exist**.

Read the current `daemon/config.py` in full before editing — the anchors above drift.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_config.py`'s preset-routing section (the `test_offline_preset_*`, `test_balanced_preset_*`, `test_quality_preset_*`, `test_every_task_is_routed_by_every_preset_except_voice` tests and the `make_settings` `hosted_provider` default) with the two-axis contract. Update `make_settings` first:

```python
def make_settings(**kwargs: Any) -> Settings:
    """A provider is supplied unless a test is about its absence (DAEMON_PROVIDER
    has no default; a hosted task with no provider fails at startup, ADR 0007/0014)."""
    kwargs.setdefault("provider", "anthropic")
    return Settings(_env_file=None, **kwargs)
```

Then the routing tests:

```python
from daemon.config import ConfigError, Route, Settings, providers_for
from daemon.tasks import Task


def _hosted_tasks() -> set[Task]:
    return {Task.CHAT_TEXT, Task.RECALL_ESCALATION, Task.REFLECTION, Task.PERSONA_RULE}


def test_ollama_provider_routes_everything_local() -> None:
    # Reconstructs the old `offline` preset exactly.
    settings = make_settings(provider="ollama")
    assert settings.routing == {
        Task.CHAT_TEXT: "ollama", Task.RECALL_ESCALATION: "ollama",
        Task.PROACTIVE_JUDGE: "ollama", Task.REFLECTION: "ollama",
        Task.PERSONA_RULE: "ollama", Task.EMBED: "ollama",
    }


def test_hosted_with_local_judge_reconstructs_balanced() -> None:
    settings = make_settings(provider="anthropic", proactive_judge_local=True, anthropic_model="c")
    assert settings.routing[Task.PROACTIVE_JUDGE] == "ollama"
    for task in _hosted_tasks():
        assert settings.routing[task] == "anthropic"
    assert settings.routing[Task.EMBED] == "ollama"


def test_hosted_with_hosted_judge_reconstructs_quality() -> None:
    settings = make_settings(provider="gemini", proactive_judge_local=False, gemini_model="g")
    assert settings.routing[Task.PROACTIVE_JUDGE] == "gemini"


def test_embed_is_always_ollama() -> None:
    for provider in ("ollama", "anthropic", "openai", "gemini", "openai_compatible"):
        s = Settings(_env_file=None, provider=provider,
                     anthropic_model="c", openai_model="o", gemini_model="g",
                     openai_compatible_model="q",
                     openai_compatible_base_url="https://x/v1")
        assert s.routing[Task.EMBED] == "ollama"


def test_a_stale_preset_key_is_refused_loudly() -> None:
    with pytest.raises(ConfigError, match="DAEMON_PRESET"):
        Settings(_env_file=None, DAEMON_PRESET="balanced", DAEMON_PROVIDER="anthropic",
                 DAEMON_ANTHROPIC_MODEL="c")


def test_a_hosted_provider_without_one_says_to_run_setup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_PROVIDER"):
        Settings(_env_file=None)  # provider "" and a hosted task routed


def test_providers_for_uses_the_two_axes_not_a_preset() -> None:
    # hosted chat + local judge: only the chat provider's key is needed.
    assert providers_for(provider="anthropic", proactive_judge_local=True,
                         voice_enabled=False, voice_provider="gemini") == ["anthropic"]
    # ollama: no hosted key at all.
    assert providers_for(provider="ollama", proactive_judge_local=True,
                         voice_enabled=False, voice_provider="gemini") == ["ollama"]
    # voice on adds the voice provider (ADR 0012 behaviour, unchanged).
    got = providers_for(provider="ollama", proactive_judge_local=True,
                       voice_enabled=True, voice_provider="openai")
    assert "openai" in got
```

Keep the existing voice-axis tests from #85 (`test_the_offline_preset_can_have_hosted_voice` etc.) but rename `preset="offline"` → `provider="ollama"` and drop the `preset=` kwarg. Keep `test_voice_off_routes_no_voice_task_under_any_preset` as a loop over providers (`for provider in ("ollama","anthropic","gemini")`).

- [ ] **Step 2: Run the tests — verify they fail**

Run: `python3 -m pytest tests/test_config.py -q`
Expected: failures at `Settings(... provider=...)` (unexpected kwarg) and `providers_for(provider=...)` (unexpected kwarg) — the new API does not exist yet.

- [ ] **Step 3: Delete the preset machinery, add the two axes**

In `daemon/config.py`:
- Delete `HOSTED = "hosted"` and its docstring, the `PRESETS` dict, `preset_providers`, `DEFAULT_PRESET`, `DEFAULT_HOSTED_PROVIDER`. Keep `VOICE_TASKS`, `HOSTED_PROVIDERS`, `PROVIDER_KEY_ENV`.
- Delete the `preset` and `hosted_provider` fields. Add:

```python
    provider: str = Field(default="", alias="DAEMON_PROVIDER")
    """Which provider answers conversation, recall, reflection and persona rules -
    `ollama` for fully local, or one of the hosted names. Empty by default: a hosted
    task with no provider fails at startup pointing at `daemon setup`, rather than
    quietly becoming Claude (ADR 0007, carried into ADR 0014). `ollama` here means the
    old `offline` preset - every hosted role resolves local. Per-task `route_overrides`
    still win over this."""

    proactive_judge_local: bool = Field(default=True, alias="DAEMON_PROACTIVE_JUDGE_LOCAL")
    """Run PROACTIVE_JUDGE on the local model instead of the provider. True by default
    (the old `balanced`): the judge fires on a 5-minute tick, so hosted cost accrues
    whether or not it ever speaks. False is the old `quality`. Ignored when `provider`
    is `ollama` - everything is local then anyway. It moves this one task and no other;
    REFLECTION stays on the provider regardless (that was the only balanced/quality
    difference)."""
```

- Rewrite `routing`:

```python
    # Fixed task -> role. "provider" means DAEMON_PROVIDER; "judge" is local unless the
    # toggle says otherwise; embed is always local; voice is its own axis (ADR 0012).
    _HOSTED_ROLE_TASKS = (
        Task.CHAT_TEXT, Task.RECALL_ESCALATION, Task.REFLECTION, Task.PERSONA_RULE,
    )

    @property
    def routing(self) -> dict[Task, str]:
        """Effective Task -> provider name, computed from the two axes (ADR 0014),
        then explicit overrides on top."""
        resolved: dict[Task, str] = {task: self.provider for task in self._HOSTED_ROLE_TASKS}
        resolved[Task.PROACTIVE_JUDGE] = (
            OLLAMA if (self.proactive_judge_local or self.provider == OLLAMA) else self.provider
        )
        resolved[Task.EMBED] = OLLAMA
        # Voice: added off DAEMON_VOICE_ENABLED, mapped to DAEMON_VOICE_PROVIDER (ADR 0012).
        if self.voice_enabled:
            resolved[Task.CHAT_VOICE] = self.voice_provider
        return {**resolved, **self.route_overrides}
```

(Confirm `OLLAMA` is the module constant name for the string `"ollama"` — it is used in `PROVIDER_KEY_ENV`. Import/reference it as the file already does.)

- Rewrite `_check`'s preset block (the `if self.preset not in PRESETS` … `elif self.hosted_provider not in HOSTED_PROVIDERS` span) as:

```python
        if "DAEMON_PRESET" in self._raw_env_keys():
            raise ConfigError(
                "DAEMON_PRESET has been removed. Set DAEMON_PROVIDER (ollama | "
                f"{' | '.join(HOSTED_PROVIDERS)}) and DAEMON_PROACTIVE_JUDGE_LOCAL "
                "instead - see docs/adr/0014-provider-is-the-axis.md."
            )
        if self.provider and self.provider != OLLAMA and self.provider not in HOSTED_PROVIDERS:
            raise ConfigError(
                f"unknown DAEMON_PROVIDER {self.provider!r}; expected ollama or one of "
                f"{', '.join(HOSTED_PROVIDERS)}"
            )
        if not self.provider and self._routes_a_hosted_task():
            raise ConfigError(
                "a hosted task is routed but DAEMON_PROVIDER is empty; run `daemon setup` "
                f"to choose one (ollama | {' | '.join(HOSTED_PROVIDERS)})"
            )
```

For the stale-preset guard, `Settings` needs to see the raw env keys. pydantic-settings drops unknown keys by default. Simplest robust approach: add `DAEMON_PRESET` as an explicit ignored field and check it — `_stale_preset: str = Field(default="", alias="DAEMON_PRESET")` — then the guard is `if self._stale_preset:`. Use that instead of `_raw_env_keys()` (which does not exist). Rewrite the guard to `if self._stale_preset:`. `_routes_a_hosted_task()` is a small helper: `return any(self.routing.get(t) not in (None, OLLAMA) for t in self._HOSTED_ROLE_TASKS)` — but `routing` reads `provider`; when `provider==""`, the hosted-role tasks map to `""`, which is neither None nor OLLAMA, so this returns True. Good: empty provider + hosted role = error. When `provider=="ollama"`, all roles are OLLAMA → False → no error.

- `provider_model`, `active_tasks`, `route_for`: these read `self.routing` and `getattr(self, f"{provider}_model")`. They do **not** reference `preset`/`hosted_provider` directly except in messages. Grep for `self.preset`, `self.hosted_provider`, `PRESETS`, `HOSTED`, `preset` in error strings and fix each. `route_for`'s `f"preset {self.preset!r} does not route {task.value}"` becomes `f"no route for {task.value}"`.

- Add `providers_for` reshaped:

```python
def providers_for(
    *,
    provider: str,
    proactive_judge_local: bool,
    voice_enabled: bool,
    voice_provider: str,
) -> list[str]:
    """Providers onboarding must ask keys for, from the two axes. `ollama` and `""`
    contribute no hosted key. Voice contributes `voice_provider` only when voice is on
    (ADR 0012). `proactive_judge_local` is accepted for symmetry with the axes even
    though every hosted role already implies the provider's key - it changes nothing
    about which *keys* are needed, only which model runs the judge."""
    providers: set[str] = set()
    if provider and provider != OLLAMA:
        providers.add(provider)
    elif provider == OLLAMA:
        providers.add(OLLAMA)
    if voice_enabled:
        providers.add(voice_provider)
    return sorted(providers)
```

- Add near the other enumerated constants:

```python
MODEL_SUGGESTIONS: dict[str, tuple[str, ...]] = {
    "ollama": ("qwen3:14b", "gemma3:4b"),
    "anthropic": ("claude-opus-5", "claude-sonnet-5"),
    "openai": ("gpt-5.2", "gpt-5.1"),
    "gemini": ("gemini-3.6-flash", "gemini-3.1-pro-preview"),
}
"""Datalist suggestions for the admin's chat-model field, newest first. NOT VALIDATED -
unlike PRESETS or GEMINI_LIVE_VOICES, this is not a constraint: a model id absent here
must still save (Settings validates model ids only as non-empty), because a just-released
id would otherwise be unselectable. Do not add a membership check."""
```

- [ ] **Step 4: Run the tests — verify they pass**

Run: `python3 -m pytest tests/test_config.py -q`
Expected: PASS. Then the whole suite `python3 -m pytest -q` will still fail in `test_setup.py`/`test_admin.py`/`test_acceptance.py` (they pass `preset=`/`hosted_provider=`) — that is Tasks 2–3's job. Confirm `tests/test_config.py`, `tests/test_wake.py` are green and `ruff check daemon/config.py` is clean.

- [ ] **Step 5: Commit**

```bash
git add daemon/config.py tests/test_config.py tests/test_wake.py
git commit -m "config: two axes (provider + proactive-judge-local) replace the preset table"
```

---

### Task 2: wizard — provider-first onboarding

**Files:**
- Modify: `daemon/setup.py` (`run()` step 2 ~1929–1935; `_choose_preset`/`PRESET_CHOICES`/`PRESET_ORDER`/`DEFAULT_PRESET`; `_choose_hosted`→`_choose_provider`; `needs_for` ~951–970)
- Test: `tests/test_setup.py`

**Interfaces:**
- Consumes: Task 1's `providers_for(*, provider, proactive_judge_local, voice_enabled, voice_provider)`, `Settings.provider`, `Settings.proactive_judge_local`.
- Produces: `.env` written with `DAEMON_PROVIDER` + `DAEMON_PROACTIVE_JUDGE_LOCAL`, never `DAEMON_PRESET`/`DAEMON_HOSTED_PROVIDER`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_setup.py`, replace the preset-menu tests (`test_the_folded_preset_menu_*`, `test_the_reasoning_is_one_keypress_away_*`, and preset walkthroughs) with provider-first equivalents. The exact scripted-answer sequences depend on the new `run()` order, so write these `needs_for`-level tests first (they are order-independent) and fix walkthroughs in Step 4:

```python
def keys(**env: str) -> set[str]:
    return {need.key for need in setup.needs_for(env)}


def test_ollama_provider_asks_no_hosted_key() -> None:
    k = keys(DAEMON_PROVIDER="ollama")
    assert "ANTHROPIC_API_KEY" not in k and "OPENAI_API_KEY" not in k
    assert "DAEMON_OLLAMA_MODEL" in k  # the local chat model is still asked


def test_hosted_provider_asks_its_key_only() -> None:
    k = keys(DAEMON_PROVIDER="anthropic")
    assert "ANTHROPIC_API_KEY" in k
    assert "GEMINI_API_KEY" not in k and "OPENAI_API_KEY" not in k


def test_a_stale_preset_env_still_names_the_provider_keys() -> None:
    # needs_for reads DAEMON_PROVIDER, not DAEMON_PRESET; a leftover preset is inert here
    # (Settings is what refuses it loudly - Task 1).
    k = keys(DAEMON_PROVIDER="gemini", DAEMON_PRESET="balanced")
    assert "GEMINI_API_KEY" in k
```

- [ ] **Step 2: Run — verify they fail**

Run: `python3 -m pytest tests/test_setup.py -k "provider_asks or stale_preset_env" -q`
Expected: FAIL — `needs_for` still calls `providers_for(preset=…)` and reads `DAEMON_PRESET`.

- [ ] **Step 3: Reshape `needs_for` and the wizard**

- `needs_for`: replace the `preset = …` / `providers_for(preset=…, hosted=…)` head with:

```python
    provider = env.get("DAEMON_PROVIDER", "")
    proactive_judge_local = _truthy(env.get("DAEMON_PROACTIVE_JUDGE_LOCAL", "true"))
    voice_on = _truthy(env.get("DAEMON_VOICE_ENABLED", ""))
    voice_provider = env.get("DAEMON_VOICE_PROVIDER", "") or _settings_default("DAEMON_VOICE_PROVIDER")
    providers = providers_for(
        provider=provider, proactive_judge_local=proactive_judge_local,
        voice_enabled=voice_on, voice_provider=voice_provider,
    )
```

  Then every `hosted == OPENAI`/`hosted == GEMINI` chat check in `needs_for` becomes `provider == OPENAI`/`provider == GEMINI`, and the `if HOSTED in PRESETS[preset].values()` guards become `provider not in ("", OLLAMA)`. The `DAEMON_OLLAMA_MODEL` need (`if OLLAMA in _chat_providers(...)`) becomes `if provider == OLLAMA`. Keep the voice-provider gating from #85 intact (it already keys off `voice_provider`).
- Delete `PRESET_CHOICES`, `PRESET_ORDER`, `DEFAULT_PRESET`, `_choose_preset`.
- Rename `_choose_hosted` → `_choose_provider(self, env, updates) -> str`: it offers `ollama | anthropic | openai | gemini | openai_compatible` (a `Choice` list; the `ollama` entry carries the folded privacy explanation the old offline preset had — "everything local, no keys, voice still opt-in"), writes `DAEMON_PROVIDER`, and calls `_choose_compatible_endpoint` when `openai_compatible` is picked (as `_choose_hosted` already does). Add a `_choose_background(self, provider, env, updates) -> None` that, **only when `provider not in ("", OLLAMA)`**, asks the one yes/no and writes `DAEMON_PROACTIVE_JUDGE_LOCAL`.
- `run()` step 2: replace the preset/hosted/voice trio with:

```python
        self._step(2, "How should Daemon think?")
        provider = self._choose_provider(env, updates)
        self._choose_background(provider, env, updates)
        voice = self._choose_voice(env, updates)
```

  and the `merged` dict writes `"DAEMON_PROVIDER": provider` (drop `DAEMON_PRESET`, `DAEMON_HOSTED_PROVIDER`).

- [ ] **Step 4: Run — fix walkthroughs, verify green**

Run `python3 -m pytest tests/test_setup.py -q`. The scripted `drive(...)` walkthroughs now have a different question order (provider before the background toggle; no preset menu). Update each script's answer list to the new order; the ollama walkthroughs drop the hosted-key answers. **Do not** weaken assertions — update the sequence, keep what each test proves. Confirm the ollama entry still prints the privacy trade (`assert "..." in result.out`).

- [ ] **Step 5: Commit**

```bash
git add daemon/setup.py tests/test_setup.py
git commit -m "setup: the wizard asks provider first, then how hard it should think"
```

---

### Task 3: admin settings_io — the editable surface catches up

**Files:**
- Modify: `daemon/admin/settings_io.py` (`STR_FIELDS`, `options`, and add the D9 note)
- Test: `tests/test_admin.py`

**Interfaces:**
- Consumes: Task 1's `Settings.provider`, `Settings.proactive_judge_local`, `config.MODEL_SUGGESTIONS`, `config.HOSTED_PROVIDERS`.
- Produces: GET `/settings` payload with `editable.provider`, `editable.proactive_judge_local`, `editable.ollama_model`; `options.providers` (= `["", *HOSTED_PROVIDERS, "ollama"]`), `options.model_suggestions`; `editable.off_provider_note` (D9, a string or null). No `preset`/`hosted_provider` key.

- [ ] **Step 1: Write the failing tests**

```python
def test_settings_offers_the_provider_axis_not_a_preset(tmp_path: Path) -> None:
    got = _get_settings(tmp_path, provider="gemini", gemini_model="g")  # helper builds Settings+payload
    assert "preset" not in got["editable"] and "hosted_provider" not in got["editable"]
    assert got["editable"]["provider"] == "gemini"
    assert got["editable"]["proactive_judge_local"] is True
    assert got["options"]["providers"][0] == ""
    assert "ollama" in got["options"]["providers"]
    assert got["options"]["model_suggestions"]["gemini"]  # non-empty tuple/list


def test_patch_sets_ollama_as_the_provider_and_its_model(tmp_path: Path) -> None:
    # provider=ollama writes DAEMON_PROVIDER and DAEMON_OLLAMA_MODEL, nothing else.
    env_path = _patch(tmp_path, {"provider": "ollama", "ollama_model": "qwen3:14b"})
    text = env_path.read_text()
    assert "DAEMON_PROVIDER=ollama" in text and "DAEMON_OLLAMA_MODEL=qwen3:14b" in text
    assert "DAEMON_PRESET" not in text


def test_the_off_provider_note_appears_only_for_out_of_band_routing(tmp_path: Path) -> None:
    plain = _get_settings(tmp_path, provider="gemini", gemini_model="g")
    assert plain["editable"]["off_provider_note"] is None
    routed = _get_settings(tmp_path, provider="gemini", gemini_model="g",
                           anthropic_model="c", anthropic_api_key="k",
                           route_overrides={"reflection": "anthropic"})
    assert "anthropic" in routed["editable"]["off_provider_note"]
```

Use the existing test helpers in `tests/test_admin.py` for building a `Settings` and calling `current_settings_payload` / `apply_patch` (mirror the existing `test_patch_sets_the_gemini_live_voice` shape).

- [ ] **Step 2: Run — verify they fail**

Run: `python3 -m pytest tests/test_admin.py -k "provider_axis or ollama_as_the_provider or off_provider_note" -q`
Expected: FAIL — `provider`/`ollama_model` not editable, no `off_provider_note`.

- [ ] **Step 3: Implement**

In `daemon/admin/settings_io.py`:
- In `STR_FIELDS`: remove `"preset"`; rename `"hosted_provider": "DAEMON_HOSTED_PROVIDER"` → `"provider": "DAEMON_PROVIDER"`; add `"ollama_model": "DAEMON_OLLAMA_MODEL"`.
- In `current_settings_payload`'s `options`: remove `"presets"`; replace `"hosted_providers"` with `"providers": ["", *HOSTED_PROVIDERS, "ollama"]`; add `"model_suggestions": {k: list(v) for k, v in MODEL_SUGGESTIONS.items()}`.
- Add `editable["proactive_judge_local"]` (it is a BOOL field — make sure it flows through the bool handling; add `proactive_judge_local` to `BOOL_FIELDS` with alias `DAEMON_PROACTIVE_JUDGE_LOCAL`).
- Compute the D9 note:

```python
    off = sorted({
        p for t, p in settings.routing.items()
        if p not in ("", settings.provider) and p != OLLAMA and t is not Task.CHAT_VOICE
    })
    editable["off_provider_note"] = (
        f"route_overrides / fallback send work to {', '.join(off)} — edit in .env" if off else None
    )
```

  (Voice's provider is expected to differ from the chat provider — exclude `CHAT_VOICE`. Exclude `OLLAMA` and the empty string so the note fires only on a genuine hosted off-provider.)

- [ ] **Step 4: Run — verify green**

Run: `python3 -m pytest tests/test_admin.py -q`. Some existing admin tests reference `preset`/`hosted_provider` — update them. Confirm green + ruff clean.

- [ ] **Step 5: Commit**

```bash
git add daemon/admin/settings_io.py tests/test_admin.py
git commit -m "admin: the settings payload speaks provider, not preset"
```

---

### Task 4: admin — live model lists, in a threadpool

**Files:**
- Modify: `daemon/admin/routes.py` (`get_settings`), `daemon/admin/settings_io.py` (accept an injected `model_lists`)
- Test: `tests/test_admin.py`

**Interfaces:**
- Consumes: `daemon.setup.check_anthropic/check_openai/check_gemini` (sync, return `Verdict` with `.models: dict[str, tuple[str,...]]`), Task 3's payload.
- Produces: `options.model_lists: dict[str, list[str]]` for the chat-three providers whose key is set; `[]` for any that failed. `get_settings` stays non-blocking.

- [ ] **Step 1: Write the failing test** (fake the probe — no network in the suite)

```python
async def test_live_model_lists_fall_back_to_empty_when_a_probe_raises(monkeypatch, tmp_path):
    from daemon.admin import routes
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(routes, "check_gemini", boom)
    lists = await routes._live_model_lists(_settings(tmp_path, provider="gemini",
                                                     gemini_model="g", gemini_api_key="k"))
    assert lists.get("gemini", None) == []          # failure → empty, never raises


async def test_live_model_lists_skip_providers_without_a_key(tmp_path):
    from daemon.admin import routes
    lists = await routes._live_model_lists(_settings(tmp_path, provider="gemini", gemini_model="g"))
    assert "anthropic" not in lists                 # no key → not probed
```

- [ ] **Step 2: Run — verify they fail** (`_live_model_lists` undefined)

Run: `python3 -m pytest tests/test_admin.py -k live_model_lists -q`

- [ ] **Step 3: Implement**

In `daemon/admin/routes.py`:

```python
import asyncio
from daemon.setup import check_anthropic, check_openai, check_gemini

_PROBES = {
    "anthropic": (check_anthropic, "anthropic_api_key", "anthropic_model", "DAEMON_ANTHROPIC_MODEL"),
    "openai": (check_openai, "openai_api_key", "openai_model", "DAEMON_OPENAI_MODEL"),
    "gemini": (check_gemini, "gemini_api_key", "gemini_model", "DAEMON_GEMINI_MODEL"),
}


async def _live_model_lists(settings) -> dict[str, list[str]]:
    """The chat-three providers' live model ids, keyed by provider, for the admin
    datalist. Runs each sync check_* in a thread (never in the event loop - a 15s
    HTTP_TIMEOUT there would freeze the daemon), concurrently, empty on any failure."""
    async def one(name, fn, key_attr, model_attr, env_key):
        key = getattr(settings, key_attr, "")
        if not key:
            return name, None
        try:
            # check_gemini takes (key); the others take (key, model).
            args = (key,) if name == "gemini" else (key, getattr(settings, model_attr, ""))
            verdict = await asyncio.to_thread(fn, *args)
            return name, list(verdict.models.get(env_key, ()))
        except Exception:
            return name, []
    pairs = await asyncio.gather(*(one(n, *cfg) for n, cfg in _PROBES.items()))
    return {n: ids for n, ids in pairs if ids is not None}
```

Then in `get_settings`: `payload = current_settings_payload(settings, env_path)` → add `payload["options"]["model_lists"] = await _live_model_lists(settings)`. (Confirm `check_gemini`'s real signature — it is `check_gemini(key)` per `daemon/setup.py`; the others are `check_*(key, model)`.)

- [ ] **Step 4: Run — verify green** (`python3 -m pytest tests/test_admin.py -q`, ruff clean)

- [ ] **Step 5: Commit**

```bash
git add daemon/admin/routes.py daemon/admin/settings_io.py tests/test_admin.py
git commit -m "admin: live model lists for the hosted three, off the event loop"
```

---

### Task 5: admin front-end — the provider-first MODEL card

**Files:**
- Modify: `daemon/admin/static/index.html` (`renderSettings` MODEL card ~1066–1076; `collectPatch`; add `brainField`, `fieldModel`, `brainPatch`)
- Test: `tests/test_admin.py` (structural — extract the JS function body from served HTML, as `test_voice_sensitivities_are_wired_through_the_provider_aware_block` does)

**Interfaces:**
- Consumes: Task 3/4's payload (`editable.provider`, `editable.proactive_judge_local`, `editable.ollama_model`, `editable.openai_compatible_*`, `options.providers`, `options.model_suggestions`, `options.model_lists`, `editable.off_provider_note`).
- Produces: a MODEL card whose every model field renders from inside `brainField(`, and a patch that emits `provider` / `proactive_judge_local` / `ollama_model` / `<provider>_model` / compat's two fields.

- [ ] **Step 1: Write the failing structural tests**

```python
def test_model_fields_render_from_brainfield_not_rendersettings(tmp_path):
    html = _served_index()  # the shell HTML string, as the existing test reads it
    start = html.index("function brainField(")
    body = html[start:html.index("\nfunction ", start + 1)]
    for name in ("provider", "proactive_judge_local", "ollama_model",
                 "openai_compatible_base_url"):
        assert name in body, f"{name} must render from inside brainField()"
    # the flat eight-box MODEL card is gone
    assert "fieldStr('preset'" not in html and "fieldStr('hosted_provider'" not in html


def test_the_chat_model_field_prefers_the_live_list(tmp_path):
    html = _served_index()
    assert "model_lists" in html and "model_suggestions" in html  # datalist prefers live, falls back
```

- [ ] **Step 2: Run — verify they fail**

- [ ] **Step 3: Implement** (vanilla JS, match the file's existing helper style — `fieldStr`, `voiceField`, `nameCell`, `val`, `pend`, the `#voice-field`/`display:contents` re-render trick)

- Replace the `card('MODEL', …)` block with `card('MODEL', brainField(val('provider', e.provider), e, o))`.
- `brainField(provider, e, o)`: renders the provider `<select>` (`o.providers`), then:
  - provider `""` → a "choose a provider" caption, no model field, no checkbox.
  - provider `ollama` → `fieldModel('ollama_model', e.ollama_model, [], [])` (text, no datalist), a "everything runs on this machine" caption, no checkbox.
  - hosted → `fieldModel('<provider>_model', e['<provider>_model'], o.model_lists[provider]||[], o.model_suggestions[provider]||[])`, the `proactive_judge_local` toggle (reuse `fieldToggle`), and for `openai_compatible` **also** `fieldStr('openai_compatible_base_url', …, <vendor-URL datalist>)`.
  - Always: the "Recall embeddings always run locally on Ollama (bge-m3)." caption, and `e.off_provider_note` as a read-only line when non-null.
  - Wrap in `<div id="brain-field" style="display:contents">…</div>`; on the provider `<select>` change, re-render `#brain-field` whole (copy the `voiceField` change handler at ~1137).
- `fieldModel(name, running, liveList, suggestions)`: a text `<input>` + a `<datalist>` (unique id) whose `<option>`s are `liveList.length ? liveList : suggestions`. The current value is always valid (free text).
- The provider select and the toggle carry `data-brain="provider"` / are handled by `brainPatch`; the toggle already works via `BOOLS` if it has `data-f="proactive_judge_local"` — simplest: give the toggle a normal `data-f` (so `collectPatch` handles it) and only the provider select needs `data-brain`. In `collectPatch`, after the generic loop, add: read `#brain-field select[data-brain="provider"]`; if it differs from `val('provider', e.provider)`, set `patch.provider`. Model inputs keep normal `data-f`.

- [ ] **Step 4: Run — verify green** + **manual smoke** (per repo rule): `python3 -m pytest tests/test_admin.py -q`, ruff/check_docs clean.

- [ ] **Step 5: Commit**

```bash
git add daemon/admin/static/index.html tests/test_admin.py
git commit -m "admin: the MODEL card asks provider first, then that provider's model"
```

---

### Task 6: the paper trail — ADR, docs, the preset sweep

**Files:**
- Create: `docs/adr/0014-provider-is-the-axis.md`
- Modify: `docs/adr/README.md`, `docs/PLAN.md` (§3.2), `docs/ARCHITECTURE.md` (routing), `.env.example`, `daemon/CLAUDE.md`, and any remaining `preset` mention flagged by the sweep
- Test: none new; `tests/test_acceptance.py` and any other file still using `preset=`/`DAEMON_PRESET` must be updated to green.

**Interfaces:** none code-facing.

- [ ] **Step 1: Write ADR 0014** (`# 0014 — The provider is the axis; presets are gone`) in the house format (Context · Decision · Consequences · What would change our mind). Context: presets encoded a table the user never saw; after provider-first on both surfaces, `DAEMON_PRESET` had no caller. Decision: the two axes (D1), the rename + loud stale-preset guard (D4), amends ADR 0007 (the provider-axis half replaces the preset-axis half; "no default provider" stands), composes with ADR 0012 (voice axis). What would change our mind: a need for per-task provider control common enough that `route_overrides` (hand-edit) is too coarse.

- [ ] **Step 2: Index row** in `docs/adr/README.md` after `0013`.

- [ ] **Step 3: `docs/PLAN.md §3.2`** — replace the three-row preset table and its prose with the two axes (Korean; match the file). State: provider (ollama = fully local, §7's promise), the one judge toggle, embed always local, voice its own axis (ADR 0012/0014).

- [ ] **Step 4: `.env.example`** — drop `DAEMON_PRESET`, rename `DAEMON_HOSTED_PROVIDER`→`DAEMON_PROVIDER` (note it accepts `ollama`), add `DAEMON_PROACTIVE_JUDGE_LOCAL`. `docs/ARCHITECTURE.md` routing section and `daemon/CLAUDE.md` (`config.py` line says "the three presets") — correct both.

- [ ] **Step 5: The sweep**

```bash
grep -rnE "DAEMON_PRESET|DAEMON_HOSTED_PROVIDER|\bpreset\b|PRESETS|preset_providers" \
  --include=*.py --include=*.md --include=*.example . | grep -v node_modules
```

Every hit outside this plan and the design doc is falsified — fix it (code already handled in Tasks 1–5; here it is docs + any straggler test). `tests/test_acceptance.py` builds settings with `preset=`/`DAEMON_PRESET`; update to `provider=`/`DAEMON_PROVIDER`.

- [ ] **Step 6: Full gate + commit**

```bash
python3 -m pytest && python3 -m ruff check . && python3 scripts/check_docs.py
git add docs/ .env.example daemon/CLAUDE.md tests/
git commit -m "docs: the provider is the axis - the paper trail drops the preset"
```

---

## Verification (whole change — the real paths, not only the suite)

- [ ] `python3 -m pytest` green; `ruff check .` clean; `check_docs.py` ok.
- [ ] The sweep from Task 6 Step 5 returns only this plan and the design doc.
- [ ] **`daemon setup`** from a scratch dir: pick each provider incl. `ollama` and `openai_compatible`; confirm the written `.env` has `DAEMON_PROVIDER`/`DAEMON_PROACTIVE_JUDGE_LOCAL`, never `DAEMON_PRESET`, and that `ollama` asks no hosted key.
- [ ] **`daemon doctor`** on a `DAEMON_PROVIDER=ollama` + voice-on `.env`: reports all-local chat + `chat_voice->gemini`. On a `.env` still containing `DAEMON_PRESET=balanced`: fails loudly naming the two new keys.
- [ ] **`daemon setup --check`** and **`daemon doctor`** agree across provider×judge×voice combinations (the #85 regression surface).
- [ ] **Live admin** (drive the real page, per `qa-drive-the-live-ux`): open Settings → PROVIDER shows the six options → pick `gemini` → the model datalist is populated (live if key set) → toggle the judge → Save & Restart → reopen and the values stuck. Pick `openai_compatible` → base_url + model both show. Pick `ollama` → model is text, no judge toggle, voice toggle still editable.
