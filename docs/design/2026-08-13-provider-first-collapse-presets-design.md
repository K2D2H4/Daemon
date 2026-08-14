# Provider-first everywhere — collapse the presets into an axis

**Status:** approved design (grilled 2026-08-13), pre-implementation
**Base:** `origin/main` after #85 (voice is its own axis, ADR 0012) and #86
(openai_compatible, v0.1.44). So `HOSTED_PROVIDERS` is
`(anthropic, openai, gemini, openai_compatible)`, `Settings.routing` adds `CHAT_VOICE`
off `voice_enabled`, and `providers_for`/`needs_for` are already voice-provider-aware.

**Supersedes** the blocked `docs/design/2026-08-11-admin-model-selection-design.md`
(on branch `claude/admin-model-selection-ui-f4d937`). That design proposed deriving a
kept `DAEMON_PRESET` in the UI. The owner chose the harder path instead: **remove the
preset concept entirely.** Its rules 1/2/4/5 survive here in changed form; its rule 3
(ollama forces voice off) is void because voice is now its own axis.

## Problem

Two surfaces still speak in "presets", and one card is a wall of inert boxes.

1. **The admin MODEL card shows eight controls, one or two in force.** After the
   openai_compatible merge it is `preset`, `hosted_provider`, and six model boxes
   (`anthropic_model`, `openai_model`, `gemini_model`, `gemini_live_model`,
   `openai_realtime_model`, `openai_compatible_model`). Nothing says which are live;
   two of them (`gemini_live_model`, `openai_realtime_model`) belong to the voice
   axis, not the hosted provider, and sit in the wrong card.
2. **"preset" names nothing the owner can act on.** `balanced` and `quality` differ
   in exactly one routed task — `PROACTIVE_JUDGE`. `offline` is not a tier; it is
   "everything local". The word encodes a table the user never sees.
3. **`DAEMON_PRESET` is legacy with no user-facing surface after this change.** If
   both the admin and the wizard ask "provider first", nobody ever types a preset
   name again, so keeping three named presets in `config.py` is a table with no
   caller but the code itself.

## Goal

The preset concept leaves the codebase. Chat routing is described by two orthogonal
axes a person actually sets. Both the admin and the wizard ask the same question in
the same order — **provider, then that provider's model(s), then that provider's
extras** — and every control shown is a control currently in force.

## Decisions (from the grilling, with rationale)

Each was a real fork; the rejected side is recorded so it is not re-litigated.

| # | Decision | Rejected |
|---|---|---|
| D1 | **Collapse `PRESETS` into two axes.** `DAEMON_PROVIDER` (the chat provider, `ollama` included) + `DAEMON_PROACTIVE_JUDGE_LOCAL` (bool). `EMBED` stays `ollama` always; voice is its own axis (ADR 0012). | Keep `DAEMON_PRESET`, derive it in the UI (the superseded spec). Rejected: leaves a table with no caller; owner wants no legacy. |
| D2 | **Per-task exposure stays out.** The two axes reconstruct all three presets with zero loss; `route_overrides` (hand-edit) remains the escape hatch for finer control. | Expose each `Task`'s provider in the UI. Rejected: that is the "onboarding asks seven questions" problem `PRESETS` was created to avoid. |
| D3 | **The toggle controls `PROACTIVE_JUDGE` only** — not reflection. `REFLECTION` is hosted in both `balanced` and `quality`; only `PROACTIVE_JUDGE` varies. The superseded spec's caption ("reflection, proactive judging") was wrong. | Frame it as "background work". Rejected: false — reflection does not move. |
| D4 | **No migration code (single-owner install).** `DAEMON_HOSTED_PROVIDER` → **rename** `DAEMON_PROVIDER`; the old key is not read. Encountering a stale `DAEMON_PRESET` **raises loudly at startup** (not silently ignored). | (a) A read-and-rewrite migration — unneeded for one install. (b) Silently ignore a leftover `DAEMON_PRESET` — a `DAEMON_PRESET=offline` install would silently start dialing a hosted provider, the privacy version of ADR 0007's footgun. |
| D5 | **Provider-list widget splits by provider.** `anthropic` / `openai` / `gemini` → a **live** model list reusing the wizard's `check_*` probes. `ollama` / `openai_compatible` → free text (open/local/huge model spaces). | (a) Static datalist for all — the big three's live list is genuinely better and already built. (b) Live probe compat's `/models` — OpenRouter returns hundreds. (c) Hard `<select>` — a just-released model id becomes unselectable until a code update. |
| D6 | **The admin probes at render time, in a threadpool.** `GET /settings` runs the `check_*` probes for providers whose key is set via `asyncio.to_thread`, concurrently (`gather`), falling back to static suggestions on any failure/timeout. | (a) A "refresh models" button — more moving parts than a single-owner panel needs. (b) Call the sync `check_*` directly in the async handler — a 15s `HTTP_TIMEOUT` blocking `httpx.get` inside `async def` freezes the **whole daemon**, the PortAudio-deadlock class of bug. |
| D7 | **`openai_compatible` shows `base_url` + `model` together.** Selecting it renders both as "that provider's fields"; `base_url` is text with a datalist of `COMPATIBLE_VENDORS` URLs. No vendor sub-menu (that is the wizard's interactive nicety). | Put `base_url` in a separate row/section. Rejected: breaks the provider-first unity. |
| D8 | **Voice card unchanged; axes fully independent.** The MODEL provider governs chat only. `MODEL=ollama` does **not** disable voice. Voice keeps its current provider-first `voiceField()`; live probing is chat-three-only (realtime/Live model lists are not clean `/models` output, and "do not regress Gemini Live" from the B2 design holds). | Live-probe voice models too. Rejected: new per-provider filtering, and it touches a shipped, verified path. |
| D9 | **A read-only note for out-of-band routing.** When `Settings.routing` names a provider other than `DAEMON_PROVIDER` (via `route_overrides`/`fallback_provider`, both hand-edit-only), the card shows a one-line **read-only** note pointing at `.env`. No hidden editable fields. | (a) Drop it — mildly lies about what the daemon is running, against this page's stated posture. (b) The superseded spec's editable `<details>` — shows the model but not the override that activates it; half a picture, and implies UI management of a hand-edit feature. |
| D10 | **One PR, staged internally** (config → wizard → admin), executed subagent-driven with per-task review, exactly like #85. | Two PRs. Rejected: the config rename atomically breaks both surfaces (`providers_for`/`needs_for` signatures; `settings_io`/HTML field refs), and a legacy-free intermediate would need a shim — which D4 forbids. |

## The config contract, after

`daemon/config.py`:

- **Delete** `PRESETS`, the `HOSTED` placeholder, `preset_providers`, and the `preset`
  field. Keep `VOICE_TASKS`, `PROVIDER_KEY_ENV`, `HOSTED_PROVIDERS`.
- **New field** `provider: str = Field(default="", alias="DAEMON_PROVIDER")`. Accepts
  `""` (unset — a hosted task then raises pointing at `daemon setup`, ADR 0007) or any
  of `PROVIDER_KEY_ENV`'s keys **including `ollama`**. `DAEMON_HOSTED_PROVIDER` is not
  an alias — the key is gone.
- **New field** `proactive_judge_local: bool = Field(default=True,
  alias="DAEMON_PROACTIVE_JUDGE_LOCAL")`. Default `True` = today's `balanced`.
- **`routing` is computed, not table-driven.** The fixed task→role map:

  | Task | role |
  |---|---|
  | `CHAT_TEXT`, `RECALL_ESCALATION`, `REFLECTION`, `PERSONA_RULE` | the provider |
  | `PROACTIVE_JUDGE` | `ollama` if `proactive_judge_local` else the provider |
  | `EMBED` | always `ollama` |
  | `CHAT_VOICE` | added off `voice_enabled`, mapped to `voice_provider` (ADR 0012, unchanged) |

  where "the provider" is `DAEMON_PROVIDER`. When `provider == "ollama"`, every hosted
  role resolves to `ollama` — reconstructing `offline` exactly, `proactive_judge_local`
  then irrelevant. `route_overrides` still applied on top.
- **Stale-preset guard:** if the raw env holds `DAEMON_PRESET`, raise `ConfigError`
  at construction naming the two new keys. One check; the only place `preset` survives.
- **ADR 0014** records this: it amends ADR 0007 (the "preset axis" half is replaced by
  the provider axis; the "no default provider" half stands) and composes with ADR 0012.

## The admin MODEL card, after

```
MODEL
  PROVIDER  [ gemini            ▾ ]      GEMINI_MODEL  [ gemini-3.6-flash   ▾ ]
  ☑ 선제 발화 판단만 로컬 모델로 (5분마다 도니 비용이 쌓임)      ← proactive_judge_local
  Recall embeddings always run locally on Ollama (bge-m3).
  ⓘ route_overrides sends reflection to anthropic — edit in .env   ← D9, only when it applies
```

- `PROVIDER` options: `""` (선택 안 함) + `anthropic` `openai` `gemini`
  `openai_compatible` `ollama`. Env values, not brand names (page posture: "the screen
  says what `.env` says"). Empty stays first (fresh install claims no provider).
- Provider empty → no model field, a "choose a provider" caption, checkbox hidden.
- Provider = `ollama` → model field is `ollama_model` (text); checkbox hidden (all
  local); a caption that everything runs on this machine.
- Provider = hosted → the checkbox shows; the model field is that provider's:
  - `anthropic`/`openai`/`gemini` → text input + `<datalist>` filled **live** (D5/D6),
    falling back to static `MODEL_SUGGESTIONS`.
  - `openai_compatible` → **two** fields: `base_url` (text + vendor-URL datalist) and
    `openai_compatible_model` (text). D7.
- The current `.env` value is always offered whether or not the probe/list contains it,
  so a stale list never blocks a save (model ids validate only as non-empty).

## Live model probing (D6 — the daemon-freeze guard)

New admin path: for each of `anthropic`/`openai`/`gemini` whose key is set, run the
existing `daemon.setup.check_*` in a thread and collect the ids:

```python
async def _live_model_lists(settings) -> dict[str, list[str]]:
    import asyncio
    probes = {p: fn for p, fn in (("anthropic", ...), ("openai", ...), ("gemini", ...))
              if <settings has that provider's key>}
    async def one(p, fn):
        try:
            verdict = await asyncio.to_thread(fn, <key>, <current model>)
            return p, list(verdict.models.get(<env key>, ()))
        except Exception:
            return p, []          # fall back to static suggestions
    return dict(await asyncio.gather(*(one(p, fn) for p, fn in probes.items())))
```

`check_*` are **sync** `httpx.get`; calling them directly in the async `get_settings`
handler blocks the event loop for up to `HTTP_TIMEOUT=15s` each — freezing the channel
loop and scheduler, not just the request. `asyncio.to_thread` + `gather` keeps the loop
free (happy path ≈ one probe's latency, ~0.6s; worst case: that one Settings request is
slow, the daemon stays alive). Feeds `options.model_lists`; the front-end prefers it and
falls back to `options.model_suggestions`.

## The wizard, after (provider-first)

`daemon/setup.py` `run()` step 2 becomes **provider → (if hosted) background toggle →
(if compat) endpoint → voice**, replacing preset-first:

- Delete `PRESET_CHOICES` / `PRESET_ORDER` / `_choose_preset`. The privacy narrative
  the offline preset carried moves onto the `ollama` provider choice (a folded
  explanation on that menu entry: everything local, no keys, voice still opt-in).
- `_choose_provider` offers the five providers + the ollama-is-fully-local entry;
  `_choose_compatible_endpoint` stays (runs when compat is picked).
- After a hosted provider, ask the one background toggle (`proactive_judge_local`).
- `providers_for`/`needs_for` lose the `preset` parameter and read the two axes. The
  voice-provider gating from #85 is unchanged.

## Naming

`DAEMON_PROVIDER` (not `DAEMON_HOSTED_PROVIDER`): the axis now includes `ollama`, which
is the opposite of "hosted" — keeping `HOSTED` in the name would contradict the value.
`ollama`, not `local`: `ollama` is a real provider in `PROVIDER_KEY_ENV` with a real
`DAEMON_OLLAMA_MODEL`; `local` appears nowhere in `config.py`. Provider options are env
values, matching the rest of the page.

## Implementation shape

- **`daemon/config.py`** — the contract change above; `MODEL_SUGGESTIONS` (chat) and the
  existing voice suggestions; a docstring saying suggestions are **not validated** (every
  other enumerated constant here is a constraint, so a reader will "fix" it otherwise).
- **`daemon/setup.py`** — `ollama_model` already in the wizard; drop `preset`, add the
  provider-first flow, thread the two axes through `providers_for`/`needs_for`.
- **`daemon/admin/settings_io.py`** — drop `preset` from editable; `hosted_provider` →
  `provider` (accepts `ollama`); add `ollama_model`; `options` gains `model_suggestions`
  and (from the route) `model_lists`; the D9 note is computed server-side.
- **`daemon/admin/routes.py`** — `get_settings` awaits `_live_model_lists` (D6).
- **`daemon/admin/static/index.html`** — `brainField(provider, e, o)` (provider select +
  model field(s) + checkbox + captions + D9 note), re-rendered whole on provider change
  like `voiceField()`; `fieldModel(name, running, list, suggestions)` (datalist, prefers
  live list); compat's two-field case; the derived controls carry `data-brain`, not
  `data-f`, so `collectPatch()` skips them and a `brainPatch()` emits `provider` /
  `proactive_judge_local`.
- **`docs/`** — ADR 0014; `PLAN.md §3.2` (the preset table → the two axes);
  `ARCHITECTURE.md` routing section; `.env.example` (drop `DAEMON_PRESET`, rename the
  provider key, add the toggle); `daemon/CLAUDE.md` (`config.py` line still says
  "three presets").

## Tests

- `config`: the routing table for each (provider × `proactive_judge_local`) combination
  reconstructs the old presets exactly; `ollama` provider = all-local; a stale
  `DAEMON_PRESET` raises; `EMBED` always `ollama`; voice axis unchanged (regression).
- `setup`: the provider-first walkthrough for each provider incl. `ollama` and
  `openai_compatible`; `needs_for` asks the right keys per the two axes; the ollama
  entry still states the privacy trade.
- `admin`: all model fields render from `brainField(`/`voiceField(`, never
  `renderSettings` (rules 1/2); PATCH `provider=ollama` writes `DAEMON_PROVIDER` and
  drops nothing else; PATCH a model id absent from suggestions still saves; the D9 note
  appears only when routing names an off-provider; `_live_model_lists` falls back to
  suggestions when a probe raises (no network in the suite — fake the probe).
- The seven-file sweep from the superseded spec's out-of-scope note is now **in** scope:
  every doc/test that says "preset" is updated or removed.

## Out of scope

- **A `route_overrides` / `fallback_provider` editor.** Still `.env`-only; D9 surfaces
  them read-only.
- **A hosted embedder.** `EMBED` stays `ollama`; the caption says so.
- **Live-probing `ollama` `/api/tags` in the admin.** ollama's model field is text; its
  installed-list probe stays in the wizard. A later enhancement, not this change.

## Execution

One PR (`claude/provider-first-collapse-presets`, off current `origin/main`), staged
config → wizard → admin, subagent-driven with per-task review and a whole-branch review,
then live QA: drive `daemon setup`, `daemon doctor`, and the real admin page (select →
save → restart → sticks) — not just the unit suite (per this repo's standing rule that
nothing is "working" until the real path runs).
