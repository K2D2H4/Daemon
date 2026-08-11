# Voice is its own axis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make "local text + hosted voice" a legal configuration by keying `Task.CHAT_VOICE`'s
presence in `Settings.routing` off `DAEMON_VOICE_ENABLED` instead of off the preset table.

**Architecture:** `daemon/config.py` already treats the voice *provider* as its own axis
(`DAEMON_VOICE_PROVIDER`, overridden into `routing`). The one thing contradicting that is
`PRESETS["offline"]` having no `CHAT_VOICE` row, which the validator turns into "voice is
impossible under offline". This change adds the row when voice is on, deletes the validator
problem that becomes unreachable, teaches onboarding to ask for the voice key under every
preset, and rewrites the three places (a code comment, the wizard's preset menu, PLAN.md §3.2)
that told users the coupling was deliberate.

**Tech Stack:** Python 3.13, pydantic-settings, pytest, ruff.

## Global Constraints

- `ruff` line-length **100**, target `py313`, lint rules `E,F,I,UP,B,ASYNC`.
- `PRESETS` itself is **not** edited. No new preset, no new `.env` key, no migration.
- The privacy promise stays literally true. `docs/PLAN.md:551` already conditions it on
  *text mode* ("텍스트 모드 + 로컬 모델"), so what carries it after this change is
  `voice_enabled=false`, not the missing table row. Every comment that credits the missing
  row must be corrected, not deleted.
- Docs are English; `docs/PLAN.md` is Korean. Match the file you are editing.
- Full gate before each commit: `python3 -m pytest`, `python3 -m ruff check .`,
  `python3 scripts/check_docs.py`.

## File Structure

| File | Responsibility in this change |
|---|---|
| `daemon/config.py` | `routing` adds the voice row when voice is on; `route_for` reports "voice is off" before "not routed"; the unreachable validator problem goes; `providers_for` learns the voice provider |
| `daemon/setup.py` | `needs_for` passes the voice provider through; `_choose_voice` stops gating on the preset; the offline entry in `PRESET_CHOICES` stops saying voice is unavailable |
| `tests/test_config.py` | the new routing truth, and the two tests that assert the old contract |
| `tests/test_setup.py` | onboarding asks for the voice key under `offline`; the menu copy tests |
| `tests/test_wake.py` | a docstring that states the old premise |
| `docs/adr/0012-voice-is-its-own-axis.md` | the decision record (new) |
| `docs/adr/README.md` | its index row |
| `docs/PLAN.md` | §3.2's preset table |

---

### Task 1: `routing` carries the voice row whenever voice is on

**Files:**
- Modify: `daemon/config.py:96-103` (the comment above `PRESETS`), `daemon/config.py:790-794`
  (the validator problem), `daemon/config.py:1031-1044` (`routing`),
  `daemon/config.py:1064-1082` (`route_for`)
- Test: `tests/test_config.py`
- Modify: `tests/test_wake.py:908-914` (docstring only)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Settings.routing` contains `Task.CHAT_VOICE` whenever `voice_enabled` is true,
  mapped to `Settings.voice_provider`. `Settings.route_for(Task.CHAT_VOICE)` raises
  `ConfigError` matching `"voice is off"` when `voice_enabled` is false, under **every**
  preset. Task 2 relies on neither — it only relies on `providers_for`.

- [ ] **Step 1: Write the failing tests**

Add `providers_for` to the existing top-level import so Task 2's tests can use it:

```python
from daemon.config import PRESETS, ConfigError, Route, Settings, providers_for
```

Then replace the whole section marked `# --- voice under the offline preset -----`
(lines 127-146). Three tests go, and each for its own reason:
`test_offline_preset_refuses_voice` and `test_enabling_voice_on_the_offline_preset_fails_at_startup`
assert the contract this task deliberately changes;
`test_voice_is_refused_while_disabled_even_when_routed` is subsumed by the new
all-presets version below, and keeping both would leave two tests asserting one thing.
Put these in their place:

```python
# --- voice is its own axis (docs/adr/0012) -----------------------------------


def test_the_offline_preset_can_have_hosted_voice() -> None:
    # Local text with hosted audio is a real configuration: the privacy promise in
    # docs/PLAN.md 7 is conditioned on *text mode*, not on the preset table.
    settings = make_settings(
        preset="offline",
        voice_enabled=True,
        gemini_api_key="k",
        gemini_live_model="m",
    )

    assert settings.routing[Task.CHAT_VOICE] == "gemini"
    assert settings.route_for(Task.CHAT_VOICE) == Route(provider="gemini", model="m")


def test_offline_voice_follows_the_voice_provider_not_the_preset_table() -> None:
    settings = make_settings(
        preset="offline",
        voice_enabled=True,
        voice_provider="openai",
        openai_api_key="k",
        openai_realtime_model="gpt-realtime",
    )

    assert settings.routing[Task.CHAT_VOICE] == "openai"
    assert settings.route_for(Task.CHAT_VOICE) == Route(
        provider="openai", model="gpt-realtime"
    )


def test_voice_off_routes_no_voice_task_under_any_preset() -> None:
    for preset in PRESETS:
        settings = make_settings(preset=preset, anthropic_api_key="k", gemini_model="g")

        # One message for one situation: before this, `offline` said "does not route
        # chat_voice" while `balanced` said "voice is off" for the same reason.
        with pytest.raises(ConfigError, match="voice is off"):
            settings.route_for(Task.CHAT_VOICE)


def test_voice_on_needs_its_own_model_under_the_offline_preset_too() -> None:
    # The validator's voice-model check is what refuses voice now; the preset table
    # no longer refuses anything.
    with pytest.raises(ConfigError, match="DAEMON_GEMINI_LIVE_MODEL is empty"):
        make_settings(preset="offline", voice_enabled=True, gemini_api_key="k")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_config.py -k "voice" -v`

Expected: `test_the_offline_preset_can_have_hosted_voice` and
`test_offline_voice_follows_the_voice_provider_not_the_preset_table` FAIL at
`make_settings(...)` with `ConfigError: ... routes no voice task`.
`test_voice_off_routes_no_voice_task_under_any_preset` FAILS on the `offline` iteration with
`does not route chat_voice`. `test_voice_on_needs_its_own_model_under_the_offline_preset_too`
FAILS because the "routes no voice task" problem is reported instead.

- [ ] **Step 3: Add the voice row in `routing`**

In `daemon/config.py`, replace the body of `routing` (currently lines 1031-1044):

```python
    @property
    def routing(self) -> dict[Task, str]:
        """Effective Task -> provider name table: preset, then the chosen hosted
        provider substituted in, then explicit overrides on top."""
        resolved = {
            task: (self.hosted_provider if provider == HOSTED else provider)
            for task, provider in PRESETS[self.preset].items()
        }
        # Voice is its own axis - DAEMON_VOICE_ENABLED and DAEMON_VOICE_PROVIDER - and
        # not a property of the preset. `offline` carries no CHAT_VOICE row, but local
        # text with hosted audio is a configuration people want, so the row is *added*
        # when voice is on rather than only rewritten when the table happened to hold
        # one. What keeps docs/PLAN.md 7 true is voice being off, which is what that
        # promise has always said ("텍스트 모드 + 로컬 모델"). See docs/adr/0012.
        if self.voice_enabled or Task.CHAT_VOICE in resolved:
            resolved[Task.CHAT_VOICE] = self.voice_provider
        return {**resolved, **self.route_overrides}
```

The `or Task.CHAT_VOICE in resolved` half is load-bearing: with voice **off** under
`balanced`/`quality` the row must stay present exactly as before, so that `route_for` reports
"voice is off" rather than "not routed" and the existing routing-table tests keep passing.

- [ ] **Step 4: Delete the validator problem that is now unreachable**

In `daemon/config.py`, delete these five lines (currently 790-794):

```python
        if self.voice_enabled and not VOICE_TASKS <= self.routing.keys():
            problems.append(
                f"DAEMON_VOICE_ENABLED is on but preset {self.preset!r} routes no voice task; "
                "voice needs a hosted native-audio provider (docs/PLAN.md 3.2)"
            )
```

`routing` now always holds `CHAT_VOICE` when `voice_enabled`, so the condition can never be
true. Leave `VOICE_TASKS` itself alone — it is still used in `providers_for`,
`_provider_problems`, `active_tasks` and `route_for`.

- [ ] **Step 5: Report "voice is off" before "not routed" in `route_for`**

Replace `route_for` (currently lines 1064-1082):

```python
    def route_for(self, task: Task) -> Route:
        """Resolve one Task, or explain precisely why it cannot be served."""
        # Before the routing lookup, because voice-off is now the only reason a voice
        # task is missing from the table, and the honest answer is the switch rather
        # than the preset.
        if task in VOICE_TASKS and not self.voice_enabled:
            raise ConfigError(
                f"{task.value} was requested but voice is off (DAEMON_VOICE_ENABLED)"
            )
        provider = self.routing.get(task)
        if provider is None:
            raise ConfigError(f"preset {self.preset!r} does not route {task.value}")
        if task in VOICE_TASKS:
            # The native-audio endpoint takes its own model id (never DAEMON_*_MODEL).
            model = self.gemini_live_model if provider == "gemini" else self.openai_realtime_model
            return Route(provider=provider, model=model)
        return Route(provider=provider, model=self.provider_model(provider))
```

The `extra` local (`" - it needs a hosted native-audio provider (docs/PLAN.md 3.2)"`) is
deleted with it: a voice task can no longer reach the `provider is None` branch, so the hint
had no remaining caller.

- [ ] **Step 6: Correct the comment that credits the missing table row**

In `daemon/config.py`, the comment block above `PRESETS` currently ends with:

```python
#   - CHAT_VOICE is deliberately ABSENT from `offline`. That absence is what
#     makes the privacy promise in docs/PLAN.md 7 literally true.
```

Replace those two lines with:

```python
#   - CHAT_VOICE is absent from `offline` because no *preset* implies voice; the
#     axis is DAEMON_VOICE_ENABLED, and `Settings.routing` adds the row when that
#     is on (docs/adr/0012). What makes the promise in docs/PLAN.md 7 literally
#     true is voice being off - which is what that promise says.
```

- [ ] **Step 7: Correct the test docstring that states the old premise**

In `tests/test_wake.py`, `voice_settings()`'s docstring currently reads "`offline`
deliberately routes no voice task - that absence is what makes the privacy claim in
docs/PLAN.md 7 true - so turning voice on there fails…". Replace that sentence with:

```python
    """A configuration where voice is genuinely available.

    `balanced` rather than `offline` only because these tests predate voice being its
    own axis (docs/adr/0012) and there is no reason to churn them: both presets can
    carry voice now, and this one already has the keys the wake path expects."""
```

- [ ] **Step 8: Run the full gate**

```bash
python3 -m pytest && python3 -m ruff check . && python3 scripts/check_docs.py
```

Expected: all pass. If `tests/test_config.py::test_offline_preset_routes_everything_local`
fails, the `or Task.CHAT_VOICE in resolved` clause from Step 3 was dropped or the voice
default flipped — that test builds `offline` with voice **off**, so its routing dict must
still hold six entries and no `CHAT_VOICE`.

- [ ] **Step 9: Commit**

```bash
git add daemon/config.py tests/test_config.py tests/test_wake.py
git commit -m "config: voice is turned on, not routed by the preset"
```

---

### Task 2: onboarding asks for the voice key under every preset

**Files:**
- Modify: `daemon/config.py:191-219` (`providers_for`)
- Modify: `daemon/setup.py:192-204` (`PRESET_CHOICES`, the `offline` entry),
  `daemon/setup.py:958-962` (the `providers_for` call in `needs_for`),
  `daemon/setup.py:1651` (the `_choose_voice` call), `daemon/setup.py:1944-1968`
  (`_choose_voice`)
- Test: `tests/test_config.py`, `tests/test_setup.py`

**Interfaces:**
- Consumes: Task 1's `routing` behaviour (not directly called; the wizard writes `.env` and
  `Settings` must accept the result).
- Produces: `providers_for(preset, *, voice_enabled: bool, hosted: str = DEFAULT_HOSTED_PROVIDER,
  voice_provider: str) -> list[str]` — `voice_provider` is **required keyword-only**.
  `Wizard._choose_voice(self, env: Mapping[str, str], updates: dict[str, str]) -> bool` —
  the `preset` and `hosted` parameters are gone.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`, after the tests from Task 1:

```python
def test_providers_for_asks_for_the_voice_key_under_the_offline_preset() -> None:
    assert providers_for(
        "offline", voice_enabled=True, hosted="", voice_provider="gemini"
    ) == ["gemini", "ollama"]
    assert providers_for(
        "offline", voice_enabled=False, hosted="", voice_provider="gemini"
    ) == ["ollama"]


def test_providers_for_follows_the_voice_provider_not_the_table() -> None:
    # Reading CHAT_VOICE straight from the preset table asked a user who chose
    # OpenAI voice for a Gemini key.
    providers = providers_for(
        "balanced", voice_enabled=True, hosted="anthropic", voice_provider="openai"
    )

    assert "openai" in providers
    assert "gemini" not in providers
```

Add to `tests/test_setup.py`, next to `test_needs_come_from_the_preset_table`:

```python
def test_offline_with_voice_on_still_asks_for_the_voice_key() -> None:
    keys = {
        need.key
        for need in setup.needs_for(
            {"DAEMON_PRESET": "offline", "DAEMON_VOICE_ENABLED": "true"}
        )
    }

    assert "GEMINI_API_KEY" in keys
    # Still no hosted *chat* provider: offline resolves no HOSTED task.
    assert "DAEMON_HOSTED_PROVIDER" not in keys


def test_the_voice_key_follows_the_voice_provider() -> None:
    keys = {
        need.key
        for need in setup.needs_for(
            {
                "DAEMON_PRESET": "offline",
                "DAEMON_VOICE_ENABLED": "true",
                "DAEMON_VOICE_PROVIDER": "openai",
            }
        )
    }

    assert "OPENAI_API_KEY" in keys
    assert "GEMINI_API_KEY" not in keys
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_config.py -k providers_for tests/test_setup.py -k "voice_key or voice_provider" -v`

Expected: the `tests/test_config.py` ones FAIL with `TypeError: providers_for() got an
unexpected keyword argument 'voice_provider'`; the `tests/test_setup.py` ones FAIL on the
`assert "GEMINI_API_KEY" in keys` / `"OPENAI_API_KEY" in keys` lines, because the offline
table contributes no voice provider today.

- [ ] **Step 3: Teach `providers_for` the voice provider**

In `daemon/config.py`, replace `providers_for` (currently lines 191-219):

```python
def providers_for(
    preset: str,
    *,
    voice_enabled: bool,
    hosted: str = DEFAULT_HOSTED_PROVIDER,
    voice_provider: str,
) -> list[str]:
    """Providers a preset actually needs, so onboarding asks for those keys only.

    Voice contributes only while voice is on - that is what lets a text-only
    `balanced` install be set up without a hosted voice key (docs/PLAN.md 6.5) - and
    when it does, it contributes `voice_provider` under *every* preset, `offline`
    included (docs/adr/0012).

    `hosted` resolves the HOSTED placeholder and is required: a caller that guesses
    it asks the user for the wrong key, which is how a person who chose GPT ends up
    being asked for an Anthropic one. Pass "" before the question has been answered
    and hosted tasks simply drop out of the list.

    `voice_provider` is required for the same reason, and is not defaulted here even
    though the field has a default: reading CHAT_VOICE from the preset table instead
    asked a user who had chosen OpenAI voice for a Gemini key.
    """
    if preset not in PRESETS:
        raise ConfigError(
            f"unknown preset {preset!r}; expected one of {', '.join(sorted(PRESETS))}"
        )
    providers = {
        provider
        for task, provider in preset_providers(preset, hosted).items()
        # An unanswered question contributes nothing rather than a guess.
        if task not in VOICE_TASKS and provider != HOSTED
    }
    if voice_enabled:
        providers.add(voice_provider)
    return sorted(providers)
```

- [ ] **Step 4: Pass it through from `needs_for`**

In `daemon/setup.py`, replace the `providers_for` call (currently lines 958-962):

```python
    providers = providers_for(
        preset,
        voice_enabled=_truthy(env.get("DAEMON_VOICE_ENABLED", "")),
        hosted=hosted,
        # The wizard has no voice-provider question yet (it is admin-only), so an
        # absent value means the field default rather than "nobody chose" - unlike
        # `hosted`, this field *has* a default, and it is `gemini`.
        voice_provider=env.get("DAEMON_VOICE_PROVIDER", "")
        or _settings_default("DAEMON_VOICE_PROVIDER"),
    )
```

- [ ] **Step 5: Stop gating the voice question on the preset**

In `daemon/setup.py`, replace `_choose_voice`'s signature and delete its gate (currently
lines 1944-1957) so the function begins:

```python
    def _choose_voice(self, env: Mapping[str, str], updates: dict[str, str]) -> bool:
        # No preset gate: every preset can carry voice now (docs/adr/0012), and the
        # trade is stated in the question itself rather than by refusing to ask it.
        raw = env.get("DAEMON_VOICE_ENABLED", "")
```

Keep the rest of the function (the `was` line through `return enabled`) exactly as it is.
Then update the call site at line 1651:

```python
        voice = self._choose_voice(env, updates)
```

`preset` and `hosted` are still used by the lines above and below it, so neither local
becomes unused.

- [ ] **Step 6: Rewrite the offline menu entry**

In `daemon/setup.py`, replace the `offline` entry of `PRESET_CHOICES` (currently lines
193-205) — keeping the `Choice(...)` shape and the "Needs Ollama" line's role:

```python
    Choice(
        "offline",
        "Everything on this machine. No keys and no accounts, unless you add voice.",
        (
            "Conversation, the daily reflection and the decision to speak first all "
            "run here through Ollama. With voice off, nothing leaves the machine.",
            "Voice is the one thing you can opt into: native audio needs a hosted "
            "model, so turning it on sends audio to the provider you choose, with "
            "your own key. Leaving it off is exactly what makes the privacy promise "
            "true instead of aspirational (docs/PLAN.md 7).",
            "Needs Ollama and two local models. No API keys and no accounts until "
            "you turn voice on.",
        ),
```

- [ ] **Step 7: Update the two menu-copy tests**

In `tests/test_setup.py:300`, the assertion `assert "Voice unavailable" in flat(result.out)`
is now false by design. Replace the test body's assertions with:

```python
    assert "unless you add voice" in flat(result.out)
    # And the argument is not printed unasked: folding is what made the menu
    # short enough to read while choosing.
    assert "privacy promise" not in result.out
```

and update its comment's second sentence to "a person choosing offline must not discover
afterwards that voice costs them the promise". `test_the_reasoning_is_one_keypress_away_and_loses_nothing`
needs no change: it asserts `"privacy promise true instead of aspirational"` and
`"docs/PLAN.md 7"`, both of which the new copy still contains, plus every
`choice.summary`, which it reads from `setup.PRESET_CHOICES` rather than hardcoding.

- [ ] **Step 8: Run the full gate**

```bash
python3 -m pytest && python3 -m ruff check . && python3 scripts/check_docs.py
```

Expected: all pass. Two likely failures and their causes:
- `F401 unused import` on `GEMINI` in `daemon/setup.py` — the deleted gate was its last
  user in that module. Check with `grep -n "\bGEMINI\b" daemon/setup.py`; if the only hit is
  the import, remove it from the `from daemon.config import (...)` block.
- A wizard walkthrough test failing on answer count (e.g. `tests/test_setup.py:2280`'s
  scripted answers) — `offline` now asks the voice question it used to skip, so any script
  that drives `offline` needs one more answer (`"n"`) after the preset choice. Fix the
  script, not the wizard: being asked is the feature.

- [ ] **Step 9: Commit**

```bash
git add daemon/config.py daemon/setup.py tests/test_config.py tests/test_setup.py
git commit -m "setup: the voice question is asked under every preset, and asks for the right key"
```

---

### Task 3: the record, and the docs that promised the coupling

**Files:**
- Create: `docs/adr/0012-voice-is-its-own-axis.md`
- Modify: `docs/adr/README.md` (index row), `docs/PLAN.md:138-144` (§3.2 table)

**Interfaces:**
- Consumes: the behaviour from Tasks 1 and 2.
- Produces: nothing code-facing.

- [ ] **Step 1: Write the ADR**

Create `docs/adr/0012-voice-is-its-own-axis.md`. Follow the house format — Context ·
Decision · Consequences, plus a line saying what would change our mind (`docs/adr/README.md`):

```markdown
# 0012 — Voice is its own axis, not a preset tier

**Status:** accepted · 2026-08-11

## Context

ADR 0007 split two axes: a preset answers *where work runs*, `DAEMON_HOSTED_PROVIDER`
answers *whose model*. Voice was later given a third, `DAEMON_VOICE_PROVIDER`, and
`Settings.routing` overrode `CHAT_VOICE` with it — the config comment says "voice
provider is its own axis" in as many words.

One thing contradicted that. `PRESETS["offline"]` has no `CHAT_VOICE` row, and the
validator turned the absence into a rule: `DAEMON_VOICE_ENABLED=true` under `offline`
failed at startup with "preset routes no voice task". So **local text with hosted
voice was unconfigurable** — a combination with an obvious owner (keep conversation,
reflection and the proactive judge on this machine; use Gemini or OpenAI for spoken
turns) and no way to ask for it.

The absence was defended as what made the privacy promise true. It is not what makes
it true. `docs/PLAN.md:551` states the promise as "**text mode** + local models —
nothing leaves the device", and one line above it: "turn voice on and audio goes to
the provider you chose (BYOK); leave it off and it does not". The promise was already
conditioned on the switch. Only the table disagreed.

The admin redesign is what surfaced it: a page whose whole shape is "provider, then
that provider's model" had to grey out the voice toggle under a local provider and
explain a limitation that no requirement asked for.

## Decision

`Settings.routing` adds the `CHAT_VOICE` row whenever `voice_enabled` is on, under
every preset, mapped to `voice_provider`. The row is still kept when a preset carries
one and voice is off, so `route_for` can go on answering "voice is off
(DAEMON_VOICE_ENABLED)" instead of "this preset does not route it".

The validator problem that refused voice under `offline` is deleted as unreachable.
`providers_for` — which decides the keys onboarding asks for — contributes
`voice_provider` while voice is on, under every preset, instead of reading the preset
table's literal `CHAT_VOICE` entry. That second half also fixes a standing bug: a user
who chose OpenAI voice was asked for a Gemini key.

`PRESETS` is unchanged. No new preset, no new setting, no migration: every existing
`.env` keeps its meaning, and the configurations that used to fail at startup now
start.

## Consequences

- "Local text, hosted voice" works. `offline` + `DAEMON_VOICE_ENABLED=true` needs the
  voice provider's key and its own realtime model id, which the existing voice-model
  checks already demand.
- `offline` is no longer a promise by construction. Turning voice on there sends audio
  out, and the wizard's preset menu now says so instead of saying voice is
  unavailable. The promise moved from "this preset cannot" to "this switch is off",
  which is where PLAN.md always had it.
- One fewer message for one situation: voice-off now reports the switch under every
  preset rather than the preset under one of them.
- The admin's provider picker no longer needs to disable voice for a local text
  provider — the reason that rule existed is gone.

## What would change our mind

A voice path that could run locally. Every reason above assumes native audio means a
hosted model (ADR 0004). If a local native-audio session became real, "voice implies
something leaves the machine" stops holding and this record needs revisiting rather
than extending.
```

- [ ] **Step 2: Add the index row**

In `docs/adr/README.md`, after the `0011` row:

```markdown
| [0012](0012-voice-is-its-own-axis.md) | Voice is its own axis, not a preset tier | accepted |
```

- [ ] **Step 3: Update PLAN.md §3.2**

`docs/PLAN.md:138-144` currently says `불가` in the offline row's 음성 column. Replace the
table and the sentence under it:

```markdown
| 프리셋 | 대화 | 성찰 | 선제성 판단 | 음성 |
|---|---|---|---|---|
| **완전 오프라인** | 로컬 | 로컬 | 로컬 | 켜면 가능(상용) |
| **균형** (기본값) | 상용 | 상용 | 로컬 | 가능 |
| **품질 우선** | 상용 | 상용 | 상용 | 가능 |

고급 설정에서 개별 덮어쓰기 허용. **"완전 오프라인"이 실재해야 §7의 약속이 참이 된다** —
단 그 약속을 지탱하는 것은 프리셋 표가 아니라 `DAEMON_VOICE_ENABLED`가 꺼져 있다는 사실이다
(§7의 문구도 "텍스트 모드 + 로컬 모델"이다). 음성은 프리셋의 등급이 아니라 별도 축이다:
`docs/adr/0012`.
```

- [ ] **Step 4: Check nothing else still promises the coupling**

```bash
grep -rnE "routes no voice task|Voice unavailable|voice is not available|음성.*불가" --include="*.py" --include="*.md" --include="*.example" . | grep -v node_modules
```

Expected: no hits outside this plan file and `docs/adr/0012`. Any hit in `.env.example`,
`docs/ARCHITECTURE.md`, `daemon/CLAUDE.md` or a `README` is a doc this change falsified —
correct it in the same commit, matching that file's language.

- [ ] **Step 5: Run the full gate**

```bash
python3 -m pytest && python3 -m ruff check . && python3 scripts/check_docs.py
```

Expected: all pass. `check_docs.py` is the one that catches a path typo in the new ADR link.

- [ ] **Step 6: Commit**

```bash
git add docs/
git commit -m "docs: the promise rests on the switch, not on the preset table"
```

---

## Verification (whole change)

- [ ] `python3 -m pytest` — full suite green.
- [ ] `python3 -m ruff check .` — clean.
- [ ] `python3 scripts/check_docs.py` — documented paths exist.
- [ ] The configuration this change exists for actually starts, from a real `.env`:

```bash
mkdir -p ~/daemon-axis-check && printf 'DAEMON_PRESET=offline\nDAEMON_OLLAMA_MODEL=gemma3:4b\nDAEMON_VOICE_ENABLED=true\nDAEMON_VOICE_PROVIDER=gemini\nDAEMON_GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview\nGEMINI_API_KEY=dummy\n' > ~/daemon-axis-check/.env
```

then run `daemon doctor` from that directory (`.env` is read relative to the working
directory). Expected: `doctor` reports `chat_voice → gemini` in its routing section and does
**not** report an unknown-preset or "routes no voice task" failure. A green unit suite is not
evidence for this one — the whole point of the change is what a real `.env` is allowed to say.
