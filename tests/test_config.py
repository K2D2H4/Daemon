"""Settings, the three presets, and the clock helper.

The clock tests live here rather than in their own file because clock.py is the
other zero-dependency foundation module; keeping them together avoids a file
that holds four assertions.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from daemon import clock
from daemon.config import PRESETS, ConfigError, Route, Settings
from daemon.tasks import Task


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's exported ANTHROPIC_API_KEY must not decide whether the
    missing-key tests pass."""
    for name in list(os.environ):
        if name.startswith(("DAEMON_", "TELEGRAM_")) or name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)


def make_settings(**kwargs: Any) -> Settings:
    """A hosted provider is supplied unless a test is about its absence.

    DAEMON_HOSTED_PROVIDER has no default - a preset that needs a hosted model and
    has not been told which one fails at startup pointing at `daemon setup`,
    rather than quietly becoming Claude. That is the behaviour under test in
    test_a_hosted_preset_without_a_provider_says_to_run_setup; everywhere else it
    is scaffolding."""
    kwargs.setdefault("hosted_provider", "anthropic")
    return Settings(_env_file=None, **kwargs)


# --- preset routing tables (docs/PLAN.md 3.2) -------------------------------


def test_offline_preset_routes_everything_local() -> None:
    settings = make_settings(preset="offline")

    assert settings.routing == {
        Task.CHAT_TEXT: "ollama",
        Task.RECALL_ESCALATION: "ollama",
        Task.PROACTIVE_JUDGE: "ollama",
        Task.REFLECTION: "ollama",
        Task.PERSONA_RULE: "ollama",
        Task.EMBED: "ollama",
    }


def test_balanced_preset_keeps_the_proactive_judge_local() -> None:
    settings = make_settings(preset="balanced", anthropic_api_key="k", gemini_model="g")

    assert settings.routing == {
        Task.CHAT_TEXT: "anthropic",
        Task.CHAT_VOICE: "gemini",
        Task.RECALL_ESCALATION: "anthropic",
        Task.PROACTIVE_JUDGE: "ollama",
        Task.REFLECTION: "anthropic",
        Task.PERSONA_RULE: "anthropic",
        Task.EMBED: "ollama",
    }


def test_quality_preset_hosts_the_proactive_judge_too() -> None:
    settings = make_settings(preset="quality", anthropic_api_key="k", gemini_model="g")

    assert settings.routing == {
        Task.CHAT_TEXT: "anthropic",
        Task.CHAT_VOICE: "gemini",
        Task.RECALL_ESCALATION: "anthropic",
        Task.PROACTIVE_JUDGE: "anthropic",
        Task.REFLECTION: "anthropic",
        Task.PERSONA_RULE: "anthropic",
        Task.EMBED: "ollama",
    }


def test_every_task_is_routed_by_every_preset_except_voice() -> None:
    # tasks.py says: do not add a Task without adding it to the preset tables.
    for name, table in PRESETS.items():
        missing = set(Task) - set(table)
        assert missing <= {Task.CHAT_VOICE}, f"preset {name} does not route {missing}"


# --- failing loudly at startup ----------------------------------------------


def test_unknown_preset_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown DAEMON_PRESET"):
        make_settings(preset="cheap")


def test_missing_key_for_a_routed_provider_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY is empty"):
        make_settings(preset="balanced")


def test_offline_preset_needs_no_keys() -> None:
    assert make_settings(preset="offline").route_for(Task.CHAT_TEXT).provider == "ollama"


def test_missing_model_for_a_routed_provider_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_OLLAMA_MODEL"):
        make_settings(preset="offline", ollama_model="")


def test_unknown_provider_in_an_override_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown provider 'llamafile'"):
        make_settings(preset="offline", route_overrides={Task.REFLECTION: "llamafile"})


def test_override_requiring_a_key_fails_without_it() -> None:
    with pytest.raises(ConfigError, match="task reflection routes to 'anthropic'"):
        make_settings(preset="offline", route_overrides={Task.REFLECTION: "anthropic"})


# --- voice under the offline preset -----------------------------------------


def test_offline_preset_refuses_voice() -> None:
    settings = make_settings(preset="offline")

    with pytest.raises(ConfigError, match="does not route chat_voice"):
        settings.route_for(Task.CHAT_VOICE)


def test_enabling_voice_on_the_offline_preset_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="routes no voice task"):
        make_settings(preset="offline", voice_enabled=True)


def test_voice_is_refused_while_disabled_even_when_routed() -> None:
    settings = make_settings(preset="balanced", anthropic_api_key="k", gemini_model="g")

    with pytest.raises(ConfigError, match="voice is off"):
        settings.route_for(Task.CHAT_VOICE)


def test_disabled_voice_needs_no_voice_key_but_enabled_voice_does() -> None:
    # The M1a default install is text-only, so it must not be asked for a key
    # it will never use.
    make_settings(preset="balanced", anthropic_api_key="k")

    with pytest.raises(ConfigError, match="GEMINI_API_KEY is empty"):
        make_settings(
            preset="balanced", anthropic_api_key="k", gemini_model="g", voice_enabled=True
        )


def test_routing_table_hides_voice_until_it_is_enabled() -> None:
    settings = make_settings(preset="balanced", anthropic_api_key="k")

    assert Task.CHAT_VOICE not in settings.routing_table()
    assert settings.routing_table()[Task.CHAT_TEXT] == Route("anthropic", settings.anthropic_model)


# --- overrides and fallback -------------------------------------------------


def test_override_wins_over_the_preset() -> None:
    settings = make_settings(
        preset="balanced", anthropic_api_key="k", route_overrides={Task.CHAT_TEXT: "ollama"}
    )

    assert settings.route_for(Task.CHAT_TEXT) == Route("ollama", "qwen3:14b")
    assert settings.route_for(Task.REFLECTION).provider == "anthropic"


def test_no_fallback_by_default() -> None:
    assert make_settings(preset="offline").fallback_route() is None


def test_configured_fallback_resolves_to_a_route() -> None:
    settings = make_settings(
        preset="balanced", anthropic_api_key="k", fallback_provider="ollama"
    )

    assert settings.fallback_route() == Route("ollama", "qwen3:14b")


def test_fallback_to_a_keyless_provider_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_FALLBACK_PROVIDER"):
        make_settings(preset="offline", fallback_provider="anthropic")


# --- M1b fields: recall, voice, residency -----------------------------------


def test_recall_and_service_defaults() -> None:
    settings = make_settings(preset="offline")

    # bge-m3 is multilingual, which is the whole reason vectors are in M1b
    # (docs/PLAN.md 4.3): FTS5 alone misses inflected Korean.
    assert settings.embed_model == "bge-m3"
    assert settings.recall_limit == 6
    assert settings.recall_half_life_days == 30.0
    assert settings.service_label == "default"
    # No guessed native-audio model id: it would fail at the first voice turn
    # instead of at startup.
    assert settings.gemini_live_model == ""


def test_recall_with_no_embedding_model_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_EMBED_MODEL is empty"):
        make_settings(preset="offline", embed_model="")


def test_a_recall_limit_below_one_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_RECALL_LIMIT"):
        make_settings(preset="offline", recall_limit=0)


def test_a_non_positive_half_life_fails_at_startup() -> None:
    # Zero or negative makes the recency term undefined rather than aggressive.
    with pytest.raises(ConfigError, match="DAEMON_RECALL_HALF_LIFE_DAYS"):
        make_settings(preset="offline", recall_half_life_days=0)


def test_enabling_voice_without_a_live_model_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_GEMINI_LIVE_MODEL is empty"):
        make_settings(
            preset="balanced",
            anthropic_api_key="k",
            gemini_model="g",
            gemini_api_key="k",
            voice_enabled=True,
        )


def test_a_service_label_that_is_really_a_path_is_rejected() -> None:
    # The label becomes a filename under ~/Library/LaunchAgents.
    with pytest.raises(ConfigError, match="not a usable label"):
        make_settings(preset="offline", service_label="../../evil")


def test_new_env_vars_are_read_with_their_documented_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAEMON_PRESET", "offline")
    monkeypatch.setenv("DAEMON_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("DAEMON_RECALL_LIMIT", "3")
    monkeypatch.setenv("DAEMON_RECALL_HALF_LIFE_DAYS", "14")
    monkeypatch.setenv("DAEMON_SERVICE_LABEL", "second")
    monkeypatch.setenv("DAEMON_GEMINI_LIVE_MODEL", "gemini-live-x")

    settings = Settings(_env_file=None)

    assert settings.embed_model == "nomic-embed-text"
    assert settings.recall_limit == 3
    assert settings.recall_half_life_days == 14.0
    assert settings.service_label == "second"
    assert settings.gemini_live_model == "gemini-live-x"


# --- misc fields ------------------------------------------------------------


def test_allowlist_is_parsed_from_a_comma_separated_string() -> None:
    settings = make_settings(preset="offline", telegram_allowed_user_ids="123, 456 ,")

    assert settings.telegram_allowed_user_ids == ("123", "456")


def test_env_vars_are_read_with_their_documented_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # .env.example is the contract with the user; these names must keep working.
    monkeypatch.setenv("DAEMON_PRESET", "offline")
    monkeypatch.setenv("DAEMON_OLLAMA_MODEL", "llama3.2")
    monkeypatch.setenv("DAEMON_DATA_DIR", "/tmp/daemon-test")

    settings = Settings(_env_file=None)

    assert settings.preset == "offline"
    assert settings.route_for(Task.CHAT_TEXT).model == "llama3.2"
    assert str(settings.data_dir) == "/tmp/daemon-test"


# --- clock ------------------------------------------------------------------


def test_to_iso_ends_with_z_and_keeps_milliseconds() -> None:
    moment = datetime(2026, 8, 3, 7, 14, 0, 123456, tzinfo=UTC)

    assert clock.to_iso(moment) == "2026-08-03T07:14:00.123Z"


def test_to_iso_converts_other_offsets_to_utc() -> None:
    kst = datetime(2026, 8, 3, 16, 14, 0, tzinfo=timezone(timedelta(hours=9)))

    assert clock.to_iso(kst) == "2026-08-03T07:14:00.000Z"


def test_parse_iso_round_trips() -> None:
    moment = clock.now()

    assert clock.parse_iso(clock.to_iso(moment)) == moment.replace(
        microsecond=moment.microsecond // 1000 * 1000
    )


def test_now_is_timezone_aware_utc() -> None:
    assert clock.now().utcoffset() == timedelta(0)


# --- the hosted provider is chosen, never assumed ----------------------------


def test_a_hosted_preset_without_a_provider_says_to_run_setup() -> None:
    """It used to default to anthropic, so a configuration that never chose a
    provider silently became Claude - and the person who was never asked the
    question could not tell a choice from a fallback. Onboarding always runs, so
    the answer always exists by the time it is needed."""
    with pytest.raises(ConfigError, match="daemon setup"):
        Settings(_env_file=None, preset="balanced", ollama_model="gemma3:4b")


def test_offline_needs_no_hosted_provider() -> None:
    """Nothing in it is hosted, so the question does not apply."""
    settings = Settings(_env_file=None, preset="offline", ollama_model="gemma3:4b")

    assert settings.hosted_provider == ""
    assert set(settings.routing.values()) == {"ollama"}


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_the_chosen_provider_answers_the_hosted_tasks(provider: str) -> None:
    settings = make_settings(
        preset="balanced",
        hosted_provider=provider,
        ollama_model="gemma3:4b",
        anthropic_api_key="k",
        openai_api_key="k",
        gemini_api_key="k",
        anthropic_model="m",
        openai_model="m",
        gemini_model="m",
    )

    assert settings.routing[Task.CHAT_TEXT] == provider
    # The five-minute proactive check stays local whatever was chosen: it runs
    # whether or not it ever speaks, so hosted cost would accumulate for nothing.
    assert settings.routing[Task.PROACTIVE_JUDGE] == "ollama"
