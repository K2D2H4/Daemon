"""Settings, the two routing axes, and the clock helper.

The clock tests live here rather than in their own file because clock.py is the
other zero-dependency foundation module; keeping them together avoids a file
that holds four assertions.
"""

from __future__ import annotations

import os
import pathlib
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from daemon import clock
from daemon.config import ConfigError, Route, Settings, providers_for
from daemon.tasks import Task


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's exported ANTHROPIC_API_KEY must not decide whether the
    missing-key tests pass."""
    for name in list(os.environ):
        if name.startswith(("DAEMON_", "TELEGRAM_")) or name.endswith("_API_KEY"):
            monkeypatch.delenv(name, raising=False)


def make_settings(**kwargs: Any) -> Settings:
    """A provider is supplied unless a test is about its absence (DAEMON_PROVIDER
    has no default; a hosted task with no provider fails at startup, ADR 0007/0014)."""
    kwargs.setdefault("provider", "anthropic")
    return Settings(_env_file=None, **kwargs)


# --- the two routing axes (docs/PLAN.md 3.2, ADR 0014) ----------------------


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
    settings = make_settings(
        provider="anthropic", proactive_judge_local=True,
        anthropic_model="c", anthropic_api_key="k",
    )
    assert settings.routing[Task.PROACTIVE_JUDGE] == "ollama"
    for task in _hosted_tasks():
        assert settings.routing[task] == "anthropic"
    assert settings.routing[Task.EMBED] == "ollama"


def test_hosted_with_hosted_judge_reconstructs_quality() -> None:
    settings = make_settings(
        provider="gemini", proactive_judge_local=False,
        gemini_model="g", gemini_api_key="k",
    )
    assert settings.routing[Task.PROACTIVE_JUDGE] == "gemini"


def test_embed_is_always_ollama() -> None:
    for provider in ("ollama", "anthropic", "openai", "gemini", "openai_compatible"):
        s = Settings(_env_file=None, provider=provider,
                     anthropic_model="c", openai_model="o", gemini_model="g",
                     openai_compatible_model="q",
                     openai_compatible_base_url="https://x/v1",
                     anthropic_api_key="k", openai_api_key="k", gemini_api_key="k",
                     openai_compatible_api_key="k")
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


# --- failing loudly at startup ----------------------------------------------


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown DAEMON_PROVIDER"):
        make_settings(provider="cheap")


def test_missing_key_for_a_routed_provider_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY is empty"):
        make_settings()


def test_ollama_provider_needs_no_keys() -> None:
    assert make_settings(provider="ollama").route_for(Task.CHAT_TEXT).provider == "ollama"


def test_missing_model_for_a_routed_provider_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_OLLAMA_MODEL"):
        make_settings(provider="ollama", ollama_model="")


def test_unknown_provider_in_an_override_is_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown provider 'llamafile'"):
        make_settings(provider="ollama", route_overrides={Task.REFLECTION: "llamafile"})


def test_override_requiring_a_key_fails_without_it() -> None:
    with pytest.raises(ConfigError, match="task reflection routes to 'anthropic'"):
        make_settings(provider="ollama", route_overrides={Task.REFLECTION: "anthropic"})


# --- voice is its own axis (ADR 0012) ----------------------------------------


def test_the_offline_preset_can_have_hosted_voice() -> None:
    # Local text with hosted audio is a real configuration: the privacy promise in
    # docs/PLAN.md 7 is conditioned on *text mode*, not on the preset table.
    settings = make_settings(
        provider="ollama",
        voice_enabled=True,
        gemini_api_key="k",
        gemini_live_model="m",
    )

    assert settings.routing[Task.CHAT_VOICE] == "gemini"
    assert settings.route_for(Task.CHAT_VOICE) == Route(provider="gemini", model="m")


def test_offline_voice_follows_the_voice_provider_not_the_preset_table() -> None:
    settings = make_settings(
        provider="ollama",
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
    for provider in ("ollama", "anthropic", "gemini"):
        settings = make_settings(
            provider=provider, anthropic_api_key="k", gemini_api_key="k", gemini_model="g"
        )

        # One message for one situation: before this, `offline` said "does not route
        # chat_voice" while `balanced` said "voice is off" for the same reason.
        with pytest.raises(ConfigError, match="voice is off"):
            settings.route_for(Task.CHAT_VOICE)


def test_voice_on_needs_its_own_model_under_the_offline_preset_too() -> None:
    # The validator's voice-model check is what refuses voice now; the routing
    # computation no longer refuses anything on its own.
    with pytest.raises(ConfigError, match="DAEMON_GEMINI_LIVE_MODEL is empty"):
        make_settings(provider="ollama", voice_enabled=True, gemini_api_key="k")


def test_providers_for_asks_for_the_voice_key_under_the_offline_preset() -> None:
    assert providers_for(
        provider="ollama", proactive_judge_local=True, voice_enabled=True, voice_provider="gemini"
    ) == ["gemini", "ollama"]
    assert providers_for(
        provider="ollama", proactive_judge_local=True, voice_enabled=False, voice_provider="gemini"
    ) == ["ollama"]


def test_providers_for_follows_the_voice_provider_not_the_table() -> None:
    # Reading CHAT_VOICE straight from the provider axis asked a user who chose
    # OpenAI voice for a Gemini key.
    providers = providers_for(
        provider="anthropic", proactive_judge_local=True,
        voice_enabled=True, voice_provider="openai",
    )

    assert "openai" in providers
    assert "gemini" not in providers


def test_a_blank_endpointing_value_means_the_server_default(tmp_path: Any) -> None:
    """`KEY=` in a .env file is an empty *string*, not an absent key.

    Goes through a real .env file rather than keyword arguments, because that is the
    gap: `make_settings` passes native Python values, so 1457 tests never exercised
    the dotenv text path that turns `KEY=` into `""` - and pydantic does not coerce
    that to None for an `int | None`. A .env copied from .env.example, where these
    ship blank on purpose, took the whole daemon down with `int_parsing`, and did it
    as `pydantic.ValidationError` rather than `ConfigError` - so `daemon doctor`,
    whose whole job is explaining the breakage, printed a traceback instead.
    """
    env = tmp_path / ".env"
    env.write_text(
        "DAEMON_PROVIDER=anthropic\n"
        "ANTHROPIC_API_KEY=k\n"
        "DAEMON_VOICE_START_SENSITIVITY=\n"
        "DAEMON_VOICE_END_SENSITIVITY=\n"
        "DAEMON_VOICE_PREFIX_PADDING_MS=\n"
        "DAEMON_VOICE_SILENCE_DURATION_MS=\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=str(env))

    assert settings.voice_prefix_padding_ms is None
    assert settings.voice_silence_duration_ms is None
    assert settings.voice_start_sensitivity == ""


def test_the_voice_lines_shipped_in_the_example_env_load(tmp_path: Any) -> None:
    """The example file exists to be copied, so the lines it ships have to survive
    being read back as a .env - verbatim, not paraphrased into this test.

    Only the voice block. `DAEMON_ROUTE_OVERRIDES=` in the same file has had the
    identical empty-string problem since f97595b ("Wire M1a end to end"), where it
    raises `SettingsError` on a `dict[Task, str]` - a pre-existing defect this change
    did not cause and does not touch.
    """
    example = pathlib.Path(__file__).resolve().parents[1] / ".env.example"
    voice_lines = [
        line
        for line in example.read_text(encoding="utf-8").splitlines()
        if line.startswith("DAEMON_VOICE_")
    ]
    assert voice_lines, "the example file stopped shipping the voice settings"
    env = tmp_path / ".env"
    env.write_text(
        "\n".join([*voice_lines, "DAEMON_PROVIDER=anthropic", "ANTHROPIC_API_KEY=k"]),
        encoding="utf-8",
    )

    settings = Settings(_env_file=str(env))

    assert settings.voice_silence_duration_ms is None
    assert settings.voice_prefix_padding_ms is None


def test_a_real_endpointing_value_still_arrives_as_a_number(tmp_path: Any) -> None:
    """The blank-is-unset rule must not swallow a value someone actually set."""
    env = tmp_path / ".env"
    env.write_text(
        "DAEMON_PROVIDER=anthropic\nANTHROPIC_API_KEY=k\n"
        "DAEMON_VOICE_SILENCE_DURATION_MS=400\n"
        "DAEMON_VOICE_PREFIX_PADDING_MS=0\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=str(env))

    assert settings.voice_silence_duration_ms == 400
    # Zero is a choice, not an absence.
    assert settings.voice_prefix_padding_ms == 0


def test_a_misspelled_speech_sensitivity_fails_at_startup() -> None:
    """Left to the wire, the server closes with 1007 and the session classifies
    that as permanent - so a typo would take voice mode out entirely rather than
    fail the setting that caused it."""
    with pytest.raises(ConfigError, match="DAEMON_VOICE_START_SENSITIVITY"):
        make_settings(voice_start_sensitivity="LOW")
    with pytest.raises(ConfigError, match="DAEMON_VOICE_END_SENSITIVITY"):
        make_settings(voice_end_sensitivity="medium")


def test_an_unknown_gemini_live_voice_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_GEMINI_LIVE_VOICE"):
        make_settings(provider="ollama", gemini_live_voice="NotAVoice")


def test_a_known_gemini_live_voice_and_empty_both_construct() -> None:
    assert make_settings(provider="ollama", gemini_live_voice="Kore").gemini_live_voice == "Kore"
    assert make_settings(provider="ollama", gemini_live_voice="").gemini_live_voice == ""


def test_endpointing_is_unset_by_default_so_the_server_decides() -> None:
    """~800 ms of silence is the server's own default. A default of our own here
    would be a number nobody measured, presented as a decision."""
    settings = make_settings(anthropic_api_key="k")

    assert settings.voice_start_sensitivity == ""
    assert settings.voice_end_sensitivity == ""
    assert settings.voice_prefix_padding_ms is None
    assert settings.voice_silence_duration_ms is None


def test_disabled_voice_needs_no_voice_key_but_enabled_voice_does() -> None:
    # The M1a default install is text-only, so it must not be asked for a key
    # it will never use.
    make_settings(anthropic_api_key="k")

    with pytest.raises(ConfigError, match="GEMINI_API_KEY is empty"):
        make_settings(anthropic_api_key="k", gemini_model="g", voice_enabled=True)


def test_routing_table_hides_voice_until_it_is_enabled() -> None:
    settings = make_settings(anthropic_api_key="k")

    assert Task.CHAT_VOICE not in settings.routing_table()
    assert settings.routing_table()[Task.CHAT_TEXT] == Route("anthropic", settings.anthropic_model)


def test_offline_with_voice_on_reaches_routing_table_and_active_tasks() -> None:
    # `routing`/`route_for` are necessary but not sufficient: `routing_table()` and
    # `active_tasks` are what `LLMGateway`/`_build_providers` actually read at boot
    # (daemon/app.py), so this is the assertion that an `offline` install now
    # builds a hosted voice provider client - not just that the table says so.
    settings = make_settings(
        provider="ollama",
        voice_enabled=True,
        gemini_api_key="k",
        gemini_live_model="m",
    )

    assert Task.CHAT_VOICE in settings.active_tasks
    assert settings.routing_table()[Task.CHAT_VOICE] == Route("gemini", "m")


# --- overrides and fallback -------------------------------------------------


def test_override_wins_over_the_preset() -> None:
    settings = make_settings(
        anthropic_api_key="k", route_overrides={Task.CHAT_TEXT: "ollama"}
    )

    assert settings.route_for(Task.CHAT_TEXT) == Route("ollama", "qwen3:14b")
    assert settings.route_for(Task.REFLECTION).provider == "anthropic"


def test_no_fallback_by_default() -> None:
    assert make_settings(provider="ollama").fallback_route() is None


def test_an_empty_fallback_provider_means_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The only way a dotenv can say "unset" is an empty value, and .env.example
    ships exactly that documented as "the error propagates". `None` is the sentinel
    the rest of the code tests against, so a blank has to become one here: it used to
    stay `""`, which failed startup as an unknown provider and - had that check ever
    been passed - would have handed the gateway a fallback route to nowhere."""
    monkeypatch.setenv("DAEMON_PROVIDER", "ollama")
    monkeypatch.setenv("DAEMON_FALLBACK_PROVIDER", "")

    settings = Settings(_env_file=None)

    assert settings.fallback_provider is None
    assert settings.fallback_route() is None


def test_configured_fallback_resolves_to_a_route() -> None:
    settings = make_settings(anthropic_api_key="k", fallback_provider="ollama")

    assert settings.fallback_route() == Route("ollama", "qwen3:14b")


def test_fallback_to_a_keyless_provider_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_FALLBACK_PROVIDER"):
        make_settings(provider="ollama", fallback_provider="anthropic")


# --- M1b fields: recall, voice, residency -----------------------------------


def test_recall_and_service_defaults() -> None:
    settings = make_settings(provider="ollama")

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
        make_settings(provider="ollama", embed_model="")


def test_a_recall_limit_below_one_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_RECALL_LIMIT"):
        make_settings(provider="ollama", recall_limit=0)


def test_a_non_positive_half_life_fails_at_startup() -> None:
    # Zero or negative makes the recency term undefined rather than aggressive.
    with pytest.raises(ConfigError, match="DAEMON_RECALL_HALF_LIFE_DAYS"):
        make_settings(provider="ollama", recall_half_life_days=0)


def test_a_wake_gate_without_a_live_model_fails_at_startup() -> None:
    """Keyed on the wake gate rather than on `voice_enabled`, since the merge that
    made one switch govern the local speaker too: `/usr/bin/say` needs no model, so
    refusing to load over a missing one took the `offline` install's only voice
    feature away along with the daemon. The wake gate keeps the check because
    opening a hosted session is the only thing it does."""
    with pytest.raises(ConfigError, match="DAEMON_GEMINI_LIVE_MODEL is empty"):
        make_settings(
            anthropic_api_key="k",
            gemini_model="g",
            gemini_api_key="k",
            voice_enabled=True,
            wake_enabled=True,
            wake_aliases="벨라",
        )


def test_voice_provider_defaults_to_gemini() -> None:
    assert make_settings(provider="ollama").voice_provider == "gemini"


def test_unknown_voice_provider_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_VOICE_PROVIDER"):
        make_settings(provider="ollama", voice_provider="anthropic")


def test_openai_voice_route_uses_the_realtime_model_and_provider() -> None:
    s = make_settings(
        proactive_judge_local=False, voice_enabled=True, voice_provider="openai",
        openai_realtime_model="gpt-realtime", openai_api_key="sk-o", gemini_api_key="g",
        anthropic_api_key="a",
    )
    route = s.route_for(Task.CHAT_VOICE)
    assert route.provider == "openai"
    assert route.model == "gpt-realtime"


def test_openai_voice_requires_its_own_model_and_key() -> None:
    """Same move as the gemini one above: these guard a hosted session, so at load
    time they fire for the wake gate. `run_voice` applies the identical list when a
    session actually opens - see tests/test_voice_conversation.py."""
    wake = {"wake_enabled": True, "wake_aliases": "벨라"}
    with pytest.raises(ConfigError, match="DAEMON_OPENAI_REALTIME_MODEL"):
        make_settings(proactive_judge_local=False, voice_enabled=True, voice_provider="openai",
                      openai_api_key="sk-o", gemini_api_key="g", anthropic_api_key="a", **wake)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        make_settings(
            proactive_judge_local=False, voice_enabled=True, voice_provider="openai",
            openai_realtime_model="gpt-realtime", gemini_api_key="g", anthropic_api_key="a",
            **wake,
        )


def test_unknown_openai_realtime_voice_fails() -> None:
    with pytest.raises(ConfigError, match="DAEMON_OPENAI_REALTIME_VOICE"):
        make_settings(provider="ollama", openai_realtime_voice="not-a-voice")
    assert (
        make_settings(provider="ollama", openai_realtime_voice="alloy").openai_realtime_voice
        == "alloy"
    )
    assert make_settings(provider="ollama", openai_realtime_voice="").openai_realtime_voice == ""


def test_a_service_label_that_is_really_a_path_is_rejected() -> None:
    # The label becomes a filename under ~/Library/LaunchAgents.
    with pytest.raises(ConfigError, match="not a usable label"):
        make_settings(provider="ollama", service_label="../../evil")


def test_new_env_vars_are_read_with_their_documented_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAEMON_PROVIDER", "ollama")
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


# --- M4 fields: persona evolution -------------------------------------------


def test_persona_evolution_defaults() -> None:
    settings = make_settings(provider="ollama")

    assert settings.persona_max_active_rules == 20
    assert settings.persona_max_new_per_cycle == 3
    assert settings.persona_min_observations == 5


def test_persona_evolution_env_vars_are_read_with_their_documented_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAEMON_PROVIDER", "ollama")
    monkeypatch.setenv("DAEMON_PERSONA_MAX_ACTIVE_RULES", "10")
    monkeypatch.setenv("DAEMON_PERSONA_MAX_NEW_PER_CYCLE", "1")
    monkeypatch.setenv("DAEMON_PERSONA_MIN_OBSERVATIONS", "8")

    settings = Settings(_env_file=None)

    assert settings.persona_max_active_rules == 10
    assert settings.persona_max_new_per_cycle == 1
    assert settings.persona_min_observations == 8


@pytest.mark.parametrize(
    "field",
    ["persona_max_active_rules", "persona_max_new_per_cycle", "persona_min_observations"],
)
def test_a_negative_persona_setting_fails_at_startup(field: str) -> None:
    with pytest.raises(ConfigError, match="DAEMON_PERSONA_"):
        make_settings(provider="ollama", **{field: -1})


# --- misc fields ------------------------------------------------------------


def test_allowlist_is_parsed_from_a_comma_separated_string() -> None:
    settings = make_settings(provider="ollama", telegram_allowed_user_ids="123, 456 ,")

    assert settings.telegram_allowed_user_ids == ("123", "456")


def test_an_empty_route_overrides_means_no_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """.env.example ships `DAEMON_ROUTE_OVERRIDES=` and tells the reader to copy the
    file, so this is the value most installs actually hold. It used to kill the whole
    load - `daemon run`, `daemon doctor`, everything - with a SettingsError naming
    neither the file nor the problem, because a complex field is JSON-decoded before
    any validator runs and `json.loads("")` raises."""
    monkeypatch.setenv("DAEMON_PROVIDER", "ollama")
    monkeypatch.setenv("DAEMON_ROUTE_OVERRIDES", "")

    settings = Settings(_env_file=None)

    assert settings.route_overrides == {}
    assert settings.routing[Task.REFLECTION] == "ollama"


def test_route_overrides_reads_the_json_documented_in_env_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAEMON_PROVIDER", "ollama")
    monkeypatch.setenv("DAEMON_ROUTE_OVERRIDES", '{"reflection": "ollama"}')

    settings = Settings(_env_file=None)

    assert settings.route_overrides == {Task.REFLECTION: "ollama"}


def test_unparseable_route_overrides_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loud beats degraded: treating a typo as "no overrides" would route a task to
    the preset's provider instead of the chosen one and say nothing about it."""
    monkeypatch.setenv("DAEMON_PROVIDER", "ollama")
    monkeypatch.setenv("DAEMON_ROUTE_OVERRIDES", "reflection: ollama")

    with pytest.raises(ValidationError, match="DAEMON_ROUTE_OVERRIDES"):
        Settings(_env_file=None)


def test_env_vars_are_read_with_their_documented_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # .env.example is the contract with the user; these names must keep working.
    monkeypatch.setenv("DAEMON_PROVIDER", "ollama")
    monkeypatch.setenv("DAEMON_OLLAMA_MODEL", "llama3.2")
    monkeypatch.setenv("DAEMON_DATA_DIR", "/tmp/daemon-test")

    settings = Settings(_env_file=None)

    assert settings.provider == "ollama"
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
        Settings(_env_file=None, ollama_model="gemma3:4b")


def test_offline_needs_no_hosted_provider() -> None:
    """Nothing in it is hosted, so the question does not apply."""
    settings = Settings(_env_file=None, provider="ollama", ollama_model="gemma3:4b")

    assert set(settings.routing.values()) == {"ollama"}


@pytest.mark.parametrize("provider", ["anthropic", "openai", "gemini"])
def test_the_chosen_provider_answers_the_hosted_tasks(provider: str) -> None:
    settings = make_settings(
        provider=provider,
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


# --- proactivity brakes (M3) ------------------------------------------------
# Every one of these settings makes it speak *less*, so a value that cannot be
# understood has to fail here rather than at 03:00.


def test_quiet_hours_that_are_not_a_range_fail_at_startup() -> None:
    """The gate's honest answer to an unparseable window is to block everything,
    which would turn the product off by typo and keep it off. Loud beats degraded.
    """
    with pytest.raises(ConfigError, match="DAEMON_PROACTIVE_QUIET_HOURS"):
        make_settings(provider="ollama", proactive_quiet_hours="11pm to 9am")


def test_an_impossible_hour_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_PROACTIVE_QUIET_HOURS"):
        make_settings(provider="ollama", proactive_quiet_hours="25:00-09:00")


def test_the_default_quiet_window_wraps_midnight() -> None:
    """Wrapping is the ordinary case for this setting, not the edge case."""
    settings = make_settings(provider="ollama")
    assert settings.proactive_quiet_hours == "23:00-09:00"


def test_an_empty_quiet_window_is_allowed() -> None:
    assert make_settings(provider="ollama", proactive_quiet_hours="").proactive_quiet_hours == ""


def test_a_negative_budget_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_PROACTIVE_DAILY_BUDGET"):
        make_settings(provider="ollama", proactive_daily_budget=-1)


def test_a_zero_silence_threshold_fails_at_startup() -> None:
    """At zero every tick is a silence candidate, which is the stalker end of the
    failure the M3 gate is judged on."""
    with pytest.raises(ConfigError, match="DAEMON_PROACTIVE_SILENCE_HOURS"):
        make_settings(provider="ollama", proactive_silence_hours=0)


def test_proactivity_and_voice_are_off_by_default() -> None:
    """Both off. `voice_enabled` now doubles as the proactive-speaker switch
    (DAEMON_PROACTIVE_SPEAKER_ENABLED is gone), so a fresh install that never
    turns on proactivity, or never turns on voice, gets no sound from either
    path - not just no Telegram message."""
    settings = make_settings(provider="ollama")
    assert settings.proactive_enabled is False
    assert settings.voice_enabled is False


def test_the_speaker_switch_is_gone() -> None:
    """One switch, not two. They were split because a voice in a meeting is an
    accident and a Telegram message is not - but the gate now carries seven rules
    for exactly that, and a second switch only made "voice on" mean two things.

    `provider="ollama"` rather than a bare `Settings()`: an empty `DAEMON_PROVIDER`
    with a hosted task routed is a startup error and this test has nothing to do
    with that - it is here to fail on a stray `hasattr`, not on an unrelated
    startup check."""
    assert not hasattr(make_settings(provider="ollama"), "proactive_speaker_enabled")


def test_an_unset_legacy_speaker_switch_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pydantic must not reject an .env that still carries the old key: the user
    upgrading is the whole point, and a settings file that refuses to load takes
    the conversation loop down with it."""
    monkeypatch.setenv("DAEMON_PROACTIVE_SPEAKER_ENABLED", "true")
    settings = make_settings(provider="ollama")  # must not raise
    assert settings.voice_enabled in (True, False)


def test_an_unset_legacy_open_loop_budget_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`proactive_open_loop_budget` is gone, replaced by `proactive_kind_budgets`
    (docs/PLAN.md 6.2). The same upgrade concern as the speaker switch above: an
    .env from before this change still sets the old key, and `extra="ignore"` is
    what keeps that inert rather than a startup failure."""
    monkeypatch.setenv("DAEMON_PROACTIVE_OPEN_LOOP_BUDGET", "1")
    settings = make_settings(provider="ollama")  # must not raise
    assert not hasattr(settings, "proactive_open_loop_budget")
    assert settings.proactive_kind_budgets["open_loop"] == 2


def test_kind_budgets_reads_the_documented_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """`proactive_kind_budgets` used to have no `DAEMON_` alias at all, so nothing
    in `.env` could reach it."""
    monkeypatch.setenv("DAEMON_PROVIDER", "ollama")
    monkeypatch.setenv("DAEMON_PROACTIVE_KIND_BUDGETS", '{"open_loop": 1, "silence": 1}')

    settings = Settings(_env_file=None)

    assert settings.proactive_kind_budgets == {"open_loop": 1, "silence": 1}


def test_an_empty_kind_budgets_means_no_per_kind_ceilings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAEMON_PROVIDER", "ollama")
    monkeypatch.setenv("DAEMON_PROACTIVE_KIND_BUDGETS", "")

    settings = Settings(_env_file=None)

    assert settings.proactive_kind_budgets == {}


def test_unparseable_kind_budgets_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DAEMON_PROVIDER", "ollama")
    monkeypatch.setenv("DAEMON_PROACTIVE_KIND_BUDGETS", "open_loop: 1")

    with pytest.raises(ValidationError, match="DAEMON_PROACTIVE_KIND_BUDGETS"):
        Settings(_env_file=None)


def test_an_unknown_kind_in_the_budget_table_is_a_config_error() -> None:
    """The regression: `{"open_loop": 99, "nonsense": -5}` used to load clean with
    no validator at all."""
    with pytest.raises(ConfigError, match="unknown kind 'nonsense'"):
        make_settings(provider="ollama", proactive_kind_budgets={"nonsense": 1})


def test_a_negative_kind_ceiling_is_a_config_error() -> None:
    with pytest.raises(ConfigError, match="non-negative"):
        make_settings(provider="ollama", proactive_kind_budgets={"open_loop": -5})


def test_a_budget_of_zero_is_allowed_as_a_way_to_silence_it() -> None:
    """Zero is a legitimate answer - keep generating candidates, never speak - and
    is how someone tunes it down without losing the label history."""
    assert (
        make_settings(provider="ollama", proactive_daily_budget=0).proactive_daily_budget == 0
    )


# --- tool use ---------------------------------------------------------------


def test_tools_are_on_and_full_by_default() -> None:
    """On, because a companion that cannot open a file on the machine it lives on is
    the definition unmet (docs/PLAN.md 1). And `full`, deliberately: on the owner's
    own machine, a prompt before every action is the "chat with extra steps" the
    product exists not to be. What keeps that from being reckless is not a mode - it
    is the origin gate, which no mode can switch off (a turn that is not the owner's
    own words reaches no tool). The browser group - the one that reads an
    authenticated session the owner never named - is still off behind its own
    switch. MCP is on, but launches nothing until a server is configured."""
    settings = make_settings(provider="ollama", ollama_model="gemma3:4b")
    assert settings.tools_enabled is True
    assert settings.tools_mode == "full"
    assert settings.tools_allowlist == ()
    assert settings.tools_roots == ("~",)
    assert settings.browser_enabled is False
    assert settings.mcp_enabled is True


def test_gemini_thinking_defaults_to_low_and_rejects_junk() -> None:
    """`low` by default - a latency decision (a Gemini 3 plain tool turn is ~3x
    faster at `low` than the default, measured). A typo fails loudly at startup, the
    same as any other bad enum here, rather than 400ing on the first turn."""
    base = dict(provider="ollama", ollama_model="gemma3:4b")
    assert make_settings(**base).gemini_thinking_level == "low"
    assert make_settings(**base, gemini_thinking_level="").gemini_thinking_level == ""
    with pytest.raises(ConfigError, match="DAEMON_GEMINI_THINKING_LEVEL"):
        make_settings(**base, gemini_thinking_level="medium")


def test_tools_can_be_switched_off_entirely() -> None:
    """The other direction has to keep working, or "on by default" becomes
    "on, and no way back"."""
    settings = make_settings(
        provider="ollama", ollama_model="gemma3:4b", DAEMON_TOOLS_ENABLED=False
    )
    assert settings.tools_enabled is False


def test_an_unknown_tool_mode_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_TOOLS_MODE"):
        make_settings(provider="ollama", ollama_model="gemma3:4b", DAEMON_TOOLS_MODE="yolo")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ls", ("ls",)),
        ("ls,git status", ("ls", "git status")),
        ("ls, git status , date", ("ls", "git status", "date")),
        ('["ls","git status"]', ("ls", "git status")),
        ("", ()),
    ],
)
def test_the_allowlist_splits_on_commas_only(raw: str, expected: tuple[str, ...]) -> None:
    """Commas only, because an entry legitimately contains a space. Splitting on
    whitespace as well would turn `git status` into two useless entries - and the
    prefix `git` alone would cover `git push`."""
    settings = make_settings(
        provider="ollama", ollama_model="gemma3:4b", DAEMON_TOOLS_ALLOWLIST=raw
    )
    assert settings.tools_allowlist == expected


def test_roots_split_the_same_way() -> None:
    """A path can contain a space too."""
    settings = make_settings(
        provider="ollama",
        ollama_model="gemma3:4b",
        DAEMON_TOOLS_ROOTS="~/Documents,~/My Projects",
    )
    assert settings.tools_roots == ("~/Documents", "~/My Projects")


def test_the_default_round_cap_matches_the_loop_constant() -> None:
    """The resident's tool-round cap is `settings.tools_max_rounds`, which `app.py`
    wires into `ConversationLoop` - `loop.MAX_TOOL_ROUNDS` is only the constructor's
    fallback, unused by the assembled app. So the two must agree, or a change to the
    loop constant silently does nothing for the running daemon. Shipped exactly that
    way once: v0.1.49 bumped the constant 6 -> 25 to stop cutting real builds short,
    while this default stayed 6, so the live cap never moved and the bug reproduced
    on the owner's machine after the update."""
    from daemon.loop import MAX_TOOL_ROUNDS

    assert make_settings(provider="ollama").tools_max_rounds == MAX_TOOL_ROUNDS


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"DAEMON_TOOLS_ROOTS": ""}, "DAEMON_TOOLS_ROOTS is empty"),
        ({"DAEMON_TOOLS_TIMEOUT_SECS": 0}, "DAEMON_TOOLS_TIMEOUT_SECS"),
        ({"DAEMON_TOOLS_TIMEOUT_SECS": -1}, "DAEMON_TOOLS_TIMEOUT_SECS"),
        ({"DAEMON_TOOLS_MAX_OUTPUT": 10}, "DAEMON_TOOLS_MAX_OUTPUT"),
        ({"DAEMON_TOOLS_MAX_ROUNDS": 0}, "DAEMON_TOOLS_MAX_ROUNDS"),
    ],
)
def test_an_incoherent_tool_setup_fails_at_startup(kwargs: dict, expected: str) -> None:
    with pytest.raises(ConfigError, match=expected):
        make_settings(
            provider="ollama",
            ollama_model="gemma3:4b",
            DAEMON_TOOLS_ENABLED=True,
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"DAEMON_TOOLS_ROOTS": ""},
        {"DAEMON_TOOLS_TIMEOUT_SECS": 0},
        {"DAEMON_TOOLS_MAX_ROUNDS": 0},
    ],
)
def test_an_install_with_tools_off_is_not_held_to_a_tool_setup(kwargs: dict) -> None:
    """The same rule voice follows: a configuration that is never reached must not be
    a reason to refuse to start. Now that tools are on by default this needs the
    switch stated - which is the point, since with them *on* the setting is reached
    and an incoherent one should fail at startup rather than mid-conversation."""
    make_settings(
        provider="ollama",
        ollama_model="gemma3:4b",
        DAEMON_TOOLS_ENABLED=False,
        **kwargs,
    )  # must not raise


def test_no_new_task_was_added_for_tool_use() -> None:
    """Tool-using chat is still `chat_text`, which is why tasks.py stayed frozen. A
    new Task would need a row in `Settings.routing`, not a preset table - and every
    task but voice must be routed regardless of which axis values are chosen."""
    settings = make_settings(anthropic_api_key="k")
    missing = set(Task) - set(settings.routing)
    assert missing <= {Task.CHAT_VOICE}


def test_the_browser_has_its_own_switch() -> None:
    """Letting Daemon act on the machine and letting it read the owner's logged-in
    browser are two decisions, so they are two settings."""
    settings = make_settings(provider="ollama", ollama_model="gemma3:4b")
    assert settings.browser_enabled is False
    assert settings.browser_app == "Google Chrome"


def test_the_browser_cannot_be_on_while_tools_are_off() -> None:
    """The browser tools are tools; with the layer off nothing would register them,
    so the setting would silently do nothing."""
    with pytest.raises(ConfigError, match="DAEMON_TOOLS_ENABLED is off"):
        make_settings(
            provider="ollama",
            ollama_model="gemma3:4b",
            DAEMON_TOOLS_ENABLED=False,
            DAEMON_BROWSER_ENABLED=True,
        )


def test_an_empty_browser_app_fails_at_startup() -> None:
    with pytest.raises(ConfigError, match="DAEMON_BROWSER_APP"):
        make_settings(
            provider="ollama",
            ollama_model="gemma3:4b",
            DAEMON_TOOLS_ENABLED=True,
            DAEMON_BROWSER_ENABLED=True,
            DAEMON_BROWSER_APP="  ",
        )


def test_screen_defaults_off() -> None:
    """Seeing the screen is its own decision, separate from browser and tools -
    off until asked for, the same as browser_enabled."""
    settings = make_settings(provider="ollama", ollama_model="gemma3:4b")
    assert settings.screen_enabled is False


def test_screen_enabled_requires_tools() -> None:
    """The screen tools are tools; with the layer off nothing would register them,
    so the setting would silently do nothing (mirrors the browser check)."""
    with pytest.raises(ConfigError, match="DAEMON_TOOLS_ENABLED"):
        make_settings(
            provider="ollama",
            ollama_model="gemma3:4b",
            DAEMON_TOOLS_ENABLED=False,
            DAEMON_SCREEN_ENABLED=True,
        )


def test_a_chromium_relative_can_be_named() -> None:
    """Brave, Arc and Edge share Chrome's AppleScript dictionary, so one setting
    covers the family."""
    settings = make_settings(
        provider="ollama",
        ollama_model="gemma3:4b",
        DAEMON_TOOLS_ENABLED=True,
        DAEMON_BROWSER_ENABLED=True,
        DAEMON_BROWSER_APP="Arc",
    )
    assert settings.browser_app == "Arc"


# --- openai_compatible: a fourth hosted provider, endpoint named by the user --


COMPAT_ENV = {
    "DAEMON_PROVIDER": "openai_compatible",
    "OPENAI_COMPATIBLE_API_KEY": "sk-compat-test",
    "DAEMON_OPENAI_COMPATIBLE_MODEL": "qwen-plus",
    "DAEMON_OPENAI_COMPATIBLE_BASE_URL": "https://api.deepseek.com/v1",
}


def test_openai_compatible_is_a_choosable_hosted_provider() -> None:
    settings = Settings(_env_file=None, **COMPAT_ENV)
    route = settings.route_for(Task.CHAT_TEXT)
    assert route == Route("openai_compatible", "qwen-plus")


def test_openai_compatible_needs_a_base_url() -> None:
    """Settings raises ConfigError at construction time - see `_check` - rather
    than exposing a separate problems() accessor, so the missing endpoint has to
    surface the same way every other provider problem here does."""
    with pytest.raises(ConfigError, match="DAEMON_OPENAI_COMPATIBLE_BASE_URL"):
        Settings(_env_file=None, **{**COMPAT_ENV, "DAEMON_OPENAI_COMPATIBLE_BASE_URL": ""})


def test_openai_compatible_base_url_drops_a_trailing_slash() -> None:
    settings = Settings(
        _env_file=None,
        **{**COMPAT_ENV, "DAEMON_OPENAI_COMPATIBLE_BASE_URL": "https://api.deepseek.com/v1/"},
    )
    assert settings.openai_compatible_base_url == "https://api.deepseek.com/v1"


def test_openai_compatible_base_url_rejects_the_full_endpoint_and_says_what_to_use() -> None:
    with pytest.raises(ValidationError) as caught:
        Settings(
            _env_file=None,
            **{
                **COMPAT_ENV,
                "DAEMON_OPENAI_COMPATIBLE_BASE_URL": (
                    "https://api.deepseek.com/v1/chat/completions"
                ),
            },
        )
    assert "https://api.deepseek.com/v1" in str(caught.value)


def test_openai_compatible_base_url_must_be_http() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            **{**COMPAT_ENV, "DAEMON_OPENAI_COMPATIBLE_BASE_URL": "api.deepseek.com"},
        )
