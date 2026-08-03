"""Settings and the three routing presets.

docs/PLAN.md 3.2: routing is a per-Task table, not one global switch, and users
see three presets instead of seven questions. Advanced users override single
tasks.

Everything here fails loudly at construction time. A missing API key must not
surface as a broken conversation turn three hours later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from daemon.tasks import Task

OLLAMA = "ollama"
ANTHROPIC = "anthropic"
OPENAI = "openai"
GEMINI = "gemini"

# Env var carrying the API key, or None for a provider that needs no key.
# Providers listed here are nameable in a route; only the ones with a module
# under daemon/llm/providers/ can actually be built (M1a: ollama, anthropic).
PROVIDER_KEY_ENV: dict[str, str | None] = {
    OLLAMA: None,
    ANTHROPIC: "ANTHROPIC_API_KEY",
    OPENAI: "OPENAI_API_KEY",
    GEMINI: "GEMINI_API_KEY",
}

VOICE_TASKS = frozenset({Task.CHAT_VOICE})
"""Tasks that need a hosted native-audio model. The offline preset has none."""

# docs/PLAN.md 3.2. Notes on the non-obvious cells:
#   - RECALL_ESCALATION follows chat, because Lane 2 is a conversation turn.
#   - PROACTIVE_JUDGE stays local except under `quality`: it runs on a 5-minute
#     tick, so hosted cost accumulates whether or not it ever speaks.
#   - PERSONA_RULE follows REFLECTION; both propagate into the whole graph.
#   - CHAT_VOICE is deliberately ABSENT from `offline`. That absence is what
#     makes the privacy promise in docs/PLAN.md 7 literally true.
PRESETS: dict[str, dict[Task, str]] = {
    "offline": {
        Task.CHAT_TEXT: OLLAMA,
        Task.RECALL_ESCALATION: OLLAMA,
        Task.PROACTIVE_JUDGE: OLLAMA,
        Task.REFLECTION: OLLAMA,
        Task.PERSONA_RULE: OLLAMA,
    },
    "balanced": {
        Task.CHAT_TEXT: ANTHROPIC,
        Task.CHAT_VOICE: GEMINI,
        Task.RECALL_ESCALATION: ANTHROPIC,
        Task.PROACTIVE_JUDGE: OLLAMA,
        Task.REFLECTION: ANTHROPIC,
        Task.PERSONA_RULE: ANTHROPIC,
    },
    "quality": {
        Task.CHAT_TEXT: ANTHROPIC,
        Task.CHAT_VOICE: GEMINI,
        Task.RECALL_ESCALATION: ANTHROPIC,
        Task.PROACTIVE_JUDGE: ANTHROPIC,
        Task.REFLECTION: ANTHROPIC,
        Task.PERSONA_RULE: ANTHROPIC,
    },
}


class ConfigError(RuntimeError):
    """Bad configuration. Raised at startup, never mid-conversation."""


@dataclass(frozen=True, slots=True)
class Route:
    """Where one Task goes: which provider, and which concrete model."""

    provider: str
    model: str


class Settings(BaseSettings):
    """Loaded from the environment and `.env` - see .env.example.

    Field names are usable as keyword arguments (`Settings(preset="offline")`),
    which is how tests build a configuration without touching the environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    preset: str = Field(default="balanced", alias="DAEMON_PRESET")

    route_overrides: dict[Task, str] = Field(
        default_factory=dict, alias="DAEMON_ROUTE_OVERRIDES"
    )
    """Per-task override on top of the preset, as JSON:
    `DAEMON_ROUTE_OVERRIDES={"reflection": "ollama"}`."""

    fallback_provider: str | None = Field(default=None, alias="DAEMON_FALLBACK_PROVIDER")
    """Opt-in. When unset, a ProviderError propagates instead of being retried
    somewhere else - a silent switch to a weaker model is worse than an error."""

    voice_enabled: bool = Field(default=False, alias="DAEMON_VOICE_ENABLED")
    """docs/PLAN.md 6.5: voice is the user's choice and text mode is a complete
    product, so voice keys are only required once this is on."""

    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="DAEMON_OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen3:14b", alias="DAEMON_OLLAMA_MODEL")
    anthropic_model: str = Field(default="claude-sonnet-5", alias="DAEMON_ANTHROPIC_MODEL")
    openai_model: str = Field(default="", alias="DAEMON_OPENAI_MODEL")
    gemini_model: str = Field(default="", alias="DAEMON_GEMINI_MODEL")

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_allowed_user_ids: tuple[str, ...] = Field(
        default=(), alias="TELEGRAM_ALLOWED_USER_IDS"
    )

    data_dir: Path = Field(default=Path("./data"), alias="DAEMON_DATA_DIR")

    host: str = Field(default="127.0.0.1", alias="DAEMON_HOST")
    """Loopback by default. The HTTP surface is a local control plane, not a
    service to expose."""
    port: int = Field(default=8787, alias="DAEMON_PORT")

    @field_validator("telegram_allowed_user_ids", mode="before")
    @classmethod
    def _split_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.preset not in PRESETS:
            raise ConfigError(
                f"unknown DAEMON_PRESET {self.preset!r}; expected one of "
                f"{', '.join(sorted(PRESETS))}"
            )

        problems: list[str] = []
        for task, provider in self.route_overrides.items():
            if provider not in PROVIDER_KEY_ENV:
                problems.append(
                    f"override for {task.value} names unknown provider {provider!r} "
                    f"(known: {', '.join(sorted(PROVIDER_KEY_ENV))})"
                )

        if self.voice_enabled and not VOICE_TASKS <= self.routing.keys():
            problems.append(
                f"DAEMON_VOICE_ENABLED is on but preset {self.preset!r} routes no voice task; "
                "voice needs a hosted native-audio provider (docs/PLAN.md 3.2)"
            )

        # Only tasks that can actually be requested are validated, so an M1a
        # text-only install is not forced to hold a voice key it never uses.
        for task in self.active_tasks:
            problems.extend(self._provider_problems(self.routing[task], f"task {task.value}"))
        if self.fallback_provider is not None:
            problems.extend(
                self._provider_problems(self.fallback_provider, "DAEMON_FALLBACK_PROVIDER")
            )

        if problems:
            raise ConfigError("; ".join(problems))
        return self

    def _provider_problems(self, provider: str, context: str) -> list[str]:
        if provider not in PROVIDER_KEY_ENV:
            return [f"{context} names unknown provider {provider!r}"]
        found: list[str] = []
        key_env = PROVIDER_KEY_ENV[provider]
        if key_env is not None and not getattr(self, key_env.lower()):
            found.append(f"{context} routes to {provider!r} but {key_env} is empty")
        if not getattr(self, f"{provider}_model"):
            found.append(
                f"{context} routes to {provider!r} but no model is set "
                f"(DAEMON_{provider.upper()}_MODEL)"
            )
        return found

    @property
    def routing(self) -> dict[Task, str]:
        """Effective Task -> provider name table: preset plus overrides."""
        return {**PRESETS[self.preset], **self.route_overrides}

    @property
    def active_tasks(self) -> list[Task]:
        """Routed tasks that may actually be requested in this configuration."""
        return [
            task
            for task in self.routing
            if self.voice_enabled or task not in VOICE_TASKS
        ]

    def provider_model(self, provider: str) -> str:
        model: str = getattr(self, f"{provider}_model", "")
        if not model:
            raise ConfigError(
                f"no model configured for provider {provider!r} "
                f"(DAEMON_{provider.upper()}_MODEL)"
            )
        return model

    def route_for(self, task: Task) -> Route:
        """Resolve one Task, or explain precisely why it cannot be served."""
        provider = self.routing.get(task)
        if provider is None:
            extra = (
                " - it needs a hosted native-audio provider (docs/PLAN.md 3.2)"
                if task in VOICE_TASKS
                else ""
            )
            raise ConfigError(f"preset {self.preset!r} does not route {task.value}{extra}")
        if task in VOICE_TASKS and not self.voice_enabled:
            raise ConfigError(
                f"{task.value} was requested but voice is off (DAEMON_VOICE_ENABLED)"
            )
        return Route(provider=provider, model=self.provider_model(provider))

    def routing_table(self) -> dict[Task, Route]:
        """The whole resolved table, for handing to the gateway."""
        return {task: self.route_for(task) for task in self.active_tasks}

    def fallback_route(self) -> Route | None:
        if self.fallback_provider is None:
            return None
        return Route(
            provider=self.fallback_provider,
            model=self.provider_model(self.fallback_provider),
        )
