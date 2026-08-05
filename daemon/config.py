"""Settings and the three routing presets.

docs/PLAN.md 3.2: routing is a per-Task table, not one global switch, and users
see three presets instead of seven questions. Advanced users override single
tasks.

Everything here fails loudly at construction time. A missing API key must not
surface as a broken conversation turn three hours later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from daemon.tasks import Task
from daemon.tools.policy import MODES as TOOL_MODES

OLLAMA = "ollama"
ANTHROPIC = "anthropic"
OPENAI = "openai"
GEMINI = "gemini"

# Env var carrying the API key, or None for a provider that needs no key.
# Every name here is buildable: tests/test_reachable.py fails if one is nameable
# in a route and the app cannot construct it.
PROVIDER_KEY_ENV: dict[str, str | None] = {
    OLLAMA: None,
    ANTHROPIC: "ANTHROPIC_API_KEY",
    OPENAI: "OPENAI_API_KEY",
    GEMINI: "GEMINI_API_KEY",
}

HOSTED = "hosted"
"""Placeholder in the preset tables for "whichever hosted provider was chosen".

The presets used to name ANTHROPIC directly, which made "provider-agnostic
gateway" true of the config surface and false of the product: a user who wanted
GPT or Gemini for conversation had no way to say so. Multiplying the presets by
the providers would have given nine, so the two questions stay separate - a preset
answers *where the work runs*, and DAEMON_HOSTED_PROVIDER answers *whose model*.

CHAT_VOICE is deliberately not HOSTED: it names GEMINI because Gemini Live is the
only native-audio session implemented, and pointing it at a provider with no voice
session would fail at the first voice turn instead of at startup."""

HOSTED_PROVIDERS = ("anthropic", "openai", "gemini")
"""What DAEMON_HOSTED_PROVIDER accepts. Ollama is not here - "hosted" is the
opposite of local, and the offline preset never resolves HOSTED at all."""

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
        Task.EMBED: OLLAMA,
    },
    "balanced": {
        Task.CHAT_TEXT: HOSTED,
        Task.CHAT_VOICE: GEMINI,
        Task.RECALL_ESCALATION: HOSTED,
        Task.PROACTIVE_JUDGE: OLLAMA,
        Task.REFLECTION: HOSTED,
        Task.PERSONA_RULE: HOSTED,
        Task.EMBED: OLLAMA,
    },
    "quality": {
        Task.CHAT_TEXT: HOSTED,
        Task.CHAT_VOICE: GEMINI,
        Task.RECALL_ESCALATION: HOSTED,
        Task.PROACTIVE_JUDGE: HOSTED,
        Task.REFLECTION: HOSTED,
        Task.PERSONA_RULE: HOSTED,
        Task.EMBED: OLLAMA,
    },
}


ENV_FILE = ".env"
"""The one file credentials live in. The service unit deliberately carries no
secrets and only points at the directory holding this file (daemon/service.py),
which is also why `daemon setup` writes here and reads nothing from the shell
environment: launchd and systemd would not have it."""


SERVICE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
"""The label becomes both a launchd job name and a filename under
`~/Library/LaunchAgents`, so it is validated rather than trusted: a label
containing `/` or `..` would write the plist somewhere else entirely."""


class ConfigError(RuntimeError):
    """Bad configuration. Raised at startup, never mid-conversation."""


DEFAULT_HOSTED_PROVIDER = HOSTED_PROVIDERS[0]


def preset_providers(preset: str, hosted: str = DEFAULT_HOSTED_PROVIDER) -> dict[Task, str]:
    """A preset's table with HOSTED resolved to a real provider name.

    Anything reading PRESETS to decide what a configuration needs has to go
    through this, or it sees the placeholder and treats "hosted" as a provider.
    """
    return {
        task: (hosted if provider == HOSTED else provider)
        for task, provider in PRESETS[preset].items()
    }


def providers_for(
    preset: str,
    *,
    voice_enabled: bool,
    hosted: str = DEFAULT_HOSTED_PROVIDER,
) -> list[str]:
    """Providers a preset actually needs, so onboarding asks for those keys only.

    Voice tasks are excluded while voice is off - the same rule as
    `Settings.active_tasks`. That is what lets a text-only `balanced` install be
    set up without a hosted voice key (docs/PLAN.md 6.5).

    `hosted` resolves the HOSTED placeholder. It defaults so that a caller which
    has not yet asked the user still behaves as before, but a caller that knows
    the answer must pass it - otherwise a user who chose GPT is asked for an
    Anthropic key.
    """
    if preset not in PRESETS:
        raise ConfigError(
            f"unknown preset {preset!r}; expected one of {', '.join(sorted(PRESETS))}"
        )
    return sorted(
        {
            provider
            for task, provider in preset_providers(preset, hosted).items()
            if voice_enabled or task not in VOICE_TASKS
        }
    )


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
        env_file=ENV_FILE,
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

    hosted_provider: str = Field(default="anthropic", alias="DAEMON_HOSTED_PROVIDER")
    """Which commercial model answers wherever a preset says "hosted".
    One of HOSTED_PROVIDERS. Per-task overrides still win over this."""

    voice_enabled: bool = Field(default=False, alias="DAEMON_VOICE_ENABLED")
    """docs/PLAN.md 6.5: voice is the user's choice and text mode is a complete
    product, so voice keys are only required once this is on."""

    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="DAEMON_OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen3:14b", alias="DAEMON_OLLAMA_MODEL")
    anthropic_model: str = Field(default="claude-sonnet-5", alias="DAEMON_ANTHROPIC_MODEL")
    openai_model: str = Field(default="", alias="DAEMON_OPENAI_MODEL")
    gemini_model: str = Field(default="", alias="DAEMON_GEMINI_MODEL")

    embed_model: str = Field(default="bge-m3", alias="DAEMON_EMBED_MODEL")
    """Embedding model for recall, separate from the chat model even though both
    run on Ollama: chat may move to a hosted provider while embeddings stay local
    (Task.EMBED is local in every preset). bge-m3 is multilingual, which the
    Korean recall path depends on - docs/PLAN.md 4.3 shows FTS5 alone missing
    inflected Korean, and that is the whole reason vectors were pulled into M1b."""

    recall_limit: int = Field(default=6, alias="DAEMON_RECALL_LIMIT")
    """Top N recalled items injected per turn. Small on purpose: this text is
    prepended to every single turn, so the budget is spent here first."""

    recall_half_life_days: float = Field(default=30.0, alias="DAEMON_RECALL_HALF_LIFE_DAYS")
    """Recency decay in the recall score (docs/PLAN.md 4.3). Exposed because the
    right value depends on how much someone talks, not because it should be
    fiddled with casually."""

    gemini_live_model: str = Field(default="", alias="DAEMON_GEMINI_LIVE_MODEL")
    """Gemini Live (native audio) model id, distinct from DAEMON_GEMINI_MODEL:
    the text and realtime endpoints do not take the same ids. Deliberately has no
    default - a guessed model id fails at the first voice turn, which is exactly
    the kind of late failure this module exists to prevent."""

    service_label: str = Field(default="default", alias="DAEMON_SERVICE_LABEL")
    """Suffix of the OS service label (`ai.daemon.<label>`). Only interesting for
    a second instance with its own data dir."""

    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    telegram_dm_policy: str = Field(default="pairing", alias="DAEMON_TELEGRAM_DM_POLICY")
    """How an unknown sender is handled. `pairing` is the default because the
    alternative is asking a person to look up their own numeric id in a third bot
    and paste it into a dotenv - and with `allowlist`, an empty list refuses to
    start, which is exactly the state a first run is in."""

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_allowed_user_ids: Annotated[tuple[str, ...], NoDecode] = Field(
        default=(), alias="TELEGRAM_ALLOWED_USER_IDS"
    )
    """NoDecode because pydantic-settings JSON-decodes a complex field *before*
    field validators run. Without it `4242` raised a ValidationError and
    `4242,4243` raised a SettingsError, so only `["4242"]` loaded - meaning the
    comma-separated form documented in .env.example was impossible to write by
    hand, `dm_policy=allowlist` could not be configured at all, and `_split_ids`
    below was dead code that looked like it worked."""

    tools_enabled: bool = Field(default=False, alias="DAEMON_TOOLS_ENABLED")
    """Off by default, and that is a product decision rather than caution about
    the code: upgrading a companion should not silently hand it a shell. The user
    turns this on once they want it to reach the machine (docs/PLAN.md tool use)."""

    tools_mode: str = Field(default="ask", alias="DAEMON_TOOLS_MODE")
    """`off` | `allowlist` | `ask` | `full` - see daemon/tools/policy.py.

    `ask` rather than `allowlist` as the default for a switched-on install: the
    first sessions are when you find out what it wants to run, and `allowlist`
    answers anything unlisted with a flat refusal that teaches nobody anything."""

    tools_allowlist: Annotated[tuple[str, ...], NoDecode] = Field(
        default=(), alias="DAEMON_TOOLS_ALLOWLIST"
    )
    """Commands that run without asking, as argv prefixes: `ls,git status,date`.
    Comma-separated only - an entry legitimately contains a space, so the
    whitespace-splitting that TELEGRAM_ALLOWED_USER_IDS accepts would break
    `git status` into two useless entries."""

    tools_roots: Annotated[tuple[str, ...], NoDecode] = Field(
        default=("~",), alias="DAEMON_TOOLS_ROOTS"
    )
    """Where the file tools may look. Comma-separated. Defaults to the home
    directory: narrower is better, and anyone who wants that should say so, but a
    default of "nothing" would make the file tools look broken rather than safe."""

    tools_timeout_secs: float = Field(default=20.0, alias="DAEMON_TOOLS_TIMEOUT_SECS")
    tools_max_output: int = Field(default=4000, alias="DAEMON_TOOLS_MAX_OUTPUT")
    """Characters of tool output given to the model. Paid for on every subsequent
    round of the same turn, since the result stays in the context."""

    tools_max_rounds: int = Field(default=6, alias="DAEMON_TOOLS_MAX_ROUNDS")
    """Tool round-trips allowed in one turn before it must answer."""

    browser_enabled: bool = Field(default=False, alias="DAEMON_BROWSER_ENABLED")
    """Whether Daemon may fetch web pages and read the owner's open browser tabs.

    Its own switch rather than part of DAEMON_TOOLS_ENABLED, because this is the one
    group that reads an authenticated session: the page in front of the owner may be
    their bank. Letting it act on the machine and letting it read over their shoulder
    are two decisions, so they are two settings (the same reasoning as
    DAEMON_VOICE_ENABLED)."""

    browser_app: str = Field(default="Google Chrome", alias="DAEMON_BROWSER_APP")
    """Which browser to read. Brave, Arc and Edge share Chrome's AppleScript
    dictionary, so naming one of those works. Safari's is a different shape and is
    not supported."""

    mcp_enabled: bool = Field(default=False, alias="DAEMON_MCP_ENABLED")
    """Whether `<data_dir>/mcp.json` is read at all. Separate from
    DAEMON_TOOLS_ENABLED so an MCP server can be configured and left switched off,
    and because starting one means starting somebody else's subprocess."""

    data_dir: Path = Field(default=Path("./data"), alias="DAEMON_DATA_DIR")

    host: str = Field(default="127.0.0.1", alias="DAEMON_HOST")
    """Loopback by default. The HTTP surface is a local control plane, not a
    service to expose."""
    port: int = Field(default=8787, alias="DAEMON_PORT")

    @field_validator("telegram_allowed_user_ids", mode="before")
    @classmethod
    def _split_ids(cls, value: object) -> object:
        """Accept what a person would actually type: `4242`, `4242,4243`, or with
        spaces. A JSON list still works, for anyone who already wrote one."""
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                import json

                try:
                    return tuple(str(item).strip() for item in json.loads(text))
                except (ValueError, TypeError):
                    return ()
            return tuple(
                part.strip() for part in text.replace(",", " ").split() if part.strip()
            )
        return value

    @field_validator("tools_allowlist", "tools_roots", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Commas only, so an entry may contain spaces (`git status`, or a path with
        one in it). `NoDecode` for the same reason as the ids above: pydantic-settings
        JSON-decodes a complex field before validators run."""
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                import json

                try:
                    return tuple(str(item).strip() for item in json.loads(text))
                except (ValueError, TypeError):
                    return ()
            return tuple(part.strip() for part in text.split(",") if part.strip())
        return value

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.preset not in PRESETS:
            raise ConfigError(
                f"unknown DAEMON_PRESET {self.preset!r}; expected one of "
                f"{', '.join(sorted(PRESETS))}"
            )

        if self.hosted_provider not in HOSTED_PROVIDERS:
            raise ConfigError(
                f"unknown DAEMON_HOSTED_PROVIDER {self.hosted_provider!r}; expected one "
                f"of {', '.join(HOSTED_PROVIDERS)}"
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
        if self.voice_enabled and not self.gemini_live_model:
            problems.append(
                "DAEMON_VOICE_ENABLED is on but DAEMON_GEMINI_LIVE_MODEL is empty; "
                "the native-audio endpoint needs its own model id"
            )

        if not self.embed_model:
            problems.append("DAEMON_EMBED_MODEL is empty; recall cannot embed anything")
        if self.recall_limit < 1:
            problems.append(
                f"DAEMON_RECALL_LIMIT is {self.recall_limit}; it must be at least 1 "
                "(to switch recall off, do not configure a recall backend)"
            )
        if self.recall_half_life_days <= 0:
            problems.append(
                f"DAEMON_RECALL_HALF_LIFE_DAYS is {self.recall_half_life_days}; "
                "it must be greater than 0 or recency decay is undefined"
            )
        if self.tools_mode not in TOOL_MODES:
            problems.append(
                f"unknown DAEMON_TOOLS_MODE {self.tools_mode!r}; expected one of "
                f"{', '.join(TOOL_MODES)}"
            )
        if self.tools_enabled:
            # Only checked when tools are on: a text-only install should not have to
            # hold a coherent tool configuration it never reaches.
            if not self.tools_roots:
                problems.append(
                    "DAEMON_TOOLS_ENABLED is on but DAEMON_TOOLS_ROOTS is empty; "
                    "the file tools would have nowhere they are allowed to look"
                )
            if self.tools_timeout_secs <= 0:
                problems.append(
                    f"DAEMON_TOOLS_TIMEOUT_SECS is {self.tools_timeout_secs}; it must be "
                    "greater than 0 or every command is killed before it starts"
                )
            if self.tools_max_output < 200:
                problems.append(
                    f"DAEMON_TOOLS_MAX_OUTPUT is {self.tools_max_output}; below ~200 "
                    "characters a tool result is truncated past the point of being useful"
                )
            if self.tools_max_rounds < 1:
                problems.append(
                    f"DAEMON_TOOLS_MAX_ROUNDS is {self.tools_max_rounds}; it must be at "
                    "least 1 (to switch tools off, use DAEMON_TOOLS_ENABLED=false)"
                )
        if self.browser_enabled and not self.tools_enabled:
            problems.append(
                "DAEMON_BROWSER_ENABLED is on but DAEMON_TOOLS_ENABLED is off; the "
                "browser tools are tools, so nothing would register them"
            )
        if self.browser_enabled and not self.browser_app.strip():
            problems.append("DAEMON_BROWSER_APP is empty; name the browser to read")
        if not SERVICE_LABEL_RE.match(self.service_label):
            problems.append(
                f"DAEMON_SERVICE_LABEL {self.service_label!r} is not a usable label; "
                "expected letters, digits, dot, dash or underscore"
            )

        # Only tasks that can actually be requested are validated, so an M1a
        # text-only install is not forced to hold a voice key it never uses.
        for task in self.active_tasks:
            problems.extend(
                self._provider_problems(
                    self.routing[task], f"task {task.value}", task=task
                )
            )
        if self.fallback_provider is not None:
            problems.extend(
                self._provider_problems(self.fallback_provider, "DAEMON_FALLBACK_PROVIDER")
            )

        if problems:
            raise ConfigError("; ".join(problems))
        return self

    def _provider_problems(
        self, provider: str, context: str, *, task: Task | None = None
    ) -> list[str]:
        if provider not in PROVIDER_KEY_ENV:
            return [f"{context} names unknown provider {provider!r}"]
        found: list[str] = []
        key_env = PROVIDER_KEY_ENV[provider]
        if key_env is not None and not getattr(self, key_env.lower()):
            found.append(f"{context} routes to {provider!r} but {key_env} is empty")

        # Voice does not use the provider's ordinary model id. The native-audio
        # endpoint takes its own, and demanding DAEMON_GEMINI_MODEL for a voice
        # route made enabling voice impossible without setting an unrelated
        # variable - and once set, that value was never read by anything. The
        # voice model has its own check in the validator above, so there is
        # nothing to verify here.
        if task in VOICE_TASKS:
            return found

        # Defaulted, not bare: a provider added to PROVIDER_KEY_ENV without a
        # matching model field used to raise AttributeError from inside pydantic
        # validation, which reads as a crash rather than as the configuration
        # mistake it is.
        if not getattr(self, f"{provider}_model", ""):
            found.append(
                f"{context} routes to {provider!r} but no model is set "
                f"(DAEMON_{provider.upper()}_MODEL)"
            )
        return found

    @property
    def routing(self) -> dict[Task, str]:
        """Effective Task -> provider name table: preset, then the chosen hosted
        provider substituted in, then explicit overrides on top."""
        resolved = {
            task: (self.hosted_provider if provider == HOSTED else provider)
            for task, provider in PRESETS[self.preset].items()
        }
        return {**resolved, **self.route_overrides}

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
        if task in VOICE_TASKS:
            # The native-audio endpoint takes its own model id, which is why
            # DAEMON_GEMINI_MODEL is neither required nor read for a voice route.
            return Route(provider=provider, model=self.gemini_live_model)
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
