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

THINKING_LEVELS = ("", "low", "high")
"""What DAEMON_GEMINI_THINKING_LEVEL accepts: `low`, `high`, or empty to leave it
to the model. A Gemini 3 knob; other providers ignore it."""

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

SENSITIVITIES = ("low", "high")
"""What the two speech-sensitivity settings accept, plus empty for "the server
decides".

Repeated here rather than imported from `daemon/voice/gemini_live.py`, for the same
reason the wake defaults below are repeated: this module is foundation, and
importing the voice layer into it would invert the layering. The cost of the copy
is one line; the cost of the import is that config cannot be read without
PortAudio."""

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

QUIET_HOURS_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d-([01]\d|2[0-3]):[0-5]\d$")
"""`HH:MM-HH:MM`, or empty for "no quiet window".

Validated here rather than only where it is parsed, because the gate's honest
answer to an unparseable window is to block everything - a typo would then turn
proactivity off silently and stay off. Dying at startup with the bad value in the
message is the same choice the rest of this file makes: loud beats degraded.

A regex rather than importing the gate's parser: `config.py` is foundation and
`daemon/proactivity/` sits above it, so the import would point the wrong way."""


class ConfigError(RuntimeError):
    """Bad configuration. Raised at startup, never mid-conversation."""


DEFAULT_HOSTED_PROVIDER = ""
"""No default, expressed as a value so callers can name the absence.

It used to be "anthropic", which meant a configuration that never chose a
provider silently became Claude and the person who was never asked could not tell
a choice from a fallback. Onboarding always runs, so the answer always exists by
the time it is needed; before then this is the honest stand-in and hosted tasks
drop out of any enumeration rather than being guessed at."""


def preset_providers(preset: str, hosted: str) -> dict[Task, str]:
    """A preset's table with HOSTED resolved to a real provider name.

    Anything reading PRESETS to decide what a configuration needs has to go
    through this, or it sees the placeholder and treats "hosted" as a provider.

    `hosted` is required and has no default on purpose. It used to default to
    anthropic, which meant a configuration that never named a provider silently
    got Claude - and someone who had not been asked the question could not tell
    the difference between a choice and a fallback. Onboarding always runs, so
    the answer always exists; an empty one is a configuration to fix, not a
    value to guess.
    """
    return {
        task: (hosted if provider == HOSTED and hosted else provider)
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

    `hosted` resolves the HOSTED placeholder and is required: a caller that guesses
    it asks the user for the wrong key, which is how a person who chose GPT ends up
    being asked for an Anthropic one. Pass "" before the question has been answered
    and hosted tasks simply drop out of the list.
    """
    if preset not in PRESETS:
        raise ConfigError(
            f"unknown preset {preset!r}; expected one of {', '.join(sorted(PRESETS))}"
        )
    return sorted(
        {
            provider
            for task, provider in preset_providers(preset, hosted).items()
            # An unanswered question contributes nothing rather than a guess.
            if (voice_enabled or task not in VOICE_TASKS) and provider != HOSTED
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

    route_overrides: Annotated[dict[Task, str], NoDecode] = Field(
        default_factory=dict, alias="DAEMON_ROUTE_OVERRIDES"
    )
    """Per-task override on top of the preset, as JSON:
    `DAEMON_ROUTE_OVERRIDES={"reflection": "ollama"}`.

    NoDecode for the same reason as TELEGRAM_ALLOWED_USER_IDS below, and it was
    missed here: pydantic-settings JSON-decodes a complex field *before* field
    validators run, so the empty value .env.example ships - the value most installs
    hold, since that file says to copy it - made `json.loads("")` raise and took the
    entire load down, `daemon run` and `daemon doctor` alike, with a SettingsError
    naming neither the file nor the problem."""

    fallback_provider: str | None = Field(default=None, alias="DAEMON_FALLBACK_PROVIDER")
    """Opt-in. When unset, a ProviderError propagates instead of being retried
    somewhere else - a silent switch to a weaker model is worse than an error.

    `None` is what "unset" means to every reader of this field, and a dotenv can only
    write that as an empty value - which `_blank_is_unset` below normalises, because
    `""` reaching those readers reported a bogus unknown provider at startup."""

    hosted_provider: str = Field(default="", alias="DAEMON_HOSTED_PROVIDER")
    """Which commercial model answers wherever a preset says "hosted".

    One of HOSTED_PROVIDERS, and deliberately empty by default: a preset that
    needs a hosted provider and does not have one fails at startup pointing at
    `daemon setup`, rather than quietly becoming Claude. Per-task overrides still
    win over this."""

    voice_enabled: bool = Field(default=False, alias="DAEMON_VOICE_ENABLED")
    """docs/PLAN.md 6.5: voice is the user's choice and text mode is a complete
    product, so voice keys are only required once this is on."""

    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="DAEMON_OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen3:14b", alias="DAEMON_OLLAMA_MODEL")
    anthropic_model: str = Field(default="claude-sonnet-5", alias="DAEMON_ANTHROPIC_MODEL")
    openai_model: str = Field(default="", alias="DAEMON_OPENAI_MODEL")
    gemini_model: str = Field(default="", alias="DAEMON_GEMINI_MODEL")

    gemini_thinking_level: str = Field(default="low", alias="DAEMON_GEMINI_THINKING_LEVEL")
    """How hard a Gemini 3 model thinks before answering: `low`, `high`, or empty
    to leave it to the model. `low` by default, and that is a latency decision:
    measured on gemini-3.6-flash, the default (high) spent ~300 thinking tokens and
    ~3.6s *per call* on a plain weather lookup; `low` is ~1.3s with the tool call
    still made, so a two-call turn lands near 4s instead of 12s. Set `high` for a
    reasoning-heavy setup, or empty for a non-Gemini-3 model that rejects the field.
    Ignored by providers other than Gemini."""

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

    gemini_live_voice: str = Field(default="", alias="DAEMON_GEMINI_LIVE_VOICE")
    """Which prebuilt voice the Gemini Live session speaks in: one of
    GEMINI_LIVE_VOICES, or empty to leave it to the server. Checked at construction,
    not on the wire: an unknown name comes back as a 1007 close the session treats as
    permanent, so a typo would end voice mode rather than fail the setting."""

    # --- how the server decides a turn ended (daemon/voice/gemini_live.py) ---
    # All four are empty or None by default, and that is not laziness: an omitted
    # field leaves the server's own default, which is ~800 ms of silence before a
    # turn is considered over (https://ai.google.dev/gemini-api/docs/live-guide).
    # Until these existed the daemon sent no `realtimeInputConfig` at all, so there
    # was no knob - and the thing worth tuning is what happens *after* echo
    # cancellation, because without it lowering the sensitivity is how you buy back
    # the daemon interrupting itself, at the price of not being interruptible.

    voice_start_sensitivity: str = Field(default="", alias="DAEMON_VOICE_START_SENSITIVITY")
    """How eagerly the server decides the user has started talking: `low`, `high`,
    or empty for the server's default. `low` is the setting that stops leaked
    speaker audio from registering as a barge-in, and it is a workaround for a
    missing echo canceller rather than a preference."""

    voice_end_sensitivity: str = Field(default="", alias="DAEMON_VOICE_END_SENSITIVITY")
    """How eagerly the server decides the user has stopped: `low`, `high`, or empty."""

    voice_prefix_padding_ms: int | None = Field(
        default=None, alias="DAEMON_VOICE_PREFIX_PADDING_MS"
    )
    """How much speech must accumulate before the server commits to a start."""

    voice_silence_duration_ms: int | None = Field(
        default=None, alias="DAEMON_VOICE_SILENCE_DURATION_MS"
    )
    """Silence before a turn is over. The one number that trades response latency
    against being cut off mid-sentence, and the one worth measuring: 800 ms is the
    server's default, and `VoiceStats.first_audio_seconds` read against
    `interruptions` is what says whether a lower value cost anything - fewer seconds
    to the first answer is only a win if the count of interruptions did not move."""

    voice_barge_in: bool = Field(default=True, alias="DAEMON_VOICE_BARGE_IN")
    """Whether the owner can cut the daemon off mid-answer by speaking.

    On (the default), the microphone streams while the daemon talks - which is the
    only way a barge-in can be noticed at all, and also why a leaked syllable of
    the daemon's own speaker audio, or an "응" of agreement, kills the answer
    mid-sentence when the echo path is imperfect. Off is half-duplex: the
    microphone yields while the daemon is speaking or a tool answer is pending, so
    an answer always plays to the end and the owner talks in the gaps - the shape
    the owner's own prototype used, at the cost of not being able to interrupt by
    voice. A room where answers keep dying mid-sentence wants this off; sensitivity
    tuning (`DAEMON_VOICE_START_SENSITIVITY`) is the gentler lever to try first."""

    # --- the wake gate (daemon/voice/wake.py) -----------------------------
    # A voice session bills per minute, so an always-open one costs about 48x what
    # 30 minutes a day costs (docs/PLAN.md 6.5). These knobs describe the free
    # local listener that decides when to open a paid one. The defaults are the
    # measured ones; daemon/voice/wake.py repeats them, because this module is
    # foundation and importing the voice layer here would invert the layering.

    wake_enabled: bool = Field(default=False, alias="DAEMON_WAKE_ENABLED")
    """Off until asked for. The product is complete without it, and a microphone
    the owner did not switch on is not a default anyone should be opted into."""

    wake_aliases: Annotated[tuple[str, ...], NoDecode] = Field(
        default=(), alias="DAEMON_WAKE_ALIASES"
    )
    """What the recognizer actually returns for the wake phrase, comma-separated.

    Not the name the owner chose: an on-device recognizer never emits a coined one.
    Measured, 3 runs each, 100% stable - `헤이 데몬` came back as `헤이 대문`,
    `데몬` as `질문`, `루시야` as `루시`. So this is a per-speaker calibration, written
    by `daemon wake calibrate`, and it lives in .env rather than in the code.

    NoDecode for the same reason as TELEGRAM_ALLOWED_USER_IDS below:
    pydantic-settings JSON-decodes a complex field before field validators run, so
    without it the comma-separated form a person would type raises instead of
    loading."""

    wake_vad_threshold: float = Field(default=0.5, alias="DAEMON_WAKE_VAD_THRESHOLD")
    """Speech probability at or above which a frame counts as speech. Lowering it
    buys nothing on its own: the VAD calls a 3-note chord speech in 46.8% of frames
    already, and the recognizer stage is what makes that harmless."""

    wake_hangover_ms: int = Field(default=600, alias="DAEMON_WAKE_HANGOVER_MS")
    """Non-speech that ends a segment. Longer than the pause inside a two-word
    wake phrase, shorter than the gap between two sentences."""

    wake_pre_roll_ms: int = Field(default=300, alias="DAEMON_WAKE_PRE_ROLL_MS")
    """Audio kept from before the VAD said speech. A VAD notices speech a frame or
    two late and a wake word is one or two syllables, so the head is the match."""

    wake_min_speech_ms: int = Field(default=200, alias="DAEMON_WAKE_MIN_SPEECH_MS")
    """Below this a segment is dropped unheard - a 32 ms blip is not a wake word and
    must not cost a transcription."""

    wake_max_segment_ms: int = Field(default=3000, alias="DAEMON_WAKE_MAX_SEGMENT_MS")
    """Hard cap on one segment, so a minute of talking is neither held in memory nor
    sent to the recognizer as one blob (measured: a 2.2 s clip finalised in 760 ms)."""

    wake_cooldown_seconds: float = Field(default=5.0, alias="DAEMON_WAKE_COOLDOWN_SECONDS")
    """Quiet window after a fire, so one wake phrase cannot open two sessions."""

    # --- proactivity (M3, docs/PLAN.md 6) ---------------------------------
    # Every one of these is a way to make it speak *less*. That asymmetry is the
    # design: non-negotiable 7 makes silence the default, so the knobs are brakes.

    proactive_enabled: bool = Field(default=False, alias="DAEMON_PROACTIVE_ENABLED")
    """Off until the user turns it on. Something that decides on its own to speak
    is not a default anyone should be opted into."""

    proactive_daily_budget: int = Field(default=3, alias="DAEMON_PROACTIVE_DAILY_BUDGET")
    """Utterances per local day, all kinds. Three, from PLAN 6.2."""

    proactive_open_loop_budget: int = Field(
        default=1, alias="DAEMON_PROACTIVE_OPEN_LOOP_BUDGET"
    )
    """Of the daily budget, at most this many may be `open_loop`.

    A separate cap because open loops are the easy kind to generate: left to
    compete on equal terms they eat the whole budget and the result is a competent
    reminder app. PLAN 6.2 says the point of the product lives in the kinds that
    have no errand attached."""

    proactive_cooldown_minutes: int = Field(default=90, alias="DAEMON_PROACTIVE_COOLDOWN_MINUTES")
    """Minimum gap between two proactive utterances, whatever their kind."""

    proactive_quiet_hours: str = Field(default="23:00-09:00", alias="DAEMON_PROACTIVE_QUIET_HOURS")
    """Local `HH:MM-HH:MM` when it never speaks. Wraps midnight when start > end."""

    proactive_silence_hours: float = Field(default=20.0, alias="DAEMON_PROACTIVE_SILENCE_HOURS")
    """Hours without conversation before the `silence` kind becomes a candidate."""

    proactive_speaker_enabled: bool = Field(
        default=False, alias="DAEMON_PROACTIVE_SPEAKER_ENABLED"
    )
    """Whether it may talk out of the machine's speaker when the user is present.

    Off by default and gated separately from `proactive_enabled`, because the two
    failure costs are not comparable: an ignored Telegram message costs nothing and
    a voice in a meeting is an accident (PLAN 6.4). Telegram-only proactivity is a
    complete product; this is the addition that needs the gate to be trustworthy
    first."""

    # --- persona evolution (M4, docs/PLAN.md 5.5) -------------------------
    # No on/off switch, deliberately: the failure cost is the same shape as
    # reflection's (an AI-owned file plus a cap on top of it), and reflection has
    # no switch either.

    persona_max_active_rules: int = Field(default=20, alias="DAEMON_PERSONA_MAX_ACTIVE_RULES")
    """Ceiling on active learned rules. The weekly pass makes no model call once
    this is reached (`daemon/persona/evolve.py`'s gate 3), so a personality
    cannot grow without bound just because the process keeps running."""

    persona_max_new_per_cycle: int = Field(default=3, alias="DAEMON_PERSONA_MAX_NEW_PER_CYCLE")
    """Rules added per weekly pass, at most - a rate limit on how fast the
    learned half can change, not a quality filter."""

    persona_min_observations: int = Field(default=5, alias="DAEMON_PERSONA_MIN_OBSERVATIONS")
    """Unconsumed observations required before a pass runs at all (gate 2). A
    handful of observations must not be enough to conclude a pattern."""

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

    tools_enabled: bool = Field(default=True, alias="DAEMON_TOOLS_ENABLED")
    """On by default, because the alternative is a product that does not do what it
    says. docs/PLAN.md 1 defines Daemon as a companion that lives on your machine,
    and one that cannot open a file on it is a chat window with extra steps - so a
    default of `false` would ship the definition unmet and call it caution.

    The switch is not what makes it safe. The default `tools_mode` is `full`, so a
    guarded tool runs without asking (see that field for the trade and how to make it
    cautious again); the boundary that always holds is the origin gate - no tool at
    all runs on a turn that is not the owner's own words (tools/policy.py). The one
    capability that reads an authenticated session the owner never named - the
    browser group - stays off behind its own switch; MCP has its own switch too but
    defaults on, because it launches nothing until the owner has configured a server.

    Turning it off is one line, and `daemon doctor` says which way it is set: a
    capability the owner cannot see is the silent state this project keeps being
    bitten by."""

    tools_mode: str = Field(default="full", alias="DAEMON_TOOLS_MODE")
    """`off` | `allowlist` | `ask` | `full` - see daemon/tools/policy.py.

    `full` is the default, and it is a deliberate choice with real teeth: a guarded
    tool - `run_command`, `write_file`, `open_path` - runs without asking, so the
    model can `rm -rf` a directory or overwrite a file on the strength of one turn.
    The reason it is still the default is the same argument `tools_enabled` makes one
    line down: this is a companion that lives on the owner's own machine, and one
    that stops to ask permission before every single action it takes there is the
    "chat window with extra steps" the product exists not to be. The boundary that
    stays is the one that cannot be configured off - **no tool runs on a turn that
    is not the owner's own words** (tools/policy.py's origin gate), so a forward, an
    inline-bot result or anything recall dug up still reaches nothing.

    The cautious settings are one line away and `daemon doctor` prints which is in
    force: `ask` asks about anything that changes the machine (each request carries a
    one-shot code), `allowlist` runs only named commands and refuses the rest, `off`
    refuses every guarded tool. Pick one of those if unattended `rm -rf` is a risk
    worth a prompt to you."""

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

    mcp_enabled: bool = Field(default=True, alias="DAEMON_MCP_ENABLED")
    """Whether `<data_dir>/mcp.json` is read at all.

    On by default, for the same reason `tools_enabled` is: the switch is not what
    makes it safe, and defaulting it off only hides a capability the owner asked
    for. It reads nothing on its own - MCP does something only once the owner has
    both installed the optional `mcp` extra *and* written server blocks into
    `mcp.json`, and those two acts are the real opt-in. A machine with neither
    degrades to zero MCP tools, not to a daemon that will not boot (tools/mcp.py).

    It keeps its own switch rather than folding into DAEMON_TOOLS_ENABLED, because
    starting a stdio server means starting somebody else's subprocess: setting this
    `false` is the one line that turns every configured server off without editing
    `mcp.json`."""

    screen_enabled: bool = Field(default=False, alias="DAEMON_SCREEN_ENABLED")
    """Whether Daemon may capture the owner's screen.

    Its own switch rather than part of DAEMON_TOOLS_ENABLED or DAEMON_BROWSER_ENABLED,
    because seeing the whole screen is a distinct decision from reading the browser's
    open tabs or acting on the machine: a screenshot can show a password manager, a
    DM, or a document that neither of those touches (the same reasoning as
    DAEMON_BROWSER_ENABLED and DAEMON_VOICE_ENABLED)."""

    screen_max_px: int = Field(default=1536, alias="DAEMON_SCREEN_MAX_PX")
    """Long edge, in pixels, for an on-demand screenshot."""

    screen_frame_px: int = Field(default=1024, alias="DAEMON_SCREEN_FRAME_PX")
    """Long edge, in pixels, for a live-sharing frame - smaller than
    DAEMON_SCREEN_MAX_PX because frames are sent repeatedly, not once."""

    screen_fps: float = Field(default=1.0, alias="DAEMON_SCREEN_FPS")
    """Frames per second while live screen sharing is active."""

    screen_keepalive_secs: float = Field(default=8.0, alias="DAEMON_SCREEN_KEEPALIVE_SECS")
    """How long a live screen-sharing session stays open with no activity before
    it closes itself."""

    screen_dedup_threshold: int = Field(default=6, alias="DAEMON_SCREEN_DEDUP_THRESHOLD")
    """Maximum dhash Hamming distance for two frames to count as duplicates and be
    skipped, so an unchanging screen does not resend the same frame every tick."""

    data_dir: Path = Field(default=Path("./data"), alias="DAEMON_DATA_DIR")

    host: str = Field(default="127.0.0.1", alias="DAEMON_HOST")
    """Loopback by default. The HTTP surface is a local control plane, not a
    service to expose."""
    port: int = Field(default=8787, alias="DAEMON_PORT")

    @field_validator("fallback_provider", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """`DAEMON_FALLBACK_PROVIDER=` means no fallback, which is what .env.example
        documents it as. Normalised here rather than at each reader: `is None` is the
        test at both the startup check and `fallback_route`, and a `""` slipping past
        the second would give the gateway a fallback provider that cannot resolve.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("route_overrides", mode="before")
    @classmethod
    def _parse_overrides(cls, value: object) -> object:
        """Empty means no overrides; anything else must be the documented JSON.

        `NoDecode` on the field is what lets this run at all - see its docstring.
        Unparseable text raises rather than quietly becoming `{}`: an override that
        is dropped sends its task to the preset's provider instead of the chosen one
        and says nothing, which is the degradation this module exists to prevent.
        """
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            import json

            try:
                return json.loads(text)
            except ValueError as exc:
                raise ValueError(
                    "DAEMON_ROUTE_OVERRIDES must be JSON per task, as in "
                    f'{{"reflection": "ollama"}} - {exc}'
                ) from exc
        return value

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

    @field_validator(
        "voice_prefix_padding_ms", "voice_silence_duration_ms", mode="before"
    )
    @classmethod
    def _blank_is_the_servers_default(cls, value: object) -> object:
        """`KEY=` in a .env file is an empty *string*, not an absent key.

        pydantic does not coerce `""` to None for an `int | None`, and these two ship
        blank in .env.example on purpose - so a .env copied from it failed every
        `Settings()` with `int_parsing`, taking the whole daemon down rather than
        voice. Worse, it raised `pydantic.ValidationError` rather than `ConfigError`,
        which is the one exception `daemon doctor` catches: the command whose whole
        job is explaining the breakage printed a traceback instead.

        Not covered by the 1457 tests that passed, and the reason is worth keeping:
        `make_settings` builds `Settings(_env_file=None, **kwargs)` with native
        Python values, so nothing exercised the dotenv text path that turns `KEY=`
        into `""`. These are also the first `int | None` settings here, so there was
        no prior pattern to copy.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("wake_aliases", mode="before")
    @classmethod
    def _split_aliases(cls, value: object) -> object:
        """Accept what a person would type: `헤이 대문,루씨` or `헤이 대문, 루씨`.

        Commas only, never whitespace - unlike the numeric allowlist above, an
        alias is a *phrase*, and splitting `헤이 대문` on the space would configure
        two wake words neither of which was ever said.
        """
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

        needs_hosted = any(
            provider == HOSTED for provider in PRESETS[self.preset].values()
        )
        if not self.hosted_provider:
            if needs_hosted:
                raise ConfigError(
                    f"preset {self.preset!r} sends work to a hosted model but "
                    "DAEMON_HOSTED_PROVIDER is empty; run `daemon setup` to choose one "
                    f"({', '.join(HOSTED_PROVIDERS)})"
                )
        elif self.hosted_provider not in HOSTED_PROVIDERS:
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

        # Caught here rather than on the wire. The server answers a bad enum by
        # closing with 1007, which the session classifies as permanent - so a typo
        # in one of these would not be a bad setting, it would be voice mode gone
        # with a message about an unknown name.
        for env, chosen in (
            ("DAEMON_VOICE_START_SENSITIVITY", self.voice_start_sensitivity),
            ("DAEMON_VOICE_END_SENSITIVITY", self.voice_end_sensitivity),
        ):
            if chosen and chosen not in SENSITIVITIES:
                problems.append(
                    f"{env} is {chosen!r}; expected one of {', '.join(SENSITIVITIES)}, "
                    "or empty to leave it to the server"
                )

        if self.gemini_live_voice and self.gemini_live_voice not in GEMINI_LIVE_VOICES:
            problems.append(
                f"DAEMON_GEMINI_LIVE_VOICE is {self.gemini_live_voice!r}; expected one of "
                "the Gemini Live voices, or empty to leave it to the server"
            )

        if self.wake_enabled and not self.wake_aliases:
            problems.append(
                "DAEMON_WAKE_ENABLED is on but DAEMON_WAKE_ALIASES is empty, so nothing "
                "can ever match; the gate matches what the recognizer returns for your "
                "voice rather than the name you chose, so run `daemon wake calibrate` to "
                "measure it"
            )
        if self.wake_enabled and not self.voice_enabled:
            problems.append(
                "DAEMON_WAKE_ENABLED is on but DAEMON_VOICE_ENABLED is off; the gate exists "
                "only to open a voice session, so set DAEMON_VOICE_ENABLED=true or switch "
                "the gate off"
            )
        if not 0.0 < self.wake_vad_threshold <= 1.0:
            problems.append(
                f"DAEMON_WAKE_VAD_THRESHOLD is {self.wake_vad_threshold}; it must be within "
                "(0, 1] - at 0 every frame of silence is speech"
            )
        for name, value in (
            ("DAEMON_WAKE_HANGOVER_MS", self.wake_hangover_ms),
            ("DAEMON_WAKE_MIN_SPEECH_MS", self.wake_min_speech_ms),
            ("DAEMON_WAKE_MAX_SEGMENT_MS", self.wake_max_segment_ms),
        ):
            if value <= 0:
                problems.append(f"{name} is {value}; it must be greater than 0")
        if self.wake_pre_roll_ms < 0:
            problems.append(
                f"DAEMON_WAKE_PRE_ROLL_MS is {self.wake_pre_roll_ms}; it cannot be negative "
                "(0 keeps no audio from before speech started, which clips the wake word)"
            )
        if self.wake_cooldown_seconds < 0:
            problems.append(
                f"DAEMON_WAKE_COOLDOWN_SECONDS is {self.wake_cooldown_seconds}; it cannot be "
                "negative (0 allows one phrase to open two sessions)"
            )
        if self.wake_max_segment_ms <= self.wake_pre_roll_ms + self.wake_min_speech_ms:
            problems.append(
                f"DAEMON_WAKE_MAX_SEGMENT_MS ({self.wake_max_segment_ms}) leaves no room for "
                f"DAEMON_WAKE_PRE_ROLL_MS ({self.wake_pre_roll_ms}) plus "
                f"DAEMON_WAKE_MIN_SPEECH_MS ({self.wake_min_speech_ms}); every segment would "
                "be cut short of the length that makes it worth transcribing"
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
        quiet = self.proactive_quiet_hours.strip()
        if quiet and not QUIET_HOURS_RE.match(quiet):
            problems.append(
                f"DAEMON_PROACTIVE_QUIET_HOURS {self.proactive_quiet_hours!r} is not "
                "HH:MM-HH:MM (24-hour, local). Leave it empty for no quiet window"
            )
        if self.proactive_open_loop_budget > self.proactive_daily_budget:
            problems.append(
                f"DAEMON_PROACTIVE_OPEN_LOOP_BUDGET ({self.proactive_open_loop_budget}) is "
                f"above DAEMON_PROACTIVE_DAILY_BUDGET ({self.proactive_daily_budget}); the "
                "sub-cap exists to hold open loops *below* the overall budget"
            )
        for name, value in (
            ("DAEMON_PROACTIVE_DAILY_BUDGET", self.proactive_daily_budget),
            ("DAEMON_PROACTIVE_OPEN_LOOP_BUDGET", self.proactive_open_loop_budget),
            ("DAEMON_PROACTIVE_COOLDOWN_MINUTES", self.proactive_cooldown_minutes),
        ):
            if value < 0:
                problems.append(f"{name} is {value}; it cannot be negative")
        if self.proactive_silence_hours <= 0:
            problems.append(
                f"DAEMON_PROACTIVE_SILENCE_HOURS is {self.proactive_silence_hours}; "
                "at zero or below, every tick would be a silence candidate"
            )
        for name, value in (
            ("DAEMON_PERSONA_MAX_ACTIVE_RULES", self.persona_max_active_rules),
            ("DAEMON_PERSONA_MAX_NEW_PER_CYCLE", self.persona_max_new_per_cycle),
            ("DAEMON_PERSONA_MIN_OBSERVATIONS", self.persona_min_observations),
        ):
            if value < 0:
                problems.append(f"{name} is {value}; it cannot be negative")
        if self.tools_mode not in TOOL_MODES:
            problems.append(
                f"unknown DAEMON_TOOLS_MODE {self.tools_mode!r}; expected one of "
                f"{', '.join(TOOL_MODES)}"
            )
        if self.gemini_thinking_level not in THINKING_LEVELS:
            problems.append(
                f"unknown DAEMON_GEMINI_THINKING_LEVEL {self.gemini_thinking_level!r}; "
                f"expected one of {', '.join(repr(x) for x in THINKING_LEVELS)}"
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
        if self.screen_enabled and not self.tools_enabled:
            problems.append(
                "DAEMON_SCREEN_ENABLED is on but DAEMON_TOOLS_ENABLED is off; the "
                "screen tools are tools, so nothing would register them"
            )
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
