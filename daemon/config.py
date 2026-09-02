"""Settings and the two routing axes.

docs/PLAN.md 3.2 / ADR 0014: routing is a per-Task table, computed from two
orthogonal choices - `DAEMON_PROVIDER` (which model answers chat, recall,
reflection and persona rules) and `DAEMON_PROACTIVE_JUDGE_LOCAL` (whether the
5-minute proactive tick runs on that provider or stays local) - rather than a
named preset table. Advanced users override single tasks with
`DAEMON_ROUTE_OVERRIDES`.

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
OPENAI_COMPATIBLE = "openai_compatible"

# Env var carrying the API key, or None for a provider that needs no key.
# Every name here is buildable: tests/test_reachable.py fails if one is nameable
# in a route and the app cannot construct it.
PROVIDER_KEY_ENV: dict[str, str | None] = {
    OLLAMA: None,
    ANTHROPIC: "ANTHROPIC_API_KEY",
    OPENAI: "OPENAI_API_KEY",
    GEMINI: "GEMINI_API_KEY",
    OPENAI_COMPATIBLE: "OPENAI_COMPATIBLE_API_KEY",
}

HOSTED_PROVIDERS = ("anthropic", "openai", "gemini", "openai_compatible")
"""What DAEMON_PROVIDER accepts besides `ollama`. Ollama is not here - "hosted" is
the opposite of local, and `DAEMON_PROVIDER=ollama` never resolves to one of these.

`openai_compatible` is one name for many vendors on purpose: Qwen, Kimi,
DeepSeek, OpenRouter and a self-hosted server differ by endpoint, not by
protocol, so the endpoint is configuration and the protocol is the provider."""

MODEL_SUGGESTIONS: dict[str, tuple[str, ...]] = {
    "ollama": ("qwen3:14b", "gemma3:4b"),
    "anthropic": ("claude-opus-5", "claude-sonnet-5"),
    "openai": ("gpt-5.2", "gpt-5.1"),
    "gemini": ("gemini-3.6-flash", "gemini-3.1-pro-preview"),
}
"""Datalist suggestions for the admin's chat-model field, newest first. NOT VALIDATED -
unlike GEMINI_LIVE_VOICES or HOSTED_PROVIDERS, this is not a constraint: a model id
absent here must still save (Settings validates model ids only as non-empty), because
a just-released id would otherwise be unselectable. Do not add a membership check."""

VOICE_TASKS = frozenset({Task.CHAT_VOICE})
"""Tasks that need a hosted native-audio model. Voice is its own axis
(DAEMON_VOICE_ENABLED, ADR 0012) and carries no route unless that is on."""

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

VOICE_PROVIDERS = ("gemini", "openai")
"""Which hosted native-audio backend a voice session uses. Independent of the text
`provider`: voice-model availability is a separate axis, and being explicit
turns a mismatch into a startup error, not a first-turn failure."""

OPENAI_REALTIME_VOICES = frozenset({
    "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse",
    "marin", "cedar",
})
"""OpenAI Realtime voices. `marin`/`cedar` are gpt-realtime-only; a voice the chosen
model rejects comes back as a session error, the same class as Gemini's 1007."""

PROACTIVE_KINDS = (
    "open_loop",
    "emotional",
    "silence",
    "pattern_time",
    "association",
    "topic",
    "calendar",
)
"""The seven candidate kinds - `daemon/proactivity/base.py`'s `CandidateKind`,
repeated here for the same reason `GEMINI_LIVE_VOICES` below is: this module is
foundation and `daemon/proactivity/` sits above it, so importing would invert the
layering. Validates the keys of `DAEMON_PROACTIVE_KIND_BUDGETS` - a name check
only. `topic` belongs here even though it has no entry in
`proactive_kind_budgets`'s default table below: this tuple validates kind
*names* an owner may reference, that dict allocates per-kind *ceilings*, and the
owner explicitly does not get a `topic` ceiling (see that field's docstring).
Without `topic` here, an owner writing `DAEMON_PROACTIVE_KIND_BUDGETS` to set
any other kind's cap alongside `topic` would have that whole setting rejected as
naming an unknown kind - two task reports flagged exactly this gap before it was
closed."""

GEMINI_LIVE_TRANSPORTS = ("api_key", "vertex")
"""Which endpoint serves the Gemini Live session. Two, because neither is a
superset of the other: the faster and steadier native-audio model is Vertex-only,
and the newer generation is API-key-only (docs/design/vertex-live-transport.md).

Named here rather than imported from `daemon/voice/vertex.py` for the reason
SENSITIVITIES gives below: this module is foundation."""

VERTEX_LIVE_MODELS = ("gemini-live-2.5-flash-native-audio",)
"""Conversational live models the `vertex` transport can serve, checked on
2026-09-02 across every region that serves any. Exactly one, and it is the reason
the transport exists: measured 1430 ms to first audio against the API-key
endpoint's 3137 ms for the same model family.

A constant rather than a probe: the admin's model lists authenticate with an API
key, and this catalogue is not visible to one. `gemini-3.5-transcribe-live-preview`
is deliberately absent - it transcribes and does not speak."""

VERTEX_LIVE_LOCATIONS = ("us-central1", "us-east1", "us-east4", "us-west1", "europe-west4")
"""Regions that listed a live model on 2026-09-02. asia-northeast1,
asia-northeast3 (Seoul) and asia-southeast1 listed none, so a Korean self-hoster's
nearest region cannot serve this at all - and from Seoul us-west1 measured
identical to us-central1 (1441 ms both), so the delay is serving rather than
distance and there is nothing to buy by moving closer."""

SENSITIVITIES = ("low", "high")
"""What the two speech-sensitivity settings accept, plus empty for "the server
decides".

Repeated here rather than imported from `daemon/voice/gemini_live.py`, for the same
reason the wake defaults below are repeated: this module is foundation, and
importing the voice layer into it would invert the layering. The cost of the copy
is one line; the cost of the import is that config cannot be read without
PortAudio."""

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


def clean_base_url(value: str) -> str:
    """The compatible endpoint, normalised - or `ValueError` naming the mistake.

    Strips the trailing slash and refuses the whole endpoint URL. Vendor docs
    print `.../v1/chat/completions`, and pasting that whole line is the
    predictable mistake. Left alone the provider appends the path a second time
    and the resulting 404 explains nothing. Rejected rather than quietly
    repaired, and the message carries the value to use instead - the same choice
    QUIET_HOURS_RE makes: loud beats degraded.

    A module-level function rather than only a validator body, because
    `daemon/setup.py` has to be able to answer the same question *at the question
    that asks for it*: a bad address caught only by `Settings` surfaces one
    question later as a key that "could not be verified", which puts the message
    next to the wrong mistake. One authority, not two that can drift - the
    wizard must never accept a value startup will refuse.
    """
    text = value.strip().rstrip("/")
    if not text:
        return ""
    if not text.startswith(("http://", "https://")):
        raise ValueError(
            f"DAEMON_OPENAI_COMPATIBLE_BASE_URL must start with http:// or https:// - "
            f"got {text!r}"
        )
    if text.endswith("/chat/completions"):
        raise ValueError(
            "DAEMON_OPENAI_COMPATIBLE_BASE_URL must not include /chat/completions - "
            f"use {text.removesuffix('/chat/completions')}"
        )
    return text


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


@dataclass(frozen=True, slots=True)
class Route:
    """Where one Task goes: which provider, and which concrete model."""

    provider: str
    model: str


class Settings(BaseSettings):
    """Loaded from the environment and `.env` - see .env.example.

    Field names are usable as keyword arguments (`Settings(provider="ollama")`),
    which is how tests build a configuration without touching the environment.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

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

    stale_preset: str = Field(default="", alias="DAEMON_PRESET")
    """Not a real setting - `DAEMON_PRESET` was removed (ADR 0014). Captured here,
    rather than left to `extra="ignore"`, so a leftover value from before the rename
    can be refused loudly instead of silently dropped: a `DAEMON_PRESET=offline`
    install that was never rewritten would otherwise start dialing a hosted
    provider, the privacy version of ADR 0007's footgun. See `_check` below.

    No leading underscore, deliberately: pydantic reserves that spelling for a
    private attribute and refuses to build the class at all (`NameError`) if a
    `Field(...)` is attached to one - this field has to be an ordinary field to
    receive `DAEMON_PRESET` through the alias mechanism in the first place."""

    route_overrides: Annotated[dict[Task, str], NoDecode] = Field(
        default_factory=dict, alias="DAEMON_ROUTE_OVERRIDES"
    )
    """Per-task override on top of the computed routing, as JSON:
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

    voice_enabled: bool = Field(default=False, alias="DAEMON_VOICE_ENABLED")
    """docs/PLAN.md 6.5: voice is the user's choice and text mode is a complete
    product, so voice keys are only required once this is on.

    Also the only switch for whether a proactive utterance may leave the laptop
    speaker - `DAEMON_PROACTIVE_SPEAKER_ENABLED` used to be a second one. It was
    split off in the first place because the two failure costs are not
    comparable (PLAN 6.4): an ignored Telegram message costs nothing, a voice in
    a meeting is an accident. But `Gate._route` already carries that asymmetry
    on its own - seven rules that downgrade to Telegram rather than block - so a
    second top-level switch only bought "voice on" meaning two different things
    depending which file you read. One switch; the gate still refuses to speak
    into a meeting."""

    ollama_base_url: str = Field(default="http://127.0.0.1:11434", alias="DAEMON_OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen3:14b", alias="DAEMON_OLLAMA_MODEL")
    anthropic_model: str = Field(default="claude-sonnet-5", alias="DAEMON_ANTHROPIC_MODEL")
    openai_model: str = Field(default="", alias="DAEMON_OPENAI_MODEL")
    gemini_model: str = Field(default="", alias="DAEMON_GEMINI_MODEL")
    openai_compatible_model: str = Field(
        default="", alias="DAEMON_OPENAI_COMPATIBLE_MODEL"
    )
    openai_compatible_base_url: str = Field(
        default="", alias="DAEMON_OPENAI_COMPATIBLE_BASE_URL"
    )
    """Which OpenAI-compatible endpoint answers, up to and including the version
    segment - `https://api.deepseek.com/v1`, not the `/chat/completions` below it.

    No default, deliberately, for the reason `DAEMON_GEMINI_LIVE_MODEL` has none:
    a guessed endpoint fails at the first conversation instead of at startup."""

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
    (Task.EMBED is always `ollama`, whatever `DAEMON_PROVIDER` is). bge-m3 is
    multilingual, which the Korean recall path depends on - docs/PLAN.md 4.3 shows
    FTS5 alone missing inflected Korean, and that is the whole reason vectors were
    pulled into M1b."""

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

    gemini_live_transport: str = Field(
        default="api_key", alias="DAEMON_GEMINI_LIVE_TRANSPORT"
    )
    """Which endpoint the Gemini Live session dials: one of GEMINI_LIVE_TRANSPORTS.

    `api_key` is the default and needs nothing but GEMINI_API_KEY. `vertex` needs a
    GCP project and credentials, and buys two things measured 2026-09-02
    (docs/design/vertex-live-transport.md): `gemini-live-2.5-flash-native-audio`,
    which exists on no other endpoint, and 1430 ms to first audio against the
    API-key endpoint's 3137 ms for the same model family. It is not a strict
    upgrade - the newer generation (3.1 live) exists only on `api_key` - which is
    why this is an axis rather than a migration."""

    vertex_project: str = Field(default="", alias="DAEMON_VERTEX_PROJECT")
    """GCP project for the `vertex` transport. Required by it, unused otherwise."""

    vertex_location: str = Field(default="us-central1", alias="DAEMON_VERTEX_LOCATION")
    """Region for the `vertex` transport. Not every region serves live models: the
    Seoul region serves none, and from Seoul us-west1 measured identical to
    us-central1 (1441 ms both), so proximity buys nothing and the default stays
    where the models are."""

    vertex_credentials_path: str = Field(
        default="", alias="GOOGLE_APPLICATION_CREDENTIALS"
    )
    """Service-account key file for the `vertex` transport, or empty for
    Application Default Credentials. Named after the environment variable Google's
    own libraries read, because a self-hoster who already has one set should not
    have to set it twice."""

    voice_provider: str = Field(default="gemini", alias="DAEMON_VOICE_PROVIDER")
    """Which native-audio backend voice mode uses: one of VOICE_PROVIDERS. Not derived
    from the text provider."""

    openai_realtime_model: str = Field(default="", alias="DAEMON_OPENAI_REALTIME_MODEL")
    """OpenAI Realtime model id (e.g. gpt-realtime), distinct from DAEMON_OPENAI_MODEL:
    the realtime endpoint takes its own id. No default - a guessed id fails at the first
    voice turn, which is what this module exists to prevent."""

    openai_realtime_voice: str = Field(default="", alias="DAEMON_OPENAI_REALTIME_VOICE")
    """Which prebuilt OpenAI voice: one of OPENAI_REALTIME_VOICES, or empty for the
    server default. Checked at construction, not on the wire."""

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

    proactive_daily_budget: int = Field(default=5, alias="DAEMON_PROACTIVE_DAILY_BUDGET")
    """Utterances per local day, all kinds. PLAN 6.2's original figure was eight,
    but it was never once binding: 572 judge calls, all time, 0 utterances, so the
    number that mattered was the judge declining, not this ceiling. It becomes a
    real ceiling with the `topic` kind (ADR 0015): unlike the other four
    generators, which need the owner to have said something in particular,
    `topic` can always find material - any quiet entity plus a web search - so
    five is the number that now does what eight never had to."""

    proactive_kind_budgets: Annotated[dict[str, int], NoDecode] = Field(
        default_factory=lambda: {
            "association": 3,
            "emotional": 2,
            "open_loop": 2,
            "calendar": 2,
            "silence": 1,
            "pattern_time": 1,
        },
        alias="DAEMON_PROACTIVE_KIND_BUDGETS",
    )
    """Per-kind ceilings for one local day. Replaces the single open_loop cap.

    They sum to 9 against a daily budget of 5 on purpose: these are ceilings, not
    allocations, and the total is what binds. The shape is PLAN 6.2's - the cheap
    kind to generate (open_loop) eats the budget on equal terms and turns a
    companion into a reminder app, and the Her feeling comes from the kinds with
    no business to transact. So the two businessless kinds get the most room.

    As JSON, like DAEMON_ROUTE_OVERRIDES:
    `DAEMON_PROACTIVE_KIND_BUDGETS={"open_loop": 1, "silence": 1}`. **Replaces the
    whole table, it does not merge with the default above** - setting one kind
    without repeating the rest removes every other kind's ceiling, leaving them
    bound only by the daily total. `_check` validates keys against
    `PROACTIVE_KINDS` and values as non-negative integers, because the setting
    this one replaced had a validator and this one shipped without one:
    `{"open_loop": 99, "nonsense": -5}` used to load clean, and a typo silently
    wiping the ceilings is exactly the reminder-app failure PLAN 6.2 wrote this
    table to prevent.

    `topic` (ADR 0015) deliberately gets no entry here, and that is not an
    omission to close: the owner rejected per-kind quotas for it as artificial,
    and the sentence above already says these are ceilings while the daily total
    is what binds - `topic` is bound by that total and by
    `proactive_daily_budget` moving 8 -> 5 for exactly this reason, not by a
    ceiling of its own. Do not add one.

    `calendar` (ADR 0021) does get one, and gets `open_loop`'s number rather than
    `topic`'s absence, because it is the same kind of thing `open_loop` is: PLAN
    6.2's cheap-to-satisfy, transactional kind that turns a companion into a
    reminder app if it competes on equal terms. Two rather than one because the
    owner's measured calendar runs to two events on a busy day (2026-09-01: 7
    events in 30 days, never more than two dated the same day), and a ceiling of
    one would mean the second interview of the day is structurally unmentionable.
    Two is also still below `association`'s three, so the businessless kind keeps
    the most room, which is the shape PLAN 6.2 asks for."""

    proactive_cooldown_minutes: int = Field(default=90, alias="DAEMON_PROACTIVE_COOLDOWN_MINUTES")
    """Minimum gap between two proactive utterances, whatever their kind. Raised
    from 30 alongside the daily budget above, and for the same reason: measured
    against the live database, `open_loop` (the cheapest generator to satisfy)
    fired 3 times in 7 days, so 30 minutes was never the thing keeping this
    quiet. `topic` is not similarly rate-limited by needing the owner to say
    something - a real cooldown is what stands in for that."""

    proactive_quiet_hours: str = Field(default="23:00-09:00", alias="DAEMON_PROACTIVE_QUIET_HOURS")
    """Local `HH:MM-HH:MM` when it never speaks. Wraps midnight when start > end."""

    proactive_silence_hours: float = Field(default=12.0, alias="DAEMON_PROACTIVE_SILENCE_HOURS")
    """Hours without conversation before the `silence` kind becomes a candidate."""

    calendar_email: str = Field(default="", alias="DAEMON_CALENDAR_EMAIL")
    """The Google account whose calendar the `calendar` kind reads. Empty = off.

    A setting rather than something the code discovers, and that is the whole
    point of it. `get_events` requires `user_google_email` (measured against the
    live server: omitting it is a validation error, not a default), and the
    obvious way to find it - call `list_calendars` first and read the primary id
    off the reply - is precisely the shape ADR 0015 spent four review rounds
    removing: a lookup result becoming the next lookup's argument. So the owner
    types it once, code passes it verbatim, and nothing the server says can
    change what the next call asks for.

    Empty is the honest default. There is no address to guess, and a generator
    that silently reads nothing is the failure `daemon/proactivity/candidates.py`
    exists not to have - so `daemon doctor` names this state and
    `daemon proactive` prints it (docs/CONTRACTS.md 12)."""

    # `DAEMON_PROACTIVE_SPEAKER_ENABLED` used to live here as a second switch.
    # Removed: `voice_enabled` above now governs the speaker path too, and
    # `model_config`'s `extra="ignore"` means an old .env that still sets it
    # loads fine - the key is just inert, not honoured and not an error.

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
    openai_compatible_api_key: str = Field(default="", alias="OPENAI_COMPATIBLE_API_KEY")

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

    tools_max_rounds: int = Field(default=25, alias="DAEMON_TOOLS_MAX_ROUNDS")
    """Tool round-trips allowed in one turn before it must answer. This is the value
    the assembled app actually uses (`app.py` passes it to `ConversationLoop`), so it
    is the live cap - not `loop.MAX_TOOL_ROUNDS`, which is only the constructor's
    fallback. Keep the two in step (`test_the_default_round_cap_matches_the_loop_constant`):
    a generous last-resort ceiling, since a stuck loop is caught far earlier by
    `loop.LOOP_REPEAT_LIMIT`. Six cut honest multi-step builds short."""

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

    face_lipsync_enabled: bool = Field(default=False, alias="DAEMON_FACE_LIPSYNC_ENABLED")
    """Whether the face's mouth is rendered from the audio it is speaking.

    Off by default, and for a different reason than DAEMON_BROWSER_ENABLED,
    DAEMON_SCREEN_ENABLED and DAEMON_VOICE_ENABLED above: those are reach decisions,
    and this one is a cost decision. It reads nothing new and shows nothing `/face`
    does not already show - it wants 1.70GB of pre-converted weights on disk and
    realises 1.62GB of them for 693ms the first time it speaks
    (2026-08-25-face-design.md, "전송과 로딩"; the 1.8/1.75GB the earlier prose quoted
    predates that measurement). An install that never turns it on should pay nothing
    for it existing - the same reasoning that has `daemon/app.py` import the face
    routes lazily.

    A live toggle on purpose (2026-08-26-face-lipsync-design.md §6): the pass mark is
    whether the mouth *feels* right, and that can only be judged by turning it off and
    on inside one conversation. `/face/manifest` answers with it so the page follows
    the switch rather than guessing, and turning it off leaves the pre-rendered clips
    of face v1 - they are not a fallback, they are still the other half of the face."""

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
        is dropped sends its task to the computed route's provider instead of the
        chosen one and says nothing, which is the degradation this module exists to
        prevent.
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

    @field_validator("proactive_kind_budgets", mode="before")
    @classmethod
    def _parse_kind_budgets(cls, value: object) -> object:
        """Empty means no per-kind ceilings (bound only by the daily total);
        anything else must be the documented JSON. Same shape as
        `_parse_overrides` and for the same reason: `NoDecode` on the field is
        what lets this run at all, and unparseable text raises rather than
        quietly becoming `{}` - which here would silently remove every ceiling
        instead of leaving the default table in place. `_check` below still
        validates the keys and values once this has produced a dict."""
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            import json

            try:
                return json.loads(text)
            except ValueError as exc:
                raise ValueError(
                    "DAEMON_PROACTIVE_KIND_BUDGETS must be JSON per kind, as in "
                    f'{{"open_loop": 1, "silence": 1}} - {exc}'
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

    @field_validator("openai_compatible_base_url", mode="before")
    @classmethod
    def _clean_base_url(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        return clean_base_url(value)

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.stale_preset:
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
                f"to choose one (ollama | {', '.join(HOSTED_PROVIDERS)})"
            )

        problems: list[str] = []
        for task, provider in self.route_overrides.items():
            if provider not in PROVIDER_KEY_ENV:
                problems.append(
                    f"override for {task.value} names unknown provider {provider!r} "
                    f"(known: {', '.join(sorted(PROVIDER_KEY_ENV))})"
                )

        # A typo in the provider name is wrong whatever else is on, so this one
        # is unconditional. What a *session* additionally needs lives in
        # `voice_session_problems` - see the comment on the wake checks below for
        # why that set is not applied here on `voice_enabled` alone.
        if self.voice_provider not in VOICE_PROVIDERS:
            problems.append(
                f"DAEMON_VOICE_PROVIDER is {self.voice_provider!r}; expected one of "
                f"{', '.join(VOICE_PROVIDERS)}"
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

        if self.openai_realtime_voice and self.openai_realtime_voice not in OPENAI_REALTIME_VOICES:
            problems.append(
                f"DAEMON_OPENAI_REALTIME_VOICE is {self.openai_realtime_voice!r}; expected one of "
                "the OpenAI Realtime voices, or empty to leave it to the server"
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
        # On `voice_enabled`, at load time, and that took two goes to settle.
        #
        # These checks briefly moved off `voice_enabled` onto `wake_enabled`. The
        # reason was real: the switch had just come to mean two things - "a hosted
        # session may run" and "a proactive line may leave the local speaker" -
        # and `/usr/bin/say` needs no route, no model and no key. Under the old
        # preset table, `offline` could satisfy the second and never the first, so
        # demanding both at load did not degrade that install, it stopped
        # `Settings` from loading at all, which stops the daemon.
        #
        # ADR 0012 removed the premise. Voice is its own axis now: turning it on
        # *adds* the CHAT_VOICE route rather than asking the preset for one, so
        # there is no configuration where voice is on and a hosted session is
        # impossible. That makes load time the right place again, and the earlier
        # move unnecessary rather than wrong. What survives from it is this list
        # having one home (`voice_session_problems`); `run_voice` no longer
        # re-applies it, because by then `Settings` has already validated.
        if self.voice_enabled:
            problems += [
                f"DAEMON_VOICE_ENABLED is on with {problem}"
                for problem in self.voice_session_problems()
            ]
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
        for name, value in (
            ("DAEMON_PROACTIVE_DAILY_BUDGET", self.proactive_daily_budget),
            ("DAEMON_PROACTIVE_COOLDOWN_MINUTES", self.proactive_cooldown_minutes),
        ):
            if value < 0:
                problems.append(f"{name} is {value}; it cannot be negative")
        if self.proactive_silence_hours <= 0:
            problems.append(
                f"DAEMON_PROACTIVE_SILENCE_HOURS is {self.proactive_silence_hours}; "
                "at zero or below, every tick would be a silence candidate"
            )
        for kind, ceiling in self.proactive_kind_budgets.items():
            if kind not in PROACTIVE_KINDS:
                problems.append(
                    f"DAEMON_PROACTIVE_KIND_BUDGETS names unknown kind {kind!r}; "
                    f"expected one of {', '.join(PROACTIVE_KINDS)}"
                )
            elif not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 0:
                problems.append(
                    f"DAEMON_PROACTIVE_KIND_BUDGETS[{kind!r}] is {ceiling!r}; it must "
                    "be a non-negative integer"
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
        if provider == OPENAI_COMPATIBLE and not self.openai_compatible_base_url:
            found.append(
                f"{context} routes to {provider!r} but no endpoint is set "
                "(DAEMON_OPENAI_COMPATIBLE_BASE_URL)"
            )
        return found

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

    def _routes_a_hosted_task(self) -> bool:
        """Whether any of the always-routed tasks resolves to something other than
        `ollama` - used only to decide whether an empty `DAEMON_PROVIDER` is an error.
        With `provider=""`, every hosted-role task maps to `""` here, which is neither
        `None` nor `OLLAMA`, so this is True and the empty provider is refused. With
        `provider="ollama"`, every role is `OLLAMA`, so this is False."""
        return any(self.routing.get(t) not in (None, OLLAMA) for t in self._HOSTED_ROLE_TASKS)

    def voice_session_problems(self) -> list[str]:
        """What stops a *hosted voice session* from running, as clauses a caller
        prefixes with its own context. Empty when nothing does.

        **One caller: `_check`, at load time, guarded on `voice_enabled`.** Say so
        before adding a clause here - anything that is only decidable when a
        session opens does not belong in this list, because nothing re-runs it
        then. A rotated key or an endpoint that went unreachable after startup
        will not be caught here; that is the session's own problem to report.

        It briefly had two callers, and the detour is worth knowing because the
        failure it worked around can return. `voice_enabled` came to mean two
        things when the speaker switch merged into it - "a hosted session may run"
        and "a proactive line may come out of the local speaker" - and only the
        first needs any of this. `/usr/bin/say` needs neither a route nor a model
        nor a key, so checking at load on `voice_enabled` alone stopped `Settings`
        from loading under the `offline` preset, which stops the daemon, and made
        docs/PLAN.md 7's promise about local proactive speech unreachable on the
        one preset the promise is about. The workaround was to check late instead:
        at load only when `wake_enabled`, and again in `run_voice`.

        ADR 0012 removed the premise rather than the symptom - `routing` now
        *adds* the CHAT_VOICE row whenever voice is on, because voice is its own
        axis and never was a property of the preset - so there is no longer a
        configuration where voice is on and a session is impossible. Load time
        became right again, `run_voice`'s repeat became a branch nothing can
        reach, and the first clause in this list (does the preset route a voice
        task) became one that cannot fire. All three are gone. A check that cannot
        fire is worse than no check, because the next reader budgets for it.

        What remains is the half that was always real:
        the chosen provider's own model and key.
        """
        problems: list[str] = []
        # The chosen provider's own realtime model plus that provider's key. The
        # text model (DAEMON_*_MODEL) is neither required nor read for voice.
        if self.voice_provider == "gemini" and not self.gemini_live_model:
            problems.append(
                "DAEMON_VOICE_PROVIDER=gemini but DAEMON_GEMINI_LIVE_MODEL is empty; "
                "the native-audio endpoint needs its own id"
            )
        # The *resolved* voice route, not the axis. `route_overrides` can send
        # chat_voice to gemini while DAEMON_VOICE_PROVIDER says openai (see
        # `routing`), and `run_voice` builds a Gemini session for whatever the route
        # resolves to - so reading the axis here left exactly that combination
        # unvalidated, and the missing project then surfaced as a ValueError raised
        # inside the reconnect loop rather than as a setting that failed to load.
        voice_route = self.routing.get(Task.CHAT_VOICE, self.voice_provider)
        if voice_route == "gemini":
            if self.gemini_live_transport not in GEMINI_LIVE_TRANSPORTS:
                problems.append(
                    f"DAEMON_GEMINI_LIVE_TRANSPORT is {self.gemini_live_transport!r}; "
                    f"expected one of {', '.join(GEMINI_LIVE_TRANSPORTS)}"
                )
            # Checked here rather than at the handshake: the Vertex URI is built
            # from the project and region, and a missing one is a 404 mid-sentence
            # instead of a setting that failed to load.
            if self.gemini_live_transport == "vertex":
                if not self.vertex_project:
                    problems.append(
                        "DAEMON_GEMINI_LIVE_TRANSPORT=vertex but DAEMON_VERTEX_PROJECT "
                        "is empty; the Vertex endpoint addresses models by project"
                    )
                if not self.vertex_location:
                    problems.append(
                        "DAEMON_GEMINI_LIVE_TRANSPORT=vertex but DAEMON_VERTEX_LOCATION "
                        "is empty; live models are served from specific regions only"
                    )
            # The model and the endpoint are one decision, and the admin console
            # makes them two clicks apart: switching to vertex rewrites the model
            # id, and switching back leaves it. Only this direction is refused,
            # because it is the only one we are certain about - VERTEX_LIVE_MODELS
            # goes stale the day Google ships another Vertex live model, and a
            # config that refuses to load over a *new* id would be worse than the
            # 1008. `daemon doctor` reports the uncertain direction instead.
            elif self.gemini_live_model in VERTEX_LIVE_MODELS:
                problems.append(
                    f"DAEMON_GEMINI_LIVE_MODEL is {self.gemini_live_model!r}, which only "
                    "the vertex transport serves - the API-key endpoint closes 1008 "
                    '"is not found" for it. Set DAEMON_GEMINI_LIVE_TRANSPORT=vertex, or '
                    "choose a model that endpoint lists"
                )
        if self.voice_provider == "openai":
            if not self.openai_realtime_model:
                problems.append(
                    "DAEMON_VOICE_PROVIDER=openai but DAEMON_OPENAI_REALTIME_MODEL is "
                    "empty; the realtime endpoint needs its own id"
                )
            if not self.openai_api_key:
                problems.append(
                    "DAEMON_VOICE_PROVIDER=openai but OPENAI_API_KEY is empty"
                )
        return problems

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
        # Before the routing lookup, because voice-off is now the only reason a voice
        # task is missing from the table, and the honest answer is the switch rather
        # than the routing.
        if task in VOICE_TASKS and not self.voice_enabled:
            raise ConfigError(
                f"{task.value} was requested but voice is off (DAEMON_VOICE_ENABLED)"
            )
        provider = self.routing.get(task)
        if provider is None:
            raise ConfigError(f"no route for {task.value}")
        if task in VOICE_TASKS:
            # The native-audio endpoint takes its own model id (never DAEMON_*_MODEL).
            model = self.gemini_live_model if provider == "gemini" else self.openai_realtime_model
            return Route(provider=provider, model=model)
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
