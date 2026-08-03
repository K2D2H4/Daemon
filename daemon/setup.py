"""First-run onboarding: `daemon setup`.

Onboarding used to be "copy `.env.example`, edit it by hand, then go find your
numeric Telegram id with @userinfobot". That is a developer procedure, not a
product (docs/PLAN.md decision 1), and it fails *late*: nothing checked an API
key until the first conversation turn needed it, so a key with a trailing space
looked like a broken companion rather than a typo. **Everything this wizard
accepts is verified against the provider before it is written**, and the file is
written once, at the end, after the user has seen which keys change.

Three more rules the shape of this module comes from:

  * It asks only what the chosen preset's routing table actually requires
    (docs/PLAN.md 3.2), so an `offline` install is never asked for a hosted key
    and a text-only install is never asked for a voice key.
  * Credentials are read from and written to `./.env` only, never the shell
    environment. The OS service unit carries no secrets and sees no shell
    (daemon/service.py), so a key that exists only as an exported variable would
    work in the terminal and break under launchd - the worst kind of "it worked
    yesterday".
  * It never asks for a Telegram user id, and it never decides who the owner is.
    Trust is `channels/pairing.py`'s to grant; the two steps after the file is
    written only drive it.

Those two steps are why this module does not end at the file. A first run used to
finish with a written `.env` and two chores left: pair a phone across two
terminals, and hand-write `persona/seed.md` - which nothing creates, so an
install that skipped it talked like a stock assistant instead of like anyone
(docs/PLAN.md 5). Both are onboarding, so both happen here:

  * `persona/seed.md` from three questions, plus the anchor items docs/PLAN.md
    5.4 fixes there. Human-owned, so an existing file is never rewritten.
  * pairing, in this process. The pairing code exists to carry an identity
    between the channel and a terminal the stranger cannot reach; during setup
    those are the same process, so the code would be transcription with nothing
    to protect. What still holds is that the numeric id is what gets approved and
    that `Pairing` decides ownership.

Synchronous on purpose: the rest of the I/O path is async because it shares a
loop with the conversation, but this module's I/O is a human typing, and one
blocking request per answered question reads better than an event loop wrapped
around `input()`.

Every side effect is a seam - stdin/stdout (`Prompt`), the network probes
(`Checks`), the browser (`opener`), the file path (`env_path`) - so the tests
need no network, no keys and no browser.
"""

from __future__ import annotations

import getpass
import json
import os
import sys
import time
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import httpx
from pydantic import ValidationError

from daemon.config import (
    ANTHROPIC,
    DEFAULT_HOSTED_PROVIDER,
    ENV_FILE,
    GEMINI,
    HOSTED,
    HOSTED_PROVIDERS,
    OLLAMA,
    OPENAI,
    PRESETS,
    ConfigError,
    Settings,
    preset_providers,
    providers_for,
)
from daemon.fs import FILE_MODE, secure_dir, secure_file
from daemon.tasks import Task

if TYPE_CHECKING:  # imported lazily at runtime - see `Wizard._pair_here`
    from daemon.channels.pairing import Approval, Pairing

OK = 0
PROBLEM = 1

HTTP_TIMEOUT = 15.0
"""Short. A wizard waiting on a hung endpoint is worse than a wizard that says
the endpoint is unreachable and asks again."""

ANTHROPIC_KEYS_URL = "https://console.anthropic.com/settings/keys"
OPENAI_KEYS_URL = "https://platform.openai.com/api-keys"
AI_STUDIO_URL = "https://aistudio.google.com/apikey"
BOTFATHER_URL = "https://t.me/BotFather"

OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
"""Duplicated from daemon/llm/providers/openai.py rather than imported: nothing
outside `daemon/llm/providers/` may import a provider (docs/CONTRACTS.md 4), and
the providers are async while this module is deliberately not."""

DEFAULT_OLLAMA_MODEL = "gemma3:4b"
"""Not a reasoning model, and not `Settings.ollama_model`'s default. docs/PLAN.md
3.2.1 measured 1.7 s against 11.8 s for the same three Korean questions, the
difference being a chain of thought that is discarded unread. A fresh install
should land on the usable side of that."""
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_LIVE_MODEL = "gemini-live-2.5-flash-preview"
DEFAULT_OPENAI_MODEL = "gpt-5.1"
"""Offered, then checked against the account's own model list before it is
written - so a default that goes stale is a sentence here and a hint naming the
ids that do exist, rather than a 404 at the first message."""

PRESET_ORDER = ("offline", "balanced", "quality")
DEFAULT_PRESET = "balanced"

HOSTED_HELP: dict[str, tuple[str, ...]] = {
    "anthropic": (
        "The default, and what the presets above were measured against. Reads",
        "long conversations well, which is what the daily reflection does before",
        "its conclusions propagate into the whole memory graph.",
    ),
    "openai": (
        "Pick this if it is the account you already pay for. Nothing else about",
        "Daemon changes - same presets, same memory, same everything.",
    ),
    "gemini": (
        "The one that shares a key with voice: native-audio voice is Google's",
        "either way, so choosing it here means one key and one bill instead of",
        "two. Note the Standard-key trap below if you have an older key.",
    ),
}
"""What separates them, rather than which model ids they publish - ids change
every few months and are the wrong thing to choose a vendor by."""

PRESET_HELP: dict[str, tuple[str, ...]] = {
    "offline": (
        "Everything runs on this machine through Ollama - conversation,",
        "reflection, and the decision to speak first. Nothing leaves the machine.",
        "Voice is not available: native-audio voice means a hosted model is both",
        "the brain and the voice, and leaving that out is exactly what makes the",
        "privacy promise true instead of aspirational.",
        "Needs: Ollama and two local models. No API keys, no accounts.",
    ),
    "balanced": (
        "Conversation and the daily reflection go to a hosted model, because",
        "reflection quality propagates into the whole memory graph. The 'should I",
        "speak?' check stays local - it runs every five minutes whether or not it",
        "ever speaks, so hosted cost would accumulate for nothing.",
        "Voice can be turned on. Needs: one hosted API key (whose is the next",
        "question), plus Ollama for the local check and for recall embeddings.",
    ),
    "quality": (
        "Everything hosted, including the five-minute proactive check. Best",
        "quality, highest running cost. Embeddings stay local in every preset,",
        "so recall still wants Ollama - `daemon doctor` checks that.",
        "Voice can be turned on. Needs: one hosted API key.",
    ),
}

GEMINI_STANDARD_KEY_HINT = (
    "This key looks like a Google 'Standard' API key. The Gemini API already "
    "refuses unrestricted Standard keys and will refuse the remaining ones in "
    f"September 2026. Create a new one at {AI_STUDIO_URL} - keys made there now "
    "are auth keys, which is what this needs."
)

PAIRING_NOTE = (
    "No Telegram allowlist yet, and that is on purpose: nobody is asked for a",
    "numeric user id here. Start the daemon, message your bot, and it answers with",
    "a pairing code. Until you pair, nobody can reach it.",
    "",
    "  daemon run                     - in one terminal, then message the bot",
    "  daemon pairing approve <code>  - in another, with the code it replied",
)
"""Printed when pairing did not happen here - declined, timed out, interrupted.

Still the documented route, and still the only one available once this process is
gone: the wizard's shortcut works because it holds the token and the database at
the same moment, which nothing else does.
"""

PAIRING_POLICY = "pairing"
"""The `DAEMON_TELEGRAM_DM_POLICY` value this step belongs to (see
`channels/telegram.DM_POLICIES`). Under `allowlist` there is nothing to pair:
ids come from the file, and an empty list is refused at startup."""

PAIRING_POLL_SECONDS = 20
"""Server-side long-poll window per `getUpdates` while waiting for the owner.
Shorter than the channel's 30s so a Ctrl-C is never far away."""

PAIRING_WAIT_SECONDS = 180.0
"""How long the wizard waits for that first message.

Long enough to pick up a phone, unlock it, search for a bot and type something -
that is tens of seconds, and three minutes leaves room for the search going
badly. Not longer: a wizard nobody came back to should hand the terminal back,
and PAIRING_NOTE loses nothing by being read a minute later."""

DEFAULT_PERSONA_NAME = "Daemon"

VOICE_PRESETS = (
    "Short and dry. I do not pad answers, and I do not perform enthusiasm.",
    "Warm and talkative. I say more than the minimum when there is more to say.",
    "Playful. I take a joke and give one back, and I am not solemn about myself.",
)
"""Three examples, because a seed is a seed - a character sheet at first run is
onboarding fatigue paid before the product has earned any.

The first one is the default deliberately. Left with no seed at all, a small
local model settles into the eager-helper register ("how may I help you today?"),
which is the one voice a companion cannot have: it is the voice of software."""

PERSONA_LINE_LIMIT = 200
"""Characters kept per answer. A seed that is mostly one answer is not a seed,
and this file is prepended to every single turn."""

SEED_NOTE = """<!--
This file is yours. Daemon reads it on every turn and never writes to it.

`daemon setup` wrote it once, from three answers. Edit it by hand whenever you
like - it is re-read per turn, so a change lands on the next message with no
restart. What Daemon works out about talking to you accumulates separately, in
persona/learned.md, which is the half you only read. Nothing being able to touch
this half is what stops a personality from dissolving into whatever the last
month of conversation happened to imply (docs/PLAN.md 5.1).
-->"""

SEED_ANCHOR = (
    "# Constant",
    "",
    "Not preferences and not up for revision. Daemon cannot write to this file, so",
    "nothing it learns later can quietly drop these (docs/PLAN.md 5.4):",
    "",
    "- I do not simply agree. When I think the user is wrong, I say so plainly.",
    "- I am someone with my own view, not a mirror of whoever is talking to me.",
)
"""Written into every seed, whatever the answers were.

A quality device before an ethical one: a character that only agrees is dull and
unconvincing, and the same property is the structural answer to over-fitting to
one person's opinions. It survives because it is in the file the AI cannot edit -
no separate feature required."""


class Cancelled(Exception):
    """The user stopped (Ctrl-C, Ctrl-D, or a closed stdin). Nothing is written."""


# --- probes ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Verdict:
    """Result of checking one credential. `detail` is safe to print - it never
    carries the secret, only what the provider said about it."""

    ok: bool
    detail: str
    hint: str = ""
    subject: str = ""
    """Who the provider says the credential belongs to, when it says: the bot's
    `@username` for Telegram. Not a secret, and not trusted for anything - it is
    printed so the user knows which bot to message next."""


@dataclass(frozen=True, slots=True)
class Updates:
    """One `getUpdates` batch, or why there is none."""

    ok: bool
    updates: tuple[dict[str, Any], ...] = ()
    detail: str = ""


@dataclass(frozen=True, slots=True)
class OllamaState:
    reachable: bool
    detail: str
    models: tuple[str, ...] = ()


def check_anthropic(key: str, model: str) -> Verdict:
    """Validate the key against the model list.

    `GET /v1/models` is the cheapest call that proves the key works: it costs no
    tokens and, unlike a one-token message, it also tells us whether the
    configured model id exists - the other late failure this wizard exists to
    prevent.
    """
    try:
        response = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            params={"limit": 100},
            timeout=HTTP_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return Verdict(False, f"could not reach api.anthropic.com: {exc}")

    if response.status_code in (401, 403):
        return Verdict(
            False,
            f"Anthropic rejected the key (HTTP {response.status_code}).",
            hint=f"Copy it again from {ANTHROPIC_KEYS_URL} - it starts with 'sk-ant-'.",
        )
    if response.status_code != 200:
        return Verdict(False, f"api.anthropic.com returned HTTP {response.status_code}")

    ids = _string_field(response, "data", "id")
    if model and ids and model not in ids:
        return Verdict(
            True,
            f"key works, but {model!r} is not in your model list",
            hint="Set DAEMON_ANTHROPIC_MODEL in .env to one of: " + ", ".join(sorted(ids)[:5]),
        )
    return Verdict(True, "key works")


def check_openai(key: str, model: str) -> Verdict:
    """Validate the key, and the model id, against the account's model list.

    Same call and same reasoning as `check_anthropic`: `GET /v1/models` costs no
    tokens and is the only cheap way to find out that a model id this install will
    ask for is not one this account can use.
    """
    try:
        response = httpx.get(
            OPENAI_MODELS_URL,
            headers={"authorization": f"Bearer {key}"},
            timeout=HTTP_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return Verdict(False, f"could not reach api.openai.com: {exc}")

    if response.status_code in (401, 403):
        return Verdict(
            False,
            f"OpenAI rejected the key (HTTP {response.status_code}).",
            hint=f"Copy it again from {OPENAI_KEYS_URL} - it starts with 'sk-'.",
        )
    if response.status_code != 200:
        return Verdict(False, f"api.openai.com returned HTTP {response.status_code}")

    ids = _string_field(response, "data", "id")
    if model and ids and model not in ids:
        return Verdict(
            True,
            f"key works, but {model!r} is not in your model list",
            hint="Set DAEMON_OPENAI_MODEL in .env to one of: " + ", ".join(sorted(ids)[:5]),
        )
    return Verdict(True, "key works")


def check_gemini(key: str) -> Verdict:
    """Validate the key, and name the Standard-key trap when that is the cause.

    The key goes in the `x-goog-api-key` header rather than the documented `?key=`
    query parameter: a secret in a URL ends up in proxy logs.
    """
    try:
        response = httpx.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": key},
            timeout=HTTP_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        return Verdict(False, f"could not reach generativelanguage.googleapis.com: {exc}")

    if response.status_code == 403:
        # 403/PERMISSION_DENIED on a syntactically fine key is the Standard-key
        # rejection; an outright bad key comes back 400/API_KEY_INVALID.
        return Verdict(False, "Google refused the key (HTTP 403).", hint=GEMINI_STANDARD_KEY_HINT)
    if response.status_code == 400:
        return Verdict(
            False,
            "Google says the key is not valid (HTTP 400).",
            hint=f"Create one at {AI_STUDIO_URL} and paste the whole string.",
        )
    if response.status_code != 200:
        return Verdict(
            False, f"generativelanguage.googleapis.com returned HTTP {response.status_code}"
        )
    return Verdict(True, "key works")


def check_telegram(token: str) -> Verdict:
    """`getMe`: proves the token and gives us the bot's name to show back."""
    try:
        response = httpx.get(
            f"https://api.telegram.org/bot{token}/getMe", timeout=HTTP_TIMEOUT
        )
    except httpx.HTTPError as exc:
        # httpx puts the request URL in its messages, and for the Bot API the
        # token IS part of the URL.
        return Verdict(False, f"could not reach api.telegram.org: {_redact(str(exc), token)}")

    if response.status_code in (401, 404):
        return Verdict(
            False,
            "Telegram rejected the token.",
            hint=f"Ask @BotFather for it again at {BOTFATHER_URL} (/mybots -> API Token).",
        )
    if response.status_code != 200:
        return Verdict(False, f"api.telegram.org returned HTTP {response.status_code}")

    try:
        result = response.json().get("result") or {}
        username = str(result.get("username", ""))
    except ValueError:
        return Verdict(False, "api.telegram.org returned a non-JSON body")
    return Verdict(
        True,
        f"connected to @{username}" if username else "token works",
        subject=f"@{username}" if username else "",
    )


def fetch_updates(token: str, offset: int | None, timeout: int) -> Updates:
    """One long poll of `getUpdates`, for the pairing step below.

    The channel owns the real polling loop (daemon/channels/telegram.py). This is
    the same endpoint borrowed for the couple of minutes in which the wizard is
    the only thing holding the token, and it stays here rather than in the channel
    because there is no channel yet: nothing has been paired, so a TelegramChannel
    built now would have nobody it is allowed to hear.

    Every update this returns is one the daemon will never be offered - passing
    `offset` is what confirms the previous batch server-side. `Wizard._pair` is
    what makes that safe, by saving the cursor.
    """
    params: dict[str, Any] = {
        "timeout": timeout,
        # A JSON string, not a list: httpx would render a one-element list as
        # `allowed_updates=message`, which the Bot API refuses to parse.
        "allowed_updates": json.dumps(["message"]),
    }
    if offset is not None:
        params["offset"] = offset
    try:
        response = httpx.get(
            f"https://api.telegram.org/bot{token}/getUpdates",
            params=params,
            # Must outlast the server-side wait, or every poll looks like a failure.
            timeout=timeout + HTTP_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        # Same reason as check_telegram: the token is in the path, so httpx's own
        # message is not printable.
        detail = _redact(str(exc), token)
        return Updates(False, detail=f"lost contact with api.telegram.org: {detail}")
    if response.status_code != 200:
        return Updates(False, detail=f"api.telegram.org returned HTTP {response.status_code}")
    try:
        result = response.json().get("result")
    except ValueError:
        return Updates(False, detail="api.telegram.org returned a non-JSON body")
    if not isinstance(result, list):
        return Updates(False, detail="getUpdates answered without a list of updates")
    return Updates(True, tuple(item for item in result if isinstance(item, dict)))


def check_ollama(base_url: str) -> OllamaState:
    """Reachability plus the installed model list, in two cheap local calls."""
    root = base_url.rstrip("/")
    try:
        version = httpx.get(f"{root}/api/version", timeout=HTTP_TIMEOUT)
        tags = httpx.get(f"{root}/api/tags", timeout=HTTP_TIMEOUT)
    except httpx.HTTPError as exc:
        return OllamaState(False, f"not reachable at {root}: {exc}")
    if version.status_code != 200:
        return OllamaState(False, f"{root} answered HTTP {version.status_code}")
    label = "unknown version"
    try:
        label = f"v{version.json().get('version', '?')}"
    except ValueError:
        pass
    installed = _string_field(tags, "models", "name")
    return OllamaState(True, f"reachable at {root} ({label})", installed)


@dataclass(frozen=True, slots=True)
class Checks:
    """The network probes, in one injectable bundle."""

    anthropic: Callable[[str, str], Verdict] = check_anthropic
    openai: Callable[[str, str], Verdict] = check_openai
    gemini: Callable[[str], Verdict] = check_gemini
    telegram: Callable[[str], Verdict] = check_telegram
    ollama: Callable[[str], OllamaState] = check_ollama
    updates: Callable[[str, int | None, int], Updates] = fetch_updates


def _string_field(response: httpx.Response, container: str, key: str) -> tuple[str, ...]:
    try:
        items = response.json().get(container) or []
    except ValueError:
        return ()
    return tuple(
        str(item[key])
        for item in items
        if isinstance(item, dict) and isinstance(item.get(key), str)
    )


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "<token>") if secret else text


# --- what a preset requires --------------------------------------------------


@dataclass(frozen=True, slots=True)
class Need:
    """One `.env` key the chosen preset requires, and how to obtain it."""

    key: str
    label: str
    why: str
    url: str = ""
    default: str = ""
    secret: bool = False
    skippable: bool = False
    """The wizard accepts an empty answer. `--check` still reports it missing."""
    silent: bool = False
    """Required by the configuration but not a decision anyone should be asked to
    make; the wizard writes `default` without a question. `--check` still reports
    it, so what this command considers complete stays identical to what startup
    considers valid."""

    @property
    def blocking(self) -> bool:
        """Does an empty answer leave the process with no value at all?

        Answered by Settings rather than by opinion, because that is what decides
        whether the daemon can start - and therefore what `--check`'s exit code
        may claim. `DAEMON_OLLAMA_MODEL` is the one interesting case: it *has* a
        built-in default, so it is not blocking, but that default is the reasoning
        model docs/PLAN.md 3.2.1 measured at 11.8 s against gemma3's 1.7 s. The
        wizard offers it anyway; `--check` reports it without failing.
        """
        return not _settings_default(self.key)


def needs_for(env: Mapping[str, str]) -> list[Need]:
    """The keys this configuration is missing an answer for, in asking order."""
    preset = env.get("DAEMON_PRESET", "") or DEFAULT_PRESET
    # The preset tables hold a HOSTED placeholder, so the chosen provider has to be
    # passed in: defaulting it would ask a user who picked GPT for an Anthropic key,
    # and reading the table raw would ask them for a key for "hosted".
    hosted = env.get("DAEMON_HOSTED_PROVIDER", "") or DEFAULT_HOSTED_PROVIDER
    providers = providers_for(
        preset,
        voice_enabled=_truthy(env.get("DAEMON_VOICE_ENABLED", "")),
        hosted=hosted,
    )
    needs: list[Need] = []

    if OLLAMA in _chat_providers(preset, hosted):
        needs.append(
            Need(
                key="DAEMON_OLLAMA_MODEL",
                label="local chat model",
                why="Ollama model used for conversation. Do not use a reasoning model here.",
                default=DEFAULT_OLLAMA_MODEL,
            )
        )
    if ANTHROPIC in providers:
        needs.append(
            Need(
                key="ANTHROPIC_API_KEY",
                label="Anthropic API key",
                why="Your preset sends conversation and reflection to Claude. Your key, "
                "your account, your bill.",
                url=ANTHROPIC_KEYS_URL,
                secret=True,
            )
        )
        # No model question: DAEMON_ANTHROPIC_MODEL is the one hosted model id
        # Settings has a default for, and `check_anthropic` verifies that default
        # against the account rather than assuming it.
    if OPENAI in providers:
        needs.append(
            Need(
                key="OPENAI_API_KEY",
                label="OpenAI API key",
                why="You chose GPT for the hosted work. Your key, your account, your bill.",
                url=OPENAI_KEYS_URL,
                secret=True,
            )
        )
        needs.append(
            Need(
                key="DAEMON_OPENAI_MODEL",
                label="OpenAI model id",
                why="Which model answers. Settings has no default here, so an empty "
                "value would refuse to start.",
                default=DEFAULT_OPENAI_MODEL,
            )
        )
    if GEMINI in providers:
        # Gemini can be here for two unrelated reasons - it is the hosted chat
        # provider, or voice is on and native audio is Google's either way - and
        # the questions differ, so the reason has to be named rather than assumed.
        # It was assumed, back when voice was the only way to get here.
        gemini_chat = hosted == GEMINI and HOSTED in PRESETS[preset].values()
        voice_on = _truthy(env.get("DAEMON_VOICE_ENABLED", ""))
        if gemini_chat and voice_on:
            why = "One key covers both: Gemini answers, and voice is Google's too."
        elif gemini_chat:
            why = "You chose Gemini for the hosted work. Your key, your account, your bill."
        else:
            why = "Voice is on, so audio goes to Google's native-audio model - with your key."
        needs.append(
            Need(
                key="GEMINI_API_KEY",
                label="Gemini API key",
                why=why,
                url=AI_STUDIO_URL,
                secret=True,
            )
        )
        if voice_on:
            needs.append(
                Need(
                    key="DAEMON_GEMINI_LIVE_MODEL",
                    label="Gemini Live model id",
                    why="The realtime audio endpoint takes its own model id, not the text one.",
                    default=DEFAULT_GEMINI_LIVE_MODEL,
                )
            )
        needs.append(
            Need(
                key="DAEMON_GEMINI_MODEL",
                label="Gemini text model id",
                why="Which model answers."
                if gemini_chat
                else "Required alongside the Live id; not used by voice itself.",
                default=DEFAULT_GEMINI_MODEL,
                # Settings resolves a model for every routed task, and the voice
                # task routes to gemini, so this must be non-empty even when only
                # voice brought us here. Asking twice for one capability is noise,
                # so in that case it is filled in rather than asked.
                silent=not gemini_chat,
            )
        )
    needs.append(
        Need(
            key="TELEGRAM_BOT_TOKEN",
            label="Telegram bot token",
            why="How you reach Daemon from your phone. Send /newbot to @BotFather, "
            "then paste the token it gives you.",
            url=BOTFATHER_URL,
            secret=True,
            skippable=True,
        )
    )
    return [need for need in needs if not env.get(need.key)]


def _chat_providers(preset: str, hosted: str = DEFAULT_HOSTED_PROVIDER) -> set[str]:
    """Providers doing real generation, i.e. excluding embeddings.

    The distinction matters for exactly one question: whether setup should insist
    on a reachable Ollama. Under `quality` only `Task.EMBED` is local, and an
    embedding model that is not pulled yet degrades recall rather than blocking
    the first conversation - so it belongs to `daemon doctor`, not to a wizard
    standing between the user and their first message.
    """
    # Through preset_providers, not PRESETS directly: the tables hold a HOSTED
    # placeholder now, and reading them raw would ask the user for a key for a
    # provider called "hosted".
    return {
        provider
        for task, provider in preset_providers(preset, hosted).items()
        if task is not Task.EMBED
    }


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


# --- .env reading and writing ------------------------------------------------


def parse_env(text: str) -> dict[str, str]:
    """Enough of the dotenv format for what we write: `KEY=value`, `#` comments."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def merge_env(existing: str, updates: Mapping[str, str]) -> str:
    """Existing file with `updates` applied in place, everything else untouched.

    Comments, ordering and keys this wizard knows nothing about all survive: the
    file is the user's, and it is the only copy of their credentials.
    """
    pending = dict(updates)
    lines: list[str] = []
    for line in existing.splitlines():
        key = line.strip().partition("=")[0].strip()
        if "=" in line and not line.strip().startswith("#") and key in pending:
            lines.append(f"{key}={pending.pop(key)}")
        else:
            lines.append(line)
    if pending:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# written by `daemon setup`")
        lines.extend(f"{key}={value}" for key, value in pending.items())
    return "\n".join(lines) + "\n"


def write_private_file(path: Path, content: str) -> None:
    """Owner-only, and atomic.

    Owner-only because both files this writes are private - a list of API keys,
    and the description of a person's companion - and 0600 from the moment the
    file exists rather than a create-then-chmod window. Atomic because a wizard
    that half-wrote someone's existing `.env` would be worse than one that never
    ran: the replace either happens or it does not. The same property is what
    makes a Ctrl-C during the persona questions cost nothing.
    """
    temporary = path.with_name(f"{path.name}.daemon-setup")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        secure_file(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():  # only reachable if the write or replace failed
            temporary.unlink()
    secure_file(path)


def mask(value: str) -> str:
    """Last four characters, which is enough to recognise a key you pasted and
    not enough to use it. Nothing else ever prints a secret."""
    if not value:
        return "(empty)"
    return f"...{value[-4:]}" if len(value) > 4 else "(set)"


# --- the persona seed ---------------------------------------------------------


def seed_markdown(name: str, voice: str, address: str = "") -> str:
    """The `persona/seed.md` a first run starts from.

    Two sections, and the split is the point: the top is the user's answers, the
    bottom is `SEED_ANCHOR`, marked as ours and explained in the file itself,
    because someone who opens this later has to be able to tell what they chose
    from what the product fixed.
    """
    lines = [
        SEED_NOTE,
        "",
        "# Who I am",
        "",
        f"- My name is {name}.",
        f"- How I talk: {voice}",
    ]
    if address:
        # Omitted rather than defaulted: inventing how someone wants to be
        # addressed is worse than not mentioning it.
        lines.append(f"- How I address the user: {address}")
    return "\n".join([*lines, "", *SEED_ANCHOR]) + "\n"


def one_line(answer: str, limit: int = PERSONA_LINE_LIMIT) -> str:
    """An answer, flattened to something that cannot restructure the file.

    This text is pasted into a file that is prepended to every prompt as a system
    message, so a multi-line answer could open its own heading and write a second
    `# Constant` section - i.e. edit the anchor by typing into a question about
    tone. Collapsing whitespace removes that without changing a word of what was
    actually said. Brackets go the same way: `daemon/loop.py` fences quoted
    material with bracketed markers, and the seed is trusted text that nothing
    downstream strips, so a bracket typed here should not be able to pose as one.
    """
    flat = " ".join(answer.split()).replace("[", "(").replace("]", ")")
    return flat[:limit].rstrip() if len(flat) > limit else flat


# --- who just messaged the bot ------------------------------------------------

NAME_LIMIT = 48
"""Characters of display name shown. Enough to recognise yourself, not enough to
push the numeric id - the part that is actually being approved - off the line."""


@dataclass(frozen=True, slots=True)
class Sender:
    """Everything the wizard is willing to know about an inbound message."""

    id: str
    """The numeric Telegram user id, as text. The only thing approval is about."""
    label: str
    """Display name and @username, for recognition only. Attacker-chosen."""


def sender_of(update: Mapping[str, Any]) -> Sender | None:
    """Who sent this update - and nothing else about it.

    Deliberately no way to reach the message body from the return value. The body
    is private (this is someone's first words to their companion, not log
    material) and it is also the one part of an update a stranger writes: text
    from an unidentified sender does not belong on the terminal where the owner is
    answering a yes/no question.
    """
    message = update.get("message")
    if not isinstance(message, dict):
        return None  # edits, channel posts, callback queries
    user = message.get("from")
    if not isinstance(user, dict):
        return None
    sender_id = user.get("id")
    # `isinstance(True, int)` is True, and a malformed body could carry one.
    if not isinstance(sender_id, int) or isinstance(sender_id, bool):
        return None
    name = " ".join(
        part
        for part in (str(user.get(key) or "").strip() for key in ("first_name", "last_name"))
        if part
    )
    username = str(user.get("username") or "").strip()
    label = " ".join(part for part in (name, f"@{username}" if username else "") if part)
    return Sender(id=str(sender_id), label=printable(label) or "(no name given)")


def printable(text: str) -> str:
    """A display name, made safe to put on a terminal.

    Every character of it was chosen by whoever sent the message, and it is being
    printed one line above a `y/N` prompt: control characters can rewrite that
    line, and a name of a thousand spaces can scroll it away. `str.isprintable`
    keeps Korean, emoji and accents, and drops exactly the escapes and newlines
    that could redraw the question.
    """
    kept = "".join(character for character in text if character.isprintable()).strip()
    return f"{kept[:NAME_LIMIT]}..." if len(kept) > NAME_LIMIT else kept


def approve_sender(pairing: Pairing, sender_id: str) -> Approval | None:
    """Approve `sender_id` through the ordinary pairing path, code and all.

    `screen()` is what opens a pending request; `approve()` is what turns one into
    an allowlist entry and, for the first one only and atomically, into ownership.
    Writing that row here instead would mean this wizard held its own opinion
    about who the owner is, and "the first approval is the owner, once" would then
    live in two places - the sort of duplicate that stays correct until it does
    not. So a code is generated and spent immediately, and nobody reads it aloud.

    None when the sender is already approved: there is nothing left to pair.
    """
    from daemon.channels.pairing import PairingError

    if pairing.screen(sender_id).allowed:
        return None
    code = next(
        (request.code for request in pairing.pending() if request.sender_id == sender_id),
        "",
    )
    if not code:
        # screen() declined to issue one: BOOTSTRAP_MAX_PENDING strangers got
        # there first, all within the hour.
        raise PairingError(
            f"could not open a pairing request for id={sender_id} - too many are "
            "already waiting. `daemon pairing list` after `daemon run` still works"
        )
    return pairing.approve(code)


# --- prompting ---------------------------------------------------------------


class Prompt:
    """stdin/stdout, behind one object so the flow can be driven by a test."""

    def __init__(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        self._in = stdin if stdin is not None else sys.stdin
        self._out = stdout if stdout is not None else sys.stdout

    def say(self, text: str = "") -> None:
        print(text, file=self._out)

    def ask(self, question: str, *, default: str = "", secret: bool = False) -> str:
        label = f"{question} [{default}]: " if default else f"{question}: "
        if secret and self._in is sys.stdin and self._in.isatty():
            # Keeps a pasted key out of the scrollback. Only on a real terminal;
            # getpass would go looking for a tty a test does not have.
            try:
                answer = getpass.getpass(label, stream=self._out)
            except (EOFError, KeyboardInterrupt) as exc:
                raise Cancelled("no input") from exc
        else:
            self._out.write(label)
            self._out.flush()
            line = self._in.readline()
            if line == "":
                raise Cancelled("no input")
            answer = line.strip()
        return answer or default

    def ask_yes_no(self, question: str, *, default: bool = False) -> bool:
        hint = "Y/n" if default else "y/N"
        while True:
            answer = self.ask(f"{question} ({hint})").strip().lower()
            if not answer:
                return default
            if answer in {"y", "yes"}:
                return True
            if answer in {"n", "no"}:
                return False
            self.say("Please answer y or n.")

    def ask_choice(self, question: str, options: Sequence[str], *, default: str) -> str:
        while True:
            answer = self.ask(question, default=default).strip().lower()
            if answer in options:
                return answer
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return options[int(answer) - 1]
            self.say(f"Pick one of: {', '.join(options)}.")


# --- the wizard --------------------------------------------------------------

MAX_ATTEMPTS = 3
"""Tries per credential before giving up. A fourth prompt is not going to help,
and abandoning the run writes nothing."""


@dataclass
class Wizard:
    env_path: Path
    prompt: Prompt
    checks: Checks = field(default_factory=Checks)
    opener: Callable[[str], object] = webbrowser.open
    bot_handle: str = field(default="", init=False)
    """The bot's `@username`, kept from verifying the token so the pairing step
    can say which bot to message."""

    def run(self) -> int:
        existing_text = self.env_path.read_text(encoding="utf-8") if self.env_path.exists() else ""
        env = parse_env(existing_text)
        updates: dict[str, str] = {}
        say = self.prompt.say

        say("Daemon setup")
        say("Nothing is written until the end, and nothing is written anywhere but")
        say(f"{self.env_path}.")
        say()

        preset = self._choose_preset(env, updates)
        hosted = self._choose_hosted(preset, env, updates)
        voice = self._choose_voice(preset, env, updates)

        merged = {
            **env,
            **updates,
            "DAEMON_PRESET": preset,
            "DAEMON_HOSTED_PROVIDER": hosted,
            "DAEMON_VOICE_ENABLED": str(voice).lower(),
        }
        for need in needs_for(merged):
            value = self._fill(need, merged)
            if value:
                updates[need.key] = value

        self._check_ollama(preset, {**merged, **updates})

        if not updates:
            say("Nothing to change in .env - this install is already configured.")
            say("`daemon doctor` checks the parts a file cannot tell you about.")
            # Still goes through _finish: a complete `.env` says nothing about
            # whether a persona seed exists or whether a phone was ever paired,
            # and this is the only command that offers either. Returning here is
            # what made an interrupted first run unrecoverable.
            return self._finish(env)
        if not self._confirm(updates, env):
            say("Nothing was written.")
            return PROBLEM

        write_private_file(self.env_path, merge_env(existing_text, updates))
        say(f"Wrote {self.env_path} (mode 0600).")
        return self._finish({**env, **updates})

    # --- steps ---------------------------------------------------------------

    def _choose_preset(self, env: Mapping[str, str], updates: dict[str, str]) -> str:
        current = env.get("DAEMON_PRESET", "")
        if current in PRESETS:
            self.prompt.say(f"Preset: {current} (already in .env, keeping it).")
            self.prompt.say()
            return current

        self.prompt.say("How should Daemon think? You can change this later.")
        for index, name in enumerate(PRESET_ORDER, start=1):
            self.prompt.say(f"  {index}) {name}")
            for line in PRESET_HELP[name]:
                self.prompt.say(f"     {line}")
        self.prompt.say()
        preset = self.prompt.ask_choice(
            "Preset", PRESET_ORDER, default=DEFAULT_PRESET
        )
        updates["DAEMON_PRESET"] = preset
        self.prompt.say()
        return preset

    def _choose_hosted(self, preset: str, env: Mapping[str, str], updates: dict[str, str]) -> str:
        """Which commercial provider answers wherever the preset says "hosted".

        A second axis rather than nine presets: a preset says *where* work runs,
        this says *whose model* runs it. Asked right after the preset because it
        decides which key the next few questions are about - and not asked at all
        under `offline`, which resolves no hosted task and would be answering a
        question about a bill nobody is going to get.
        """
        if HOSTED not in PRESETS[preset].values():
            self.prompt.say(f"Nothing in the {preset} preset talks to a hosted model, so there")
            self.prompt.say("is no provider to choose and no API key to paste.")
            self.prompt.say()
            return DEFAULT_HOSTED_PROVIDER

        current = env.get("DAEMON_HOSTED_PROVIDER", "")
        if current in HOSTED_PROVIDERS:
            self.prompt.say(f"Hosted provider: {current} (already in .env, keeping it).")
            self.prompt.say()
            return current

        self.prompt.say("Whose model should the hosted work go to? The preset decided where")
        self.prompt.say("work runs; this decides who runs it. Changeable later, and it is one")
        self.prompt.say("key either way - Daemon is not a reseller, you bring your own.")
        for index, name in enumerate(HOSTED_PROVIDERS, start=1):
            self.prompt.say(f"  {index}) {name}")
            for line in HOSTED_HELP[name]:
                self.prompt.say(f"     {line}")
        self.prompt.say()
        chosen = self.prompt.ask_choice(
            "Provider", HOSTED_PROVIDERS, default=DEFAULT_HOSTED_PROVIDER
        )
        updates["DAEMON_HOSTED_PROVIDER"] = chosen
        self.prompt.say()
        return chosen

    def _choose_voice(self, preset: str, env: Mapping[str, str], updates: dict[str, str]) -> bool:
        if GEMINI not in providers_for(preset, voice_enabled=True):
            self.prompt.say(f"Voice is not part of the {preset} preset: native audio has to")
            self.prompt.say("run on a hosted model, and leaving it out is what makes 'nothing")
            self.prompt.say("leaves this machine' literally true. Text mode is the whole product.")
            self.prompt.say()
            return False

        if env.get("DAEMON_VOICE_ENABLED"):
            enabled = _truthy(env["DAEMON_VOICE_ENABLED"])
            self.prompt.say(f"Voice: {'on' if enabled else 'off'} (already in .env, keeping it).")
            self.prompt.say()
            return enabled

        self.prompt.say("Turn voice on? Audio then goes to Google's native-audio model,")
        self.prompt.say("with your own key. Off means no audio ever leaves this machine,")
        self.prompt.say("and text mode loses nothing by it.")
        enabled = self.prompt.ask_yes_no("Enable voice", default=False)
        updates["DAEMON_VOICE_ENABLED"] = str(enabled).lower()
        self.prompt.say()
        return enabled

    def _fill(self, need: Need, env: Mapping[str, str]) -> str:
        if need.silent:
            return need.default
        self.prompt.say(f"{need.label} ({need.key})")
        self.prompt.say(f"  {need.why}")
        if need.url:
            self.prompt.say(f"  Opening {need.url}")
            try:
                self.opener(need.url)
            except Exception:
                # A headless box, or no browser at all. Not a setup failure: the
                # URL is on screen either way.
                self.prompt.say("  (could not open a browser - use the link above)")

        for attempt in range(1, MAX_ATTEMPTS + 1):
            value = self.prompt.ask(f"  {need.key}", default=need.default, secret=need.secret)
            if not value:
                if need.skippable:
                    self.prompt.say("  Skipped. `daemon setup` again when you have it.")
                    self.prompt.say()
                    return ""
                self.prompt.say("  This one is required.")
                continue

            verdict = self._verify(need, value, env)
            if verdict.ok:
                if verdict.detail:
                    shown = mask(value) if need.secret else value
                    self.prompt.say(f"  ok: {verdict.detail} ({shown})")
                if verdict.hint:
                    self.prompt.say(f"  note: {verdict.hint}")
                self.prompt.say()
                return value

            self.prompt.say(f"  {verdict.detail}")
            if verdict.hint:
                self.prompt.say(f"  {verdict.hint}")
            if attempt < MAX_ATTEMPTS:
                self.prompt.say("  Try again.")

        raise Cancelled(f"{need.key} could not be verified after {MAX_ATTEMPTS} tries")

    def _verify(self, need: Need, value: str, env: Mapping[str, str]) -> Verdict:
        """Verify now, so a bad key is a sentence here instead of a broken
        conversation later.

        Model ids are taken at their word: there is no free call that proves a
        Gemini Live id, and the Anthropic model id is checked as part of checking
        the key. An empty detail means "nothing worth printing".
        """
        if need.key == "ANTHROPIC_API_KEY":
            # Their configured model if they have one, so the check is about the
            # model this install will actually ask for.
            model = env.get("DAEMON_ANTHROPIC_MODEL") or _config_default("anthropic_model")
            return self.checks.anthropic(value, model)
        if need.key == "OPENAI_API_KEY":
            return self.checks.openai(value, env.get("DAEMON_OPENAI_MODEL") or DEFAULT_OPENAI_MODEL)
        if need.key == "GEMINI_API_KEY":
            return self.checks.gemini(value)
        if need.key == "TELEGRAM_BOT_TOKEN":
            verdict = self.checks.telegram(value)
            # getMe already told us the handle; the pairing step would otherwise
            # have to ask Telegram a second time for something we were just told.
            self.bot_handle = verdict.subject or self.bot_handle
            return verdict
        return Verdict(True, "")

    def _check_ollama(self, preset: str, env: Mapping[str, str]) -> None:
        if OLLAMA not in _chat_providers(preset):
            embed_model = env.get("DAEMON_EMBED_MODEL") or DEFAULT_EMBED_MODEL
            self.prompt.say("Nothing in this preset needs a local chat model. Recall")
            self.prompt.say("embeddings are local in every preset though, so `ollama pull")
            self.prompt.say(f"{embed_model}` is still worth doing - `daemon doctor` checks it.")
            self.prompt.say()
            return

        base_url = env.get("DAEMON_OLLAMA_BASE_URL") or _config_default("ollama_base_url")
        state = self.checks.ollama(base_url)
        if not state.reachable:
            self.prompt.say(f"Ollama: {state.detail}")
            self.prompt.say("  Install it from https://ollama.com, then run `ollama serve`.")
            self.prompt.say("  Setup continues; `daemon doctor` re-checks this.")
            self.prompt.say()
            return

        self.prompt.say(f"Ollama: {state.detail}")
        wanted = [
            env.get("DAEMON_OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
            env.get("DAEMON_EMBED_MODEL") or DEFAULT_EMBED_MODEL,
        ]
        missing = [model for model in wanted if not _installed(state.models, model)]
        for model in wanted:
            if model not in missing:
                self.prompt.say(f"  {model}: installed")
        if missing:
            # Deliberately not run for them: these are gigabytes, and a wizard
            # should not start a download the user did not ask for.
            self.prompt.say("  Missing models. Run these yourself (they are large):")
            for model in missing:
                self.prompt.say(f"    ollama pull {model}")
        self.prompt.say()

    def _confirm(self, updates: Mapping[str, str], env: Mapping[str, str]) -> bool:
        self.prompt.say(f"These keys change in {self.env_path}:")
        secrets = {need.key for need in _all_needs() if need.secret}
        for key, value in updates.items():
            shown = mask(value) if key in secrets else value
            before = env.get(key)
            was = "" if before is None else f" (was {mask(before) if key in secrets else before})"
            self.prompt.say(f"  {key}={shown}{was}")
        self.prompt.say("Everything else in the file is left alone.")
        return self.prompt.ask_yes_no("Write it", default=True)

    def _finish(self, env: Mapping[str, str]) -> int:
        say = self.prompt.say
        say()
        try:
            settings = Settings(_env_file=self.env_path)
        except (ConfigError, ValidationError) as exc:
            # The same validation `daemon run` does, run here, on the file we just
            # wrote. This is the last chance to fail before "it worked in setup"
            # turns into a service that will not start - which is the class of bug
            # this whole module exists to close, so it is not allowed to be a
            # traceback either.
            say("The file is written, but the configuration is not usable yet:")
            for line in str(exc).splitlines():
                say(f"  {line}")
            say("Fix it with `daemon setup` again, by editing .env, or ask `daemon doctor`.")
            return PROBLEM

        # Everything below reads settings rather than `env`: the data dir and the
        # DM policy both have defaults that are not in the file, and both decide
        # where these two steps write and whether they run at all.
        unpaired = bool(
            settings.telegram_bot_token
            and not settings.telegram_allowed_user_ids
            and settings.telegram_dm_policy == PAIRING_POLICY
        )
        try:
            self._seed_persona(settings)
            if unpaired and self._pair_here(settings):
                unpaired = False
        except (Cancelled, KeyboardInterrupt):
            # Both files are written atomically or not at all, so there is nothing
            # half-done to repair - and run()'s handler must not see this, because
            # it would print "nothing was written" about a `.env` that is on disk.
            say()
            say("Stopped there. Everything already written above stays written.")
            say()
        if unpaired:
            for line in PAIRING_NOTE:
                say(line)
            say()

        say("Next:")
        say("  daemon doctor     - checks Ollama, the data dir and the schema")
        say("  daemon run        - runs it here, in this terminal")
        say("  daemon install    - keeps it running after you close the terminal or reboot")
        return OK

    # --- the persona seed ----------------------------------------------------

    def _seed_persona(self, settings: Settings) -> None:
        """Write `persona/seed.md`, unless there is already one.

        Nothing else in Daemon creates this file - `loop.py` reads it if it is
        there and otherwise sends no persona at all, which is how an install ends
        up sounding like a stock assistant rather than like anyone (docs/PLAN.md
        5). So it belongs to first-run, next to the questions the user is already
        answering, and not to a paragraph of documentation asking them to write
        markdown before they have said hello.
        """
        say = self.prompt.say
        path = settings.data_dir / "persona" / "seed.md"
        if path.exists():
            # Human-owned (docs/PLAN.md 5.1), and re-running setup is not consent
            # to overwrite a personality someone has been editing for a month.
            say(f"{path} already exists, so it is left exactly as it is.")
            say("That file is yours - nothing here and nothing in Daemon rewrites it.")
            say()
            return

        say("Who should this be?")
        say("Three questions, all skippable with Enter, and none of them final:")
        say(f"they write {path}, which you own and can edit at any time.")
        say()
        name = one_line(self.prompt.ask("  Name", default=DEFAULT_PERSONA_NAME))
        say()
        say("  How should it talk? A number, or write your own line.")
        for index, preset in enumerate(VOICE_PRESETS, start=1):
            say(f"    {index}) {preset}")
        voice = self._voice(self.prompt.ask("  Voice", default="1"))
        say()
        say("  How should it address you? Enter skips this.")
        say("  In Korean it is half the voice: 반말 or 존댓말, and what to call you.")
        address = one_line(self.prompt.ask("  Address", default=""))

        try:
            secure_dir(path.parent)
            write_private_file(path, seed_markdown(name or DEFAULT_PERSONA_NAME, voice, address))
        except OSError as exc:
            # `DAEMON_DATA_DIR` can point somewhere that cannot be created - that is
            # what `daemon doctor` checks for, and this command runs before doctor
            # has ever been useful. A sentence, not a traceback.
            say()
            say(f"Could not write {path}: {exc}")
            say("Daemon runs without a seed; `daemon setup` writes it once the data")
            say("directory is reachable.")
            say()
            return
        say()
        say(f"Wrote {path} (mode 0600).")
        say("It also carries two fixed lines - that this is someone with a view of")
        say("their own, who will disagree with you. Those are in the file Daemon")
        say("cannot write to, which is the only reason they survive (docs/PLAN.md 5.4).")
        say()

    def _voice(self, answer: str) -> str:
        """A preset by number, or whatever they typed, or the default."""
        picked = answer.strip()
        if picked.isdigit() and 1 <= int(picked) <= len(VOICE_PRESETS):
            return VOICE_PRESETS[int(picked) - 1]
        return one_line(picked) or VOICE_PRESETS[0]

    # --- pairing, in this process --------------------------------------------

    def _pair_here(self, settings: Settings) -> bool:
        """Offer to pair now, and do it. True if someone was approved.

        Raises `Cancelled` (or lets `KeyboardInterrupt` through) if the user leaves
        mid-wait; `_finish` treats that as "stop asking me things", because by now
        the only irreversible thing this command does has already happened.
        """
        say = self.prompt.say
        say("Telegram is configured, but Daemon does not know who you are yet -")
        say("it answers nobody until someone is paired.")
        say("The documented way to fix that needs two terminals: `daemon run` in")
        say("one, then `daemon pairing approve` in the other with the code the bot")
        say("replies. The code exists to carry your id between two processes. This")
        say("is one process holding both the token and the database, so it can")
        say("watch for your message and simply ask whether it was you.")
        if not self.prompt.ask_yes_no("Pair your Telegram account now", default=True):
            return False
        try:
            return self._pair(settings)
        except OSError as exc:
            # Same reason as the seed above: approval is stored in the data dir,
            # and the data dir is configuration that can be wrong.
            say(f"  Could not open the pairing database: {exc}")
            return False

    def _pair(self, settings: Settings) -> bool:
        # Imported here, not at module scope: `daemon setup` has to run on an
        # install whose configuration does not load yet, and app.py pulls in the
        # whole server to be able to name one filename.
        from daemon.app import DB_FILENAME
        from daemon.channels.pairing import Pairing, PairingError
        from daemon.channels.telegram import TelegramChannel
        from daemon.memory.store import Store

        say = self.prompt.say
        token = settings.telegram_bot_token
        handle = self.bot_handle
        if not handle:
            # The token was already in the file, so getMe never ran this time.
            # Worth one call: telling someone to go and message a bot whose token
            # was revoked would spend the whole wait on a bot that cannot hear.
            verdict = self.checks.telegram(token)
            if not verdict.ok:
                say(f"  {verdict.detail}")
                return False
            handle = verdict.subject or "your bot"

        say(f"  Send any message to {handle} from your phone - anything at all.")
        say(f"  Waiting up to {PAIRING_WAIT_SECONDS / 60:.0f} minute(s). Ctrl-C stops waiting.")
        say()

        store = Store.open(settings.data_dir / DB_FILENAME)
        pairing = Pairing(store, TelegramChannel.name)
        offset: int | None = None
        refused: set[str] = set()
        try:
            deadline = time.monotonic() + PAIRING_WAIT_SECONDS
            while True:
                batch = self.checks.updates(token, offset, PAIRING_POLL_SECONDS)
                if not batch.ok:
                    # One failure ends it rather than retrying for three minutes:
                    # the token worked seconds ago, so this is the network, and the
                    # user should hear that instead of watching a spinner.
                    say(f"  {batch.detail}")
                    return False
                for update in batch.updates:
                    update_id = update.get("update_id")
                    if isinstance(update_id, int):
                        offset = update_id + 1
                    sender = sender_of(update)
                    if sender is None or sender.id in refused:
                        # Asked about them already; further messages from the same
                        # id are not a second question.
                        continue
                    if not self._is_you(sender):
                        refused.add(sender.id)
                        say("  Still waiting, then - someone else got there first.")
                        say()
                        continue
                    try:
                        approval = approve_sender(pairing, sender.id)
                    except PairingError as exc:
                        say(f"  {exc}")
                        return False
                    if approval is None:
                        say(f"  id={sender.id} is already paired - nothing to do.")
                        return True
                    say(f"  Paired. id={approval.sender_id} may talk to Daemon; nobody else can.")
                    if not approval.is_owner:
                        say("  Added as a guest: this channel already had an owner, and")
                        say("  ownership is granted once and never transferred.")
                    return True
                if time.monotonic() >= deadline:
                    say("  Nothing arrived, so this is not the moment.")
                    return False
        finally:
            if offset is not None:
                # Not optional. Passing `offset` above is what confirms updates
                # server-side, so these are spent - but the last batch is not
                # confirmed until the next call, and a daemon starting with no
                # cursor asks for everything Telegram still holds. Without this
                # line the daemon's first act is to answer the message that was
                # only ever meant to identify you.
                store.save_cursor(TelegramChannel.name, offset)
            store.close()

    def _is_you(self, sender: Sender) -> bool:
        say = self.prompt.say
        say(f"  A message just arrived from id={sender.id}  (name: {sender.label})")
        say("  What gets approved is the id. The name is whatever that account")
        say("  typed into its own profile, so treat it as a hint and not as proof.")
        say("  What they wrote is not shown: it is not mine to print, and an")
        say("  unidentified stranger's words have no business on this screen.")
        return self.prompt.ask_yes_no(f"  Is id={sender.id} you", default=True)


def _all_needs() -> list[Need]:
    """Every key any configuration can require, deduplicated: used to decide
    whether a value may be printed, and to list what `--check` found already set.

    Every hosted provider, not just the default one. Iterating presets alone left
    `OPENAI_API_KEY` out of the set of keys known to be secret, which is the sort
    of omission that ends with a key printed in the change list.
    """
    seen: dict[str, Need] = {}
    for preset in PRESET_ORDER:
        for hosted in HOSTED_PROVIDERS:
            for need in needs_for(
                {
                    "DAEMON_PRESET": preset,
                    "DAEMON_HOSTED_PROVIDER": hosted,
                    "DAEMON_VOICE_ENABLED": "true",
                }
            ):
                seen.setdefault(need.key, need)
    return list(seen.values())


def _config_default(field_name: str) -> str:
    """A Settings default, read from Settings rather than copied. A wizard that
    verified a different model id than the one startup will use would be a
    convincing way to reintroduce the bug this module removes."""
    return str(Settings.model_fields[field_name].default)


def _settings_default(env_key: str) -> str:
    """Same, looked up by the environment variable name. Empty for a key Settings
    has no default for - which is exactly `Need.blocking`."""
    for info in Settings.model_fields.values():
        if info.alias == env_key:
            return str(info.default or "")
    return ""


def _installed(models: Sequence[str], wanted: str) -> bool:
    """Ollama reports `name:tag`; a bare name means the `latest` tag."""
    full = wanted if ":" in wanted else f"{wanted}:latest"
    return any(model in (wanted, full) for model in models)


# --- entry points ------------------------------------------------------------


def run(
    *,
    check_only: bool = False,
    env_path: Path | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    checks: Checks | None = None,
    opener: Callable[[str], object] = webbrowser.open,
) -> int:
    path = Path(env_path) if env_path is not None else Path.cwd() / ENV_FILE
    prompt = Prompt(stdin, stdout)
    if check_only:
        return report(path, prompt)
    try:
        return Wizard(
            env_path=path,
            prompt=prompt,
            checks=checks if checks is not None else Checks(),
            opener=opener,
        ).run()
    except Cancelled as exc:
        prompt.say()
        prompt.say(f"Stopped ({exc}). {path} was not touched.")
        return PROBLEM
    except KeyboardInterrupt:
        prompt.say()
        prompt.say(f"Stopped. {path} was not touched.")
        return PROBLEM


def report(path: Path, prompt: Prompt) -> int:
    """`--check`: asks nothing, opens nothing, calls nobody.

    Local and deterministic so it is usable from CI and from documentation. It
    answers "is this file complete?", which is why liveness - is Ollama up, is
    the schema current - stays with `daemon doctor`.
    """
    env = parse_env(path.read_text(encoding="utf-8")) if path.exists() else {}
    prompt.say(f"{path}: {'found' if path.exists() else 'missing'}")
    preset = env.get("DAEMON_PRESET", "") or DEFAULT_PRESET
    if preset not in PRESETS:
        prompt.say(f"[FAIL] DAEMON_PRESET={preset!r} is not a preset")
        return PROBLEM
    voice = _truthy(env.get("DAEMON_VOICE_ENABLED", ""))
    default_note = "" if env.get("DAEMON_PRESET") else " (default, not set in the file)"
    hosted = env.get("DAEMON_HOSTED_PROVIDER", "") or DEFAULT_HOSTED_PROVIDER
    prompt.say(
        f"preset: {preset}{default_note}, hosted provider: {hosted}, "
        f"voice {'on' if voice else 'off'}"
    )

    missing = needs_for(env)
    for need in _all_needs():
        value = env.get(need.key, "")
        if value:
            prompt.say(f"[ok]      {need.key} = {mask(value) if need.secret else value}")
    for need in missing:
        tag = "[missing]" if need.blocking else "[offered]"
        prompt.say(f"{tag} {need.key} - {need.label}: {need.why}")

    blocking = [need for need in missing if need.blocking]
    if blocking:
        prompt.say(f"{len(blocking)} item(s) missing. Run `daemon setup` to fill them in.")
        return PROBLEM
    if missing:
        # Everything left has a working built-in default, so this install starts.
        prompt.say("Nothing missing; `daemon setup` would offer the [offered] items above.")
    else:
        prompt.say("Nothing missing.")
    prompt.say("`daemon doctor` checks Ollama, the data dir and the schema.")
    return OK
