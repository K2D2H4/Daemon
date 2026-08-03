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
  * It reads and writes `./.env` only, never the shell environment. The OS
    service unit carries no secrets and sees no shell (daemon/service.py), so a
    key that exists only as an exported variable would work in the terminal and
    break under launchd - the worst kind of "it worked yesterday".
  * It never asks for a Telegram user id. Pairing is a channel concern; the
    wizard takes the token and says how to pair.

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
import os
import sys
import webbrowser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import httpx
from pydantic import ValidationError

from daemon.config import (
    ANTHROPIC,
    ENV_FILE,
    GEMINI,
    OLLAMA,
    PRESETS,
    ConfigError,
    Settings,
    providers_for,
)
from daemon.fs import FILE_MODE, secure_file
from daemon.tasks import Task

OK = 0
PROBLEM = 1

HTTP_TIMEOUT = 15.0
"""Short. A wizard waiting on a hung endpoint is worse than a wizard that says
the endpoint is unreachable and asks again."""

ANTHROPIC_KEYS_URL = "https://console.anthropic.com/settings/keys"
AI_STUDIO_URL = "https://aistudio.google.com/apikey"
BOTFATHER_URL = "https://t.me/BotFather"

DEFAULT_OLLAMA_MODEL = "gemma3:4b"
"""Not a reasoning model, and not `Settings.ollama_model`'s default. docs/PLAN.md
3.2.1 measured 1.7 s against 11.8 s for the same three Korean questions, the
difference being a chain of thought that is discarded unread. A fresh install
should land on the usable side of that."""
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_LIVE_MODEL = "gemini-live-2.5-flash-preview"

PRESET_ORDER = ("offline", "balanced", "quality")
DEFAULT_PRESET = "balanced"

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
        "Conversation and the daily reflection go to Claude, because reflection",
        "quality propagates into the whole memory graph. The 'should I speak?'",
        "check stays local - it runs every five minutes whether or not it ever",
        "speaks, so hosted cost would accumulate for nothing.",
        "Voice can be turned on. Needs: an Anthropic key, plus Ollama for the",
        "local check and for recall embeddings.",
    ),
    "quality": (
        "Everything hosted, including the five-minute proactive check. Best",
        "quality, highest running cost. Embeddings stay local in every preset,",
        "so recall still wants Ollama - `daemon doctor` checks that.",
        "Voice can be turned on. Needs: an Anthropic key.",
    ),
}

GEMINI_STANDARD_KEY_HINT = (
    "This key looks like a Google 'Standard' API key. The Gemini API already "
    "refuses unrestricted Standard keys and will refuse the remaining ones in "
    "September 2026. Create a new key in AI Studio - keys made there now are "
    "auth keys, which is what this needs."
)

PAIRING_NOTE = (
    "No Telegram allowlist yet, and that is on purpose: nobody is asked for a",
    "numeric user id here. Start the daemon, message your bot, and it answers with",
    "a pairing code. Until you pair, nobody can reach it.",
)


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
    return Verdict(True, f"connected to @{username}" if username else "token works")


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
    gemini: Callable[[str], Verdict] = check_gemini
    telegram: Callable[[str], Verdict] = check_telegram
    ollama: Callable[[str], OllamaState] = check_ollama


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
    providers = providers_for(preset, voice_enabled=_truthy(env.get("DAEMON_VOICE_ENABLED", "")))
    needs: list[Need] = []

    if OLLAMA in _chat_providers(preset):
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
    if GEMINI in providers:
        needs.append(
            Need(
                key="GEMINI_API_KEY",
                label="Gemini API key",
                why="Voice is on, so audio goes to Google's native-audio model - with your key.",
                url=AI_STUDIO_URL,
                secret=True,
            )
        )
        needs.append(
            Need(
                key="DAEMON_GEMINI_LIVE_MODEL",
                label="Gemini Live model id",
                why="The realtime audio endpoint takes its own model id, not the text one.",
                default=DEFAULT_GEMINI_LIVE_MODEL,
            )
        )
        # Settings resolves every routed task's model, and the voice task routes
        # to gemini, so this must be non-empty even though voice reads the Live id
        # above. Asking twice for one capability is noise, so it is filled in.
        needs.append(
            Need(
                key="DAEMON_GEMINI_MODEL",
                label="Gemini text model id",
                why="Required alongside the Live id; not used by voice itself.",
                default=DEFAULT_GEMINI_MODEL,
                silent=True,
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


def _chat_providers(preset: str) -> set[str]:
    """Providers doing real generation, i.e. excluding embeddings.

    The distinction matters for exactly one question: whether setup should insist
    on a reachable Ollama. Under `quality` only `Task.EMBED` is local, and an
    embedding model that is not pulled yet degrades recall rather than blocking
    the first conversation - so it belongs to `daemon doctor`, not to a wizard
    standing between the user and their first message.
    """
    return {provider for task, provider in PRESETS[preset].items() if task is not Task.EMBED}


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


def write_env(path: Path, content: str) -> None:
    """Owner-only, and atomic.

    Owner-only because this file is a list of API keys, and 0600 from the moment
    it exists rather than a create-then-chmod window. Atomic because a wizard
    that half-wrote someone's existing `.env` would be worse than one that never
    ran: the replace either happens or it does not.
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
        voice = self._choose_voice(preset, env, updates)

        merged = {**env, **updates, "DAEMON_PRESET": preset, "DAEMON_VOICE_ENABLED": str(voice)}
        for need in needs_for(merged):
            value = self._fill(need)
            if value:
                updates[need.key] = value

        self._check_ollama(preset, {**merged, **updates})

        if not updates:
            say("Nothing to change - this install is already configured.")
            say("`daemon doctor` checks the parts a file cannot tell you about.")
            return OK
        if not self._confirm(updates, env):
            say("Nothing was written.")
            return PROBLEM

        write_env(self.env_path, merge_env(existing_text, updates))
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

    def _fill(self, need: Need) -> str:
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

            verdict = self._verify(need, value)
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

    def _verify(self, need: Need, value: str) -> Verdict:
        """Verify now, so a bad key is a sentence here instead of a broken
        conversation later.

        Model ids are taken at their word: there is no free call that proves a
        Gemini Live id, and the Anthropic model id is checked as part of checking
        the key. An empty detail means "nothing worth printing".
        """
        if need.key == "ANTHROPIC_API_KEY":
            return self.checks.anthropic(value, _config_default("anthropic_model"))
        if need.key == "GEMINI_API_KEY":
            return self.checks.gemini(value)
        if need.key == "TELEGRAM_BOT_TOKEN":
            return self.checks.telegram(value)
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
        if env.get("TELEGRAM_BOT_TOKEN") and not env.get("TELEGRAM_ALLOWED_USER_IDS"):
            for line in PAIRING_NOTE:
                say(line)
            say()

        try:
            Settings(_env_file=self.env_path)
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

        say("Next:")
        say("  daemon doctor     - checks Ollama, the data dir and the schema")
        say("  daemon run        - runs it here, in this terminal")
        say("  daemon install    - keeps it running after you close the terminal or reboot")
        return OK


def _all_needs() -> list[Need]:
    """Every key any preset can require, deduplicated: used to decide whether a
    value may be printed, and to list what `--check` found already set."""
    seen: dict[str, Need] = {}
    for preset in PRESET_ORDER:
        for need in needs_for({"DAEMON_PRESET": preset, "DAEMON_VOICE_ENABLED": "true"}):
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
    prompt.say(f"preset: {preset}{default_note}, voice {'on' if voice else 'off'}")

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
