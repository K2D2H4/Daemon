"""Read the editable settings, and write a patch to `.env` - validated first.

docs/design decision 3, and the M5 test gate: a patch is committed to `.env`
**only** if a candidate `Settings` built from current-plus-patch constructs
cleanly. `Settings` fails loudly at construction (daemon/config.py), so building
one is the whole validation: an unknown provider, a non-numeric limit, a voice
switch with no realtime model id all raise there, before a single byte is
written. On failure the caller returns 400 and the file is untouched.

Secrets are indirect in both directions. GET reports `"set"`/`null`, never the
value - the loopback admin has no auth (decision 1), so a value that never leaves
the process cannot leak from it. PATCH accepts a new secret and writes it, but an
empty one is left alone rather than clearing a working key by accident.

The writer is the existing atomic 0600 one (`fs.write_private_replace`, via
`setup.merge_env` so comments and unknown keys survive); this module invents
neither the format nor the durability.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daemon.admin.google_accounts import authenticated_accounts
from daemon.config import (
    GEMINI_LIVE_TRANSPORTS,
    GEMINI_LIVE_VOICES,
    HOSTED_PROVIDERS,
    MODEL_SUGGESTIONS,
    OLLAMA,
    OPENAI_REALTIME_VOICES,
    SENSITIVITIES,
    VERTEX_LIVE_LOCATIONS,
    VERTEX_LIVE_MODELS,
    VOICE_PROVIDERS,
    Settings,
)
from daemon.fs import write_private_replace
from daemon.setup import merge_env, parse_env
from daemon.tasks import Task
from daemon.tools.policy import MODES as TOOL_MODES

# Editable field -> the `.env` key it is stored under. The env key is also the
# keyword `Settings(**candidate)` accepts (config aliases are these names, with
# populate_by_name on), so one map serves reading, validating and writing.
STR_FIELDS: dict[str, str] = {
    "provider": "DAEMON_PROVIDER",
    "tools_mode": "DAEMON_TOOLS_MODE",
    "gemini_live_voice": "DAEMON_GEMINI_LIVE_VOICE",
    # Model ids, one per provider the `provider` list offers - all of them, not
    # just the one in use: `DAEMON_OPENAI_MODEL` has no default, so a page that
    # let you pick `openai` without letting you name its model could only ever
    # answer the choice with a 400 the page itself could not fix.
    "ollama_model": "DAEMON_OLLAMA_MODEL",
    "anthropic_model": "DAEMON_ANTHROPIC_MODEL",
    "openai_model": "DAEMON_OPENAI_MODEL",
    "gemini_model": "DAEMON_GEMINI_MODEL",
    # The realtime endpoint does not take the text endpoint's id (config.py), which
    # is why voice has its own.
    "gemini_live_model": "DAEMON_GEMINI_LIVE_MODEL",
    # Which endpoint serves that model. Web-configurable because the two endpoints
    # carry *different model catalogues* - the fast native-audio id is Vertex-only,
    # the newer generation is API-key-only - so choosing a live model and choosing a
    # transport are one decision, and leaving half of it in a text editor would mean
    # picking a model this page cannot reach (docs/design/vertex-live-transport.md).
    "gemini_live_transport": "DAEMON_GEMINI_LIVE_TRANSPORT",
    "vertex_project": "DAEMON_VERTEX_PROJECT",
    "vertex_location": "DAEMON_VERTEX_LOCATION",
    # A path, not a secret: the file it names is the credential, and this only says
    # where to look. Empty means Application Default Credentials.
    "vertex_credentials_path": "GOOGLE_APPLICATION_CREDENTIALS",
    "openai_compatible_model": "DAEMON_OPENAI_COMPATIBLE_MODEL",
    # The endpoint belongs beside the model for the same reason the model ids do:
    # a page that lets you choose `openai_compatible` without letting you name its
    # address could only ever answer the choice with an error it cannot fix.
    "openai_compatible_base_url": "DAEMON_OPENAI_COMPATIBLE_BASE_URL",
    # Voice provider is its own axis (config.py): gemini or openai. Its own realtime
    # model + voice, exactly like the gemini pair, so voice is fully web-configurable.
    "voice_provider": "DAEMON_VOICE_PROVIDER",
    "openai_realtime_model": "DAEMON_OPENAI_REALTIME_MODEL",
    "openai_realtime_voice": "DAEMON_OPENAI_REALTIME_VOICE",
    # Both halves of the pair: `Settings` validates each against SENSITIVITIES, and
    # offering only the start one would leave the end one hand-edit-only for no
    # reason a reader could infer.
    "voice_start_sensitivity": "DAEMON_VOICE_START_SENSITIVITY",
    "voice_end_sensitivity": "DAEMON_VOICE_END_SENSITIVITY",
    # Which account's calendar the `calendar` proactive kind reads (ADR 0021).
    # A plain string field rather than a choice: `options.calendar_accounts`
    # below suggests the ones already authenticated, but an owner may name one
    # that is not yet, so the page offers a datalist and not a select. Empty is
    # a real value - it is how the kind is switched off.
    "calendar_email": "DAEMON_CALENDAR_EMAIL",
}
BOOL_FIELDS: dict[str, str] = {
    "voice_enabled": "DAEMON_VOICE_ENABLED",
    "mcp_enabled": "DAEMON_MCP_ENABLED",
    "browser_enabled": "DAEMON_BROWSER_ENABLED",
    # The switches the Overview already reports on. Showing a proactivity budget,
    # a wake gate and a screen capability while leaving their on/off in a text
    # editor is the same gap this milestone opened to close.
    "tools_enabled": "DAEMON_TOOLS_ENABLED",
    "screen_enabled": "DAEMON_SCREEN_ENABLED",
    "wake_enabled": "DAEMON_WAKE_ENABLED",
    "voice_barge_in": "DAEMON_VOICE_BARGE_IN",
    "proactive_enabled": "DAEMON_PROACTIVE_ENABLED",
    # The `proactive_judge_local` axis (docs/adr/0014): whether PROACTIVE_JUDGE
    # runs on the local model instead of `provider`. A bool like the switches
    # above, not a STR_FIELD - it flows through the same read/patch handling.
    "proactive_judge_local": "DAEMON_PROACTIVE_JUDGE_LOCAL",
}
LIST_FIELDS: dict[str, str] = {"wake_aliases": "DAEMON_WAKE_ALIASES"}
"""Comma-separated in `.env` and tuple-valued on `Settings`. Reported and accepted
as the comma-separated form, which is what `daemon wake calibrate` writes and what a
person types - the JSON array a naive round-trip would produce is neither."""
INT_FIELDS: dict[str, str] = {"recall_limit": "DAEMON_RECALL_LIMIT"}
FLOAT_FIELDS: dict[str, str] = {"recall_half_life_days": "DAEMON_RECALL_HALF_LIFE_DAYS"}
SECRET_FIELDS: dict[str, str] = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "gemini_api_key": "GEMINI_API_KEY",
    "telegram_bot_token": "TELEGRAM_BOT_TOKEN",
    "openai_compatible_api_key": "OPENAI_COMPATIBLE_API_KEY",
}
ROUTE_OVERRIDES = "route_overrides"
ROUTE_OVERRIDES_ENV = "DAEMON_ROUTE_OVERRIDES"

EDITABLE = {
    *STR_FIELDS,
    *BOOL_FIELDS,
    *LIST_FIELDS,
    *INT_FIELDS,
    *FLOAT_FIELDS,
    *SECRET_FIELDS,
    ROUTE_OVERRIDES,
}


class PatchError(ValueError):
    """A patch that cannot be applied. Its message is safe to return to the
    client - it names the field or the validation failure, never a secret."""


def pending_values(settings: Settings, env_path: Path) -> dict[str, Any]:
    """Editable values `.env` holds that differ from the ones this process is running.

    The running daemon does not hot-reload, so a saved patch is invisible to
    `app.state.settings` until a restart. Without this the page reloads showing the
    pre-save values with nothing to say why, and a save that worked perfectly reads
    as a save that was lost - measured by doing exactly that.

    Reported *beside* the running values, never instead of them: the admin's one job
    is not to lie about what the daemon is currently doing.
    """
    if not env_path.exists():
        return {}
    try:
        saved = Settings(_env_file=None, **parse_env(env_path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 - a file we cannot build from has nothing to report
        return {}
    pending: dict[str, Any] = {}
    for name in (*STR_FIELDS, *BOOL_FIELDS, *INT_FIELDS, *FLOAT_FIELDS):
        if getattr(saved, name) != getattr(settings, name):
            pending[name] = getattr(saved, name)
    for name in LIST_FIELDS:
        if getattr(saved, name) != getattr(settings, name):
            pending[name] = ",".join(getattr(saved, name))
    return pending


def current_settings_payload(settings: Settings, env_path: Path | None = None) -> dict[str, Any]:
    """The GET /settings body: editable values (secrets masked), read-only
    display, the option lists the front-end offers choices from, and anything
    `.env` holds that this process has not restarted into."""
    editable: dict[str, Any] = {
        name: getattr(settings, name) for name in (*STR_FIELDS, *BOOL_FIELDS, *INT_FIELDS)
    }
    editable["recall_half_life_days"] = settings.recall_half_life_days
    for name in LIST_FIELDS:
        editable[name] = ",".join(getattr(settings, name))
    editable[ROUTE_OVERRIDES] = {
        task.value: provider for task, provider in settings.route_overrides.items()
    }
    for name in SECRET_FIELDS:
        # "set"/null, never the value. The one place this module could leak.
        editable[name] = "set" if getattr(settings, name) else None

    # D9: a read-only note when work can land on a provider other than
    # `provider` - only `route_overrides` (folded into `settings.routing`) and
    # `fallback_provider` (a separate attribute - `routing` does not fold it in,
    # it is only consulted via `fallback_route()`/`routing_table()`) can do
    # that, both hand-edit-only. CHAT_VOICE is excluded: its provider is
    # `voice_provider`, expected to differ from the chat provider, not an
    # out-of-band route.
    off = {
        p for t, p in settings.routing.items()
        if p not in ("", settings.provider) and p != OLLAMA and t is not Task.CHAT_VOICE
    }
    fallback = settings.fallback_provider
    if fallback and fallback not in ("", settings.provider, OLLAMA):
        off.add(fallback)
    editable["off_provider_note"] = (
        f"route_overrides / fallback send work to {', '.join(sorted(off))} — edit in .env"
        if off else None
    )

    return {
        "editable": editable,
        "pending": pending_values(settings, env_path) if env_path is not None else {},
        "readonly": {
            "host": settings.host,
            "port": settings.port,
            "data_dir": str(settings.data_dir),
        },
        "options": {
            "providers": ["", *HOSTED_PROVIDERS, "ollama"],
            "model_suggestions": {k: list(v) for k, v in MODEL_SUGGESTIONS.items()},
            "tool_modes": list(TOOL_MODES),
            "gemini_live_voices": ["", *sorted(GEMINI_LIVE_VOICES)],
            "gemini_live_transports": list(GEMINI_LIVE_TRANSPORTS),
            # What each transport can actually serve. The API-key list is probed
            # live in routes.py (`model_lists`); Vertex's cannot be, because that
            # probe authenticates with an API key and the Vertex catalogue is not
            # visible to one - so its ids are named, as measured across every
            # region that serves them.
            "vertex_live_models": list(VERTEX_LIVE_MODELS),
            "vertex_locations": list(VERTEX_LIVE_LOCATIONS),
            "voice_providers": list(VOICE_PROVIDERS),
            "openai_realtime_voices": ["", *sorted(OPENAI_REALTIME_VOICES)],
            # Empty is a real choice - "leave it to the server" (config.py).
            "sensitivities": ["", *SENSITIVITIES],
            # Suggestions, not a closed set - see `calendar_email` above and
            # `daemon/admin/google_accounts.py` for why this cannot come from the
            # server itself. An empty list is the ordinary state on an install
            # with no google server and renders as a plain text field.
            "calendar_accounts": authenticated_accounts(),
        },
    }


def _reject_newlines(value: str) -> None:
    """`.env` is line-oriented, so a value with a newline injects extra `KEY=value`
    lines - bypassing the EDITABLE allowlist and the validate-before-write check
    entirely (finding #2). `{"anthropic_api_key": "x\\nDAEMON_HOST=0.0.0.0"}` would
    bind the admin to every interface on the next boot. Refuse it at the source, for
    every editable value and every indirectly-written secret."""
    if "\n" in value or "\r" in value:
        raise PatchError("a value may not contain a newline")


def _env_value(field: str, value: Any) -> str:
    """One editable value serialised the way `.env` stores it."""
    if field in BOOL_FIELDS:
        return "true" if bool(value) else "false"
    if field == ROUTE_OVERRIDES:
        if not isinstance(value, Mapping):
            raise PatchError("route_overrides must be an object of task -> provider")
        # Keys validated as real tasks here so a typo is a 400, not a silently
        # dropped override at startup (the failure config.py's NoDecode guards).
        out: dict[str, str] = {}
        for task, provider in value.items():
            try:
                Task(str(task))
            except ValueError as exc:
                raise PatchError(f"route_overrides names unknown task {task!r}") from exc
            # json.dumps would *escape* a newline rather than emit one, so this line
            # never injects - but a newline in a routing value is malformed input, and
            # rejecting it keeps the rule uniform across every field.
            _reject_newlines(str(task))
            _reject_newlines(str(provider))
            out[str(task)] = str(provider)
        return json.dumps(out)
    text = str(value)
    _reject_newlines(text)
    return text


def write_env_secret(env_path: Path, key: str, value: str) -> None:
    """Write one secret into `.env` under `key`, 0600, leaving everything else.

    The MCP key-auth connect flow calls this before persisting the server block:
    the value lives in `.env` (0600) and `mcp.json` keeps only the variable *name*
    (daemon/tools/mcp.py), so the config file stays shareable. Same writer, same
    quoting rule and same durability as the settings patch above - `merge_env`
    preserves comments and unknown keys because the file is the owner's only copy
    of their credentials."""
    # Same newline guard as the settings patch (finding #2): a secret with a newline
    # would inject an extra `.env` line under the writer's nose.
    _reject_newlines(value)
    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    write_private_replace(env_path, merge_env(existing, {key: value}))


@dataclass(frozen=True, slots=True)
class PatchResult:
    candidate: Settings
    changed: dict[str, str]
    """`.env` keys actually written (differing from the current file)."""


def apply_patch(patch: Mapping[str, Any], env_path: Path) -> PatchResult:
    """Validate current-plus-patch, then write only the changed keys.

    Raises `PatchError` before writing anything if the patch names a field that
    is not editable, or if the resulting `Settings` will not construct. The order
    matters and is the contract: construct, and only a successful construction
    reaches `write_private_replace`.
    """
    unknown = set(patch) - EDITABLE
    if unknown:
        raise PatchError(f"not editable: {', '.join(sorted(unknown))}")

    existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    current = parse_env(existing)

    updates: dict[str, str] = {}
    for field, value in patch.items():
        env_key = (
            ROUTE_OVERRIDES_ENV
            if field == ROUTE_OVERRIDES
            else {
                **STR_FIELDS,
                **BOOL_FIELDS,
                **LIST_FIELDS,
                **INT_FIELDS,
                **FLOAT_FIELDS,
                **SECRET_FIELDS,
            }[field]
        )
        if field in SECRET_FIELDS and (value is None or str(value).strip() == ""):
            # An empty secret means "leave the working key alone", not "clear it".
            continue
        updates[env_key] = _env_value(field, value)

    merged = {**current, **updates}
    try:
        # Init kwargs outrank every settings source, so the patched value is the
        # one validated even if the operator's shell exported an older one. A bad
        # value raises here - config.py's ConfigError, or pydantic's own parse
        # error - and nothing below runs.
        candidate = Settings(_env_file=None, **merged)
    except Exception as exc:  # noqa: BLE001 - every construction failure is a 400
        raise PatchError(str(exc)) from exc

    changed = {key: val for key, val in updates.items() if current.get(key) != val}
    if changed:
        write_private_replace(env_path, merge_env(existing, changed))
    return PatchResult(candidate=candidate, changed=changed)
