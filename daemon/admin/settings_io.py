"""Read the editable settings, and write a patch to `.env` - validated first.

docs/design decision 3, and the M5 test gate: a patch is committed to `.env`
**only** if a candidate `Settings` built from current-plus-patch constructs
cleanly. `Settings` fails loudly at construction (daemon/config.py), so building
one is the whole validation: an unknown preset, a non-numeric limit, a voice
switch with no voice route all raise there, before a single byte is written. On
failure the caller returns 400 and the file is untouched.

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

from daemon.config import (
    HOSTED_PROVIDERS,
    PRESETS,
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
    "preset": "DAEMON_PRESET",
    "hosted_provider": "DAEMON_HOSTED_PROVIDER",
    "tools_mode": "DAEMON_TOOLS_MODE",
}
BOOL_FIELDS: dict[str, str] = {
    "voice_enabled": "DAEMON_VOICE_ENABLED",
    "mcp_enabled": "DAEMON_MCP_ENABLED",
    "browser_enabled": "DAEMON_BROWSER_ENABLED",
}
INT_FIELDS: dict[str, str] = {"recall_limit": "DAEMON_RECALL_LIMIT"}
FLOAT_FIELDS: dict[str, str] = {"recall_half_life_days": "DAEMON_RECALL_HALF_LIFE_DAYS"}
SECRET_FIELDS: dict[str, str] = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "gemini_api_key": "GEMINI_API_KEY",
}
ROUTE_OVERRIDES = "route_overrides"
ROUTE_OVERRIDES_ENV = "DAEMON_ROUTE_OVERRIDES"

EDITABLE = {
    *STR_FIELDS,
    *BOOL_FIELDS,
    *INT_FIELDS,
    *FLOAT_FIELDS,
    *SECRET_FIELDS,
    ROUTE_OVERRIDES,
}


class PatchError(ValueError):
    """A patch that cannot be applied. Its message is safe to return to the
    client - it names the field or the validation failure, never a secret."""


def current_settings_payload(settings: Settings) -> dict[str, Any]:
    """The GET /settings body: editable values (secrets masked), read-only
    display, and the option lists the front-end offers choices from."""
    editable: dict[str, Any] = {
        name: getattr(settings, name) for name in (*STR_FIELDS, *BOOL_FIELDS, *INT_FIELDS)
    }
    editable["recall_half_life_days"] = settings.recall_half_life_days
    editable[ROUTE_OVERRIDES] = {
        task.value: provider for task, provider in settings.route_overrides.items()
    }
    for name in SECRET_FIELDS:
        # "set"/null, never the value. The one place this module could leak.
        editable[name] = "set" if getattr(settings, name) else None
    return {
        "editable": editable,
        "readonly": {
            "host": settings.host,
            "port": settings.port,
            "data_dir": str(settings.data_dir),
        },
        "options": {
            "presets": sorted(PRESETS),
            "hosted_providers": list(HOSTED_PROVIDERS),
            "tool_modes": list(TOOL_MODES),
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
            else {**STR_FIELDS, **BOOL_FIELDS, **INT_FIELDS, **FLOAT_FIELDS, **SECRET_FIELDS}[
                field
            ]
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
