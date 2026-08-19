"""Which tools a voice session may be offered.

The native-audio Gemini Live model emits a flat-argument tool call reliably and a
nested-argument one almost never - it fakes the result instead (measured:
evals/voice_write_nudge_spike.py, docs/superpowers/specs/2026-08-14-async-delegation-design.md).
So a voice session is offered only flat-schema tools plus `delegate_task`, which
routes the rest to the text path. `is_flat_schema` is that gate.

**An optional scalar is not nesting, and treating it as nesting cost most of the
tool set (2026-08-19).** `Optional[str]` renders as `anyOf: [{string}, {null}]`,
which is what FastMCP and pydantic emit for every defaulted argument - so one
`page_token=None` hid a whole tool. Of the google server's 27 tools only 9 reached
voice; 8 of the 18 missing were excluded by nothing but a nullable scalar,
`search_gmail_messages` among them. The owner asked his daemon to find an email,
it could list labels and nothing else, and it invented a `gmail search` shell
command instead of saying so.

Measured before relaxing it, 20 trials per arm against the live model: offered
`search_gmail_messages` with its real schema, the audio model emitted a correct
call with both required arguments **20/20**, identically whether the `anyOf` was
left as shipped or folded to a plain type. The wall is real for genuine structure
and was never real for a nullable scalar.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DELEGATE_TOOL_NAME = "delegate_task"
"""The one non-flat-work tool a voice session gets. Defined once; consumed by
`Companion.specs(surface="voice")` and by the tool itself."""

_PRIMITIVE_TYPES = frozenset({"string", "number", "integer", "boolean"})
_COMPOSERS = ("$ref", "anyOf", "oneOf", "allOf")
_UNIONS = ("anyOf", "oneOf")
"""The two composers that can still describe a single fillable value. `allOf` and
`$ref` cannot - they are composition proper, and stay disqualifying wherever they
appear."""


def _is_primitive(schema: Mapping[str, Any]) -> bool:
    """One argument the audio model can fill: a scalar, or a scalar that may be
    omitted.

    The second case is the whole of the 2026-08-19 correction. An argument with a
    default is `Optional[T]` in the server's own signature and reaches us as either
    `anyOf: [{"type": T}, {"type": "null"}]` or JSON Schema's `type: [T, "null"]`.
    Both describe one scalar the model may leave out, which is what it does with
    them anyway.

    A union stays disqualifying the moment a branch is *not* scalar - which is not
    hypothetical, it is the next tool along: `send_gmail_message`'s `to` is
    `anyOf: [string, array, null]`, one address or a list of them, and that is the
    shape the model fakes rather than fills. A union of nothing but `null` is
    likewise refused: there is no type left to fill.
    """
    if "$ref" in schema or "allOf" in schema:
        return False
    for key in _UNIONS:
        branches = schema.get(key)
        if branches is None:
            continue
        if not isinstance(branches, list) or not branches:
            return False
        kinds = [b.get("type") if isinstance(b, dict) else None for b in branches]
        if not all(kind in _PRIMITIVE_TYPES or kind == "null" for kind in kinds):
            return False
        if not any(kind in _PRIMITIVE_TYPES for kind in kinds):
            return False
        return True
    declared = schema.get("type")
    if isinstance(declared, list):
        kinds = [kind for kind in declared]
        return all(kind in _PRIMITIVE_TYPES or kind == "null" for kind in kinds) and any(
            kind in _PRIMITIVE_TYPES for kind in kinds
        )
    return declared in _PRIMITIVE_TYPES


def is_flat_schema(parameters: Mapping[str, Any]) -> bool:
    """True if every argument is one value the audio model can fill - no nested
    object or array, no composition, though a scalar that may be omitted counts.

    The top level is stricter than a property: a composer *on the schema itself*
    ("exactly one of these two argument sets") describes the shape of the call
    rather than the shape of one value, and there is nothing to relax about it.
    """
    if any(key in parameters for key in _COMPOSERS):
        return False
    properties = parameters.get("properties")
    if not properties:
        # No declared arguments: a no-arg tool is trivially callable.
        return True
    if not isinstance(properties, dict):
        return False
    return all(
        isinstance(schema, dict) and _is_primitive(schema) for schema in properties.values()
    )
