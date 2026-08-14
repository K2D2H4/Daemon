"""Which tools a voice session may be offered.

The native-audio Gemini Live model emits a flat-argument tool call reliably and a
nested-argument one almost never - it fakes the result instead (measured:
evals/voice_write_nudge_spike.py, docs/superpowers/specs/2026-08-14-async-delegation-design.md).
So a voice session is offered only flat-schema tools plus `delegate_task`, which
routes the rest to the text path. `is_flat_schema` is that gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DELEGATE_TOOL_NAME = "delegate_task"
"""The one non-flat-work tool a voice session gets. Defined once; consumed by
`Companion.specs(surface="voice")` and by the tool itself."""

_PRIMITIVE_TYPES = frozenset({"string", "number", "integer", "boolean"})
_COMPOSERS = ("$ref", "anyOf", "oneOf", "allOf")


def is_flat_schema(parameters: Mapping[str, Any]) -> bool:
    """True if every argument is a primitive - no nested object or array, no
    composition. A flat schema is one the audio model can actually fill."""
    if any(key in parameters for key in _COMPOSERS):
        return False
    properties = parameters.get("properties")
    if not properties:
        # No declared arguments: a no-arg tool is trivially callable.
        return not any(key in parameters for key in _COMPOSERS)
    if not isinstance(properties, dict):
        return False
    for schema in properties.values():
        if not isinstance(schema, dict):
            return False
        if any(key in schema for key in _COMPOSERS):
            return False
        if schema.get("type") not in _PRIMITIVE_TYPES:
            return False
    return True
