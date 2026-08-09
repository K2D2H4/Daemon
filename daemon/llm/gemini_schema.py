"""Narrow a JSON Schema to what Gemini's function declarations accept.

Two Gemini clients send tool schemas on the wire and both must apply this: the
REST provider (`llm/providers/gemini.py`) and the Live voice session
(`voice/gemini_live.py`). It lives here, in the neutral `llm` layer, so the voice
client can share it without importing a concrete provider.

A `ToolSpec.parameters` is forwarded from its source untouched - an MCP server's
own `inputSchema` is its own business (`tools/mcp.py`) - and those routinely carry
`additionalProperties`, `title`, `$schema` or `$ref`. Gemini rejects every one of
them, closing the request (REST: 400; Live socket: 1007), so a single connected
MCP server broke *every* tool-enabled turn - even "hello", because tools are
offered on every owner turn. Anthropic and OpenAI tolerate the extra keywords;
this narrowing is Gemini's alone.
"""

from __future__ import annotations

from typing import Any

_SCHEMA_KEYS = frozenset(
    {
        "type", "description", "nullable", "enum", "items", "properties",
        "required", "minItems", "maxItems", "minimum", "maximum",
        "minLength", "maxLength", "pattern", "anyOf",
    }
)
"""The JSON-Schema keywords Gemini's function declarations accept (an OpenAPI 3.0
subset). Everything else - `additionalProperties`, `$schema`, `title`, `$ref`,
`default`, `format` - is rejected."""


def gemini_schema(node: Any) -> Any:
    """Reduce a JSON Schema to the subset Gemini's function declarations accept."""
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {k: gemini_schema(v) for k, v in value.items()}
        elif key == "items":
            out[key] = gemini_schema(value)
        elif key == "anyOf" and isinstance(value, list):
            out[key] = [gemini_schema(v) for v in value]
        else:
            out[key] = value
    # JSON Schema allows `type: ["string", "null"]`; Gemini wants one type plus a
    # `nullable` flag.
    declared = node.get("type")
    if isinstance(declared, list):
        concrete = [t for t in declared if t != "null"]
        out["type"] = concrete[0] if concrete else "string"
        if "null" in declared:
            out["nullable"] = True
    # An object schema Gemini can read has to say so, even when the server left the
    # `type` implicit and only gave `properties`.
    if "properties" in out and "type" not in out:
        out["type"] = "object"
    return out
