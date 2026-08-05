"""Tool contract - how Daemon reaches the machine it lives on.

A tool is a named thing the model may ask for, with a JSON Schema for its
arguments and an async body. Nothing here decides whether a call is *allowed*:
that is `policy.py`, and keeping the two apart is what lets the policy be tested
without a filesystem and the tools be tested without a policy.

Two properties every tool must have, because the runner relies on them:

  * `risk` - 'safe' means read-only, local, and with no effect the owner would
    want to be asked about. Everything else is 'guarded' and goes through policy.
    A tool that is unsure about itself is 'guarded'.
  * `preview(arguments)` - one line naming what the call will actually do. It is
    what the approval message shows and what the audit row records, so it must
    contain the specifics (the command, the path) and not a description of the
    tool in general.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from daemon.llm.base import ToolSpec

Risk = Literal["safe", "guarded"]


class ToolError(RuntimeError):
    """A tool refused or failed in a way the model should be told about.

    The message goes back to the model as the tool's result, so it is written for
    that reader: what went wrong and what would work instead. Distinct from
    `ProviderError` - a failing tool is an ordinary event in a turn, not a failed
    turn.
    """


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    name: str
    content: str
    ok: bool = True
    elapsed_ms: int = 0


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec
    risk: Risk

    def preview(self, arguments: Mapping[str, Any]) -> str:
        """One line describing this specific call, for approval and audit."""
        ...

    async def run(self, arguments: Mapping[str, Any]) -> str:
        """Do the thing. Raise `ToolError` for anything the model should see."""
        ...


@runtime_checkable
class Executable(Protocol):
    """A tool whose action reduces to a command line.

    Split out from `Tool` because allowlist matching and approval binding are only
    meaningful for these: 'let it run `git status` without asking' is a statement
    about argv. A guarded tool that is not `Executable` can be allowed or asked
    about, but never allowlisted - which is why `allowlist` mode refuses
    `write_file` outright rather than pretending to match it.
    """

    def argv(self, arguments: Mapping[str, Any]) -> list[str]:
        """The command this call would run, already split. Raises `ToolError` if
        the arguments do not parse into one."""
        ...


class Registry:
    """The tools this process has, by name.

    Registration order is preserved so the schemas the model sees are stable
    between turns: a set would reorder them and quietly defeat prefix caching on
    the hosted providers.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        name = tool.spec.name
        if name in self._tools:
            # Two tools answering to one name means one of them is unreachable, and
            # which one would depend on import order. An MCP server whose tool
            # collides with a built-in is the realistic way to get here.
            raise ValueError(f"tool {name!r} is already registered")
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def closeables(self) -> list[Any]:
        """Tools holding something that has to be released at shutdown.

        `fetch_page` owns an HTTP client. Leaking one per restart is the same defect
        the lifespan already avoids for providers and the embedder.
        """
        return [tool for tool in self._tools.values() if hasattr(tool, "aclose")]

    def __len__(self) -> int:
        return len(self._tools)


def canonical_arguments(arguments: Mapping[str, Any]) -> str:
    """Arguments as stable JSON, for fingerprinting and for the audit column.

    Sorted keys so the same call fingerprints the same way whichever order the
    model happened to emit, and `default=str` so an argument that is not JSON
    serialisable produces a fingerprint instead of an exception - a bad call still
    has to be recorded.
    """
    return json.dumps(dict(arguments), sort_keys=True, ensure_ascii=False, default=str)
