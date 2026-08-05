"""MCP client: other people's tools, under our policy.

An MCP server is one more way to reach the machine, so its tools are registered
`guarded` and go through `policy.py` exactly like `run_command` does. A server
that could bypass the approval path would make the whole policy decorative.

Two containment decisions:

  * **The SDK is only imported here.** `mcp` has a v2 in beta with a different
    client surface, so the blast radius of that landing is one file. It is also an
    optional dependency (the `mcp` extra), and a missing import degrades to "no
    MCP tools" rather than to a daemon that will not boot.
  * **A failing server is skipped, never fatal.** Same rule the recall stack
    follows in `app.py:_build_recall` and for the same reason: an npx download that
    times out must not cost the user their conversation loop.

Config lives in `<data_dir>/mcp.json`, in the shape everything else in this
ecosystem uses, so a server block can be copied from another client's config:

    {
      "servers": {
        "filesystem": {"command": "npx",
                       "args": ["-y", "@modelcontextprotocol/server-filesystem", "~"]},
        "notes":      {"url": "https://example.com/mcp"}
      }
    }

Per-server `"safe": ["tool_name"]` marks read-only tools so they skip the approval
path. It is opt-in per tool and never a wildcard: the owner is asserting something
about a specific tool they have read about, not about a server they trust.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daemon.llm.base import ToolSpec
from daemon.tools.base import Registry, Risk, ToolError

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "mcp.json"
NAME_SEPARATOR = "__"
"""`<server>__<tool>`. Namespaced because two servers legitimately both offer
`search`, and `Registry.register` refuses a collision rather than shadowing one."""

STARTUP_TIMEOUT = 30.0
"""Per server. An `npx` server downloads its package on first run, which is slow
once and fast afterwards; unbounded would mean a typo in a command hangs startup."""

CALL_TIMEOUT = 60.0


@dataclass(frozen=True, slots=True)
class ServerConfig:
    name: str
    command: str = ""
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = ()  # type: ignore[assignment]
    url: str = ""
    safe: frozenset[str] = frozenset()

    @property
    def is_remote(self) -> bool:
        return bool(self.url)


@dataclass(frozen=True, slots=True)
class McpConfig:
    servers: list[ServerConfig]
    rejected: dict[str, str]
    """Blocks that could not be read, by name, with why.

    Carried out rather than only logged: a server the user configured and that is
    not available has to show up on `/health`, or the difference between "a typo in
    mcp.json" and "the model chose not to use that tool" is invisible - which is the
    exact failure `_tools_health` exists to prevent.
    """


def load_config(data_dir: Path) -> McpConfig:
    """Read `mcp.json`, or return nothing configured.

    Every problem here is reported and skipped rather than raised. This file is
    hand-edited, and a trailing comma in it must not stop the daemon starting -
    the user would have no way to see why.
    """
    path = Path(data_dir) / CONFIG_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return McpConfig([], {})
    except (OSError, ValueError) as exc:
        logger.error("%s could not be read, so no MCP servers are configured: %s", path, exc)
        return McpConfig([], {CONFIG_FILENAME: f"could not be read: {exc}"})

    servers = raw.get("servers") if isinstance(raw, dict) else None
    if not isinstance(servers, dict):
        logger.error("%s has no 'servers' object; no MCP servers are configured", path)
        return McpConfig([], {CONFIG_FILENAME: "has no 'servers' object"})

    configs: list[ServerConfig] = []
    rejected: dict[str, str] = {}
    for name, block in servers.items():
        problem = ""
        if not isinstance(block, dict):
            problem = "is not an object"
        elif NAME_SEPARATOR in name:
            # Otherwise `a__b` + tool `c` and `a` + tool `b__c` collide.
            problem = f"may not contain {NAME_SEPARATOR!r} in its name"
        if problem:
            logger.error("MCP server %r %s; skipping it", name, problem)
            rejected[name] = problem
            continue
        assert isinstance(block, dict)  # narrowed above

        command = str(block.get("command", "") or "")
        url = str(block.get("url", "") or "")
        if bool(command) == bool(url):
            problem = "needs exactly one of 'command' or 'url'"
            logger.error("MCP server %r %s; skipping it", name, problem)
            rejected[name] = problem
            continue
        configs.append(
            ServerConfig(
                name=name,
                command=command,
                args=tuple(str(a) for a in block.get("args", ()) or ()),
                env={str(k): str(v) for k, v in (block.get("env") or {}).items()},
                url=url,
                safe=frozenset(str(s) for s in block.get("safe", ()) or ()),
            )
        )
    return McpConfig(configs, rejected)


class McpTool:
    """One remote tool, wearing the local `Tool` protocol.

    Not `Executable`: there is no argv to allowlist, so in `ask` mode every call is
    a question and in `allowlist` mode every call is refused. That is the honest
    mapping - "only run allowlisted commands" cannot be satisfied by something whose
    action we cannot name.
    """

    def __init__(self, bridge: McpBridge, server: str, name: str, spec: ToolSpec, risk: Risk):
        self._bridge = bridge
        self._server = server
        self._remote_name = name
        self.spec = spec
        self.risk = risk

    def preview(self, arguments: Mapping[str, Any]) -> str:
        shown = json.dumps(dict(arguments), ensure_ascii=False, default=str)[:160]
        return f"{self._server}: {self._remote_name}({shown})"

    async def run(self, arguments: Mapping[str, Any]) -> str:
        return await self._bridge.call(self._server, self._remote_name, arguments)


class McpBridge:
    """Owns the sessions and their teardown.

    One `AsyncExitStack` for every server, closed in reverse: a stdio server is a
    child process, and leaking one per restart is how a machine ends up with forty
    of them.
    """

    def __init__(self, config: McpConfig | list[ServerConfig]) -> None:
        # A bare list is accepted so a test can build a bridge from one server
        # without also stating that nothing was rejected.
        resolved = McpConfig(config, {}) if isinstance(config, list) else config
        self._configs = resolved.servers
        self._stack = AsyncExitStack()
        self._sessions: dict[str, Any] = {}
        self.failures: dict[str, str] = dict(resolved.rejected)
        """Server name -> why it is not available, for `/health`. A silently absent
        tool is indistinguishable from a model that chose not to use it. Seeded with
        the blocks that never parsed, then added to as servers fail to start."""

    async def start(self, registry: Registry) -> int:
        """Connect every configured server and register what it offers.

        Returns how many tools landed. Never raises: the caller is startup.
        """
        if not self._configs:
            return 0
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            for config in self._configs:
                self.failures[config.name] = "the 'mcp' extra is not installed"
            logger.error(
                "%d MCP server(s) configured but the mcp package is missing "
                "(pip install 'daemon-ai[mcp]'): %s",
                len(self._configs),
                exc,
            )
            return 0

        registered = 0
        for config in self._configs:
            try:
                session = await self._connect(
                    config, ClientSession, StdioServerParameters, stdio_client
                )
                self._sessions[config.name] = session
                # Inside the same guard as the connect, deliberately. It used to sit
                # after it, so a server that connected and then hung on `list_tools`
                # raised out of `start()` and cost every *remaining* server its tools
                # as well - one slow server disabling the others.
                registered += await self._register(config, session, registry)
            except Exception as exc:
                # Anything at all: a missing executable, a protocol mismatch, a
                # server that exits immediately, a TLS failure, a listing that never
                # arrives.
                self.failures[config.name] = str(exc) or exc.__class__.__name__
                logger.error("MCP server %r did not start: %s", config.name, exc)
                continue
        return registered

    async def _connect(
        self,
        config: ServerConfig,
        client_session: Any,
        stdio_params: Any,
        stdio_client: Any,
    ) -> Any:
        import asyncio

        if config.is_remote:
            from mcp.client.streamable_http import streamablehttp_client

            # The first two of the three streams; the third is a session-id getter
            # this code has no use for.
            read, write, *_ = await self._stack.enter_async_context(
                streamablehttp_client(config.url)
            )
        else:
            read, write = await self._stack.enter_async_context(
                stdio_client(
                    stdio_params(
                        command=config.command,
                        args=list(config.args),
                        # Inherit, then overlay: a server started with an empty
                        # environment has no PATH and no HOME, and most of them
                        # need both.
                        env={**os.environ, **dict(config.env)},
                    )
                )
            )
        session = await self._stack.enter_async_context(client_session(read, write))
        await asyncio.wait_for(session.initialize(), timeout=STARTUP_TIMEOUT)
        return session

    async def _register(self, config: ServerConfig, session: Any, registry: Registry) -> int:
        import asyncio

        listing = await asyncio.wait_for(session.list_tools(), timeout=STARTUP_TIMEOUT)
        count = 0
        for tool in getattr(listing, "tools", ()):
            local_name = f"{config.name}{NAME_SEPARATOR}{tool.name}"
            schema = getattr(tool, "inputSchema", None)
            spec = ToolSpec(
                name=local_name,
                description=(getattr(tool, "description", "") or f"{tool.name} via {config.name}"),
                # Forwarded untouched. A server's schema is its own business, and
                # rewriting it here is how a valid call starts getting rejected.
                parameters=(
                    schema
                    if isinstance(schema, dict)
                    else {"type": "object", "properties": {}}
                ),
            )
            risk: Risk = "safe" if tool.name in config.safe else "guarded"
            try:
                registry.register(McpTool(self, config.name, tool.name, spec, risk))
            except ValueError as exc:
                logger.error("MCP tool %s could not be registered: %s", local_name, exc)
                continue
            count += 1
        logger.info("MCP server %r offered %d tool(s)", config.name, count)
        return count

    async def call(self, server: str, name: str, arguments: Mapping[str, Any]) -> str:
        import asyncio

        session = self._sessions.get(server)
        if session is None:
            raise ToolError(f"the MCP server {server!r} is not connected")
        try:
            result = await asyncio.wait_for(
                session.call_tool(name, dict(arguments)), timeout=CALL_TIMEOUT
            )
        except TimeoutError:
            raise ToolError(f"{server}: {name} did not answer within {CALL_TIMEOUT:.0f}s") from None
        except Exception as exc:
            raise ToolError(f"{server}: {name} failed: {exc}") from exc

        text = _text_of(result)
        if getattr(result, "isError", False):
            raise ToolError(text or f"{server}: {name} reported an error")
        return text or "(no output)"

    async def aclose(self) -> None:
        self._sessions.clear()
        try:
            await self._stack.aclose()
        except Exception:
            # Shutdown: a server that dies while being closed has nothing left to
            # break, and raising here would mask whatever else the lifespan is
            # unwinding.
            logger.exception("closing MCP sessions failed")


def _text_of(result: Any) -> str:
    """The text of a tool result, across the shapes the SDK has used.

    `structuredContent` on newer servers, a list of content blocks otherwise, and
    a bare string from something home-made. Reaching for whichever is present beats
    pinning one and breaking on an upgrade.
    """
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and structured:
        return json.dumps(structured, ensure_ascii=False, default=str)
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or ():
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
            continue
        data = getattr(block, "data", None)
        # A non-text block (an image) has no useful rendering for a text turn, so
        # it is named rather than dumped.
        parts.append(f"[{getattr(block, 'type', 'content')} block]" if data else str(block))
    return "\n".join(parts).strip()
