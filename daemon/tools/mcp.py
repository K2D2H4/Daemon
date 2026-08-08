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

import asyncio
import json
import logging
import os
import shutil
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daemon.fs import write_private_replace
from daemon.llm.base import ToolSpec
from daemon.mcp_catalog import CatalogEntry
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
    key_env: str = ""
    """The NAME of the environment variable holding this server's secret, never the
    value. The value lives in `.env` (0600); `mcp.json` stores only the name, so the
    config file stays shareable. At connect time it becomes one child env var for a
    stdio server or an `Authorization: Bearer` header for a url one."""

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
                key_env=str(block.get("key_env", "") or ""),
            )
        )
    return McpConfig(configs, rejected)


_EXTRA_BIN_DIRS = ("~/.local/bin", "/opt/homebrew/bin", "/usr/local/bin")
"""Where user-installed launchers live but a supervised process cannot see them.

A LaunchAgent/systemd service runs with a minimal PATH (`/usr/bin:/bin:...`), which
omits `~/.local/bin` - exactly where `uv tool install` puts `uvx`. So a catalog
server configured as `uvx mcp-server-<x>` died with `[Errno 2] No such file or
directory` under the service while working fine from a shell `daemon run`. These
dirs are added when resolving the command and to the child's PATH."""

_SAFE_STDIO_ENV = ("PATH", "HOME")
"""The only variables inherited by a stdio child, and deliberately not `os.environ`.

The environment Daemon runs in holds its own provider secrets - `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`, the Telegram token - and a curated third-party
MCP server has no business reading any of them, so the old `{**os.environ, ...}`
merge is refused. What a `uvx` server does need is a `PATH` to be found on and a
`HOME`; those are not secrets, so they pass through by name. Everything else the
server sees is what the owner wrote in `mcp.json` plus the single secret they named
for it. Allowlist, not denylist: a new secret in Daemon's environment is invisible
to servers by default, never leaked until someone adds it here on purpose."""


def _secret_value(config: ServerConfig, secret: str | None) -> str | None:
    """The secret for this server: the caller's if given, else read from `.env` by
    the name the config carries. None when the server names no key or the named
    variable is unset - the honest 'no secret configured', not an empty one."""
    if not config.key_env:
        return None
    if secret is not None:
        return secret
    return os.environ.get(config.key_env)


def _augmented_path() -> str:
    """The current PATH with the user bin dirs appended, so `uvx` (and whatever it
    shells out to) is findable even under a service's minimal PATH."""
    extra = os.pathsep.join(os.path.expanduser(d) for d in _EXTRA_BIN_DIRS)
    current = os.environ.get("PATH", "")
    return os.pathsep.join(p for p in (current, extra) if p)


def _resolve_command(command: str) -> str:
    """Absolute path to `command`, searching the user bin dirs too, or `command`
    unchanged if not found (so a bad command still fails honestly, by name).

    Handing subprocess an absolute path is what actually fixes the service case:
    on POSIX the executable is looked up against the *parent's* PATH, not the child
    env we pass, so augmenting the child's PATH alone would not let it find `uvx`."""
    if os.path.isabs(command):
        return command
    return shutil.which(command, path=_augmented_path()) or command


def _stdio_env(config: ServerConfig, secret: str | None = None) -> dict[str, str]:
    """The environment a stdio child is started with: the safe passthrough (with the
    user bin dirs added to PATH), then the static env the owner set in `mcp.json`,
    then at most the one named secret."""
    env = {name: os.environ[name] for name in _SAFE_STDIO_ENV if name in os.environ}
    env["PATH"] = _augmented_path()
    env.update({str(k): str(v) for k, v in dict(config.env).items()})
    value = _secret_value(config, secret)
    if config.key_env and value is not None:
        env[config.key_env] = value
    return env


def _bearer_headers(config: ServerConfig, secret: str | None = None) -> dict[str, str] | None:
    """`Authorization: Bearer <value>` for a url server whose secret is available, or
    None. This header path is net-new: the SDK's `streamablehttp_client` accepts
    `headers=`, and passing the owner's key here is what authenticates a hosted
    server without ever writing the value into `mcp.json`."""
    value = _secret_value(config, secret)
    if config.key_env and value is not None:
        return {"Authorization": f"Bearer {value}"}
    return None


def _leaf_message(exc: BaseException) -> str:
    """A readable reason from a connect failure, unwrapping ExceptionGroups.

    The streamable-HTTP client runs inside an anyio task group, so a failed url
    connection surfaces as `unhandled errors in a taskgroup (1 sub-exception)` -
    which hides the 401 or closed stream that actually happened (a url server with a
    missing bearer looked exactly like this). Dig to the first leaf and report it."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return str(exc) or exc.__class__.__name__


def _mcp_pin() -> tuple[str, ...]:
    """`--with mcp==<the version this daemon speaks>`, or empty if unknown.

    A bare `uvx mcp-server-<x>` resolves the server's `mcp` dependency freshly, and
    the reference servers still `from mcp.shared.exceptions import McpError` while a
    newer `mcp` has renamed it - so the server dies at import before it can speak a
    word, and the only symptom is a connect that "did not connect". Pinning the
    server's `mcp` to the exact one the daemon imports keeps both ends on one
    protocol library. Empty tuple if `mcp` somehow has no metadata: no pin is better
    than a broken argument, and the connect then fails honestly as before."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return ("--with", f"mcp=={version('mcp')}")
    except PackageNotFoundError:
        return ()


def server_config_from_catalog(entry: CatalogEntry) -> ServerConfig:
    """Turn a trusted catalog entry into a `ServerConfig` the bridge can connect.

    The field mapping lives here, in one place, so the admin route need not know how
    a catalog `kind` becomes a command or a url. Only structured fields cross over -
    a `key_env` name, never a secret (CONTRACTS 13).

    A `uvx` server is launched with the daemon's own `mcp` pinned in (see
    `_mcp_pin`), so a curated catalog server actually starts rather than failing on a
    resolved-too-new `mcp`. The pin is a plain argv prefix - structured, no secret,
    and persisted transparently into `mcp.json`."""
    args = tuple(entry.args)
    if entry.kind == "uvx":
        args = (*_mcp_pin(), *args)
    return ServerConfig(
        name=entry.name,
        command=entry.command,
        args=args,
        url=entry.url,
        key_env=entry.key_env or "",
    )


def _block_of(config: ServerConfig) -> dict[str, Any]:
    """The `mcp.json` block for one server - structured fields only, no secret."""
    block: dict[str, Any] = {}
    if config.is_remote:
        block["url"] = config.url
    else:
        block["command"] = config.command
        if config.args:
            block["args"] = list(config.args)
    if config.env:
        block["env"] = dict(config.env)
    if config.safe:
        block["safe"] = sorted(config.safe)
    if config.key_env:
        block["key_env"] = config.key_env
    return block


def _read_raw(path: Path) -> dict[str, Any]:
    """The current `mcp.json` as a dict, `{}` only when the file is genuinely absent.

    Absence is fine - the first `save_server` starts the file. But a file that
    *exists* and does not parse (the trailing comma `load_config`'s docstring calls
    out) must not be treated as empty: the next `save_server`/`remove_server` would
    then rewrite the file with only its one change, silently discarding every block
    the owner hand-edited. So a parse or shape error raises, for the route to
    surface, rather than the connect flow quietly clobbering the file.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise ToolError(
            f"{path} exists but is not valid JSON ({exc}); fix or remove it before "
            "adding or removing a server, so its other blocks are not lost"
        ) from exc
    if not isinstance(raw, dict):
        raise ToolError(
            f"{path} exists but is not a JSON object; fix or remove it before "
            "adding or removing a server, so its other blocks are not lost"
        )
    return raw


def save_server(data_dir: Path, config: ServerConfig) -> None:
    """Write (or replace) one server's block in `mcp.json`, atomically and 0600.

    The connect flow calls this *before* connecting: a connect that then fails
    leaves the entry, so `/health` can report 'configured but not connected' rather
    than the install vanishing. Owner-only because the file lives under the data dir
    with everything else private, and atomic because a half-written `mcp.json` would
    read as no servers at all on the next start.

    Only `_block_of`'s structured fields are written - crucially `key_env`, the name
    of the variable, never its value."""
    path = Path(data_dir) / CONFIG_FILENAME
    raw = _read_raw(path)
    servers = raw.get("servers")
    if not isinstance(servers, dict):
        servers = {}
    servers[config.name] = _block_of(config)
    raw["servers"] = servers
    write_private_replace(path, json.dumps(raw, ensure_ascii=False, indent=2) + "\n")


def remove_server(data_dir: Path, name: str) -> bool:
    """Drop one server's block from `mcp.json`, returning whether it was there.

    The persisted half of `disconnect_server`. Idempotent: removing an absent server
    is not an error, so a disconnect that half-happened is safe to retry."""
    path = Path(data_dir) / CONFIG_FILENAME
    raw = _read_raw(path)
    servers = raw.get("servers")
    if not isinstance(servers, dict) or name not in servers:
        return False
    del servers[name]
    raw["servers"] = servers
    write_private_replace(path, json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
    return True


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


class _ServerLink:
    """One server's transport + session, opened and closed in a single task.

    The SDK's stdio/http transports and `ClientSession` are anyio cancel scopes, and
    anyio raises `RuntimeError: Attempted to exit cancel scope in a different task
    than it was entered in` if they are `__aexit__`'d anywhere but the task that
    `__aenter__`'d them. A hot connect runs in one request task and its disconnect in
    another, so splitting enter and exit across them orphaned the stdio child every
    time (the RuntimeError was swallowed as "closing failed"). `_run` keeps both ends
    together: it opens the contexts, hands the session back over `_ready`, waits for
    `_close`, and only then exits them - all in the one task. The request tasks
    (`connect_server`, `disconnect_server`, `aclose`) only ever signal across.
    """

    def __init__(
        self, bridge: McpBridge, config: ServerConfig, *, secret: str | None, auth: Any
    ) -> None:
        self._bridge = bridge
        self._config = config
        self._secret = secret
        self._auth = auth
        self._session: Any = None
        self._error: BaseException | None = None
        self._ready = asyncio.Event()
        self._close = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def open(self) -> Any:
        """Start the owning task and block until the session is up, or the connect
        failed. Returns the session; on failure raises exactly what the connect
        raised - the task has already torn its own partial stack down in that task,
        so nothing is orphaned."""
        self._task = asyncio.create_task(self._run())
        await self._ready.wait()
        if self._error is not None:
            await self._task
            raise self._error
        return self._session

    async def close(self) -> None:
        """Signal the owning task to exit the contexts, and wait for it. Idempotent:
        a second call just re-sets an already-set event and awaits a done task."""
        self._close.set()
        if self._task is not None:
            await self._task

    async def _run(self) -> None:
        try:
            self._session = await self._bridge._connect(
                self._config, secret=self._secret, auth=self._auth
            )
        except BaseException as exc:  # noqa: BLE001 - relayed to `open`, task ends clean
            # `_connect` has already closed its own partial stack, in this same task.
            self._error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            await self._close.wait()
        finally:
            # Same task that entered the stack, so anyio permits the exit.
            await self._bridge._close_stack(self._config.name)


class McpBridge:
    """Owns the sessions and their teardown, one exit stack *per server*.

    A stdio server is a child process, so every server gets its own `AsyncExitStack`
    that closes only it: that is what lets one server connect or disconnect while the
    others stay up, which the admin "add a server" flow needs and a single shared
    stack could not give. Leaking a stack per restart is how a machine ends up with
    forty orphaned children, so `disconnect_server` and `aclose` both close them.

    That stack is entered *and exited in a single dedicated task per server* (see
    `_ServerLink`). The SDK's stdio/http transports and `ClientSession` are anyio
    cancel scopes, and anyio refuses to exit one in a different task than opened it -
    so a hot connect (a request task) followed by a disconnect (another request task)
    used to raise `Attempted to exit cancel scope in a different task`, which the
    teardown's `except` swallowed while the stdio child was left running. The task
    keeps enter and exit together; the request tasks only ever *signal* it.

    `register`, `unregister` and `specs` are synchronous and need no lock, but a rare
    cross-turn install (the admin route connecting while a turn is mid-flight) could
    race the session and stack dicts, so those mutations are serialised by
    `self._lock`.
    """

    def __init__(self, config: McpConfig | list[ServerConfig]) -> None:
        # A bare list is accepted so a test can build a bridge from one server
        # without also stating that nothing was rejected.
        resolved = McpConfig(config, {}) if isinstance(config, list) else config
        self._configs = resolved.servers
        self._stacks: dict[str, AsyncExitStack] = {}
        self._connections: dict[str, _ServerLink] = {}
        """Server name -> the task that owns its transport, so the same task that
        opened the stack is the one that closes it (see `_ServerLink`)."""
        self._sessions: dict[str, Any] = {}
        self._registered: dict[str, tuple[str, ...]] = {}
        """Server name -> the local tool names it put in the registry, so a
        disconnect removes exactly those and nothing it does not own."""
        self._registry: Registry | None = None
        """The registry `start` was handed. Kept so a later hot connect/disconnect
        adds to and removes from the same one the app built."""
        self._lock = asyncio.Lock()
        self.failures: dict[str, str] = dict(resolved.rejected)
        """Server name -> why it is not available, for `/health`. A silently absent
        tool is indistinguishable from a model that chose not to use it. Seeded with
        the blocks that never parsed, then added to as servers fail to start."""

    async def start(self, registry: Registry) -> int:
        """Connect every configured server and register what it offers.

        Returns how many tools landed. Never raises: the caller is startup. Records
        the registry so `connect_server`/`disconnect_server` act on the same one.
        """
        self._registry = registry
        if not self._configs:
            return 0

        registered = 0
        for config in self._configs:
            try:
                registered += await self._bring_up(config, registry)
            except Exception as exc:
                # Anything at all: a missing executable, a protocol mismatch, a
                # server that exits immediately, a TLS failure, a listing that never
                # arrives.
                self.failures[config.name] = _leaf_message(exc)
                logger.error("MCP server %r did not start: %s", config.name, exc)
                continue
        return registered

    async def connect_server(
        self, config: ServerConfig, *, secret: str | None = None, auth: Any = None
    ) -> int:
        """Connect one server after startup and register its tools; returns how many.

        The caller persists `mcp.json` first (see `save_server`), then calls this. A
        failure is recorded in `failures` and re-raised, so the route can report it
        while `/health` still shows "configured but not connected". A reconnect of a
        name already present tears the old registration down first, so `register`'s
        collision guard does not refuse it. `auth` is the 2b seam: an
        `OAuthClientProvider` passed here reaches `streamablehttp_client(auth=...)`
        without any further rewrite.
        """
        if self._registry is None:
            raise ToolError("the MCP bridge has not started; there is no registry to register into")
        async with self._lock:
            await self._teardown(config.name)
            try:
                landed = await self._bring_up(
                    config, self._registry, secret=secret, auth=auth
                )
            except Exception as exc:
                reason = _leaf_message(exc)
                self.failures[config.name] = reason
                logger.error("MCP server %r did not connect: %s", config.name, exc)
                # Re-raised as a clean, unwrapped message so the admin route reports
                # the real reason rather than "unhandled errors in a taskgroup".
                raise ToolError(reason) from exc
            self.failures.pop(config.name, None)
            logger.info("MCP server %r connected, %d tool(s)", config.name, landed)
            return landed

    async def disconnect_server(self, name: str) -> None:
        """Unregister a server's tools and close its session and stack. Idempotent -
        disconnecting a server that is not connected is a no-op, so it is safe to
        call after a persisted removal whether or not the connect ever succeeded."""
        async with self._lock:
            await self._teardown(name)
            self.failures.pop(name, None)

    async def _bring_up(
        self,
        config: ServerConfig,
        registry: Registry,
        *,
        secret: str | None = None,
        auth: Any = None,
    ) -> int:
        # The transport is opened in a dedicated task that will also close it, so
        # anyio's "exit the cancel scope in the task that entered it" rule holds even
        # when the later disconnect comes from a different request task. See
        # `_ServerLink`.
        link = _ServerLink(self, config, secret=secret, auth=auth)
        session = await link.open()
        self._connections[config.name] = link
        self._sessions[config.name] = session
        # Registration is inside the same call as the connect, deliberately. It used
        # to sit after it in `start`, so a server that connected and then hung on
        # `list_tools` raised out and cost every *remaining* server its tools as well
        # - one slow server disabling the others.
        try:
            return await self._register(config, session, registry)
        except BaseException:
            # The session came up but its tools would not list; tear the whole
            # connection down (in its own task) so its child process is not orphaned.
            await self._teardown(config.name)
            raise

    async def _teardown(self, name: str) -> None:
        """Undo one server: unregister its tools, drop its session, close its stack.
        Runs under `self._lock` (its callers hold it). The stack is closed by the
        server's own task (`_ServerLink.close`), never inline here, so a disconnect
        can run from any request task without tripping anyio's same-task exit rule."""
        if self._registry is not None:
            for tool_name in self._registered.get(name, ()):
                self._registry.unregister(tool_name)
        self._registered.pop(name, None)
        self._sessions.pop(name, None)
        link = self._connections.pop(name, None)
        if link is not None:
            await link.close()

    async def _close_stack(self, name: str) -> None:
        """Close and drop one server's exit stack. Called only from that server's own
        task (`_ServerLink._run`), which is the task that entered the stack - so
        anyio's same-task exit rule is satisfied and the stdio child is terminated."""
        stack = self._stacks.pop(name, None)
        if stack is None:
            return
        try:
            await stack.aclose()
        except Exception:
            logger.exception("closing MCP session %r failed", name)

    async def _connect(
        self, config: ServerConfig, *, secret: str | None = None, auth: Any = None
    ) -> Any:
        """Open one session on its own stack. Raises, and the caller records it
        against this server; the partial stack is closed first so a failed connect
        leaks no child process."""
        try:
            from mcp import ClientSession as client_session
            from mcp import StdioServerParameters as stdio_params
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise ImportError(
                f"the 'mcp' extra is not installed (pip install 'daemon-ai[mcp]'): {exc}"
            ) from exc

        stack = AsyncExitStack()
        try:
            if config.is_remote:
                from mcp.client.streamable_http import streamablehttp_client

                # The first two of the three streams; the third is a session-id
                # getter this code has no use for. `headers` carries the bearer
                # secret; `auth` is the OAuth provider seam for 2b (None today).
                read, write, *_ = await stack.enter_async_context(
                    streamablehttp_client(
                        config.url,
                        headers=_bearer_headers(config, secret),
                        auth=auth,
                    )
                )
            else:
                read, write = await stack.enter_async_context(
                    stdio_client(
                        stdio_params(
                            # Resolved to an absolute path so a service's minimal
                            # PATH does not turn `uvx` into [Errno 2].
                            command=_resolve_command(config.command),
                            args=list(config.args),
                            # Not `os.environ`: the safe passthrough plus the owner's
                            # static env plus at most the one named secret. See
                            # `_stdio_env` for why the full environment is refused.
                            env=_stdio_env(config, secret),
                        )
                    )
                )
            session = await stack.enter_async_context(client_session(read, write))
            await asyncio.wait_for(session.initialize(), timeout=STARTUP_TIMEOUT)
        except BaseException:
            # A connect that got partway has a live child or socket on the stack;
            # close it before the error propagates rather than orphaning it.
            await stack.aclose()
            raise
        self._stacks[config.name] = stack
        return session

    async def _register(self, config: ServerConfig, session: Any, registry: Registry) -> int:
        listing = await asyncio.wait_for(session.list_tools(), timeout=STARTUP_TIMEOUT)
        count = 0
        landed: list[str] = []
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
            landed.append(local_name)
            count += 1
        # Recorded so `disconnect_server` unregisters exactly what this connect added.
        self._registered[config.name] = tuple(landed)
        logger.info("MCP server %r offered %d tool(s)", config.name, count)
        return count

    async def call(self, server: str, name: str, arguments: Mapping[str, Any]) -> str:
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
        self._registered.clear()
        # Each link closes its own stack in the task that opened it (`_close_stack`
        # swallows a stack that dies mid-close - shutdown has nothing left to break,
        # and raising would mask whatever else the lifespan is unwinding). Awaiting
        # them from the lifespan task is safe precisely because the exit happens over
        # in each server's task, not here.
        for link in list(self._connections.values()):
            await link.close()
        self._connections.clear()
        self._stacks.clear()


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
