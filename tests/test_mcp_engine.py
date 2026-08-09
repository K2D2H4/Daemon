"""Phase 2a MCP engine: hot-reload, per-server lifecycle, secret indirection.

No network and no `uvx` download. `_connect` is monkeypatched to hand back an
in-process fake session with the SDK's shape - exactly as
`tests/test_mcp.py::test_start_connects_and_registers` already does - so what is
exercised is the bridge's own machinery: that connecting and disconnecting one
server leaves another untouched, that the named secret becomes exactly one child
env var (and Daemon's own provider keys never do), and that a url server is reached
with an `Authorization: Bearer` header built from that env var. The two "wiring"
tests go one step further and monkeypatch the SDK transport itself, to prove
`_connect` actually hands `headers=`/`auth=`/`env=` on rather than computing them
and dropping them.
"""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path
from typing import Any

import pytest

from daemon.llm.base import ToolSpec
from daemon.mcp_catalog import CatalogEntry
from daemon.tools import mcp as mcp_module
from daemon.tools.base import Registry, Tool, ToolError
from daemon.tools.mcp import (
    McpBridge,
    ServerConfig,
    _bearer_headers,
    _leaf_message,
    _resolve_command,
    _stdio_env,
    load_config,
    remove_server,
    save_server,
    server_config_from_catalog,
)
from tests.test_mcp import Listing, RemoteTool, Session, write_config


class _Dummy:
    """A minimal Tool, for exercising the registry without a bridge."""

    def __init__(self, name: str) -> None:
        self.spec = ToolSpec(name, "dummy", {"type": "object", "properties": {}})
        self.risk = "safe"

    def preview(self, arguments: Any) -> str:
        return self.spec.name

    async def run(self, arguments: Any) -> str:
        return "ok"


# --- Registry.unregister ----------------------------------------------------


def test_unregister_removes_a_registered_tool() -> None:
    registry = Registry()
    registry.register(_Dummy("a"))
    assert registry.unregister("a") is True
    assert registry.get("a") is None
    assert len(registry) == 0


def test_unregister_an_absent_tool_is_a_safe_noop() -> None:
    """A disconnect that half-happened must be safe to retry, so removing a name
    that is not there is not an error."""
    registry = Registry()
    assert registry.unregister("never-registered") is False


def test_unregister_then_register_the_same_name_succeeds() -> None:
    """The whole point of the inverse: a reconnecting server must be able to take
    its old name back, which `register`'s collision guard would otherwise refuse."""
    registry = Registry()
    registry.register(_Dummy("notes__search"))
    registry.unregister("notes__search")
    registry.register(_Dummy("notes__search"))  # would raise before unregister
    assert registry.names() == ("notes__search",)


# --- per-server connect / disconnect isolation ------------------------------


async def _started_bridge(
    monkeypatch: pytest.MonkeyPatch, sessions: dict[str, Session]
) -> tuple[McpBridge, Registry]:
    bridge = McpBridge([])
    registry = Registry()
    await bridge.start(registry)  # no configs; sets the registry the hot path uses

    async def connect(config: ServerConfig, *, secret: Any = None, auth: Any = None) -> Any:
        return sessions[config.name]

    monkeypatch.setattr(bridge, "_connect", connect)
    return bridge, registry


async def test_connecting_a_server_registers_its_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge, registry = await _started_bridge(monkeypatch, {"alpha": Session([RemoteTool("read")])})
    landed = await bridge.connect_server(ServerConfig(name="alpha", command="x"))
    assert landed == 1
    assert registry.names() == ("alpha__read",)
    assert isinstance(registry.get("alpha__read"), Tool)


async def test_disconnecting_one_server_leaves_another_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = {
        "alpha": Session([RemoteTool("read"), RemoteTool("write")]),
        "beta": Session([RemoteTool("search")]),
    }
    bridge, registry = await _started_bridge(monkeypatch, sessions)
    await bridge.connect_server(ServerConfig(name="alpha", command="x"))
    await bridge.connect_server(ServerConfig(name="beta", command="y"))
    assert set(registry.names()) == {"alpha__read", "alpha__write", "beta__search"}

    await bridge.disconnect_server("alpha")
    assert registry.names() == ("beta__search",)
    assert "alpha" not in bridge._sessions
    assert "beta" in bridge._sessions


async def test_a_failed_hot_connect_is_recorded_and_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Persist-first means the mcp.json entry stays; the failure has to surface on
    /health as 'configured but not connected', which reads `bridge.failures`."""
    bridge = McpBridge([])
    await bridge.start(Registry())

    async def connect(config: ServerConfig, *, secret: Any = None, auth: Any = None) -> Any:
        raise RuntimeError("it exited immediately")

    monkeypatch.setattr(bridge, "_connect", connect)
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
        await bridge.connect_server(ServerConfig(name="oops", command="x"))
    assert bridge.failures.get("oops") == "it exited immediately"


async def test_a_successful_connect_clears_a_stale_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge, _ = await _started_bridge(monkeypatch, {"notes": Session([RemoteTool("read")])})
    bridge.failures["notes"] = "would not start last time"
    await bridge.connect_server(ServerConfig(name="notes", command="x"))
    assert "notes" not in bridge.failures


async def test_reconnecting_replaces_rather_than_colliding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second connect of the same name must not raise on the registry's collision
    guard; it tears the old registration down first."""
    bridge, registry = await _started_bridge(
        monkeypatch, {"notes": Session([RemoteTool("read")])}
    )
    await bridge.connect_server(ServerConfig(name="notes", command="x"))
    await bridge.connect_server(ServerConfig(name="notes", command="x"))
    assert registry.names() == ("notes__read",)


async def test_connect_before_start_refuses() -> None:
    """`connect_server` needs the registry `start` handed it. Called before start,
    it must refuse rather than silently register into nothing."""
    bridge = McpBridge([])
    with pytest.raises(ToolError):
        await bridge.connect_server(ServerConfig(name="x", command="y"))


async def test_disconnecting_an_unknown_server_is_a_noop() -> None:
    bridge = McpBridge([])
    await bridge.start(Registry())
    await bridge.disconnect_server("never-connected")  # must not raise


async def test_is_connected_tracks_live_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/servers` reads `is_connected`, so it must follow the live session, not
    `failures`: false before the connect, true after, false again after disconnect."""
    bridge, _ = await _started_bridge(monkeypatch, {"notes": Session([RemoteTool("read")])})
    assert not bridge.is_connected("notes")
    await bridge.connect_server(ServerConfig(name="notes", command="x"))
    assert bridge.is_connected("notes")
    assert bridge.connected_names() == ("notes",)
    await bridge.disconnect_server("notes")
    assert not bridge.is_connected("notes")


# --- oauth servers reconnect at startup through the provider factory ---------


async def test_startup_reconnects_an_oauth_server_with_its_stored_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persisted auth="oauth" server must reconnect at startup through its stored
    token, with no browser step: the factory builds the non-interactive provider, and
    the bridge hands it to `_connect` as `auth=`. A valid token connects silently."""
    sentinel = object()
    seen: dict[str, Any] = {}

    def factory(config: ServerConfig) -> Any:
        seen["config"] = config
        return sentinel

    bridge = McpBridge(
        [ServerConfig(name="notion", url="https://mcp.notion.com/mcp", auth="oauth")],
        oauth_provider_factory=factory,
    )

    async def connect(config: ServerConfig, *, secret: Any = None, auth: Any = None) -> Any:
        seen["auth"] = auth
        return Session([RemoteTool("search")])

    monkeypatch.setattr(bridge, "_connect", connect)
    registry = Registry()
    landed = await bridge.start(registry)

    assert landed == 1
    assert seen["config"].name == "notion"  # the factory was consulted for the oauth server
    assert seen["auth"] is sentinel  # its provider reached the transport as auth=
    assert bridge.is_connected("notion")  # a live session, no browser step
    assert "notion" not in bridge.failures


async def test_startup_oauth_reconnect_without_a_valid_token_fails_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing or expired-unrefreshable token makes the non-interactive provider's
    redirect handler raise; the connect must degrade to configured-but-not-connected,
    never a hang or a raise out of `start`."""
    bridge = McpBridge(
        [ServerConfig(name="notion", url="https://mcp.notion.com/mcp", auth="oauth")],
        oauth_provider_factory=lambda config: object(),
    )

    async def connect(config: ServerConfig, *, secret: Any = None, auth: Any = None) -> Any:
        raise ToolError("notion: reauthorize it in the admin")

    monkeypatch.setattr(bridge, "_connect", connect)
    landed = await bridge.start(Registry())  # must not raise

    assert landed == 0
    assert not bridge.is_connected("notion")
    assert "notion" in bridge.failures
    assert "reauthorize" in bridge.failures["notion"]


async def test_a_non_oauth_server_never_consults_the_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory fires only for auth="oauth"; a plain url or stdio server connects
    with auth=None exactly as before, so the seam cannot change their behaviour."""
    calls: list[ServerConfig] = []
    bridge = McpBridge(
        [ServerConfig(name="fetch", command="uvx", args=("mcp-server-fetch",))],
        oauth_provider_factory=lambda config: calls.append(config) or object(),
    )
    seen: dict[str, Any] = {}

    async def connect(config: ServerConfig, *, secret: Any = None, auth: Any = None) -> Any:
        seen["auth"] = auth
        return Session([RemoteTool("fetch")])

    monkeypatch.setattr(bridge, "_connect", connect)
    await bridge.start(Registry())

    assert calls == []  # factory untouched for a non-oauth server
    assert seen["auth"] is None


# --- per-server exit stacks close independently -----------------------------


class _FakeStack:
    def __init__(self, name: str, closed: list[str]) -> None:
        self._name = name
        self._closed = closed

    async def aclose(self) -> None:
        self._closed.append(self._name)


def _stack_recording_connect(
    bridge: McpBridge, closed: list[str]
) -> Any:
    """A fake `_connect` that puts a recording stack in `bridge._stacks`, the way the
    real `_connect` leaves one there - so the connection's own task has a stack to
    close and the test can watch it be closed."""

    async def connect(config: ServerConfig, *, secret: Any = None, auth: Any = None) -> Any:
        bridge._stacks[config.name] = _FakeStack(config.name, closed)  # type: ignore[assignment]
        return Session([])

    return connect


async def test_disconnect_closes_only_that_servers_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = McpBridge([])
    await bridge.start(Registry())
    closed: list[str] = []
    monkeypatch.setattr(bridge, "_connect", _stack_recording_connect(bridge, closed))
    await bridge.connect_server(ServerConfig(name="a", command="x"))
    await bridge.connect_server(ServerConfig(name="b", command="y"))

    await bridge.disconnect_server("a")
    assert closed == ["a"]
    assert "b" in bridge._stacks and "a" not in bridge._stacks


async def test_aclose_closes_every_server_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = McpBridge([])
    await bridge.start(Registry())
    closed: list[str] = []
    monkeypatch.setattr(bridge, "_connect", _stack_recording_connect(bridge, closed))
    await bridge.connect_server(ServerConfig(name="a", command="x"))
    await bridge.connect_server(ServerConfig(name="b", command="y"))

    await bridge.aclose()
    assert sorted(closed) == ["a", "b"]
    assert not bridge._stacks


async def test_a_server_is_torn_down_in_the_task_that_opened_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix, proven rather than assumed: anyio requires the stdio/session context
    to be exited in the task it was entered in. A connect from one request task and a
    disconnect from a *different* request task - the split that raised 'exit cancel
    scope in a different task' and orphaned the child - must still tear down in the
    single owning task, and a connect-then-aclose must too."""
    import mcp
    import mcp.client.stdio as stdio

    class _AffinityStdioCM:
        """Records the task its enter and exit run in, so the test can assert they
        match. A plain object() stands in for each of the (read, write) streams."""

        def __init__(self, record: dict[str, Any]) -> None:
            self._record = record

        async def __aenter__(self) -> tuple[Any, Any]:
            self._record["enter"] = asyncio.current_task()
            return (object(), object())

        async def __aexit__(self, *exc: Any) -> bool:
            self._record["exit"] = asyncio.current_task()
            return False

    class _AffinitySessionCM:
        async def __aenter__(self) -> _AffinitySessionCM:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def initialize(self) -> None:
            return None

        async def list_tools(self) -> Any:
            return Listing([])

    def wire(record: dict[str, Any]) -> None:
        monkeypatch.setattr(mcp, "StdioServerParameters", lambda **_: object())
        monkeypatch.setattr(stdio, "stdio_client", lambda _p: _AffinityStdioCM(record))
        monkeypatch.setattr(mcp, "ClientSession", lambda r, w: _AffinitySessionCM())

    # connect-then-disconnect, deliberately from two different request tasks.
    record: dict[str, Any] = {}
    wire(record)
    bridge = McpBridge([])
    await bridge.start(Registry())
    config = ServerConfig(name="t", command="uvx")
    await asyncio.create_task(bridge.connect_server(config))
    enter_task = record["enter"]
    assert enter_task is not None
    assert enter_task is not asyncio.current_task()  # a dedicated task, not the caller
    assert "exit" not in record  # still open
    await asyncio.create_task(bridge.disconnect_server("t"))
    assert record["exit"] is enter_task  # exited where it was entered, not in the caller

    # connect-then-aclose closes it in the entering task too.
    record = {}
    wire(record)
    bridge = McpBridge([])
    await bridge.start(Registry())
    await asyncio.create_task(bridge.connect_server(config))
    await bridge.aclose()
    assert record["exit"] is record["enter"]


# --- secret indirection: stdio env ------------------------------------------


def test_stdio_env_injects_only_the_named_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """The security property this whole indirection exists for: a curated MCP
    server gets the one secret the owner named for it and never Daemon's own
    provider keys, which is what merging `os.environ` would have handed it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-daemons-own-secret")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-123")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    config = ServerConfig(name="t", command="uvx", args=("x",), key_env="TAVILY_API_KEY")

    env = _stdio_env(config)

    assert env["TAVILY_API_KEY"] == "tvly-123"
    assert "ANTHROPIC_API_KEY" not in env  # the leak we refuse
    # The base PATH is preserved and the user bin dirs are added, so `uvx` is findable.
    assert "/usr/bin" in env["PATH"].split(os.pathsep)
    assert os.path.expanduser("~/.local/bin") in env["PATH"].split(os.pathsep)


def test_stdio_env_does_not_carry_a_secret_it_was_not_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-daemons-own-secret")
    config = ServerConfig(name="t", command="uvx", key_env="NOT_SET_ANYWHERE")
    env = _stdio_env(config)
    assert "NOT_SET_ANYWHERE" not in env
    assert "ANTHROPIC_API_KEY" not in env


def test_an_explicit_secret_overrides_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The connect flow can hand the value straight through (the owner just typed
    it) rather than round-tripping it via the process environment."""
    monkeypatch.setenv("TAVILY_API_KEY", "from-env")
    config = ServerConfig(name="t", command="uvx", key_env="TAVILY_API_KEY")
    assert _stdio_env(config, secret="explicit")["TAVILY_API_KEY"] == "explicit"


def test_static_env_from_mcp_json_is_still_passed() -> None:
    config = ServerConfig(name="t", command="uvx", env={"FOO": "bar"})
    assert _stdio_env(config)["FOO"] == "bar"


def test_stdio_env_passes_declared_passthrough_names_but_not_others(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that runs its own OAuth (workspace/Google) needs specific env vars the
    allowlist omits; `env_passthrough` names them. The allowlist still holds: a var it
    did not name - here Daemon's own provider key - is never forwarded."""
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "gcid")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "gcsecret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-daemons-own-secret")
    config = ServerConfig(
        name="google",
        command="uvx",
        env_passthrough=("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET"),
    )

    env = _stdio_env(config)

    assert env["GOOGLE_OAUTH_CLIENT_ID"] == "gcid"
    assert env["GOOGLE_OAUTH_CLIENT_SECRET"] == "gcsecret"
    assert "ANTHROPIC_API_KEY" not in env  # not named, so still refused


def test_stdio_env_omits_an_unset_passthrough_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A declared name that is not set stays absent rather than becoming an empty
    string - the honest 'not configured' the secret path already makes."""
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID", raising=False)
    config = ServerConfig(name="g", command="uvx", env_passthrough=("GOOGLE_OAUTH_CLIENT_ID",))
    assert "GOOGLE_OAUTH_CLIENT_ID" not in _stdio_env(config)


# --- finding uvx under a service's minimal PATH -----------------------------


def test_stdio_env_adds_the_user_bin_dirs_to_path() -> None:
    """A LaunchAgent/systemd PATH omits ~/.local/bin, where uvx lives, so the child's
    PATH is augmented or a uvx server dies with [Errno 2]."""
    path = _stdio_env(ServerConfig(name="t", command="uvx"))["PATH"]
    assert os.path.expanduser("~/.local/bin") in path.split(os.pathsep)


def test_resolve_command_finds_a_launcher_in_a_user_bin_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`uvx` on a bare service PATH is unfindable; resolution searches the user bin
    dirs and hands subprocess an absolute path (POSIX looks the executable up on the
    parent's PATH, so augmenting only the child env would not be enough)."""
    fake = tmp_path / "uvx"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(mcp_module, "_EXTRA_BIN_DIRS", (str(tmp_path),))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # a service-like minimal PATH
    assert _resolve_command("uvx") == str(fake)


def test_resolve_command_leaves_an_unknown_command_to_fail_by_name() -> None:
    missing = "definitely-not-a-real-binary-xyz"
    assert _resolve_command(missing) == missing


def test_resolve_command_passes_an_absolute_path_through() -> None:
    assert _resolve_command("/usr/bin/env") == "/usr/bin/env"


def test_leaf_message_unwraps_a_taskgroup_exception_group() -> None:
    """A url connect failure arrives wrapped in an anyio ExceptionGroup whose own
    message is the useless 'unhandled errors in a taskgroup'; the real 401/closed
    stream is a leaf. `_leaf_message` reports the leaf."""
    inner = RuntimeError("401 Unauthorized")
    group = BaseExceptionGroup("unhandled errors in a task group", [inner])
    assert _leaf_message(group) == "401 Unauthorized"
    assert _leaf_message(RuntimeError("plain")) == "plain"


# --- secret indirection: bearer header --------------------------------------


def test_bearer_header_is_built_from_the_named_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-999")
    config = ServerConfig(name="t", url="https://mcp.example.com", key_env="TAVILY_API_KEY")
    assert _bearer_headers(config) == {"Authorization": "Bearer tvly-999"}


def test_a_url_server_with_no_key_env_gets_no_header() -> None:
    config = ServerConfig(name="t", url="https://mcp.example.com")
    assert _bearer_headers(config) is None


def test_a_named_but_unset_secret_yields_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """A url server whose key is not in `.env` connects without a header rather than
    with `Bearer None` - the failure is 'no secret configured', not a broken header."""
    config = ServerConfig(name="t", url="https://x", key_env="NOT_SET_ANYWHERE")
    assert _bearer_headers(config) is None


# --- config carries the env-var name (never the value) ----------------------


async def test_key_env_name_is_read_from_mcp_json(data_dir: Path) -> None:
    write_config(
        data_dir,
        {"servers": {"tavily": {"url": "https://x/mcp", "key_env": "TAVILY_API_KEY"}}},
    )
    (config,) = load_config(data_dir).servers
    assert config.key_env == "TAVILY_API_KEY"


async def test_key_env_defaults_to_empty_when_absent(data_dir: Path) -> None:
    write_config(data_dir, {"servers": {"fs": {"command": "uvx", "args": ["x"]}}})
    (config,) = load_config(data_dir).servers
    assert config.key_env == ""


# --- persisting mcp.json (write first, then connect) ------------------------


async def test_save_server_writes_the_env_name_never_the_secret(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mcp.json must stay shareable: the whole indirection is pointless if the
    secret value lands in the config file next to its name."""
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret-value")
    save_server(data_dir, ServerConfig(name="tavily", url="https://x", key_env="TAVILY_API_KEY"))
    text = (data_dir / "mcp.json").read_text(encoding="utf-8")
    assert "TAVILY_API_KEY" in text  # the name is stored
    assert "tvly-secret-value" not in text  # the value never is


async def test_save_server_is_additive(data_dir: Path) -> None:
    save_server(data_dir, ServerConfig(name="a", command="uvx", args=("x",)))
    save_server(data_dir, ServerConfig(name="b", url="https://x/mcp"))
    assert {s.name for s in load_config(data_dir).servers} == {"a", "b"}


async def test_save_server_round_trips_through_load_config(data_dir: Path) -> None:
    save_server(
        data_dir,
        ServerConfig(
            name="fetch",
            command="uvx",
            args=("mcp-server-fetch",),
            safe=frozenset({"fetch"}),
        ),
    )
    (config,) = load_config(data_dir).servers
    assert config.command == "uvx"
    assert config.args == ("mcp-server-fetch",)
    assert config.safe == frozenset({"fetch"})


async def test_env_passthrough_round_trips_through_mcp_json(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A restart rebuilds servers from mcp.json, so the passthrough names must survive
    the write/read - otherwise `google` reconnects blind and its OAuth vars vanish.
    Names are stored (like key_env); no secret value is written."""
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "gcsecret")
    save_server(
        data_dir,
        ServerConfig(
            name="google",
            command="uvx",
            args=("workspace-mcp",),
            env_passthrough=("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET"),
        ),
    )
    text = (data_dir / "mcp.json").read_text(encoding="utf-8")
    assert "GOOGLE_OAUTH_CLIENT_SECRET" in text  # the name persists
    assert "gcsecret" not in text  # the value never does
    (config,) = load_config(data_dir).servers
    assert config.env_passthrough == ("GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET")


async def test_save_server_writes_owner_only(data_dir: Path) -> None:
    save_server(data_dir, ServerConfig(name="a", command="uvx"))
    mode = (data_dir / "mcp.json").stat().st_mode & 0o777
    assert mode == stat.S_IRUSR | stat.S_IWUSR  # 0o600


async def test_remove_server_drops_only_that_block(data_dir: Path) -> None:
    save_server(data_dir, ServerConfig(name="a", command="uvx"))
    save_server(data_dir, ServerConfig(name="b", url="https://x/mcp"))
    assert remove_server(data_dir, "a") is True
    assert {s.name for s in load_config(data_dir).servers} == {"b"}


async def test_remove_absent_server_reports_false(data_dir: Path) -> None:
    save_server(data_dir, ServerConfig(name="a", command="uvx"))
    assert remove_server(data_dir, "never-there") is False


# --- an unparseable mcp.json is never silently overwritten -------------------


async def test_save_server_refuses_to_clobber_an_unparseable_config(data_dir: Path) -> None:
    """A trailing comma - the exact case `load_config`'s docstring calls out - must
    not cost the owner every hand-edited block: `save_server` raises rather than
    rewriting the file with only the new server."""
    bad = '{"servers": {"fs": {"command": "uvx"},}}'  # trailing comma
    (data_dir / "mcp.json").write_text(bad, encoding="utf-8")
    with pytest.raises(ToolError):
        save_server(data_dir, ServerConfig(name="new", url="https://x/mcp"))
    # The file is untouched - nothing was discarded.
    assert (data_dir / "mcp.json").read_text(encoding="utf-8") == bad


async def test_remove_server_refuses_to_clobber_an_unparseable_config(data_dir: Path) -> None:
    bad = '{"servers": {"fs": {"command": "uvx"},}}'
    (data_dir / "mcp.json").write_text(bad, encoding="utf-8")
    with pytest.raises(ToolError):
        remove_server(data_dir, "fs")
    assert (data_dir / "mcp.json").read_text(encoding="utf-8") == bad


async def test_save_server_refuses_a_config_that_is_not_an_object(data_dir: Path) -> None:
    """A file that parses but is a list, not an object, is just as unsafe to
    overwrite - it is still something the owner wrote."""
    (data_dir / "mcp.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ToolError):
        save_server(data_dir, ServerConfig(name="new", command="uvx"))


async def test_save_server_starts_fresh_when_the_file_is_absent(data_dir: Path) -> None:
    """Absence is not corruption: the first save creates the file."""
    assert not (data_dir / "mcp.json").exists()
    save_server(data_dir, ServerConfig(name="a", command="uvx"))
    assert {s.name for s in load_config(data_dir).servers} == {"a"}


# --- catalog -> ServerConfig -------------------------------------------------


def test_a_uvx_catalog_entry_becomes_a_stdio_config() -> None:
    entry = CatalogEntry(
        name="fetch", kind="uvx", description="", command="uvx", args=("mcp-server-fetch",)
    )
    config = server_config_from_catalog(entry)
    assert config.name == "fetch"
    # uvx servers are pinned to the daemon's own mcp (see `_mcp_pin`); the tool
    # itself is still the last argument.
    assert config.command == "uvx"
    assert config.args[0] == "--with" and config.args[-1] == "mcp-server-fetch"
    assert not config.is_remote


def test_a_url_catalog_entry_becomes_a_remote_config_with_its_key_env() -> None:
    entry = CatalogEntry(
        name="tavily",
        kind="url",
        description="",
        url="https://x/mcp",
        key_env="TAVILY_API_KEY",
        auth="key",
    )
    config = server_config_from_catalog(entry)
    assert config.is_remote and config.url == "https://x/mcp"
    assert config.key_env == "TAVILY_API_KEY"


# --- wiring: the SDK actually receives headers / auth / env ------------------


class _FakeTransportCM:
    """Async CM standing in for `streamablehttp_client(...)`: yields the (read,
    write, get-session-id) triple the SDK returns."""

    async def __aenter__(self) -> tuple[Any, Any, Any]:
        return (object(), object(), lambda: None)

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakeStdioCM:
    """Async CM standing in for `stdio_client(...)`: yields the (read, write) pair
    the stdio transport returns (no session-id getter, unlike streamable-HTTP)."""

    def __init__(self, closed: list[bool] | None = None) -> None:
        self._closed = closed

    async def __aenter__(self) -> tuple[Any, Any]:
        return (object(), object())

    async def __aexit__(self, *exc: Any) -> bool:
        if self._closed is not None:
            self._closed.append(True)
        return False


class _FakeSessionCM:
    """Async CM standing in for `ClientSession(read, write)` and the session it
    yields - the same object plays both roles."""

    async def __aenter__(self) -> _FakeSessionCM:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def initialize(self) -> None:
        return None


async def test_connect_hands_headers_and_auth_to_the_streamable_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The net-new code path: `_connect` computes a bearer header and must actually
    pass it (and the OAuth `auth=` seam) to the SDK, not compute and drop it."""
    import mcp
    import mcp.client.streamable_http as streamable

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-abc")
    captured: dict[str, Any] = {}

    def fake_client(url: str, *, headers: Any = None, auth: Any = None, **_: Any) -> Any:
        captured.update(url=url, headers=headers, auth=auth)
        return _FakeTransportCM()

    monkeypatch.setattr(streamable, "streamablehttp_client", fake_client)
    monkeypatch.setattr(mcp, "ClientSession", lambda read, write: _FakeSessionCM())

    bridge = McpBridge([])
    sentinel = object()
    config = ServerConfig(name="tavily", url="https://x/mcp", key_env="TAVILY_API_KEY")
    session = await bridge._connect(config, auth=sentinel)

    assert isinstance(session, _FakeSessionCM)
    assert captured["url"] == "https://x/mcp"
    assert captured["headers"] == {"Authorization": "Bearer tvly-abc"}
    assert captured["auth"] is sentinel  # the 2b OAuth provider seam
    assert "tavily" in bridge._stacks  # the per-server stack was kept


async def test_connect_hands_the_minimal_env_to_the_stdio_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mcp
    import mcp.client.stdio as stdio

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-daemons-own")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-xyz")
    captured: dict[str, Any] = {}

    def fake_params(*, command: str, args: Any, env: Any) -> Any:
        captured.update(command=command, args=args, env=env)
        return object()

    def fake_client(params: Any) -> Any:
        return _FakeStdioCM()

    monkeypatch.setattr(mcp, "StdioServerParameters", fake_params)
    monkeypatch.setattr(stdio, "stdio_client", fake_client)
    monkeypatch.setattr(mcp, "ClientSession", lambda read, write: _FakeSessionCM())

    bridge = McpBridge([])
    config = ServerConfig(name="t", command="uvx", args=("x",), key_env="TAVILY_API_KEY")
    await bridge._connect(config)

    # Resolved to an absolute path where uvx is installed, or left bare if not found.
    assert captured["command"].endswith("uvx")
    assert captured["env"]["TAVILY_API_KEY"] == "tvly-xyz"
    assert "ANTHROPIC_API_KEY" not in captured["env"]


async def test_connect_closes_the_stack_if_the_session_will_not_initialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connect that fails mid-way must not leak its stdio child: the partial stack
    is closed before the error propagates."""
    import mcp
    import mcp.client.stdio as stdio

    closed: list[bool] = []

    class _BrokenSessionCM(_FakeSessionCM):
        async def initialize(self) -> None:
            raise RuntimeError("handshake failed")

    monkeypatch.setattr(mcp, "StdioServerParameters", lambda **_: object())
    monkeypatch.setattr(stdio, "stdio_client", lambda _params: _FakeStdioCM(closed))
    monkeypatch.setattr(mcp, "ClientSession", lambda read, write: _BrokenSessionCM())

    bridge = McpBridge([])
    config = ServerConfig(name="t", command="uvx")
    with pytest.raises(RuntimeError):
        await bridge._connect(config)
    assert closed == [True]  # the transport was torn down
    assert "t" not in bridge._stacks  # and no dangling stack was kept
