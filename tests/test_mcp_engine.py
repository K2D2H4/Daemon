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

import stat
from pathlib import Path
from typing import Any

import pytest

from daemon.llm.base import ToolSpec
from daemon.mcp_catalog import CatalogEntry
from daemon.tools.base import Registry, Tool, ToolError
from daemon.tools.mcp import (
    McpBridge,
    ServerConfig,
    _bearer_headers,
    _stdio_env,
    load_config,
    remove_server,
    save_server,
    server_config_from_catalog,
)
from tests.test_mcp import RemoteTool, Session, write_config


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


# --- per-server exit stacks close independently -----------------------------


class _FakeStack:
    def __init__(self, name: str, closed: list[str]) -> None:
        self._name = name
        self._closed = closed

    async def aclose(self) -> None:
        self._closed.append(self._name)


async def test_disconnect_closes_only_that_servers_stack() -> None:
    bridge = McpBridge([])
    await bridge.start(Registry())
    closed: list[str] = []
    bridge._stacks["a"] = _FakeStack("a", closed)  # type: ignore[assignment]
    bridge._stacks["b"] = _FakeStack("b", closed)  # type: ignore[assignment]
    bridge._sessions["a"] = object()
    bridge._sessions["b"] = object()

    await bridge.disconnect_server("a")
    assert closed == ["a"]
    assert "b" in bridge._stacks and "a" not in bridge._stacks


async def test_aclose_closes_every_server_stack() -> None:
    bridge = McpBridge([])
    closed: list[str] = []
    bridge._stacks["a"] = _FakeStack("a", closed)  # type: ignore[assignment]
    bridge._stacks["b"] = _FakeStack("b", closed)  # type: ignore[assignment]
    await bridge.aclose()
    assert sorted(closed) == ["a", "b"]
    assert not bridge._stacks


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
    assert env.get("PATH") == "/usr/bin:/bin"  # but `uvx` must still be findable


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


# --- catalog -> ServerConfig -------------------------------------------------


def test_a_uvx_catalog_entry_becomes_a_stdio_config() -> None:
    entry = CatalogEntry(
        name="fetch", kind="uvx", description="", command="uvx", args=("mcp-server-fetch",)
    )
    config = server_config_from_catalog(entry)
    assert config.name == "fetch"
    assert config.command == "uvx" and config.args == ("mcp-server-fetch",)
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

    assert captured["command"] == "uvx"
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
