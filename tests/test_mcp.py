"""The MCP bridge: config reading, registration, and the policy applying to it.

No subprocess and no socket. `McpBridge.call` is exercised against a stand-in
session with the SDK's shape, which is the part that matters here - that a remote
tool wears the local `Tool` protocol, gets namespaced, and is guarded unless the
owner said otherwise. Whether the SDK can start `npx` is the SDK's test.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from daemon.llm.base import ToolSpec
from daemon.memory.store import Store
from daemon.tools.base import Registry, Tool, ToolError
from daemon.tools.mcp import McpBridge, McpTool, ServerConfig, load_config
from daemon.tools.policy import ToolPolicy

OWNER = "42"


def write_config(data_dir: Path, payload: Any) -> None:
    (data_dir / "mcp.json").write_text(json.dumps(payload), encoding="utf-8")


# --- config -----------------------------------------------------------------


async def test_no_config_file_means_no_servers(data_dir: Path) -> None:
    assert load_config(data_dir).servers == []


async def test_a_stdio_server_is_read(data_dir: Path) -> None:
    write_config(
        data_dir,
        {
            "servers": {
                "fs": {
                    "command": "npx",
                    "args": ["-y", "server-filesystem"],
                    "env": {"A": "b"},
                }
            }
        },
    )
    (config,) = load_config(data_dir).servers
    assert config.name == "fs"
    assert config.command == "npx"
    assert config.args == ("-y", "server-filesystem")
    assert dict(config.env) == {"A": "b"}
    assert not config.is_remote


async def test_a_remote_server_is_read(data_dir: Path) -> None:
    write_config(data_dir, {"servers": {"notes": {"url": "https://example.com/mcp"}}})
    (config,) = load_config(data_dir).servers
    assert config.is_remote and config.url == "https://example.com/mcp"


async def test_broken_json_is_reported_and_skipped(
    data_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """This file is hand-edited. A trailing comma in it must not stop the daemon
    starting, because the user would have no way to find out why."""
    (data_dir / "mcp.json").write_text('{"servers": {"a": {},}}', encoding="utf-8")
    with caplog.at_level("ERROR"):
        assert load_config(data_dir).servers == []
    assert any("could not be read" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    "block",
    [
        {},  # neither command nor url
        {"command": "npx", "url": "https://x/mcp"},  # both
        "not-an-object",
    ],
)
async def test_an_unusable_server_block_is_skipped(
    data_dir: Path, block: Any, caplog: pytest.LogCaptureFixture
) -> None:
    write_config(data_dir, {"servers": {"a": block}})
    with caplog.at_level("ERROR"):
        assert load_config(data_dir).servers == []


async def test_a_server_name_may_not_contain_the_separator(
    data_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Otherwise server `a__b` tool `c` and server `a` tool `b__c` are the same
    registered name."""
    write_config(data_dir, {"servers": {"a__b": {"command": "x"}}})
    with caplog.at_level("ERROR"):
        assert load_config(data_dir).servers == []


async def test_a_bad_block_does_not_take_a_good_one_with_it(data_dir: Path) -> None:
    write_config(data_dir, {"servers": {"bad": {}, "good": {"command": "npx"}}})
    config = load_config(data_dir)
    assert [server.name for server in config.servers] == ["good"]
    assert "bad" in config.rejected


async def test_a_rejected_block_reaches_health(data_dir: Path) -> None:
    """Skipping it quietly would make a typo in mcp.json indistinguishable from a
    model choosing not to use that tool - the failure `_tools_health` exists for."""
    write_config(data_dir, {"servers": {"typo": {}}})
    bridge = McpBridge(load_config(data_dir))
    assert await bridge.start(Registry()) == 0
    assert "typo" in bridge.failures
    assert "command" in bridge.failures["typo"]


async def test_unreadable_json_reaches_health_too(data_dir: Path) -> None:
    (data_dir / "mcp.json").write_text("{oops", encoding="utf-8")
    bridge = McpBridge(load_config(data_dir))
    assert "mcp.json" in bridge.failures


async def test_a_file_with_no_servers_key_is_reported(
    data_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_config(data_dir, {"mcpServers": {"a": {"command": "x"}}})
    with caplog.at_level("ERROR"):
        assert load_config(data_dir).servers == []
    assert any("no 'servers'" in record.message for record in caplog.records)


# --- registration -----------------------------------------------------------


class Listing:
    def __init__(self, tools: list[Any]) -> None:
        self.tools = tools


class RemoteTool:
    def __init__(self, name: str, schema: Any = None, description: str = "") -> None:
        self.name = name
        self.description = description
        self.inputSchema = schema


class Result:
    def __init__(
        self, content: Any = None, *, structured: Any = None, is_error: bool = False
    ) -> None:
        self.content = content
        self.structuredContent = structured
        self.isError = is_error


class Block:
    def __init__(self, text: str) -> None:
        self.text = text
        self.type = "text"


class Session:
    """Stands in for `mcp.ClientSession` - only the three methods used."""

    def __init__(self, tools: list[Any], result: Any = None) -> None:
        self._tools = tools
        self._result = result or Result([Block("done")])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> Listing:
        return Listing(self._tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        return self._result


async def bridge_with(config: ServerConfig, session: Session) -> tuple[McpBridge, Registry]:
    bridge = McpBridge([config])
    bridge._sessions[config.name] = session
    registry = Registry()
    await bridge._register(config, session, registry)
    return bridge, registry


async def test_remote_tools_are_namespaced_by_server() -> None:
    """Two servers legitimately both offer `search`, and the registry refuses a
    collision rather than shadowing one."""
    config = ServerConfig(name="notes", command="x")
    _, registry = await bridge_with(config, Session([RemoteTool("search")]))
    assert registry.names() == ("notes__search",)


async def test_a_remote_tool_is_guarded_by_default() -> None:
    """An MCP server is one more way to reach this machine, so its tools go through
    the same policy as run_command."""
    config = ServerConfig(name="notes", command="x")
    _, registry = await bridge_with(config, Session([RemoteTool("write")]))
    tool = registry.get("notes__write")
    assert tool is not None and tool.risk == "guarded"


async def test_a_tool_the_owner_marked_safe_skips_approval() -> None:
    config = ServerConfig(name="notes", command="x", safe=frozenset({"search"}))
    _, registry = await bridge_with(config, Session([RemoteTool("search"), RemoteTool("write")]))
    assert registry.get("notes__search").risk == "safe"  # type: ignore[union-attr]
    assert registry.get("notes__write").risk == "guarded"  # type: ignore[union-attr]


async def test_a_remote_schema_is_forwarded_untouched() -> None:
    """Rewriting a server's schema here is how a valid call starts being rejected."""
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    config = ServerConfig(name="notes", command="x")
    _, registry = await bridge_with(config, Session([RemoteTool("search", schema)]))
    assert registry.get("notes__search").spec.parameters == schema  # type: ignore[union-attr]


async def test_a_missing_schema_becomes_an_empty_object() -> None:
    """Some servers omit it, and every provider needs *something* shaped like a
    schema or the request is rejected before it leaves."""
    config = ServerConfig(name="notes", command="x")
    _, registry = await bridge_with(config, Session([RemoteTool("ping", None)]))
    assert registry.get("notes__ping").spec.parameters == {  # type: ignore[union-attr]
        "type": "object",
        "properties": {},
    }


async def test_a_remote_tool_satisfies_the_local_protocol() -> None:
    config = ServerConfig(name="notes", command="x")
    _, registry = await bridge_with(config, Session([RemoteTool("search")]))
    assert isinstance(registry.get("notes__search"), Tool)


async def test_a_colliding_tool_name_is_logged_not_fatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = ServerConfig(name="notes", command="x")
    bridge = McpBridge([config])
    registry = Registry()
    session = Session([RemoteTool("search"), RemoteTool("search")])
    with caplog.at_level("ERROR"):
        landed = await bridge._register(config, session, registry)
    assert landed == 1
    assert len(registry) == 1


# --- calling ----------------------------------------------------------------


async def test_calling_a_remote_tool_returns_its_text() -> None:
    config = ServerConfig(name="notes", command="x")
    session = Session([RemoteTool("search")], Result([Block("두 건 찾았어")]))
    bridge, registry = await bridge_with(config, session)
    out = await registry.get("notes__search").run({"q": "발표"})  # type: ignore[union-attr]
    assert out == "두 건 찾았어"
    assert session.calls == [("search", {"q": "발표"})]


async def test_structured_content_is_preferred_when_present() -> None:
    config = ServerConfig(name="notes", command="x")
    session = Session([RemoteTool("search")], Result(structured={"hits": 2}))
    bridge, registry = await bridge_with(config, session)
    out = await registry.get("notes__search").run({})  # type: ignore[union-attr]
    assert json.loads(out) == {"hits": 2}


async def test_a_server_side_error_becomes_a_tool_error() -> None:
    """So the runner records it as a failed call and the model is told, rather than
    the turn dying."""
    config = ServerConfig(name="notes", command="x")
    session = Session([RemoteTool("search")], Result([Block("no such note")], is_error=True))
    bridge, registry = await bridge_with(config, session)
    with pytest.raises(ToolError) as caught:
        await registry.get("notes__search").run({})  # type: ignore[union-attr]
    assert "no such note" in str(caught.value)


async def test_a_raising_session_becomes_a_tool_error() -> None:
    class Broken(Session):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            raise RuntimeError("the pipe is closed")

    config = ServerConfig(name="notes", command="x")
    bridge, registry = await bridge_with(config, Broken([RemoteTool("search")]))
    with pytest.raises(ToolError) as caught:
        await registry.get("notes__search").run({})  # type: ignore[union-attr]
    assert "the pipe is closed" in str(caught.value)


async def test_calling_a_disconnected_server_is_a_tool_error() -> None:
    bridge = McpBridge([])
    tool = McpTool(
        bridge, "gone", "search", ToolSpec("gone__search", "x", {"type": "object"}), "guarded"
    )
    with pytest.raises(ToolError) as caught:
        await tool.run({})
    assert "not connected" in str(caught.value)


# --- startup and the policy -------------------------------------------------


async def test_no_configured_servers_means_no_work(data_dir: Path) -> None:
    bridge = McpBridge([])
    assert await bridge.start(Registry()) == 0
    assert not bridge.failures


async def test_a_server_that_will_not_start_is_recorded_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The same rule `_build_recall` follows: an npx download that times out must
    not cost the user their conversation loop. `/health` reads `failures`."""
    bridge = McpBridge([ServerConfig(name="broken", command="definitely-not-a-real-binary")])
    with caplog.at_level("ERROR"):
        landed = await bridge.start(Registry())
    assert landed == 0
    assert "broken" in bridge.failures


async def test_closing_a_bridge_that_never_started_is_fine() -> None:
    """Shutdown runs whether or not startup got anywhere."""
    await McpBridge([]).aclose()


async def test_a_remote_tool_goes_through_the_origin_gate(db: sqlite3.Connection) -> None:
    """The point of registering them as ordinary tools: an MCP server cannot be a
    way around the one rule that has no configuration."""
    store = Store(db)
    config = ServerConfig(name="notes", command="x")
    _, registry = await bridge_with(config, Session([RemoteTool("write")]))
    tool = registry.get("notes__write")
    assert tool is not None

    policy = ToolPolicy(store, mode="full")
    assert policy.decide(tool, {}, origin="untrusted").verdict == "deny"
    assert policy.decide(tool, {}, origin="owner").verdict == "allow"


async def test_a_remote_tool_cannot_be_allowlisted(db: sqlite3.Connection) -> None:
    """It has no argv, so 'run only allowlisted commands' cannot be satisfied for
    it. `ask` asks; `allowlist` refuses."""
    store = Store(db)
    config = ServerConfig(name="notes", command="x")
    _, registry = await bridge_with(config, Session([RemoteTool("write")]))
    tool = registry.get("notes__write")
    assert tool is not None
    assert ToolPolicy(store, mode="ask").decide(tool, {}, origin="owner").verdict == "ask"
    assert (
        ToolPolicy(store, mode="allowlist").decide(tool, {}, origin="owner").verdict == "deny"
    )


# --- start(), the path the app actually calls --------------------------------


async def test_start_connects_and_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_register` and `call` were covered through stubs, but `start()` - the method
    `_build_tools` calls - was not, so the loop that walks the configured servers and
    records failures was shipping unexercised."""
    config = ServerConfig(name="notes", command="x")
    bridge = McpBridge([config])
    session = Session([RemoteTool("search"), RemoteTool("write")])

    async def connect(cfg: Any, **_: Any) -> Any:
        return session

    monkeypatch.setattr(bridge, "_connect", connect)
    registry = Registry()
    assert await bridge.start(registry) == 2
    assert registry.names() == ("notes__search", "notes__write")
    assert not bridge.failures


async def test_one_server_failing_does_not_stop_the_next(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    good = ServerConfig(name="good", command="x")
    bad = ServerConfig(name="bad", command="y")
    bridge = McpBridge([bad, good])

    async def connect(cfg: Any, **_: Any) -> Any:
        if cfg.name == "bad":
            raise RuntimeError("it exited immediately")
        return Session([RemoteTool("search")])

    monkeypatch.setattr(bridge, "_connect", connect)
    registry = Registry()
    with caplog.at_level("ERROR"):
        assert await bridge.start(registry) == 1
    assert registry.names() == ("good__search",)
    assert bridge.failures == {"bad": "it exited immediately"}


async def test_a_server_that_never_lists_its_tools_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    class Slow(Session):
        async def list_tools(self) -> Any:
            await asyncio.sleep(30)
            return Listing([])

    config = ServerConfig(name="slow", command="x")
    bridge = McpBridge([config])

    async def connect(cfg: Any, **_: Any) -> Any:
        return Slow([])

    monkeypatch.setattr(bridge, "_connect", connect)
    monkeypatch.setattr("daemon.tools.mcp.STARTUP_TIMEOUT", 0.05)
    assert await bridge.start(Registry()) == 0
    assert "slow" in bridge.failures


async def test_a_remote_call_that_hangs_is_given_up_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    class Slow(Session):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            await asyncio.sleep(30)

    config = ServerConfig(name="notes", command="x")
    bridge, registry = await bridge_with(config, Slow([RemoteTool("search")]))
    monkeypatch.setattr("daemon.tools.mcp.CALL_TIMEOUT", 0.05)
    with pytest.raises(ToolError) as caught:
        await registry.get("notes__search").run({})  # type: ignore[union-attr]
    assert "did not answer" in str(caught.value)


async def test_a_non_text_block_is_named_not_dumped() -> None:
    class ImageBlock:
        type = "image"
        data = b"\x89PNG"

    config = ServerConfig(name="notes", command="x")
    bridge, registry = await bridge_with(
        config, Session([RemoteTool("shot")], Result([ImageBlock()]))
    )
    out = await registry.get("notes__shot").run({})  # type: ignore[union-attr]
    assert "[image block]" in out


async def test_a_string_content_result_is_passed_through() -> None:
    config = ServerConfig(name="notes", command="x")
    bridge, registry = await bridge_with(
        config, Session([RemoteTool("search")], Result("plain string"))
    )
    assert await registry.get("notes__search").run({}) == "plain string"  # type: ignore[union-attr]


async def test_an_empty_result_says_so() -> None:
    config = ServerConfig(name="notes", command="x")
    bridge, registry = await bridge_with(
        config, Session([RemoteTool("search")], Result([]))
    )
    assert "(no output)" in await registry.get("notes__search").run({})  # type: ignore[union-attr]


async def test_closing_reports_a_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Shutdown: a server that dies while being closed has nothing left to break, and
    raising here would mask whatever else the lifespan is unwinding."""
    bridge = McpBridge([])

    class BoomStack:
        async def aclose(self) -> None:
            raise RuntimeError("the pipe was already gone")

    # Per-server stacks now: one whose close raises must not stop the rest.
    bridge._stacks["notes"] = BoomStack()  # type: ignore[assignment]
    with caplog.at_level("ERROR"):
        await bridge.aclose()
    assert any("closing MCP session" in r.message for r in caplog.records)
