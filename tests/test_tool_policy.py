"""The gate in front of the machine.

Everything here is deterministic and offline by construction - the policy makes no
model calls and no network calls, which is the property that makes it worth
trusting. The origin gate gets the most attention because it is the one rule that
cannot be configured off, and a regression in it turns "look at this forwarded
message" into a shell.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from daemon import clock
from daemon.llm.base import ToolSpec
from daemon.tools.base import ToolError
from daemon.tools.builtin import PathScope, RunCommand, WriteFile
from daemon.tools.policy import (
    ANY_CHANNEL,
    APPROVAL_TTL,
    MODES,
    Command,
    ToolPolicy,
    fingerprint,
    parse_command,
)

OWNER = "5502877373"


class StubStore:
    """The approval tables, in memory.

    A stub rather than the `db` fixture for the pure-policy tests: what is being
    checked is the decision, and a real sqlite file would only make a failure
    harder to read. `test_tools.py` exercises the same policy against the real
    store.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.granted: dict[str, list[str]] = {}
        self.grants: dict[str, list[str]] = {}
        self.expired = 0

    def create_tool_approval(
        self,
        *,
        code: str,
        channel: str,
        sender_id: str,
        tool: str,
        arguments: str,
        fingerprint: str,
        preview: str,
        created_at: datetime,
        expires_at: datetime,
    ) -> bool:
        if code in self.rows:
            return False
        self.rows[code] = {
            "code": code,
            "channel": channel,
            "sender_id": sender_id,
            "tool": tool,
            "arguments": arguments,
            "fingerprint": fingerprint,
            "preview": preview,
            "state": "pending",
            "expires_at": expires_at,
        }
        return True

    def spend_tool_approval(
        self, code: str, *, sender_id: str, denied: bool, now: datetime
    ) -> Any:
        row = self.rows.get(code)
        if row is None or row["state"] != "pending" or row["sender_id"] != sender_id:
            return None
        if row["expires_at"] <= now:
            return None
        row["state"] = "denied" if denied else "spent"
        return row

    def expire_tool_approvals(self, *, now: datetime) -> int:
        self.expired += 1
        return 0

    def count_pending_tool_approvals(self, *, now: datetime) -> int:
        return sum(1 for r in self.rows.values() if r["state"] == "pending")

    def add_tool_allowlist_entry(self, tool: str, pattern: str, *, now: datetime) -> None:
        self.granted.setdefault(tool, []).append(pattern)

    def tool_allowlist(self, tool: str) -> list[str]:
        return list(self.granted.get(tool, ()))

    def tool_grants(self, tool: str) -> list[str]:
        return list(self.grants.get(tool, ()))


@pytest.fixture
def scope(tmp_path: Path) -> PathScope:
    return PathScope([tmp_path])


@pytest.fixture
def run_command(scope: PathScope) -> RunCommand:
    return RunCommand(scope)


@pytest.fixture
def store() -> StubStore:
    return StubStore()


def policy(store: StubStore, **kw: Any) -> ToolPolicy:
    return ToolPolicy(store, **kw)


# --- the origin gate --------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("origin", ["untrusted", "agent", "system", ""])
async def test_a_turn_that_is_not_the_owners_words_reaches_no_tool(
    store: StubStore, run_command: RunCommand, mode: str, origin: str
) -> None:
    """The rule that has no configuration. `full` is in the parametrisation on
    purpose: it is the mode a user reaches for when they are tired of being asked,
    and it must not be a way to hand a forwarded message a shell."""
    decided = policy(store, mode=mode, allowlist=["ls"]).decide(
        run_command, {"command": "ls"}, origin=origin
    )
    assert decided.verdict == "deny"
    assert origin in decided.reason


@pytest.mark.parametrize("mode", MODES)
async def test_the_origin_gate_covers_read_only_tools_too(
    store: StubStore, scope: PathScope, mode: str
) -> None:
    """`read_file` is 'safe' and normally needs no approval, but safe means "the
    owner would not want to be asked", not "anyone may ask". Reading a file is how
    a forwarded message would exfiltrate one."""
    from daemon.tools.builtin import ReadFile

    decided = policy(store, mode=mode).decide(
        ReadFile(scope), {"path": "notes.md"}, origin="untrusted"
    )
    assert decided.verdict == "deny"


async def test_the_owner_is_the_only_origin_that_passes(
    store: StubStore, run_command: RunCommand
) -> None:
    decided = policy(store, mode="ask", allowlist=["ls"]).decide(
        run_command, {"command": "ls"}, origin="owner"
    )
    assert decided.verdict == "allow"


# --- modes ------------------------------------------------------------------


async def test_disabled_refuses_everything(store: StubStore, run_command: RunCommand) -> None:
    decided = policy(store, mode="full", enabled=False).decide(
        run_command, {"command": "ls"}, origin="owner"
    )
    assert decided.verdict == "deny"
    assert "DAEMON_TOOLS_ENABLED" in decided.reason


async def test_disabled_reports_its_mode_as_off(store: StubStore) -> None:
    """So one read answers "what will happen" instead of needing two."""
    assert policy(store, mode="ask", enabled=False).mode == "off"


async def test_off_refuses_guarded_but_still_reads(
    store: StubStore, run_command: RunCommand, scope: PathScope
) -> None:
    from daemon.tools.builtin import ListDir

    guard = policy(store, mode="off")
    assert guard.decide(run_command, {"command": "ls"}, origin="owner").verdict == "deny"
    assert guard.decide(ListDir(scope), {"path": "."}, origin="owner").verdict == "allow"


async def test_allowlist_mode_refuses_a_miss_rather_than_asking(
    store: StubStore, run_command: RunCommand
) -> None:
    decided = policy(store, mode="allowlist", allowlist=["ls"]).decide(
        run_command, {"command": "curl example.com"}, origin="owner"
    )
    assert decided.verdict == "deny"
    assert "allowlist" in decided.reason


async def test_ask_mode_asks_about_a_miss(store: StubStore, run_command: RunCommand) -> None:
    decided = policy(store, mode="ask", allowlist=["ls"]).decide(
        run_command, {"command": "curl example.com"}, origin="owner"
    )
    assert decided.verdict == "ask"


async def test_full_allows_a_miss(store: StubStore, run_command: RunCommand) -> None:
    decided = policy(store, mode="full").decide(
        run_command, {"command": "curl example.com"}, origin="owner"
    )
    assert decided.verdict == "allow"


async def test_full_mode_says_so_at_construction(
    store: StubStore, caplog: pytest.LogCaptureFixture
) -> None:
    """A user who set this deserves to see it in the log they will read after
    something goes wrong."""
    with caplog.at_level("WARNING"):
        policy(store, mode="full")
    assert any("mode=full" in record.message for record in caplog.records)


# --- allowlist matching -----------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "command", "expected"),
    [
        ("ls", "ls", "allow"),
        ("ls", "ls -la /tmp", "allow"),
        ("git status", "git status", "allow"),
        ("git status", "git status --short", "allow"),
        # A prefix entry is a prefix, not a family: approving `git status` must not
        # be a way to reach `git push`.
        ("git status", "git push", "ask"),
        ("ls", "lsof", "ask"),
        ("ls", "curl example.com", "ask"),
    ],
)
async def test_an_allowlist_entry_matches_a_command_prefix(
    store: StubStore, run_command: RunCommand, entry: str, command: str, expected: str
) -> None:
    decided = policy(store, mode="ask", allowlist=[entry]).decide(
        run_command, {"command": command}, origin="owner"
    )
    assert decided.verdict == expected


@pytest.mark.parametrize(
    "command",
    [
        "/bin/ls -la",
        "./ls",
        "../ls",
        "/tmp/anything/ls",
        "subdir/ls",
    ],
)
async def test_a_path_never_matches_an_allowlisted_name(
    store: StubStore, run_command: RunCommand, command: str
) -> None:
    """The hole this closed. `ls` on the allowlist means the `ls` PATH resolves to,
    not any file that happens to be called `ls`.

    Matching on basename let `mode=allowlist` + `ls` run `/tmp/anything/ls` **without
    asking**; measured, it printed PWNED. Combined with `write_file` that is a way
    past the allowlist entirely. `/bin/ls` being asked about is the price, and it is
    the right way round - an explicit path is exactly the case worth a look.
    """
    decided = policy(store, mode="ask", allowlist=["ls"]).decide(
        run_command, {"command": command}, origin="owner"
    )
    assert decided.verdict == "ask"

    refused = policy(store, mode="allowlist", allowlist=["ls"]).decide(
        run_command, {"command": command}, origin="owner"
    )
    assert refused.verdict == "deny"


async def test_a_bare_name_still_matches(store: StubStore, run_command: RunCommand) -> None:
    """The other direction, so the fix above cannot quietly refuse everything."""
    decided = policy(store, mode="allowlist", allowlist=["ls"]).decide(
        run_command, {"command": "ls -la /tmp"}, origin="owner"
    )
    assert decided.verdict == "allow"


async def test_a_standing_grant_is_not_bypassable_by_path(
    store: StubStore, run_command: RunCommand
) -> None:
    """`/approve … always` grants a command, and the same rule has to hold for it -
    otherwise the durable grant is the weaker one."""
    store.granted["run_command"] = ["date"]
    assert (
        policy(store, mode="ask").decide(
            run_command, {"command": "/tmp/evil/date"}, origin="owner"
        ).verdict
        == "ask"
    )


async def test_a_standing_grant_is_honoured(
    store: StubStore, run_command: RunCommand
) -> None:
    store.granted["run_command"] = ["date"]
    decided = policy(store, mode="ask").decide(
        run_command, {"command": "date"}, origin="owner"
    )
    assert decided.verdict == "allow"


async def test_a_non_command_tool_can_never_be_allowlisted(
    store: StubStore, scope: PathScope
) -> None:
    """`write_file` has no argv, so 'run only allowlisted commands' cannot be
    satisfied for it. `ask` asks; `allowlist` refuses, rather than pretending a
    pattern matched."""
    write = WriteFile(scope)
    arguments = {"path": "notes.md", "content": "hi"}
    assert policy(store, mode="ask").decide(write, arguments, origin="owner").verdict == "ask"
    assert (
        policy(store, mode="allowlist", allowlist=["write_file"])
        .decide(write, arguments, origin="owner")
        .verdict
        == "deny"
    )


# --- per-tool grants --------------------------------------------------------
# The axis the argv allowlist could not express. Which tools were stuck was
# measured rather than assumed, because the guess was wrong in both directions:
# `notify` is `safe` and never reaches the mode check at all, and `open_path`
# *does* implement `argv`. Of the ten built-in and browser tools exactly one -
# `write_file` - is guarded and not allowlistable. The group where it bites is
# MCP: `McpTool` has no argv by construction and its risk is `guarded` unless
# `mcp.json` names it safe, so `mode=allowlist` refused every remote tool an
# owner might add, with no setting that could change it.


ARGUMENTS = {"path": "notes.md", "content": "hi"}


class StubMcpTool:
    """A guarded tool with no argv, the shape every MCP tool has.

    Here rather than `McpTool` itself because the policy reads three things - the
    name, the risk, and whether it is `Executable` - and constructing a real one
    would drag in a bridge and a server config to test none of them.
    """

    risk = "guarded"
    spec = ToolSpec(name="jira_create_issue", description="", parameters={})

    def preview(self, arguments: Any) -> str:
        return "jira: create an issue"

    async def run(self, arguments: Any) -> str:  # pragma: no cover - never reached
        return ""


async def test_a_granted_tool_runs_in_allowlist_mode(
    store: StubStore, scope: PathScope
) -> None:
    """The hole this table exists for. Before it, `allowlist` was a permanent
    refusal for every tool without an argv, whatever the owner said."""
    store.grants["write_file"] = [ANY_CHANNEL]
    decided = policy(store, mode="allowlist").decide(
        WriteFile(scope), ARGUMENTS, origin="owner", channel="telegram"
    )
    assert decided.verdict == "allow"
    assert "grant" in decided.reason


async def test_a_granted_tool_is_not_asked_about_in_ask_mode(
    store: StubStore, scope: PathScope
) -> None:
    """A standing grant is the answer to "stop asking me about this one"."""
    store.grants["write_file"] = [ANY_CHANNEL]
    decided = policy(store, mode="ask").decide(
        WriteFile(scope), ARGUMENTS, origin="owner", channel="telegram"
    )
    assert decided.verdict == "allow"


async def test_a_granted_mcp_tool_runs_in_allowlist_mode(store: StubStore) -> None:
    """The group the hole actually cost: an MCP tool has no argv by construction,
    so before this table `mode=allowlist` refused every remote tool an owner
    added, whatever they configured."""
    store.grants["jira_create_issue"] = [ANY_CHANNEL]
    decided = policy(store, mode="allowlist").decide(
        StubMcpTool(), {"summary": "hi"}, origin="owner", channel="telegram"
    )
    assert decided.verdict == "allow"


async def test_a_grant_for_one_tool_says_nothing_about_another(
    store: StubStore, scope: PathScope
) -> None:
    """Keyed by tool name, so granting one is not granting the set."""
    store.grants["jira_create_issue"] = [ANY_CHANNEL]
    decided = policy(store, mode="allowlist").decide(
        WriteFile(scope), ARGUMENTS, origin="owner", channel="telegram"
    )
    assert decided.verdict == "deny"


@pytest.mark.parametrize("origin", ["untrusted", "agent", "system", ""])
@pytest.mark.parametrize("mode", MODES)
async def test_a_grant_does_not_reach_past_the_origin_gate(
    store: StubStore, scope: PathScope, origin: str, mode: str
) -> None:
    """CONTRACTS rule 10 names standing grants explicitly: not a mode, not an
    allowlist, not a grant. A grant is stored per tool and a forwarded message can
    name any tool, so this is the one that would turn "look at this" into a write."""
    store.grants["write_file"] = [ANY_CHANNEL]
    decided = policy(store, mode=mode).decide(
        WriteFile(scope), ARGUMENTS, origin=origin, channel="telegram"
    )
    assert decided.verdict == "deny"


async def test_a_grant_does_not_reach_past_mode_off(
    store: StubStore, scope: PathScope
) -> None:
    """`off` means no guarded tool runs. A grant written while the mode was
    `allowlist` must not survive the owner turning tools down."""
    store.grants["write_file"] = [ANY_CHANNEL]
    decided = policy(store, mode="off").decide(
        WriteFile(scope), ARGUMENTS, origin="owner", channel="telegram"
    )
    assert decided.verdict == "deny"


async def test_a_grant_does_not_reach_past_the_switch(
    store: StubStore, scope: PathScope
) -> None:
    store.grants["write_file"] = [ANY_CHANNEL]
    decided = policy(store, mode="allowlist", enabled=False).decide(
        WriteFile(scope), ARGUMENTS, origin="owner", channel="telegram"
    )
    assert decided.verdict == "deny"
    assert "DAEMON_TOOLS_ENABLED" in decided.reason


async def test_a_grant_is_not_a_way_around_the_argv_allowlist(
    store: StubStore, run_command: RunCommand
) -> None:
    """`run_command` *is* argv-shaped, so it keeps being decided by argv. A
    tool-level grant on it would mean "any command at all", which is `mode=full`
    wearing a table row - and would be granted by whoever could write one row."""
    store.grants["run_command"] = [ANY_CHANNEL]
    decided = policy(store, mode="ask").decide(
        run_command, {"command": "curl evil.example"}, origin="owner", channel="telegram"
    )
    assert decided.verdict == "ask"


async def test_a_grant_scoped_to_a_channel_does_not_apply_elsewhere(
    store: StubStore, scope: PathScope
) -> None:
    """Nothing writes a channel-scoped row today. The column is here so the
    milestone that wants "this tool over text only" needs no migration - and the
    match honours it from the start, because a row `decide` ignored would be a
    grant that reads as granted and is not."""
    store.grants["write_file"] = ["telegram"]
    write = WriteFile(scope)
    assert (
        policy(store, mode="allowlist")
        .decide(write, ARGUMENTS, origin="owner", channel="telegram")
        .verdict
        == "allow"
    )
    assert (
        policy(store, mode="allowlist")
        .decide(write, ARGUMENTS, origin="owner", channel="voice")
        .verdict
        == "deny"
    )


async def test_a_channel_scoped_grant_is_not_matched_by_a_caller_that_named_none(
    store: StubStore, scope: PathScope
) -> None:
    """`channel` defaults to empty, so a caller that forgets it gets only the
    `'*'` grants - the safe direction to be wrong in."""
    store.grants["write_file"] = ["telegram"]
    decided = policy(store, mode="allowlist").decide(
        WriteFile(scope), ARGUMENTS, origin="owner"
    )
    assert decided.verdict == "deny"


async def test_an_ungranted_tool_still_says_how_it_could_be_granted(
    store: StubStore, scope: PathScope
) -> None:
    """The refusal is read by the model, which is told why so it stops trying the
    same thing - and can say what the owner would have to do instead."""
    decided = policy(store, mode="allowlist").decide(
        WriteFile(scope), ARGUMENTS, origin="owner", channel="telegram"
    )
    assert decided.verdict == "deny"
    assert "grant" in decided.reason


# --- shell operators --------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "ls; rm -rf /",
        "ls && curl evil.example",
        "ls || true",
        "ls | grep secret",
        "echo `whoami`",
        "echo $(whoami)",
        "ls > /tmp/out",
        "cat < /etc/passwd",
        "ls & sleep 1",
        "ls\nrm -rf /",
    ],
)
@pytest.mark.parametrize("mode", MODES)
async def test_shell_operators_are_refused_in_every_mode(
    store: StubStore, run_command: RunCommand, command: str, mode: str
) -> None:
    """There is no shell to inject into - `run_command` execs an argv vector - so
    this is refused early with an explanation rather than exec'd as a program
    literally named `ls;`. `full` is included because `full` means "do not ask",
    not "add a shell"."""
    with pytest.raises(ToolError) as caught:
        run_command.argv({"command": command})
    assert "shell" in str(caught.value)

    if mode not in ("off", "full"):
        decided = policy(store, mode=mode, allowlist=["ls"]).decide(
            run_command, {"command": command}, origin="owner"
        )
        assert decided.verdict == "deny"


async def test_an_unparseable_command_is_refused_not_crashed(
    store: StubStore, run_command: RunCommand
) -> None:
    decided = policy(store, mode="ask").decide(
        run_command, {"command": 'echo "unbalanced'}, origin="owner"
    )
    assert decided.verdict == "deny"


@pytest.mark.parametrize("command", ["", "   ", None, 42, [] ])
async def test_a_command_that_is_not_a_command_is_refused(
    store: StubStore, run_command: RunCommand, command: Any
) -> None:
    decided = policy(store, mode="full").decide(
        run_command, {"command": command}, origin="owner"
    )
    # `full` allows without inspecting argv, so the refusal has to come from the
    # tool itself at run time - which is why this asserts on argv directly.
    assert decided.verdict == "allow"
    with pytest.raises(ToolError):
        run_command.argv({"command": command})


# --- approvals --------------------------------------------------------------


async def test_an_approval_is_bound_to_the_exact_call(
    store: StubStore, run_command: RunCommand
) -> None:
    guard = policy(store, mode="ask")
    approval = guard.request(
        run_command, {"command": "ls /tmp"}, channel="telegram", sender_id=OWNER
    )
    assert approval is not None
    assert guard.verify("run_command", {"command": "ls /tmp"}, approval_fingerprint(store))
    # The whole point: the approved code does not authorise a different command.
    assert not guard.verify("run_command", {"command": "rm -rf /"}, approval_fingerprint(store))


def approval_fingerprint(store: StubStore) -> str:
    (row,) = store.rows.values()
    return str(row["fingerprint"])


async def test_a_fingerprint_distinguishes_the_cwd_too(run_command: RunCommand) -> None:
    """cwd decides what a relative path means, so it is part of the binding."""
    assert fingerprint("run_command", {"command": "ls", "cwd": "/tmp"}) != fingerprint(
        "run_command", {"command": "ls", "cwd": "/etc"}
    )


async def test_key_order_does_not_change_the_fingerprint() -> None:
    """Models emit arguments in whatever order they like; the same call must bind
    the same way."""
    assert fingerprint("t", {"a": 1, "b": 2}) == fingerprint("t", {"b": 2, "a": 1})


async def test_a_code_can_be_spent_only_once(
    store: StubStore, run_command: RunCommand
) -> None:
    guard = policy(store, mode="ask")
    approval = guard.request(
        run_command, {"command": "ls"}, channel="telegram", sender_id=OWNER
    )
    assert approval is not None
    command = Command(code=approval.code, denied=False, always=False)
    assert guard.claim(command, sender_id=OWNER) is not None
    assert guard.claim(command, sender_id=OWNER) is None


async def test_an_expired_code_cannot_be_spent(
    store: StubStore, run_command: RunCommand, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = datetime(2026, 8, 3, 7, 14, tzinfo=UTC)
    monkeypatch.setattr(clock, "now", lambda: start)
    guard = policy(store, mode="ask")
    approval = guard.request(
        run_command, {"command": "ls"}, channel="telegram", sender_id=OWNER
    )
    assert approval is not None

    monkeypatch.setattr(clock, "now", lambda: start + APPROVAL_TTL + timedelta(seconds=1))
    assert guard.claim(Command(approval.code, denied=False, always=False), sender_id=OWNER) is None


async def test_another_sender_cannot_spend_the_owners_code(
    store: StubStore, run_command: RunCommand
) -> None:
    """A guest may be allowlisted for conversation without being allowed to answer
    an approval addressed to the owner."""
    guard = policy(store, mode="ask")
    approval = guard.request(
        run_command, {"command": "ls"}, channel="telegram", sender_id=OWNER
    )
    assert approval is not None
    assert guard.claim(Command(approval.code, False, False), sender_id="99999") is None


async def test_an_unknown_code_says_nothing_about_which_codes_exist(
    store: StubStore, run_command: RunCommand
) -> None:
    guard = policy(store, mode="ask")
    guard.request(run_command, {"command": "ls"}, channel="telegram", sender_id=OWNER)
    assert guard.claim(Command("AAAAAAAA", False, False), sender_id=OWNER) is None


async def test_denying_marks_the_code_spent_as_well(
    store: StubStore, run_command: RunCommand
) -> None:
    """Otherwise a `/deny` followed by an `/approve` on the same code would run it."""
    guard = policy(store, mode="ask")
    approval = guard.request(
        run_command, {"command": "ls"}, channel="telegram", sender_id=OWNER
    )
    assert approval is not None
    denied = guard.claim(Command(approval.code, denied=True, always=False), sender_id=OWNER)
    assert denied is not None and denied.denied
    assert guard.claim(Command(approval.code, False, False), sender_id=OWNER) is None


async def test_always_grants_the_whole_command_not_the_program(
    store: StubStore, run_command: RunCommand
) -> None:
    """`always` said of `git status` is about `git status`. Widening it to `git`
    would grant `git push` on the strength of a read-only approval."""
    guard = policy(store, mode="ask")
    approval = guard.request(
        run_command, {"command": "git status"}, channel="telegram", sender_id=OWNER
    )
    assert approval is not None
    guard.claim(Command(approval.code, denied=False, always=True), sender_id=OWNER)
    assert store.granted["run_command"] == ["git status"]
    assert (
        guard.decide(run_command, {"command": "git push"}, origin="owner").verdict == "ask"
    )


async def test_always_is_not_granted_when_denied(
    store: StubStore, run_command: RunCommand
) -> None:
    guard = policy(store, mode="ask")
    approval = guard.request(
        run_command, {"command": "git status"}, channel="telegram", sender_id=OWNER
    )
    assert approval is not None
    guard.claim(Command(approval.code, denied=True, always=True), sender_id=OWNER)
    assert not store.granted


async def test_always_on_a_non_command_tool_grants_nothing(
    store: StubStore, scope: PathScope
) -> None:
    """There is no argv to remember, and inventing one would grant more than was
    approved. The call still runs; only the standing grant is skipped."""
    guard = policy(store, mode="ask")
    approval = guard.request(
        WriteFile(scope),
        {"path": "notes.md", "content": "hi"},
        channel="telegram",
        sender_id=OWNER,
    )
    assert approval is not None
    claimed = guard.claim(Command(approval.code, denied=False, always=True), sender_id=OWNER)
    assert claimed is not None and not claimed.denied
    assert not store.granted


async def test_requesting_clears_lapsed_codes_first(
    store: StubStore, run_command: RunCommand
) -> None:
    guard = policy(store, mode="ask")
    guard.request(run_command, {"command": "ls"}, channel="telegram", sender_id=OWNER)
    assert store.expired == 1


async def test_a_code_that_cannot_be_minted_is_reported_as_none(
    run_command: RunCommand, caplog: pytest.LogCaptureFixture
) -> None:
    """The caller's fallback is to refuse the call, so failing to ask must not look
    like permission."""

    class Full(StubStore):
        def create_tool_approval(self, **kw: Any) -> bool:
            return False

    with caplog.at_level("ERROR"):
        approval = policy(Full(), mode="ask").request(
            run_command, {"command": "ls"}, channel="telegram", sender_id=OWNER
        )
    assert approval is None
    assert any("approval code" in record.message for record in caplog.records)


# --- parsing the command ----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "code", "denied", "always"),
    [
        ("/approve A3F2K9QT", "A3F2K9QT", False, False),
        ("  /approve a3f2k9qt  ", "A3F2K9QT", False, False),
        ("/APPROVE a3f2k9qt", "A3F2K9QT", False, False),
        ("/approve A3F2K9QT always", "A3F2K9QT", False, True),
        ("/approve A3F2K9QT ALWAYS", "A3F2K9QT", False, True),
        ("/deny A3F2K9QT", "A3F2K9QT", True, False),
    ],
)
async def test_approval_commands_are_recognised(
    text: str, code: str, denied: bool, always: bool
) -> None:
    parsed = parse_command(text)
    assert parsed == Command(code=code, denied=denied, always=always)


@pytest.mark.parametrize(
    ("text", "denied"),
    [
        ("/approve", False),
        ("  /APPROVE  ", False),
        ("/deny", True),
    ],
)
async def test_a_bare_command_still_parses_with_no_code(text: str, denied: bool) -> None:
    """A `/approve` with no code is still the control-plane verb, and must parse -
    to an empty code - rather than return None. Returning None here fell through to
    the model as conversation, which answered it by re-issuing the guarded call and
    minting yet another approval code: the loop the owner could not escape by typing
    `/approve` again. `loop._approve` turns the empty code into a nudge, not a call."""
    assert parse_command(text) == Command(code="", denied=denied, always=False)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "hello",
        "tell me about /approve A3F2K9QT",
        "approve A3F2K9QT",
        "/approved A3F2K9QT",
    ],
)
async def test_ordinary_conversation_is_not_an_approval(text: str) -> None:
    """A false positive here would swallow a message the user meant for the model,
    so the first token must be exactly the command. A bare `/approve` clears that
    bar - the first token *is* the command - and is covered above, not here."""
    assert parse_command(text) is None


# --- paths coverage found nothing exercising ---------------------------------


async def test_the_enabled_flag_is_readable(store: StubStore) -> None:
    assert policy(store, mode="ask").enabled is True
    assert policy(store, mode="ask", enabled=False).enabled is False


async def test_a_command_that_parses_to_nothing_is_refused(
    store: StubStore, scope: PathScope
) -> None:
    """`argv()` normally catches this, so the guard is reached only by a tool whose
    argv is empty for some other reason. Refusing beats indexing argv[0]."""

    class Empty(RunCommand):
        def argv(self, arguments: Any) -> list[str]:
            return []

    decided = policy(store, mode="ask").decide(
        Empty(scope), {"command": "whatever"}, origin="owner"
    )
    assert decided.verdict == "deny"
    assert "empty command" in decided.reason


async def test_a_claim_with_corrupt_arguments_still_returns(
    store: StubStore, run_command: RunCommand
) -> None:
    """The arguments travel through the database as JSON text. A row someone edited
    by hand must not raise out of the approval path - the fingerprint check downstream
    is what refuses it."""
    guard = policy(store, mode="ask")
    approval = guard.request(
        run_command, {"command": "ls"}, channel="telegram", sender_id=OWNER
    )
    assert approval is not None
    store.rows[approval.code]["arguments"] = "{not json"

    claimed = guard.claim(Command(approval.code, denied=False, always=False), sender_id=OWNER)
    assert claimed is not None
    assert claimed.arguments == {}
    # And the binding no longer matches, so nothing runs off the back of it.
    assert not guard.verify(claimed.tool, claimed.arguments, claimed.fingerprint)


async def test_a_claim_whose_arguments_are_not_an_object_is_emptied(
    store: StubStore, run_command: RunCommand
) -> None:
    guard = policy(store, mode="ask")
    approval = guard.request(
        run_command, {"command": "ls"}, channel="telegram", sender_id=OWNER
    )
    assert approval is not None
    store.rows[approval.code]["arguments"] = '["a", "list"]'
    claimed = guard.claim(Command(approval.code, denied=False, always=False), sender_id=OWNER)
    assert claimed is not None and claimed.arguments == {}


async def test_a_broken_preview_does_not_stop_a_code_being_minted(
    store: StubStore, caplog: pytest.LogCaptureFixture
) -> None:
    """The caller normally hands the preview in already-guarded. This is the fallback
    for one that does not, and it must not be the thing that stops the owner being
    asked."""
    from daemon.llm.base import ToolSpec

    class BadPreview:
        risk = "guarded"
        spec = ToolSpec(name="odd", description="x", parameters={"type": "object"})

        def preview(self, arguments: Any) -> str:
            raise RuntimeError("preview is broken")

        async def run(self, arguments: Any) -> str:
            return "ran"

    with caplog.at_level("ERROR"):
        approval = policy(store, mode="ask").request(
            BadPreview(), {"a": 1}, channel="telegram", sender_id=OWNER
        )
    assert approval is not None
    assert "odd(" in approval.preview
