"""Whether a tool call may run. Deterministic, and decided before the model call.

The shape is ported from OpenClaw's exec approvals, which had the important
details already worked out: modes rather than a boolean, an approval bound to the
*exact* call rather than to the tool, and standing grants persisted rather than
held in memory. What is dropped is its `auto` mode, which uses a reviewer model -
this layer makes no model calls at all, and that is the property that makes it
trustworthy.

What is added is the first rule below, which neither benchmark has to solve.

**The origin gate.** Daemon's inbound messages can carry someone else's words:
`InboundMessage.authored_by_sender` is False for a forward or an inline-bot
result, and the loop records those as `origin='untrusted'` (channels/base.py).
Recall then replays arbitrary old text into every prompt, including anything ever
forwarded (loop.py `_recalled`). Without a gate, "look at this message" becomes a
way to hand a stranger a shell, so:

    no tool runs on a turn whose origin is not 'owner' - in every mode, `full`
    included, with no way to configure it off.

That is why `decide` takes an origin, and why nothing below it - not a mode, not
an allowlist, not a standing grant - can reach past it. (The `_enabled` check runs
first, because tools being switched off refuses everything anyway; origin is the first
check that can distinguish one turn from another.)
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import PurePath
from typing import Any, Literal, Protocol

from daemon import clock
from daemon.channels.pairing import generate_code, normalise_code
from daemon.tools.base import Executable, Tool, ToolError, canonical_arguments

logger = logging.getLogger(__name__)

Mode = Literal["off", "allowlist", "ask", "full"]
MODES: tuple[Mode, ...] = ("off", "allowlist", "ask", "full")
"""`off` refuses every guarded tool · `allowlist` runs matches and refuses the
rest · `ask` runs matches and asks about the rest · `full` runs everything."""

Verdict = Literal["allow", "ask", "deny"]

APPROVAL_TTL = timedelta(minutes=5)
"""Short on purpose, and much shorter than a pairing code's hour. A pairing code
is read off one screen and typed on another; this one is answered in the
conversation that is already open, and a stale approval for a command whose
context has moved on is worse than being asked again."""

CODE_ATTEMPTS = 5
"""Retries when a generated code collides with a live one."""

MAX_PENDING = 10
"""Live approvals allowed at once, for the reason `channels/pairing.MAX_PENDING`
exists: every extra live code is another chance for a guess to land, and a model
that asks for the same guarded thing on every round of every turn would otherwise
mint them without limit. Beyond this, minting fails and the call is refused - the
safe direction."""

APPROVE_COMMAND = "/approve"
DENY_COMMAND = "/deny"
ALWAYS = "always"
"""`/approve CODE always` also writes a standing allowlist entry."""

ANY_CHANNEL = "*"
"""A grant that is not about one channel. The only value written today.

The alternative - no channel column, add one later - is the migration
`docs/CONTRACTS.md` says these tables exist to avoid, and `decide` matches on the
column from the start rather than after: a row it ignored would be a grant that
reads as granted and is not, which is this project's worst failure shape."""


@dataclass(frozen=True, slots=True)
class Decision:
    verdict: Verdict
    reason: str
    """Written for two readers: the audit row, and - when the verdict is `deny` -
    the model, which is told why so it stops trying the same thing."""


@dataclass(frozen=True, slots=True)
class Approval:
    code: str
    expires_at: datetime
    preview: str


@dataclass(frozen=True, slots=True)
class Claimed:
    """A pending approval that has just been granted or refused."""

    tool: str
    arguments: dict[str, Any]
    preview: str
    denied: bool
    fingerprint: str


@dataclass(frozen=True, slots=True)
class Command:
    """A parsed `/approve` or `/deny` message."""

    code: str
    denied: bool
    always: bool


class PolicyStore(Protocol):
    """The slice of `memory.store.Store` this module needs.

    A protocol rather than the class, so the policy can be tested against a stub
    and so nothing here imports storage.

    Named for the policy rather than for approvals, which is what it was called
    while approvals were all it carried. Two standing things are now read through
    it and they are different axes - `tool_allowlist` is an argv pattern,
    `tool_grants` is a whole tool - so a name that promised only one of them was
    the wrong name to add the second to.
    """

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
    ) -> bool: ...

    def spend_tool_approval(
        self, code: str, *, sender_id: str, denied: bool, now: datetime
    ) -> Any: ...

    def expire_tool_approvals(self, *, now: datetime) -> int: ...

    def count_pending_tool_approvals(self, *, now: datetime) -> int: ...

    def add_tool_allowlist_entry(self, tool: str, pattern: str, *, now: datetime) -> None: ...

    def tool_allowlist(self, tool: str) -> list[str]: ...

    def tool_grants(self, tool: str) -> list[str]:
        """Channels this tool is granted on, `ANY_CHANNEL` meaning all of them."""
        ...


def fingerprint(tool: str, arguments: Mapping[str, Any]) -> str:
    """Bind an approval to one exact call.

    Everything that decides what runs is in the arguments - the command, its
    arguments, the working directory - so hashing them canonically is what stops
    an approval for `ls /tmp` from authorising anything else. OpenClaw binds cwd
    and argv separately for the same reason; here they are one dict.
    """
    payload = f"{tool}\x00{canonical_arguments(arguments)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_command(text: str) -> Command | None:
    """Read `/approve [CODE] [always]` or `/deny [CODE]`, or return None.

    Recognised before the model sees the message (loop.py), which is why the first
    token must be *exactly* one of the two commands: anything else - `/approved`,
    `approve`, "tell me about /approve" - is ordinary conversation and must reach
    the model untouched.

    A bare `/approve` with no code still parses, to `code=""`. That is the fix for
    the loop the owner could not escape: returning None here sent the bare command
    on to the model as conversation, which answered it by re-issuing the guarded
    call, so the runner minted *another* approval code and asked all over again -
    every `/approve` making it worse and the pending code never spendable. As a
    command it lands in `loop._approve`, which turns the empty code into a nudge to
    include one, with no model call and nothing run.
    """
    parts = text.strip().split()
    if not parts:
        return None
    head = parts[0].lower()
    if head not in (APPROVE_COMMAND, DENY_COMMAND):
        return None
    return Command(
        code=normalise_code(parts[1]) if len(parts) > 1 else "",
        denied=head == DENY_COMMAND,
        always=len(parts) > 2 and parts[2].lower() == ALWAYS,
    )


class ToolPolicy:
    def __init__(
        self,
        store: PolicyStore,
        *,
        mode: Mode = "ask",
        allowlist: Sequence[str] = (),
        enabled: bool = True,
    ) -> None:
        self._store = store
        self._mode: Mode = mode
        # Entries are argv prefixes: 'ls', 'git status'. Normalised once, at
        # construction, so matching a call is a comparison and not a parse.
        self._allowlist = tuple(
            tuple(entry.split()) for entry in allowlist if entry and entry.split()
        )
        self._enabled = enabled
        if enabled and mode == "full":
            logger.warning(
                "tool policy mode=full: guarded tool calls run without approval. "
                "The origin gate still holds; nothing else does."
            )

    @property
    def mode(self) -> Mode:
        """The mode in force. `off` while tools are disabled, so one read answers
        "what will happen" rather than needing two."""
        return self._mode if self._enabled else "off"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def decide(
        self,
        tool: Tool,
        arguments: Mapping[str, Any],
        *,
        origin: str,
        channel: str = "",
    ) -> Decision:
        """The whole decision. No model call, and no I/O beyond standing grants."""
        if not self._enabled:
            return Decision("deny", "tool use is switched off (DAEMON_TOOLS_ENABLED)")

        if origin != "owner":
            # The gate this module exists for. Deliberately above the mode check,
            # so `full` cannot reach past it.
            return Decision(
                "deny",
                f"this turn's origin is {origin!r}, not the owner's own words; "
                "tools only run on what the owner said themselves",
            )

        if tool.risk == "safe":
            return Decision("allow", "read-only tool")

        if self._mode == "off":
            return Decision("deny", "tool mode is 'off'; no guarded tool runs")
        if self._mode == "full":
            return Decision("allow", "tool mode is 'full'")

        if not isinstance(tool, Executable):
            # Nothing to match an argv pattern against, so there is no such thing
            # as an allowlisted `write_file`. The grant is the other axis, and it
            # is read *first*: without it `allowlist` was a permanent refusal for
            # every tool with no argv, and nothing a user could configure reached
            # it. That was `write_file` among the built-ins and - the group it
            # actually cost - every MCP tool, which has no argv by construction
            # (tools/mcp.py) and is `guarded` unless mcp.json names it safe.
            granted = self._granted(tool.spec.name, channel)
            if granted is not None:
                return Decision("allow", f"standing grant: {tool.spec.name} on {granted}")
            if self._mode == "ask":
                return Decision("ask", f"{tool.spec.name} is not allowlistable")
            return Decision(
                "deny",
                f"{tool.spec.name} is not a command, so it cannot be allowlisted, "
                "and tool mode is 'allowlist'. The owner would have to grant the "
                "tool itself; asking again with the same arguments will not help",
            )

        try:
            argv = tool.argv(arguments)
        except ToolError as exc:
            return Decision("deny", str(exc))
        if not argv:
            return Decision("deny", "empty command")

        matched = self._match(tool.spec.name, argv)
        if matched is not None:
            return Decision("allow", f"allowlisted: {matched}")
        if self._mode == "ask":
            return Decision("ask", f"`{' '.join(argv)}` is not allowlisted")
        return Decision(
            "deny", f"{argv[0]} is not allowlisted and tool mode is 'allowlist'"
        )

    def _granted(self, tool_name: str, channel: str) -> str | None:
        """Which channel's grant covers this call, or None.

        Deliberately not consulted for an `Executable` tool. A tool-level grant on
        `run_command` would mean "any command at all" - `mode=full` wearing one
        table row, and reachable by anything that can write one - so a command
        keeps being decided by its argv. That asymmetry is the whole reason there
        are two tables rather than one.

        An empty `channel` matches only `ANY_CHANNEL`, so a caller that does not
        say where it is calling from gets the narrower answer rather than the
        wider one.
        """
        channels = self._store.tool_grants(tool_name)
        if ANY_CHANNEL in channels:
            return ANY_CHANNEL
        if channel and channel in channels:
            return channel
        return None

    def _match(self, tool_name: str, argv: Sequence[str]) -> str | None:
        """The allowlist entry covering this argv, or None.

        An entry is a prefix: `git` covers `git status`, `git status` does not cover
        `git push`.

        **A path never matches.** `ls` on the allowlist means the `ls` that PATH
        resolves to, not any file that happens to be named `ls`. This used to compare
        basenames, on the reasoning that `/bin/ls` is the same command - and it was
        a hole: with `mode=allowlist` and `ls` allowed, a call to
        `/tmp/anything/ls` was let through *without asking* and ran. Measured; it
        printed PWNED. OpenClaw pins the resolved executable for the same reason.

        The cost is that `/bin/ls` now gets asked about. That is the right way round:
        a bare name is the ordinary way to write a command, and an explicit path is
        exactly the case that deserves a look.
        """
        head = argv[0]
        if PurePath(head).name != head:
            return None
        candidate = (head, *argv[1:])
        granted = (tuple(p.split()) for p in self._store.tool_allowlist(tool_name))
        for entry in (*self._allowlist, *granted):
            if len(entry) <= len(candidate) and tuple(candidate[: len(entry)]) == entry:
                return " ".join(entry)
        return None

    # --- approvals ----------------------------------------------------------

    def request(
        self,
        tool: Tool,
        arguments: Mapping[str, Any],
        *,
        channel: str,
        sender_id: str,
        preview: str | None = None,
    ) -> Approval | None:
        """Mint a one-shot code bound to this exact call, or None if none could be.

        Returning None rather than raising: the caller's fallback is to refuse the
        call, which is the safe direction to fail in.

        `preview` is passed in by the runner, which has already guarded it. Calling
        `tool.preview` here was a way for a tool with a broken preview to take the
        whole approval path down with it - the owner then never got asked, on the
        one path where not asking is the failure.
        """
        preview = preview if preview is not None else _safe_preview(tool, arguments)
        created_at = clock.now()
        expires_at = created_at + APPROVAL_TTL
        # Clear the dead ones first, so an abandoned code cannot hold its slot.
        self._store.expire_tool_approvals(now=created_at)
        if self._store.count_pending_tool_approvals(now=created_at) >= MAX_PENDING:
            logger.warning(
                "refusing to mint an approval: %d already pending", MAX_PENDING
            )
            return None
        for _ in range(CODE_ATTEMPTS):
            code = generate_code()
            if self._store.create_tool_approval(
                code=code,
                channel=channel,
                sender_id=sender_id,
                tool=tool.spec.name,
                arguments=canonical_arguments(arguments),
                fingerprint=fingerprint(tool.spec.name, arguments),
                preview=preview,
                created_at=created_at,
                expires_at=expires_at,
            ):
                return Approval(code=code, expires_at=expires_at, preview=preview)
        logger.error("could not mint a tool approval code after %d attempts", CODE_ATTEMPTS)
        return None

    def claim(self, command: Command, *, sender_id: str) -> Claimed | None:
        """Spend a pending approval. None when the code is unknown, expired, already
        spent, or addressed to a different sender - all indistinguishable on
        purpose, so a wrong code says nothing about which codes exist.

        The `always` grant is written only for a command-shaped tool, because a
        standing entry is an argv pattern and there is nothing to store otherwise.
        """
        import json

        now = clock.now()
        row = self._store.spend_tool_approval(
            command.code, sender_id=sender_id, denied=command.denied, now=now
        )
        if row is None:
            return None
        try:
            arguments = json.loads(row["arguments"])
        except (ValueError, TypeError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        claimed = Claimed(
            tool=str(row["tool"]),
            arguments=arguments,
            preview=str(row["preview"]),
            denied=command.denied,
            fingerprint=str(row["fingerprint"]),
        )
        if command.always and not command.denied:
            pattern = _argv_pattern(claimed.arguments)
            if pattern:
                self._store.add_tool_allowlist_entry(claimed.tool, pattern, now=now)
            else:
                logger.info(
                    "'always' ignored for %s: nothing argv-shaped to remember", claimed.tool
                )
        return claimed

    def verify(self, tool_name: str, arguments: Mapping[str, Any], expected: str) -> bool:
        """Re-check the binding at execution time.

        The gap between minting a code and spending it is a gap an argument could
        change across, and the fingerprint is the whole point of binding. OpenClaw
        denies a run whose bound file drifted after approval; this is the same check
        against the arguments themselves.
        """
        return fingerprint(tool_name, arguments) == expected


def _safe_preview(tool: Tool, arguments: Mapping[str, Any]) -> str:
    """Fallback for a caller that did not bring one. See `ToolRunner._preview`."""
    try:
        return tool.preview(arguments)
    except Exception:
        logger.exception("preview failed for %s", tool.spec.name)
        return f"{tool.spec.name}({canonical_arguments(arguments)[:120]})"


def _argv_pattern(arguments: Mapping[str, Any]) -> str:
    """What to remember from an approved call, for `/approve CODE always`.

    The whole command, not just the executable: `always` said of `git status` is a
    statement about `git status`, and widening it to every `git` invocation would
    grant `git push` on the strength of a read-only approval.
    """
    command = arguments.get("command")
    if isinstance(command, str) and command.split():
        return " ".join(command.split())
    return ""
