"""Decide, execute, record. One place, so no caller can skip a step.

The conversation loop hands over the tool calls a model asked for and gets back
results to feed it, approval requests to show the owner, and notices to put in
front of them. It never talks to a tool or to the policy directly - that is what
keeps "every executed call is audited" true by construction rather than by review.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any, Protocol

from daemon.llm.base import ToolCall, ToolSpec
from daemon.tools.base import Registry, Tool, ToolError, ToolResult, canonical_arguments
from daemon.tools.policy import Approval, Claimed, Command, ToolPolicy

logger = logging.getLogger(__name__)

AUDIT_EXCERPT = 500
"""Characters of tool output kept in the audit row. Enough to recognise what
happened; the full output was already given to the model and, when it mattered,
shown to the owner."""


class AuditStore(Protocol):
    """The slice of `memory.store.Store` the runner needs."""

    def record_tool_call(
        self,
        *,
        tool: str,
        arguments: str,
        preview: str,
        verdict: str,
        mode: str,
        reason: str,
        origin: str,
        channel: str,
        sender_id: str | None,
        ran: bool = False,
        ok: bool | None = None,
        output_excerpt: str | None = None,
        elapsed_ms: int | None = None,
        now: datetime | None = None,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Who this turn belongs to, for the origin gate and the audit row."""

    origin: str
    channel: str
    sender_id: str | None


@dataclass(slots=True)
class Outcome:
    results: list[ToolResult] = field(default_factory=list)
    """One per call, in order, to hand back to the model."""
    approvals: list[Approval] = field(default_factory=list)
    """Calls the owner has to authorise before anything happens.

    What actually ran is not carried back for the reply: the owner's ground-truth
    record of every executed call is the `tool_calls` audit row (`daemon tools log`),
    not a line folded into the model's prose. `_run` writes that row for every call.
    """


class ToolRunner:
    def __init__(self, registry: Registry, policy: ToolPolicy, audit: AuditStore) -> None:
        self._registry = registry
        self._policy = policy
        self._audit = audit

    def specs(self) -> tuple[ToolSpec, ...]:
        return self._registry.specs()

    def __len__(self) -> int:
        return len(self._registry)

    async def aclose(self) -> None:
        """Release what the tools hold. Never raises - this runs during shutdown,
        alongside whatever else the lifespan is unwinding."""
        for tool in self._registry.closeables():
            try:
                await tool.aclose()
            except Exception:
                logger.exception("closing tool %s failed", getattr(tool, "spec", tool))

    def claim(self, command: Command, *, sender_id: str) -> Claimed | None:
        """Spend an approval code. A passthrough on purpose: the loop holds one
        reference, to the runner, so `execute` can never be reachable by a caller
        that could not also have audited it."""
        return self._policy.claim(command, sender_id=sender_id)

    async def execute(self, calls: Sequence[ToolCall], context: TurnContext) -> Outcome:
        """Run one round of tool calls, in the order the model asked for them.

        Sequential rather than concurrent, and that is a decision rather than an
        oversight: these calls touch one filesystem, a model that asks to write a
        file and then read it back means those two in that order, and there is no
        latency budget here worth the reordering risk.
        """
        outcome = Outcome()
        for call in calls:
            tool = self._registry.get(call.name)
            if tool is None:
                # Names come from the model, so this is a hallucinated tool rather
                # than a broken registry. Told, not raised: the model can recover.
                available = ", ".join(self._registry.names())
                outcome.results.append(
                    ToolResult(
                        call_id=call.id,
                        name=call.name,
                        content=f"there is no tool called {call.name!r}. Available: {available}",
                        ok=False,
                    )
                )
                continue
            outcome.results.append(await self._one(tool, call, context, outcome))
        return outcome

    async def _one(
        self, tool: Tool, call: ToolCall, context: TurnContext, outcome: Outcome
    ) -> ToolResult:
        decision = self._policy.decide(
            tool,
            call.arguments,
            origin=context.origin,
            # The same channel that goes in the audit column, so a grant scoped to
            # one channel is matched on the value the row will later say it was
            # matched on.
            channel=context.channel,
        )
        preview = _preview(tool, call.arguments)

        if decision.verdict == "deny":
            self._audit.record_tool_call(
                tool=call.name,
                arguments=canonical_arguments(call.arguments),
                preview=preview,
                verdict="deny",
                mode=self._policy.mode,
                reason=decision.reason,
                origin=context.origin,
                channel=context.channel,
                sender_id=context.sender_id,
            )
            logger.info(
                "tool.deny tool=%s origin=%s: %s", call.name, context.origin, decision.reason
            )
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"refused: {decision.reason}",
                ok=False,
            )

        if decision.verdict == "ask":
            approval = self._policy.request(
                tool,
                call.arguments,
                channel=context.channel,
                # The gate above guarantees this turn is the owner's, so a missing
                # sender id would mean the loop passed an inconsistent context.
                sender_id=context.sender_id or "",
                preview=preview,
            )
            self._audit.record_tool_call(
                tool=call.name,
                arguments=canonical_arguments(call.arguments),
                preview=preview,
                verdict="ask",
                mode=self._policy.mode,
                reason=decision.reason,
                origin=context.origin,
                channel=context.channel,
                sender_id=context.sender_id,
            )
            if approval is None:
                return ToolResult(
                    call_id=call.id,
                    name=call.name,
                    content="this needs the owner's approval and I could not ask for it",
                    ok=False,
                )
            outcome.approvals.append(approval)
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=(
                    f"waiting: I have asked the owner to approve `{preview}`. "
                    "Nothing has run. Tell them it is waiting and stop here - the "
                    "result will come back separately once they answer."
                ),
                ok=False,
            )

        return await self._run(
            tool, call.id, call.name, call.arguments, decision.reason, context
        )

    async def resume(self, claimed: Claimed, context: TurnContext) -> ToolResult:
        """Run a call the owner has just approved.

        The fingerprint is re-checked here rather than trusted from the row: the
        arguments travelled through the database and back, and the binding is the
        only thing that makes an approval an approval.
        """
        tool = self._registry.get(claimed.tool)
        if tool is None:
            return ToolResult(
                call_id="approved",
                name=claimed.tool,
                content=f"{claimed.tool} is no longer available",
                ok=False,
            )
        if not self._policy.verify(claimed.tool, claimed.arguments, claimed.fingerprint):
            self._audit.record_tool_call(
                tool=claimed.tool,
                arguments=canonical_arguments(claimed.arguments),
                preview=claimed.preview,
                verdict="deny",
                mode=self._policy.mode,
                reason="the approved call and the stored fingerprint disagree",
                origin=context.origin,
                channel=context.channel,
                sender_id=context.sender_id,
            )
            return ToolResult(
                call_id="approved",
                name=claimed.tool,
                content="that approval does not match the call it was issued for",
                ok=False,
            )
        return await self._run(
            tool, "approved", claimed.tool, claimed.arguments, "approved by the owner", context
        )

    async def _run(
        self,
        tool: Tool,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        reason: str,
        context: TurnContext,
    ) -> ToolResult:
        started = perf_counter()
        ok = True
        try:
            content = await tool.run(arguments)
        except ToolError as exc:
            # The tool's own refusal or failure: expected, and the message is
            # written for the model.
            ok, content = False, str(exc)
        except Exception as exc:
            # A bug in a tool, or an MCP server falling over. One tool must not end
            # the turn, so it is reported as a failed call and logged with a
            # traceback for whoever has to fix it.
            logger.exception("tool %s raised", name)
            ok, content = False, f"{name} failed unexpectedly: {exc}"
        elapsed_ms = int((perf_counter() - started) * 1000)

        self._audit.record_tool_call(
            tool=name,
            arguments=canonical_arguments(arguments),
            preview=_preview(tool, arguments),
            verdict="allow",
            mode=self._policy.mode,
            reason=reason,
            origin=context.origin,
            channel=context.channel,
            sender_id=context.sender_id,
            ran=True,
            ok=ok,
            output_excerpt=content[:AUDIT_EXCERPT],
            elapsed_ms=elapsed_ms,
        )
        logger.info("tool.ran tool=%s ok=%s ms=%d", name, ok, elapsed_ms)
        return ToolResult(
            call_id=call_id, name=name, content=content, ok=ok, elapsed_ms=elapsed_ms
        )


def _preview(tool: Tool, arguments: Mapping[str, Any]) -> str:
    """A tool's own one-line preview, or something usable if it raises.

    Previews are shown to the owner in an approval request, so a tool whose preview
    throws on odd arguments must not take the approval path down with it.
    """
    try:
        return tool.preview(arguments)
    except Exception:
        logger.exception("preview failed for %s", tool.spec.name)
        return f"{tool.spec.name}({canonical_arguments(arguments)[:120]})"
