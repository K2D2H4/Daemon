"""The one tool a voice session gets for work it cannot do itself.

The native-audio model cannot call a nested-schema tool (it fakes the result -
evals/voice_write_nudge_spike.py). This tool has a single string argument it *can*
call: it queues the request durably and returns at once, so the voice turn is never
held open and the ack it speaks is true (the row is committed before this returns).
A background worker runs the request through the text agent and reports the result.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from daemon.llm.base import ToolSpec
from daemon.tools.base import ToolError
from daemon.tools.schema import DELEGATE_TOOL_NAME

DELEGATE_ACK = (
    "알겠어. 백그라운드로 처리하고, 끝나면 결과를 알려줄게. "
    "(I've queued this and will report back when it's done.)"
)
"""Fixed, not model-composed: the truthfulness of "queued" must not depend on the
model's phrasing. Returned only after the row is committed."""


class DelegateTask:
    """Implements the `Tool` protocol in daemon/tools/base.py."""

    risk = "safe"

    def __init__(
        self, enqueue: Callable[[str], int], *, notify: Callable[[], None] | None = None
    ) -> None:
        self._enqueue = enqueue
        self._notify = notify
        self.spec = ToolSpec(
            name=DELEGATE_TOOL_NAME,
            description=(
                "Hand a task off to be done in the background and reported back when "
                "finished. Use this for anything you cannot do in one direct step - "
                "creating or editing a Notion page, multi-step research, anything "
                "with structured input. Pass the owner's request in plain language; "
                "do not try to do it yourself first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": "The task to carry out, in the owner's own words.",
                    }
                },
                "required": ["request"],
            },
        )

    def preview(self, arguments: Mapping[str, Any]) -> str:
        request = str(arguments.get("request", "")).strip()
        return f"delegate_task({request[:80]})"

    async def run(self, arguments: Mapping[str, Any]) -> str:
        request = str(arguments.get("request", "")).strip()
        if not request:
            raise ToolError("delegate_task needs a non-empty request")
        self._enqueue(request)
        if self._notify is not None:
            self._notify()
        return DELEGATE_ACK
