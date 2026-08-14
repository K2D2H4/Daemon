# Async Delegation for the Voice Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the voice model reliably act on complex requests by calling one flat tool, `delegate_task`, that queues the request and runs it asynchronously through the reliable text agent, then reports the result by presence.

**Architecture:** A voice session is offered only flat-schema tools plus `delegate_task` (nested-schema tools it cannot call are withheld). `delegate_task` commits a durable row and returns immediately with a truthful ack. A single background worker claims queued rows, runs each through the existing text `ConversationLoop` behind a capture-channel, and delivers the reply by presence (speaker + Telegram when the owner is at the machine, Telegram otherwise). On restart, unfinished rows are reported rather than lost.

**Tech Stack:** Python 3.13, asyncio, sqlite3 (STRICT tables), the repo's `Companion`/`ConversationLoop`/`ToolRunner`/`Store` seams. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-14-async-delegation-design.md`

## Global Constraints

- Python 3.13; `python3 -m pytest` green, `python3 -m ruff check .` clean, `python3 scripts/check_docs.py` clean before each commit.
- Tests never touch the network, an API key, a microphone, or a speaker — use fakes (tests/CLAUDE.md). Add at least one Korean case where text is involved.
- `daemon/memory/schema.sql` is **FROZEN**; this plan adds one table, `delegated_tasks`. Declare it in the PR description. No existing table/column changes.
- Only `daemon/app.py` may import concrete implementations (layering). New modules expose functions/classes; `app.py` wires them.
- Delegated work runs with `origin="owner"` (a microphone has no relay path; the request is the owner's). The `ToolPolicy` origin gate still governs every tool the worker's loop runs.
- `delegate_task` is the exact tool name, defined once as `DELEGATE_TOOL_NAME`.

---

### Task 1: Schema classifier + delegate constant

**Files:**
- Create: `daemon/tools/schema.py`
- Test: `tests/test_tools_schema.py`

**Interfaces:**
- Produces: `DELEGATE_TOOL_NAME: str = "delegate_task"`; `is_flat_schema(parameters: Mapping[str, Any]) -> bool` — True iff every top-level property is a primitive (`string`/`number`/`integer`/`boolean`, or an `enum` of those) with no `object`/`array` property and no `$ref`/`anyOf`/`oneOf`/`allOf` anywhere at the top level.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_schema.py
from daemon.tools.schema import DELEGATE_TOOL_NAME, is_flat_schema


def test_flat_when_all_properties_are_primitive():
    assert is_flat_schema(
        {"type": "object", "properties": {"target": {"type": "string"}}}
    )
    assert is_flat_schema(
        {"type": "object", "properties": {"command": {"type": "string"},
                                          "cwd": {"type": "string"}}}
    )


def test_flat_allows_enum_and_number_and_bool():
    assert is_flat_schema(
        {"type": "object", "properties": {
            "mode": {"type": "string", "enum": ["a", "b"]},
            "count": {"type": "integer"},
            "force": {"type": "boolean"}}}
    )


def test_nested_object_property_is_not_flat():
    assert not is_flat_schema(
        {"type": "object", "properties": {"parent": {"type": "object"}}}
    )


def test_array_property_is_not_flat():
    assert not is_flat_schema(
        {"type": "object", "properties": {"pages": {"type": "array"}}}
    )


def test_composed_schema_is_not_flat():
    assert not is_flat_schema(
        {"type": "object", "properties": {"x": {"type": "string"}},
         "anyOf": [{"required": ["x"]}]}
    )


def test_empty_or_missing_properties_is_flat():
    # A no-arg tool (get_time) is trivially flat and callable over voice.
    assert is_flat_schema({"type": "object", "properties": {}})
    assert is_flat_schema({"type": "object"})


def test_delegate_name_is_the_literal():
    assert DELEGATE_TOOL_NAME == "delegate_task"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_tools_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'daemon.tools.schema'`

- [ ] **Step 3: Write minimal implementation**

```python
# daemon/tools/schema.py
"""Which tools a voice session may be offered.

The native-audio Gemini Live model emits a flat-argument tool call reliably and a
nested-argument one almost never - it fakes the result instead (measured:
evals/voice_write_nudge_spike.py, docs/superpowers/specs/2026-08-14-async-delegation-design.md).
So a voice session is offered only flat-schema tools plus `delegate_task`, which
routes the rest to the text path. `is_flat_schema` is that gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DELEGATE_TOOL_NAME = "delegate_task"
"""The one non-flat-work tool a voice session gets. Defined once; consumed by
`Companion.specs(surface="voice")` and by the tool itself."""

_PRIMITIVE_TYPES = frozenset({"string", "number", "integer", "boolean"})
_COMPOSERS = ("$ref", "anyOf", "oneOf", "allOf")


def is_flat_schema(parameters: Mapping[str, Any]) -> bool:
    """True if every argument is a primitive - no nested object or array, no
    composition. A flat schema is one the audio model can actually fill."""
    if any(key in parameters for key in _COMPOSERS):
        return False
    properties = parameters.get("properties")
    if not properties:
        # No declared arguments: a no-arg tool is trivially callable.
        return not any(key in parameters for key in _COMPOSERS)
    if not isinstance(properties, dict):
        return False
    for schema in properties.values():
        if not isinstance(schema, dict):
            return False
        if any(key in schema for key in _COMPOSERS):
            return False
        if schema.get("type") not in _PRIMITIVE_TYPES:
            return False
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_tools_schema.py -v && python3 -m ruff check daemon/tools/schema.py tests/test_tools_schema.py`
Expected: PASS, ruff clean

- [ ] **Step 5: Commit**

```bash
git add daemon/tools/schema.py tests/test_tools_schema.py
git commit -m "tools: a flat-schema classifier, so voice is offered only tools it can call"
```

---

### Task 2: Surface-aware `Companion.specs`

**Files:**
- Modify: `daemon/companion.py` (`specs` method, imports)
- Modify: `daemon/app.py:1212` (`run_voice` passes `surface="voice"`)
- Test: `tests/test_companion.py` (add cases; create the file only if it does not exist — check first with `ls tests/test_companion.py`)

**Interfaces:**
- Consumes: `DELEGATE_TOOL_NAME`, `is_flat_schema` (Task 1); the existing `ToolRunner.specs() -> tuple[ToolSpec, ...]`.
- Produces: `Companion.specs(*, origin: str, surface: str = "text") -> tuple[ToolSpec, ...]` — `surface="voice"` returns flat-schema specs plus any spec named `DELEGATE_TOOL_NAME`; `surface="text"` (default) returns every spec **except** `DELEGATE_TOOL_NAME`; non-owner origin still returns `()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_companion.py (add these; imports at top of file)
from daemon.companion import Companion
from daemon.llm.base import ToolSpec
from daemon.tools.schema import DELEGATE_TOOL_NAME


class _FakeRunner:
    def __init__(self, specs):
        self._specs = tuple(specs)

    def __len__(self):
        return len(self._specs)

    def specs(self):
        return self._specs


_FLAT = ToolSpec(name="open_path", description="",
                 parameters={"type": "object", "properties": {"target": {"type": "string"}}})
_NESTED = ToolSpec(name="notion__notion-create-pages", description="",
                   parameters={"type": "object", "properties": {"pages": {"type": "array"}}})
_DELEGATE = ToolSpec(name=DELEGATE_TOOL_NAME, description="",
                     parameters={"type": "object", "properties": {"request": {"type": "string"}}})


def _companion(tmp_path, specs):
    return Companion(_FakeMemory(), data_dir=tmp_path, tools=_FakeRunner(specs))


def test_voice_surface_drops_nested_tools_but_keeps_flat_and_delegate(tmp_path):
    c = _companion(tmp_path, [_FLAT, _NESTED, _DELEGATE])
    names = {s.name for s in c.specs(origin="owner", surface="voice")}
    assert names == {"open_path", DELEGATE_TOOL_NAME}


def test_text_surface_keeps_nested_and_drops_delegate(tmp_path):
    c = _companion(tmp_path, [_FLAT, _NESTED, _DELEGATE])
    names = {s.name for s in c.specs(origin="owner", surface="text")}
    assert names == {"open_path", "notion__notion-create-pages"}


def test_non_owner_gets_nothing_on_either_surface(tmp_path):
    c = _companion(tmp_path, [_FLAT, _DELEGATE])
    assert c.specs(origin="untrusted", surface="voice") == ()
    assert c.specs(origin="untrusted", surface="text") == ()
```

Use the existing test module's memory fake for `_FakeMemory` — check `tests/conftest.py` and how other `Companion(...)` tests construct it, and reuse that. If `Companion` requires more constructor args in this repo revision, pass them as those tests do.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_companion.py -k surface -v`
Expected: FAIL — `specs()` got an unexpected keyword argument `surface`

- [ ] **Step 3: Write minimal implementation**

In `daemon/companion.py`, add the import near the other tool imports:

```python
from daemon.tools.schema import DELEGATE_TOOL_NAME, is_flat_schema
```

Replace the `specs` method body:

```python
    def specs(self, *, origin: str, surface: str = "text") -> tuple[ToolSpec, ...]:
        """What may be offered to the model on a turn from `origin`. The origin gate.

        `surface` splits the offer by endpoint. A voice session is offered only
        flat-schema tools plus `delegate_task`: the native-audio model fakes a
        nested-argument call instead of making it (docs/superpowers/specs/
        2026-08-14-async-delegation-design.md), so a nested tool on a voice turn is
        an invitation to confabulate. The text path gets every tool except
        `delegate_task`, which it does not need - it can call the nested tools
        directly. Empty for a turn that is not the owner's own words, unchanged.
        """
        if self._tools is None or not len(self._tools) or origin != "owner":
            return ()
        all_specs = self._tools.specs()
        if surface == "voice":
            return tuple(
                spec for spec in all_specs
                if spec.name == DELEGATE_TOOL_NAME or is_flat_schema(spec.parameters)
            )
        return tuple(spec for spec in all_specs if spec.name != DELEGATE_TOOL_NAME)
```

Then in `daemon/app.py` at the `run_voice` spec call (currently `tool_specs = companion.specs(origin="owner")`, ~line 1212):

```python
        tool_specs = companion.specs(origin="owner", surface="voice")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_companion.py -k surface -v && python3 -m pytest tests/test_loop.py -q`
Expected: new tests PASS; `test_loop.py` (the text path, default `surface="text"`) still PASS — `_tool_rules` and the loop use the default and are unaffected.

- [ ] **Step 5: Commit**

```bash
git add daemon/companion.py daemon/app.py tests/test_companion.py
git commit -m "companion: split the tool offer by surface - voice gets flat tools plus delegate"
```

---

### Task 3: `delegated_tasks` table + Store methods

**Files:**
- Modify: `daemon/memory/schema.sql` (add table, after `tool_calls`)
- Modify: `daemon/memory/store.py` (add methods; import `utc_iso` is already present — it is used by `record_tool_call`)
- Test: `tests/test_store.py` (add cases)

**Interfaces:**
- Produces on `Store`:
  - `enqueue_task(*, request: str, origin: str, channel: str, sender_id: str | None, now: datetime | None = None) -> int` — inserts a `queued` row, returns its id.
  - `claim_next_queued(*, now: datetime | None = None) -> sqlite3.Row | None` — atomically flips the oldest `queued` row to `running` and returns it (or `None`).
  - `mark_task_done(task_id: int, result: str, *, now: datetime | None = None) -> None`
  - `mark_task_failed(task_id: int, error: str, *, now: datetime | None = None) -> None`
  - `pending_tasks() -> list[sqlite3.Row]` — every row still `queued` or `running`, oldest first.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py (add; reuse the module's existing `store` fixture / Store.open pattern)
def test_enqueue_then_claim_marks_running_and_returns_it(store):
    tid = store.enqueue_task(request="노션에 페이지 만들어줘", origin="owner",
                             channel="voice", sender_id=None)
    assert tid > 0
    row = store.claim_next_queued()
    assert row is not None
    assert row["id"] == tid
    assert row["request"] == "노션에 페이지 만들어줘"
    assert row["status"] == "running"
    # Only one queued row existed, so the next claim finds nothing.
    assert store.claim_next_queued() is None


def test_mark_done_records_the_result(store):
    tid = store.enqueue_task(request="r", origin="owner", channel="voice", sender_id=None)
    store.claim_next_queued()
    store.mark_task_done(tid, "만들었어요")
    (row,) = [r for r in store.pending_tasks()] or [None]
    assert row is None  # a done task is not pending
    done = store.conn.execute("SELECT status, result FROM delegated_tasks WHERE id=?", (tid,)).fetchone()
    assert done["status"] == "done"
    assert done["result"] == "만들었어요"


def test_mark_failed_records_the_error(store):
    tid = store.enqueue_task(request="r", origin="owner", channel="voice", sender_id=None)
    store.claim_next_queued()
    store.mark_task_failed(tid, "notion 400")
    row = store.conn.execute("SELECT status, error FROM delegated_tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "notion 400"


def test_pending_reports_queued_and_running_left_by_a_restart(store):
    a = store.enqueue_task(request="a", origin="owner", channel="voice", sender_id=None)
    b = store.enqueue_task(request="b", origin="owner", channel="voice", sender_id=None)
    store.claim_next_queued()  # a -> running
    pending = store.pending_tasks()
    assert [r["id"] for r in pending] == [a, b]
    assert {r["status"] for r in pending} == {"running", "queued"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_store.py -k task -v`
Expected: FAIL — `AttributeError: 'Store' object has no attribute 'enqueue_task'`

- [ ] **Step 3a: Add the table to `daemon/memory/schema.sql`** (after the `tool_calls` block and its index)

```sql
-- Owner-requested work handed off from a voice turn to the text agent and run in
-- the background (docs/superpowers/specs/2026-08-14-async-delegation-design.md).
-- Durable so a restart cannot silently swallow a promised task: the ack was
-- already spoken.
CREATE TABLE IF NOT EXISTS delegated_tasks (
    id           INTEGER PRIMARY KEY,
    request      TEXT    NOT NULL,
    status       TEXT    NOT NULL DEFAULT 'queued'
                         CHECK (status IN ('queued', 'running', 'done', 'failed')),
    result       TEXT,
    error        TEXT,
    created_ts   TEXT    NOT NULL,
    finished_ts  TEXT,
    origin       TEXT    NOT NULL,
    channel      TEXT    NOT NULL,
    sender_id    TEXT
) STRICT;

CREATE INDEX IF NOT EXISTS idx_delegated_tasks_status ON delegated_tasks (status, id);
```

- [ ] **Step 3b: Add the methods to `daemon/memory/store.py`** (near `record_tool_call`)

```python
    def enqueue_task(
        self, *, request: str, origin: str, channel: str,
        sender_id: str | None, now: datetime | None = None,
    ) -> int:
        cursor = self.conn.execute(
            "INSERT INTO delegated_tasks (request, status, created_ts, origin, channel, sender_id) "
            "VALUES (?, 'queued', ?, ?, ?, ?)",
            (request, utc_iso(now or datetime.now(UTC)), origin, channel, sender_id),
        )
        self.conn.commit()
        return int(cursor.lastrowid or 0)

    def claim_next_queued(self, *, now: datetime | None = None) -> sqlite3.Row | None:
        """Flip the oldest queued row to running and return it, atomically."""
        row = self.conn.execute(
            "SELECT * FROM delegated_tasks WHERE status='queued' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        self.conn.execute(
            "UPDATE delegated_tasks SET status='running' WHERE id=?", (row["id"],)
        )
        self.conn.commit()
        return self.conn.execute(
            "SELECT * FROM delegated_tasks WHERE id=?", (row["id"],)
        ).fetchone()

    def mark_task_done(self, task_id: int, result: str, *, now: datetime | None = None) -> None:
        self.conn.execute(
            "UPDATE delegated_tasks SET status='done', result=?, finished_ts=? WHERE id=?",
            (result, utc_iso(now or datetime.now(UTC)), task_id),
        )
        self.conn.commit()

    def mark_task_failed(self, task_id: int, error: str, *, now: datetime | None = None) -> None:
        self.conn.execute(
            "UPDATE delegated_tasks SET status='failed', error=?, finished_ts=? WHERE id=?",
            (error, utc_iso(now or datetime.now(UTC)), task_id),
        )
        self.conn.commit()

    def pending_tasks(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM delegated_tasks WHERE status IN ('queued', 'running') ORDER BY id"
        ).fetchall()
```

Confirm `UTC` and `datetime` are already imported at the top of `store.py` (they are used by `record_tool_call`); if `UTC` is imported as `from datetime import UTC`, reuse it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_store.py -k task -v && python3 -m pytest tests/test_store.py -q`
Expected: new PASS; the rest of `test_store.py` still PASS (schema only added a table).

- [ ] **Step 5: Commit**

```bash
git add daemon/memory/schema.sql daemon/memory/store.py tests/test_store.py
git commit -m "store: a durable delegated_tasks queue (enqueue/claim/mark/pending) [FROZEN schema: +1 table]"
```

---

### Task 4: The `delegate_task` tool

**Files:**
- Create: `daemon/tools/delegate.py`
- Test: `tests/test_delegate_tool.py`

**Interfaces:**
- Consumes: `DELEGATE_TOOL_NAME` (Task 1); `Store.enqueue_task` (Task 3); the `Tool` protocol (`spec`, `risk`, `preview`, `async run`).
- Produces: `class DelegateTask` with `__init__(self, enqueue: Callable[[str], int], *, notify: Callable[[], None] | None = None)` where `enqueue(request) -> task_id`; `spec: ToolSpec` named `DELEGATE_TOOL_NAME`; `risk = "safe"`; `run(arguments)` enqueues and returns the ack string `DELEGATE_ACK`.

`risk="safe"` on purpose: enqueuing is not a machine mutation — every real tool the worker's loop later runs is gated normally by the origin gate and `ToolPolicy`. Making the enqueue itself `guarded` would only add an approval the voice turn cannot answer.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_delegate_tool.py
import pytest

from daemon.tools.delegate import DELEGATE_ACK, DelegateTask
from daemon.tools.schema import DELEGATE_TOOL_NAME


async def test_run_enqueues_the_request_and_returns_the_ack():
    seen = {}
    fired = []
    tool = DelegateTask(
        enqueue=lambda request: seen.setdefault("request", request) or 7,
        notify=lambda: fired.append(True),
    )
    out = await tool.run({"request": "노션에 하위 페이지 만들어줘"})
    assert seen["request"] == "노션에 하위 페이지 만들어줘"
    assert out == DELEGATE_ACK
    assert fired == [True]  # the worker was signalled


async def test_run_rejects_an_empty_request():
    from daemon.tools.base import ToolError

    tool = DelegateTask(enqueue=lambda request: 1)
    with pytest.raises(ToolError):
        await tool.run({"request": "   "})


def test_spec_is_named_and_flat():
    from daemon.tools.schema import is_flat_schema

    tool = DelegateTask(enqueue=lambda request: 1)
    assert tool.spec.name == DELEGATE_TOOL_NAME
    assert tool.risk == "safe"
    assert is_flat_schema(tool.spec.parameters)  # the voice model must be able to call it
```

Mark the file `async` per the repo's pytest-asyncio convention (check the top of an existing `tests/test_*` that uses `async def` tests — mirror its marker/config).

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_delegate_tool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'daemon.tools.delegate'`

- [ ] **Step 3: Write minimal implementation**

```python
# daemon/tools/delegate.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_delegate_tool.py -v && python3 -m ruff check daemon/tools/delegate.py tests/test_delegate_tool.py`
Expected: PASS, ruff clean

- [ ] **Step 5: Commit**

```bash
git add daemon/tools/delegate.py tests/test_delegate_tool.py
git commit -m "tools: delegate_task - a flat tool that queues work and returns a true ack"
```

---

### Task 5: Result delivery by presence

**Files:**
- Create: `daemon/delegation.py` (this task adds `deliver_result`; Task 6 adds the worker to the same file)
- Test: `tests/test_delegation.py`

**Interfaces:**
- Consumes: `MachinePresence.read() -> Reading` (with `Reading.at_keyboard`), `LocalSpeaker.say(text) -> bool`, `Channel.send(OutboundMessage)`, `OutboundMessage(text=..., recipient_id=...)`.
- Produces: `async def deliver_result(text: str, *, presence, speaker, channel, recipient_id: str | None) -> str` — routes by presence: when the owner is at the keyboard, speak **and** send; otherwise send to the channel only. Returns the route actually achieved: one of `"both"`, `"telegram"`, `"local_speaker"`, `"none"`. Never raises — a failed speak or send degrades the route, it does not lose the result. `presence`, `speaker`, `channel` may each be `None` (a headless or channel-less install); a `None` collaborator simply contributes nothing.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_delegation.py
from daemon.channels.base import OutboundMessage
from daemon.delegation import deliver_result


class _FakeReading:
    def __init__(self, at_keyboard):
        self.at_keyboard = at_keyboard


class _FakePresence:
    def __init__(self, at_keyboard):
        self._r = _FakeReading(at_keyboard)

    async def read(self):
        return self._r


class _FakeSpeaker:
    def __init__(self, ok=True):
        self.said = []
        self._ok = ok

    async def say(self, text):
        self.said.append(text)
        return self._ok


class _FakeChannel:
    name = "telegram"

    def __init__(self, ok=True):
        self.sent = []
        self._ok = ok

    async def send(self, message: OutboundMessage):
        if not self._ok:
            raise RuntimeError("channel down")
        self.sent.append(message)


async def test_at_keyboard_speaks_and_sends():
    speaker, channel = _FakeSpeaker(), _FakeChannel()
    route = await deliver_result("만들었어요", presence=_FakePresence(True),
                                 speaker=speaker, channel=channel, recipient_id="42")
    assert route == "both"
    assert speaker.said == ["만들었어요"]
    assert channel.sent[0].text == "만들었어요"
    assert channel.sent[0].recipient_id == "42"


async def test_away_sends_to_channel_only():
    speaker, channel = _FakeSpeaker(), _FakeChannel()
    route = await deliver_result("만들었어요", presence=_FakePresence(False),
                                 speaker=speaker, channel=channel, recipient_id="42")
    assert route == "telegram"
    assert speaker.said == []
    assert len(channel.sent) == 1


async def test_channel_failure_degrades_route_not_raises():
    speaker, channel = _FakeSpeaker(), _FakeChannel(ok=False)
    route = await deliver_result("x", presence=_FakePresence(True),
                                 speaker=speaker, channel=channel, recipient_id="42")
    assert route == "local_speaker"  # spoke, send failed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_delegation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'daemon.delegation'`

- [ ] **Step 3: Write minimal implementation**

```python
# daemon/delegation.py
"""Running an owner's delegated request in the background, and reporting it back.

A voice turn hands work off through `delegate_task`; this is where that work is
actually run - through the same text `ConversationLoop` the Telegram path uses, so
a nested-schema tool the voice model could not call is called here where it can be -
and reported by presence. See docs/superpowers/specs/2026-08-14-async-delegation-design.md.
"""

from __future__ import annotations

import logging
from typing import Any

from daemon.channels.base import OutboundMessage

logger = logging.getLogger(__name__)


async def deliver_result(
    text: str, *, presence: Any, speaker: Any, channel: Any, recipient_id: str | None
) -> str:
    """Route a finished result to the owner by presence. Never raises.

    At the keyboard: speak it and send it. Away: send it. A speak or send that
    fails degrades the route rather than losing the result - the reply already
    happened, and raising here would strand it.
    """
    at_keyboard = False
    if presence is not None:
        try:
            reading = await presence.read()
            at_keyboard = bool(reading.at_keyboard)
        except Exception:
            logger.exception("delegation: presence read failed; treating as away")

    spoke = False
    if at_keyboard and speaker is not None:
        try:
            spoke = await speaker.say(text)
        except Exception:
            logger.exception("delegation: could not speak the result")

    sent = False
    if channel is not None:
        try:
            await channel.send(OutboundMessage(text=text, recipient_id=recipient_id))
            sent = True
        except Exception:
            logger.exception("delegation: could not send the result to the channel")

    if spoke and sent:
        return "both"
    if sent:
        return "telegram"
    if spoke:
        return "local_speaker"
    logger.warning("delegation: result reached nobody: %s", text[:80])
    return "none"
```

Confirm the presence `Reading` property that means "the owner is actively at the machine" is `at_keyboard` (grep `daemon/proactivity/presence.py` / `daemon/proactivity/base.py` for the `Reading` dataclass). If it is named differently, use that name and update the fakes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_delegation.py -v && python3 -m ruff check daemon/delegation.py tests/test_delegation.py`
Expected: PASS, ruff clean

- [ ] **Step 5: Commit**

```bash
git add daemon/delegation.py tests/test_delegation.py
git commit -m "delegation: report a finished result by presence - speak-and-send at the keyboard, send when away"
```

---

### Task 6: The worker + capture channel

**Files:**
- Modify: `daemon/delegation.py` (add `CaptureChannel`, `DelegationWorker`)
- Test: `tests/test_delegation.py` (add cases)

**Interfaces:**
- Consumes: `Store.claim_next_queued`, `mark_task_done`, `mark_task_failed` (Task 3); `deliver_result` (Task 5); the `Channel` protocol.
- Produces:
  - `class CaptureChannel` — a `Channel` with `name = "delegate"`, whose `async send(message)` records `self.reply = message.text`, and whose `listen()` is never used by the worker (it yields nothing).
  - `class DelegationWorker(__init__(self, store, run_request: Callable[[str], Awaitable[str]], deliver: Callable[[str, sqlite3.Row], Awaitable[None]], *, wake: asyncio.Event, poll_seconds: float = 5.0))`. `run_request(request_text) -> reply_text` runs one task to completion (Task 7 builds the real one from a text `ConversationLoop`; tests inject a fake). `deliver(reply_text, task_row)` reports it (Task 7 wraps `deliver_result`). Methods: `async drain_once() -> bool` (claim+run+mark+deliver exactly one task; returns whether one ran), and `async run()` (loop: `drain_once` until empty, then `await wake` or timeout `poll_seconds`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_delegation.py (add)
import asyncio

from daemon.delegation import CaptureChannel, DelegationWorker


async def test_capture_channel_records_the_reply():
    from daemon.channels.base import OutboundMessage

    ch = CaptureChannel()
    await ch.send(OutboundMessage(text="만들었어요"))
    assert ch.reply == "만들었어요"


async def test_worker_runs_a_queued_task_marks_done_and_delivers(store):
    tid = store.enqueue_task(request="노션에 페이지 만들어줘", origin="owner",
                             channel="voice", sender_id=None)
    delivered = []

    async def fake_run(request):
        assert request == "노션에 페이지 만들어줘"
        return "만들었어요"

    async def fake_deliver(text, task_row):
        delivered.append((text, task_row["id"]))

    worker = DelegationWorker(store, fake_run, fake_deliver, wake=asyncio.Event())
    ran = await worker.drain_once()
    assert ran is True
    assert delivered == [("만들었어요", tid)]
    row = store.conn.execute("SELECT status, result FROM delegated_tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "done" and row["result"] == "만들었어요"


async def test_worker_marks_failed_and_delivers_the_failure(store):
    tid = store.enqueue_task(request="r", origin="owner", channel="voice", sender_id=None)
    delivered = []

    async def boom(request):
        raise RuntimeError("notion 400")

    async def fake_deliver(text, task_row):
        delivered.append(text)

    worker = DelegationWorker(store, boom, fake_deliver, wake=asyncio.Event())
    ran = await worker.drain_once()
    assert ran is True
    assert delivered and "notion 400" in delivered[0]
    row = store.conn.execute("SELECT status, error FROM delegated_tasks WHERE id=?", (tid,)).fetchone()
    assert row["status"] == "failed" and "notion 400" in row["error"]


async def test_drain_once_returns_false_when_the_queue_is_empty(store):
    worker = DelegationWorker(store, None, None, wake=asyncio.Event())
    assert await worker.drain_once() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_delegation.py -k "worker or capture" -v`
Expected: FAIL — `ImportError: cannot import name 'CaptureChannel'`

- [ ] **Step 3: Write minimal implementation** (append to `daemon/delegation.py`)

```python
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from daemon.channels.base import InboundMessage  # noqa: E402 - grouped with the additions

FAILURE_PREFIX = "그 작업을 하려다 실패했어"


class CaptureChannel:
    """A `Channel` that keeps the loop's reply instead of sending it.

    `ConversationLoop.handle` ends by `channel.send(...)`; the worker wants that text,
    not a delivery. `listen()` is never driven - the worker calls `handle` directly.
    """

    name = "delegate"

    def __init__(self) -> None:
        self.reply: str | None = None

    async def send(self, message: Any) -> None:
        self.reply = message.text

    async def listen(self) -> AsyncIterator[InboundMessage]:
        return
        yield  # pragma: no cover - makes this an async generator; never driven


class DelegationWorker:
    """Runs queued delegated tasks one at a time and reports each result."""

    def __init__(
        self,
        store: Any,
        run_request: Callable[[str], Awaitable[str]] | None,
        deliver: Callable[[str, Any], Awaitable[None]] | None,
        *,
        wake: asyncio.Event,
        poll_seconds: float = 5.0,
    ) -> None:
        self._store = store
        self._run_request = run_request
        self._deliver = deliver
        self._wake = wake
        self._poll_seconds = poll_seconds

    async def drain_once(self) -> bool:
        """Claim and finish exactly one task. Returns False if the queue was empty."""
        row = self._store.claim_next_queued()
        if row is None:
            return False
        task_id, request = row["id"], row["request"]
        try:
            assert self._run_request is not None
            reply = await self._run_request(request)
            self._store.mark_task_done(task_id, reply)
        except Exception as exc:
            logger.exception("delegation: task %s failed", task_id)
            reply = f"{FAILURE_PREFIX}: {exc}"
            self._store.mark_task_failed(task_id, str(exc))
        if self._deliver is not None:
            try:
                await self._deliver(reply, row)
            except Exception:
                logger.exception("delegation: could not report task %s", task_id)
        return True

    async def run(self) -> None:
        """Drain the queue, then wait for a wake signal or the poll timeout."""
        while True:
            try:
                while await self.drain_once():
                    pass
            except Exception:
                logger.exception("delegation: worker loop error; continuing")
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_delegation.py -v && python3 -m ruff check daemon/delegation.py`
Expected: PASS, ruff clean

- [ ] **Step 5: Commit**

```bash
git add daemon/delegation.py tests/test_delegation.py
git commit -m "delegation: a worker that runs one queued task at a time via a capture channel"
```

---

### Task 7: Wire it into the app + boot recovery + reachability

**Files:**
- Modify: `daemon/delegation.py` (add `build_run_request` factory)
- Modify: `daemon/app.py` (register `delegate_task` in `_build_tools`; build + start the worker in `_lifespan`; boot recovery)
- Modify: `tests/test_reachable.py` (register the new tool/worker or add PENDING with the owning milestone)
- Test: `tests/test_delegation.py` (a `build_run_request` integration-ish test with a fake gateway)

**Interfaces:**
- Consumes: everything above; `ConversationLoop`, `Companion`, `LLMGateway`, `Store`, `MachinePresence`, `LocalSpeaker`, `_build_channel` (all already assembled in `app.py`).
- Produces: `build_run_request(*, gateway, companion_factory) -> Callable[[str], Awaitable[str]]` where `companion_factory() -> Companion` yields a fresh text-mode companion (full tools). Each call builds a `CaptureChannel`, a `ConversationLoop(capture, gateway, companion_factory())`, runs `await loop.handle(InboundMessage(text=request, sender_id=<owner>, received_at=<now>, channel="delegate", authored_by_sender=True))`, and returns `capture.reply or ""`.

- [ ] **Step 1: Write the failing test for `build_run_request`**

```python
# tests/test_delegation.py (add)
from daemon.delegation import build_run_request


async def test_build_run_request_runs_the_text_loop_and_returns_the_reply(monkeypatch):
    captured = {}

    class _FakeLoop:
        def __init__(self, channel, gateway, companion, **kw):
            captured["channel"] = channel
            self._channel = channel

        async def handle(self, inbound):
            captured["inbound"] = inbound
            await self._channel.send(type("M", (), {"text": "만들었어요"})())

    monkeypatch.setattr("daemon.delegation.ConversationLoop", _FakeLoop)
    run = build_run_request(gateway=object(), companion_factory=lambda: object())
    reply = await run("노션에 페이지 만들어줘")
    assert reply == "만들었어요"
    assert captured["inbound"].text == "노션에 페이지 만들어줘"
    assert captured["inbound"].authored_by_sender is True
    assert captured["channel"].name == "delegate"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_delegation.py -k build_run_request -v`
Expected: FAIL — `ImportError: cannot import name 'build_run_request'`

- [ ] **Step 3a: Add `build_run_request` to `daemon/delegation.py`**

```python
from daemon.clock import now as clock_now  # noqa: E402 - grouped with the additions
from daemon.loop import ConversationLoop  # noqa: E402


def build_run_request(
    *, gateway: Any, companion_factory: Callable[[], Any], owner_id: str = "owner"
) -> Callable[[str], Awaitable[str]]:
    """A `run_request` that runs one task through the real text ConversationLoop.

    A fresh CaptureChannel and companion per call: the loop records and answers as
    the text path does, and the captured reply is what gets delivered. Origin is the
    owner (`authored_by_sender=True`) - the request came from the owner by voice.
    """

    async def run_request(request: str) -> str:
        capture = CaptureChannel()
        loop = ConversationLoop(capture, gateway, companion_factory())
        await loop.handle(
            InboundMessage(
                text=request,
                sender_id=owner_id,
                received_at=clock_now(),
                channel="delegate",
                authored_by_sender=True,
            )
        )
        return capture.reply or ""

    return run_request
```

Confirm the clock helper name/import (`daemon/clock.py` — the repo's single timestamp helper; grep it and match, e.g. `from daemon.clock import now`).

- [ ] **Step 3b: Register `delegate_task` in `_build_tools` (`daemon/app.py`)**

In `_build_tools`, after the builtin tools are registered on the `Registry`, add the delegate tool wired to this store and the lifespan's wake event. `_build_tools` must accept the event; thread an optional `delegate_wake: asyncio.Event | None = None` parameter through it (default `None` so the text-only assembly and `run_voice`'s own `_build_tools` call still work). When present:

```python
    from daemon.tools.delegate import DelegateTask

    if delegate_wake is not None:
        registry.register(
            DelegateTask(
                enqueue=lambda request: store.enqueue_task(
                    request=request, origin="owner", channel="voice", sender_id=None
                ),
                notify=delegate_wake.set,
            )
        )
```

The `_build_tools` call inside `_lifespan` (~line 314) passes `delegate_wake=app.state.delegate_wake`; the `run_voice` resident shares the boot-built tool layer, so its voice sessions see the same registered `delegate_task`. (The standalone `daemon voice` CLI and the text-only path pass nothing and simply have no delegate tool — acceptable: delegation is a resident feature.)

- [ ] **Step 3c: Build and start the worker in `_lifespan` (`daemon/app.py`)**

Near where the other lifespan background tasks are created (the `conversation-loop`, `wake-gate` tasks), after `store`, `gateway`, `settings`, the channel and a `MachinePresence`/`LocalSpeaker` are available:

```python
    from daemon.delegation import DelegationWorker, build_run_request, deliver_result
    from daemon.proactivity.presence import MachinePresence
    from daemon.proactivity.speaker import LocalSpeaker

    app.state.delegate_wake = asyncio.Event()  # set BEFORE _build_tools is called

    def _companion_factory() -> Companion:
        return Companion(
            memory, data_dir=settings.data_dir, recall=recall,
            recall_limit=settings.recall_limit, resolve_id=resolve_id, tools=tools,
        )

    _run_request = build_run_request(gateway=gateway, companion_factory=_companion_factory)
    _presence = MachinePresence()
    _speaker = LocalSpeaker()
    _owner_recipient = settings.owner_chat_id  # the same recipient the proactive path sends to; use whatever app.py already uses for the owner's channel id

    async def _deliver(text: str, task_row: Any) -> None:
        await deliver_result(text, presence=_presence, speaker=_speaker,
                             channel=channel, recipient_id=_owner_recipient)

    worker = DelegationWorker(store, _run_request, _deliver, wake=app.state.delegate_wake)

    # Boot recovery: a restart may have left tasks unfinished. Report, do not resume.
    for row in store.pending_tasks():
        await _deliver(
            f"아까 부탁한 '{row['request'][:40]}' 작업을 다 못 끝내고 재시작됐어. 다시 시켜줘.",
            row,
        )
        store.mark_task_failed(row["id"], "interrupted by restart")

    app.state.delegation_task = asyncio.create_task(worker.run(), name="delegation-worker")
```

Match the real names `app.py` uses for `memory`, `recall`, `resolve_id`, `channel`, and the owner recipient id at that point in `_lifespan` (read the surrounding assembly — the `ConversationLoop(...)` block at ~343 shows `memory`, `recall`, `resolve_id`, `channel`, `gateway`, `tools` all in scope). If the owner's channel recipient id is not already a variable there, reuse whatever `_build_channel`/the proactive delivery uses to address the owner.

- [ ] **Step 3d: Reachability** — in `tests/test_reachable.py`, register that `DelegateTask`, `DelegationWorker`, `CaptureChannel`, and `build_run_request` are constructed by `app.py`, or add them to `PENDING_CLASSES`/`PENDING_WIRING` with the owning milestone. Run the reachability gate to see exactly what it wants:

Run: `python3 -m pytest tests/test_reachable.py -v`

- [ ] **Step 4: Run the whole suite + lint + docs**

Run: `python3 -m pytest -q && python3 -m ruff check . && python3 scripts/check_docs.py`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add daemon/delegation.py daemon/app.py tests/test_delegation.py tests/test_reachable.py
git commit -m "app: wire delegate_task and the delegation worker into the resident, with boot recovery"
```

---

### Task 8: Live verification (manual, real key)

**Files:**
- Use: `evals/voice_write_nudge_spike.py` (already committed) as the pattern; optionally add a `delegate` cell.

Not a unit test — the whole reason this feature exists is a live-model behaviour. After Task 7, verify on the real path (memory: "verify by running the real thing"):

- [ ] **Step 1:** Extend `evals/voice_write_nudge_spike.py` with a cell that offers the crowded set filtered to `surface="voice"` (flat tools + a `delegate_task` spec) and issues the create request; assert `delegate_task` is called (expect it fires, like the other flat tools). This closes the loop the spike opened.
- [ ] **Step 2:** Run it: `cd ~/Daemon && python3 -m evals.voice_write_nudge_spike --runs 5`. Record the `delegate_task` call rate in the spike's docstring and `evals/CLAUDE.md`.
- [ ] **Step 3:** Drive the assembled resident once by voice ("...하위 페이지 만들어줘"), and confirm from `daemon tools log` / the `delegated_tasks` table that a row went `queued → running → done` and the result was delivered — the end-to-end path the unit tests fake.

---

## Self-Review

**Spec coverage:**
- Voice gets flat tools + delegate only → Tasks 1, 2 (classifier + surface split), Task 4 (the tool), Task 7 (wiring). ✓
- Auto schema classification (not a whitelist) → Task 1. ✓
- Durable queue → Task 3. ✓
- Truthful ack after commit → Task 4 (`run` enqueues then returns `DELEGATE_ACK`). ✓
- Worker reuses the text loop via a capture channel → Tasks 5–7. ✓
- Presence-routed delivery, not on the proactive budget → Task 5 (`deliver_result` uses presence/speaker/channel directly, records nothing to the proactive utterance table). ✓
- Boot recovery reports unfinished → Task 7 Step 3c. ✓
- Frozen schema change declared → Task 3 commit message + spec note. ✓
- Testing incl. Korean + reachability → each task's tests + Task 7 Step 3d. ✓
- Live premise/verification → Task 8. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". The two "confirm the exact name" notes (presence `at_keyboard`, `daemon.clock` import, the owner recipient id) are explicit lookups against named files with a fallback, not deferred design.

**Type consistency:** `enqueue_task(*, request, origin, channel, sender_id)` used identically in Task 3 (def), Task 7 (`lambda request: store.enqueue_task(request=..., origin="owner", channel="voice", sender_id=None)`). `run_request(str)->str`, `deliver(str, Row)->None`, `wake: asyncio.Event` consistent across Tasks 6–7. `DELEGATE_TOOL_NAME`/`DELEGATE_ACK` referenced by their defining module throughout. `deliver_result(text, *, presence, speaker, channel, recipient_id)` signature identical in Tasks 5 and 7.
