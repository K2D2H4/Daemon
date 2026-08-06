# daemon/ — recipes

How to add the things this package is extended with. Each one names the step that
was actually missed the time it shipped broken; orientation is in
[CLAUDE.md](CLAUDE.md), the binding rules are in
[docs/CONTRACTS.md](../docs/CONTRACTS.md).

**A new LLM provider.** Implement `Provider` in `daemon/llm/base.py` under `daemon/llm/providers/`, name
it in `HOSTED_PROVIDERS`, build it in `daemon/app.py`, offer it in `daemon/setup.py`. Missing the last two shipped once.

**A new channel.** Implement `Channel` in `daemon/channels/base.py`; the `Cursor` it needs is already
`daemon.memory.store.Store`. **Don't** invent a second allowlist — `daemon/channels/pairing.py` owns who the owner is.

**A new tool.** Implement `Tool` in `daemon/tools/base.py` (`spec`, `risk`, `preview`, `run`) and
register it in `app._build_tools` — the `Registry` is built there, so a tool nobody registers is
the unreachable-implementation defect again. Two of those four members are what the runner leans
on: `risk` is `'safe'` only if the call is read-only, local, and has no effect the owner would want
to be asked about (**unsure means `'guarded'`**), and `preview` returns one line carrying *this
call's* specifics — the command, the path — because it is what the approval message shows and what
the `tool_calls` row records. **Do not decide anything about permission inside the tool**:
`tools/policy.py` owns that, makes no model calls, and its origin gate refuses every call on a turn
that is not the owner's in every mode, `full` included. And per CONTRACTS 13, never build code from
data — the script is a constant, the values travel as `argv`.

**A new background job.** Register it on the scheduler in `daemon/app.py`'s lifespan
and copy the reflection job or the proactivity tick: `timezone=None` to fire on *local*
time, `max_instances=1` + `coalesce=True` so a slow run cannot stack, a top-level
`except` in the tick because a raising job is logged once and the schedule then reads
healthy forever, and **a CLI command running the same object** — nobody is awake at
04:00 to read a log. Register it only when its switch is on — an absent job is a clearer
"off" than a disabled one — and give polling a floor; see the last bullet.

**Anything a model writes into a path, a score or a date.** Reflection's output names files, scales
the recall score and can retire a fact the user stated, so `daemon/reflection.py` clamps every number,
narrows keys to a fixed charset, and checks names with `entities.safe_name` **and** a boundary on the
resolved path — a blocklist is not exhaustive. Date records by the day they are *about*.

**A new `Task`.** `daemon/tasks.py` is the routing key: add it to all three presets in
`daemon/config.py` *and* give it a caller, or `tests/test_reachable.py` fails unless it is declared
PENDING with the milestone that owns it.
