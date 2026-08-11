# 0011 — core.md holds every active fact, not the injected ones

**Status:** accepted · 2026-08-11 · measured

## Context

`CuratedMemory.add` rewrites `data/memory/core.md` whole on every write, rendering
it from `self.entries(MAX_INJECTED)` — the top 50 rows by importance. One number
was doing two jobs: bounding what gets injected into every turn, and deciding what
the source-of-truth file contains.

Since the write is a whole replace, the 51st active fact was simply absent from the
next render. Measured on a throwaway database: 55 `add()` calls leave 55 active rows
in the mirror and 50 bullets in the file.

Nothing catches it afterwards. `rebuild` only inserts bodies it finds in the
markdown, so `daemon reindex` cannot notice a fact that is missing from it; recall
still serves the row from the mirror (`TRIGGER_SCAN` is `10 * MAX_INJECTED`), so the
daemon keeps quoting prose the source of truth no longer holds; and deleting the
database — the documented recovery — loses those facts for good. Measured: after
wiping the mirror, `rebuild` restored 50 of 55.

[0002](0002-one-process-sqlite-markdown.md) makes markdown the original and the
mirror the derivative. A file that silently holds less than the mirror inverts that
for every fact past the budget.

## Decision

The file holds every active fact; the budget bounds only injection.

`ALL_ACTIVE = 10_000` renders `data/memory/core.md` — the same ceiling `rebuild`
already reads with, so it is a runaway guard rather than a second budget.
`MAX_INJECTED` keeps its one job: `CuratedMemory.entries()` still defaults to it,
`recall` still truncates to it, and reflection still shows the model that many rows
(ADR 0010). None of those paths changed.

Found by a reliability audit of [0010](0010-supersession-needs-an-id-not-a-name.md)
rather than by use: the live tier holds 8 facts and reflection caps at 8 a night, so
this was roughly a week of productive passes away.

## Consequences

`data/memory/core.md` grows past 50 bullets on a machine that has accumulated more,
and a `daemon reindex` after this change restores facts a previous one would have
dropped — so post-reindex row counts can go up. Nothing in the tree assumed a
bounded file.

The two numbers can now drift apart deliberately: raising the injection budget is a
context-window decision, and it no longer silently changes what the source of truth
records.

## What would change our mind

If the file ever grows large enough that rewriting it whole on every fact becomes
the cost that matters, the answer is an append-and-compact format rather than a
smaller render — truncating the original to keep the write cheap is what this
decision rejects.
