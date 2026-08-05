# 0003 — File ownership split is the persona anchor

**Status:** accepted · 2026-08-03

## Context

A personality that adapts to one person will drift toward agreeing with them.
PLAN §5 named this risk (over-adaptation, echo chamber) without a mechanism, and a
mechanism written as a rule the AI must follow is a rule the AI can rewrite.

## Decision

Split ownership by file, and let the filesystem carry the guarantee:

```
data/persona/seed.md      human writes; code never writes
data/persona/learned.md   AI writes; human reads, or asks for a rule to be dropped
data/memory/**            AI writes
```

The prompt is assembled from both. **The anchor is not a rule, it is the fact that
the AI cannot reach `seed.md`** — including the line the wizard always appends:
*I do not simply agree. When I think you are wrong, I say so.*

Rule metadata (created_at, evidence, status, supersession) goes in SQLite columns,
not prose comments, so a model cannot forge a rule's provenance or quietly retire
one.

## Consequences

This is also the write-conflict answer: the file a person edits in Obsidian and the
file the AI rewrites are never the same file. Rate limiting becomes simple counting
— an active-rule ceiling and a per-cycle addition ceiling.

The anchor line is a quality feature, not only an ethics one. A companion that only
agrees is boring and untrustworthy, and the local models' default register is
exactly that.

## What would change our mind

Nothing about the ownership split. The specific anchor text is worth revising once
there are weeks of real observations to read.
