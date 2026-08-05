# 0001 — The memory layer is borrowed, not invented

**Status:** accepted · 2026-08-03

## Context

The first plan treated graph memory with recall scoring as the differentiator.
Reading OpenClaw's actual implementation showed it already ships the same design,
down to the same formula: similarity × exponential recency decay with a 30-day
half-life × importance 1–10. Provenance in columns, daily notes, a consolidation
pass, an Obsidian-friendly vault plugin — all of it.

## Decision

Port that design rather than reinvent it, and stop claiming memory as a
differentiator. Specifically borrowed: the five-tier split between always-injected
curated context and search-only episodic files; provenance in DB columns a model
cannot write in prose; supersession keys so facts replace instead of accumulating
contradictions; recall Lane 1 making zero model calls; and the hygiene rule that
proactive and reflection sessions cannot create durable memories.

The differentiators narrow to two: **persona self-evolution** and **autonomous
proactivity**. Neither exists anywhere else — OpenClaw's personality file is
human-edited and its own docs say the consolidation pass does not touch it, and
both it and Hermes are reactive.

## Consequences

Two months of work that would have produced a worse version of something that
exists went instead into the two things nobody has done. The cost is that README
and PLAN must not present memory as novel; doing so reads as imitation and invites
the comparison we would lose.

## What would change our mind

If the borrowed design turns out to be wrong for Korean or for one user's scale.
Partly already true — see [0005](0005-vectors-belong-in-m1b.md).
