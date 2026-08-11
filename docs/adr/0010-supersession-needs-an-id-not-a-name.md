# 0010 — Supersession keys off an id, not a name the model reinvents

**Status:** accepted · 2026-08-11 · measured

## Context

[0001](0001-borrow-the-memory-design.md) borrowed "supersession keys so facts
replace instead of accumulating contradictions". The mechanism was built
correctly: `Store.insert_entry` retires the active row holding the same
`supersession_key`, points it at its replacement, and a partial unique index
(`schema.sql:199`) keeps at most one active row per key. Unit tests passed.

Five nights of real reflection showed it never fires. The live curated tier held
three overlapping facts about the same subject:

| id | key | body |
|---|---|---|
| 1 | `user_name` | 사용자의 이름은 김대현이다 |
| 2 | `user_job` | 사용자는 개발자(벨라의 창조주/개발자)이다 |
| 5 | `user_profile` | 이름은 김대현이며, 9년차 이상의 AI/LLM Application Engineer 겸 Full-Stack Developer이다 |

The cause is in the prompt, not the store: `_reflect` sends `SYSTEM` plus the
day's transcript and nothing else, so the model cannot see which keys already
exist. Asked to name a key for a fact about the owner's identity, it invents a
fresh plausible name each night. Supersession fires on string equality between
two independent acts of naming, which is not a mechanism — it is a hope.

The cost is not the wasted injection budget (9 of 50). It is that a stale fact
survives under a different key: when the owner's job changes, `user_job` gets
superseded and `user_profile` keeps the old title, and both are injected on every
turn. That is precisely the contradiction 0001 bought supersession to prevent.

## Decision

The reflection prompt carries the active facts with their row ids, and the model
answers with `updates: <id> | null` per fact. Supersession keys off that id.

Three constraints on the shape:

* **No DELETE.** The model may ADD or UPDATE, never retire on its own. Retirement
  is only ever a side effect of a successor being written, so no path exists for a
  fact to vanish without its replacement. mem0's resolver, which does get DELETE,
  is reported silently dropping context-scoped facts it mistook for contradictions
  — a memory system's worst failure is the one it does not announce.
* **`updates` is hostile input**, like everything else in `daemon/reflection.py`: a
  non-integer, an unknown id, or an already-retired id is dropped and reported,
  and the fact is still added. A model must not be able to retire a memory by
  pointing at an arbitrary number.
* **Still one model call a night.** The fact set is bounded by the injection
  budget, so it fits in the existing call. A second resolve call would buy nothing
  and break the economy [0008](0008-three-stages-one-model-call.md) sets.

`supersession_key` stays. It still does useful work within a single night
(`_one_per_key`) and for the case where the model does reuse a key; the id is the
stronger signal layered over it. When both point somewhere, both rows retire into
the new one — otherwise the unique index rejects the insert.

No schema change: `status` and `superseded_by` already exist. Retire-not-delete is
what Zep/Graphiti call bi-temporal edge invalidation, and this schema had it from
the start. What was missing was never the storage. It was the read of its own past
on the way in.

## Consequences

Reflection stops being write-only with respect to memory. It sees what it knows
before deciding what to record, which is the step every production memory system
(mem0's ADD/UPDATE/NOOP, Hindsight's write-time entity resolution) converged on.

The prompt grows by the active fact list, bounded at 50 entries.

Merging is left conservative on purpose. Retiring id 1 into id 5 is safe because
5 contains 1 verbatim; id 2 was left active because it carries "built this daemon",
which 5 does not say. An overlap is not a contradiction, and a merge that loses a
clause is the same silent failure the DELETE ban exists to avoid.

## What would change our mind

If the model starts pointing `updates` at wrong-but-plausible ids — consolidating
facts that merely sound alike — then id-matching has the same weakness as
name-matching, one layer down, and the resolve step needs to move to its own call
with the candidate set narrowed by similarity first.
