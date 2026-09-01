# 0020 — Lip-sync makes a clip ambient

**Status:** accepted · 2026-08-31 · scopes [0017](0017-the-neutral-moment-not-the-matched-pose.md) · measured

## Context

The face plays pre-rendered clips, and until now the clip carried the mouth. A
clip with a closed mouth is a clip that is not talking, so speech had to *mean* a
clip change: the moment the daemon started speaking, the page cut to a speaking
clip — mid-clip, whatever the outgoing clip happened to be in the middle of
doing. [ADR 0017](0017-the-neutral-moment-not-the-matched-pose.md) is entirely
about surviving that cut: wait for the outgoing clip's next near-neutral moment,
and pose-match the entry for the measured 14% where that wait times out.

Lip-sync takes the mouth off the clip. The renderer generates a mouth for
whichever clip is on screen, so no particular clip is required in order to speak
and nothing has to interrupt anything — speech begins where the face already is,
and a clip can be allowed to reach its own end. The only join left is **clip end
→ the next clip's frame 0**.

That changes what a clip is *for*, which is why this is a record and not a diff.
0017's two mechanisms are not being deleted because they were wrong. They are
going unused on this path because the switch they were built to survive no longer
happens.

## The measurement

Downscaled whole-frame mean absolute difference across the join, over the owner's
real ten prepared clips. **The baseline is a clip's own loop point** — the join
the face has always made, every few seconds, that the owner has never remarked
on.

| join | min | median | p90 | max | over baseline max (2.14) |
|---|---|---|---|---|---|
| a clip's own loop point (**baseline**) | 0.88 | 1.14 | — | 2.14 | — |
| **clip end → next clip frame 0** | 1.08 | **1.41** | — | 2.18 | **3 / 90 (3%)** |
| a "near-neutral" moment → one-shot frame 0 | 0.94 | 1.51 | 5.06 | 8.95 | 85 / 252 (**34%**) |
| any moment → one-shot frame 0 (cut now) | 1.40 | 7.98 | 12.54 | 12.82 | 76 / 84 (90%) |

**Letting every clip finish is smooth by construction.** end→0 sits on the
baseline, and only 3 of 90 pairs exceed the baseline's own worst case, by
0.01–0.04. Nothing has to be built to get that; it is what the join already is.

**Waiting for a near-neutral moment does not reliably buy it.** The median
(1.51) looks fine and the distribution is the problem: p90 5.06, max 8.95, and a
third of samples worse than the baseline's worst case. `daemon/face_match.py`'s
`neutral` flag is computed against *each clip's own frame 0*, not a pose shared
across clips, so "neutral for idle2" does not imply "close to `amused`'s frame
0" — `idle2@1.75s` is flagged neutral and is 8.94 from it. The flag is also
coarser than the cut it is asked to place: the buckets are 0.5s (12 frames of the
played clip), the flag means *some* frame in the slice qualifies rather than the
frame the cut lands on, and the runtime only ever knows the bucket. That
imprecision belongs to the mechanism, not to this measurement.

**An immediate cut is far worse** — median 7.98, 90% of samples over the
baseline — and that is the thing 0017 built the wait to avoid. It did avoid it:
both middle rows beat this row comfortably. The finding here is not that the wait
fails, it is that a boundary join needs no help at all.

One scope note, because the table is narrower than its headline. Both one-shot
rows measure entry at **frame 0**, which is what a one-shot has to do — the
one-shots are excluded from the pose-match table on purpose
(`face_match.py:ONE_SHOTS`, rule 3). Neutral-wait-then-pose-matched
loop-to-loop entry, the other half of 0017's path, is **not** in this table,
and 0017's own numbers still govern it.

## Decision

**Under lip-sync a clip is ambient body motion: it plays from frame 0 to its own
end, and whatever wants to follow it waits for that boundary.** Neither of
0017's mechanisms is consulted on that path.

| | the clip carries the mouth (v1) | lip-sync carries the mouth |
|---|---|---|
| what a clip is | reactive — it must change when speech starts | ambient — body motion under a generated mouth |
| when a switch happens | mid-clip, when the event arrives | at the current clip's own end |
| the join being made | `outgoing[t]` → `incoming[seek]` | end → frame 0 |
| neutral wait, pose match | **what makes that join survivable** | **unused** — the join is already the baseline |

**ADR 0017 is not overturned. It is scoped** — the shape
[ADR 0015](0015-code-may-search-where-the-model-may-not.md) used on
non-negotiable 10 and [ADR 0018](0018-a-declared-expression-is-not-a-tool-call.md)
used on 12: the original keeps its wording for the path it describes, and a
narrower path is carved out, named and justified.

0017 governs the **v1 fallback**, which is untouched: `daemon/static/face.html`'s
own policy, what a browser gets when the renderer latches `failed` or lip-sync is
off. There the mouth is back in the clips, a clip is reactive again, and every
sentence of 0017 applies unchanged — the neutral wait, the 1.2s cap the owner
watched and kept, and pose matching on the tail. It also governs any clip
lip-sync does not drive: `speaking_loud` and `speaking_soft` have no prepared
cache deliberately, because they are chosen by loudness and driving them would
swap the clip every time the owner raised his voice.

The boundary rule has one structural consequence worth recording: clip choice
moves out of the page into a pure server module of its own (`face_clips`),
assembled by `daemon/app.py` (CONTRACTS 4). The renderer composites onto the
frame it believes is on screen, so it cannot be told which clip is up a round
trip late.

## Consequences

**The cost is expression latency, and that is the whole cost.** A one-shot queued
to the clip end arrives after half a clip on average — **4.0s** behind
idle1/2/3, **3.1s** behind `listening`, **3.0s** behind `thinking`. The owner
chose that over a picture that snaps one time in three, with both numbers in
front of him.

**He chose it on paper, and it has not yet been judged in use.** The plan's live
check asks him directly whether an expression arriving 3–4s late is acceptable in
conversation, and that answer is the one that settles this, not the table. Until
it comes in, this ADR records a decision taken on measured joins and an accepted
latency, not a verified feel.

**The premise is verified on fewer clips than it applies to.** "The mouth is
generated for whatever clip is up" is load-bearing here, and nothing has rendered
a mouth onto `sulky`, `curious` or `flourish_arms` in any path. `amused` was
rendered offline and judged good by the owner (2026-08-31); the live check's third
question asks whether the mouth still matches on clips other than `idle2`, the
single clip the previous wiring drove. A no there costs the driveable set — those
clips fall back to v1, where 0017 governs — not this decision.

**The table keeps its version 3 schema and its producer.** `neutral` and `match`
are still built into `<data_dir>/face/transitions.json` by
`daemon face-transitions` and still served, because the fallback page still reads
them. What this removes is a consumer, not a lookup.

**A `daemon/face_match.py` docstring was over-claiming and is corrected with this
record.** `ONE_SHOTS` said each one-shot is an arc from "the **shared** neutral
pose" and back; its `neutral` flag is computed against each clip's own frame 0
(`NEUTRAL_THRESHOLD_FRACTION`), which is a different claim, and `idle2@1.75s` at
8.94 from `amused`'s frame 0 is the data separating the two. Only that word came
out, plus the bucket's own imprecision written next to
`NEUTRAL_THRESHOLD_FRACTION`, where a reader meets the flag. The exclusion of
one-shots from the table stays, and stays for the reason the docstring already
gives: entering an arc mid-way destroys the arc.

## What would change our mind

**The owner's answer to the latency question.** If 3–4s late is not acceptable in
use, the fix is *not* a return to a mid-clip cut — rows three and four are what
that costs. It is the refinement he deferred ("나중에 고도화"): choose the cut
moment by measuring the join at runtime instead of trusting a bucket flag. That
buys the latency back, and it needs one-shots in the match table, which
`daemon/face_match.py` excludes today.

**0017's own condition, which reaches this record too.** If the clips are ever
regenerated as one continuous take per motion group, a real motion manifold
exists between them, and the mid-clip question reopens — for the fallback path,
which is the only path that asks it.

**A clip the renderer cannot drive well** leaves the driveable set and is reactive
again, under 0017. The boundary rule survives that; the set shrinks.
