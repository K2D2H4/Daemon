# 0017 — The neutral moment, not the matched pose

**Status:** accepted · 2026-08-26 · measured

## Context

The face plays twelve pre-rendered clips and crosses from one to another when
the daemon changes what it is doing. Running it showed loop-to-loop transitions
still reading as cuts, and the cause was measured rather than guessed: a
transition blends `outgoing[t]` into `incoming[0]`, and every clip's near-neutral
pose sits at its own start and end, so `incoming[0]` is usually far from whatever
pose `outgoing` happened to be in.

Three entry strategies, over all 56 loop-to-loop pairs of the owner's real clips
(10fps, 32×48 grey thumbnails, mean distance over a ±200ms window):

| strategy | median distance |
|---|---|
| `currentTime = 0` | 6.4 |
| same normalised phase | **9.6 — worse** |
| best-matching frame | **3.8** |

So `daemon/face_match.py` was built: [Video
Textures](https://www.researchgate.net/publication/2624304_Video_Textures)
(Schödl et al., SIGGRAPH 2000) applied *across* clips instead of within one,
picking the frame of the incoming clip that looks most like the outgoing one.

Three follow-ups then arrived on top of it, and the third overturns the order of
the first two. This record exists because that reversal is expensive to
rediscover and easy to undo by accident — the earlier reasoning is still
persuasive on its own terms.

## Decision

**Waiting for the outgoing clip's next near-neutral moment is the main
mechanism. Pose matching covers the tail where that wait times out.** The
original order was the other way round, and it was wrong.

### 1 — A window, not a frame, and a direction term on top of it

Schödl's own refinement is load-bearing here: a hand halfway *up* and a hand
halfway *down* can be pixel-identical, so single-frame matching splices them and
plays the motion backwards. A window catches it — the neighbourhoods disagree
even when the centre frames match exactly.

The shipped window was ±200ms, sized against exactly that hand example. It was
too narrow for a slower motion: the owner reported a cut landing mid-sigh, and a
breath cycle is 3–4s, so ±200ms samples about a tenth of one and cannot tell
inhaling from exhaling at the same chest position. Measured: **41% of shipped
matches had the incoming clip moving opposite to the outgoing clip** at the
splice.

Widening to ±500ms alone gets that to ~28–30% and plateaus. What fixes it is a
direction term in the cost — `cost = appearance + λ(1 − cos(dir_a, dir_b))`,
swept against this implementation's own `_directions` and `_frames`:

| λ | opposed matches | median cosine | appearance | vs λ=0 |
|---|---|---|---|---|
| 0 | 38% | 0.04 | 24.3 | — |
| 3.0 | 34% | 0.07 | — | — |
| 10 | 31% | 0.08 | — | — |
| 30 | 26% | 0.11 | 26.0 | +7% |
| **100** | **20%** | **0.13** | **27.4** | **+13%** |
| 300 | 16% | 0.16 | 35.0 | +44% |

100 is the knee. `LAMBDA_DIRECTION = 100.0`, `WINDOW = 5` (±500ms at 10fps).

**Sweep against the implementation, not a script beside it.** This constant
misled once already: a first table measured against a different appearance
normalisation made 3.0 look like a strong weight when on this implementation's
scale it was barely distinguishable from off. `_pairwise_mean_sq_dist` divides
by the pixel count for the same reason — the raw sum runs to the tens of
thousands, which would make any λ small enough to read as a sane weight
completely inert.

### 2 — The ceiling that sweep exposed matters more than the value it picked

Even at λ=300, median cosine between matched frames reaches only **0.16**.
Across every weight tried, matched motion stays essentially uncorrelated.

That is structural, not a tuning problem. Video Textures works *within one
continuous recording*, where every frame belongs to the same motion manifold and
a nearby frame is very likely to be moving a similar way. These are twelve
**independently generated** clips sharing only a neutral pose. There is no
manifold linking their motion, so a frame in one clip has no reason to move like
a frame in another however hard appearance is outweighed. Pose is recovered
well; motion barely. No larger λ closes that gap, and re-sweeping it later would
re-find the same ceiling.

### 3 — The one thing the clips genuinely share was sitting unused

Waiting for the outgoing clip to reach a near-neutral moment was ruled out
early — before follow-up 1 — on a first measurement at a different scale, where
near-neutral time looked like a two-tenths-of-a-second sliver at the very end of
each clip. Too narrow to be worth much, so pose matching was built instead.

Re-measured against `daemon/face_match.py`'s own `_frames`, at this scale, that
inverts. Near-neutral time is **wide**, not narrow:

| clip | median wait to neutral | p90 | already neutral |
|---|---|---|---|
| `idle1` | **0.0s** | 0.4s | 85% |
| `listening` | **0.0s** | 1.4s | 68% |
| `working` | **0.0s** | 1.8s | 59% |
| `thinking` | **0.0s** | 2.3s | 52% |

The median wait is zero for every loop clip: most of the time the outgoing clip
already qualifies. Thresholds of 0.15 and 0.25 of each clip's own peak departure
give the same answer, so this is not an artifact of where the line is drawn
(`NEUTRAL_THRESHOLD_FRACTION = 0.20` splits them). At a 1.2s cap the wait
expires and cuts anyway **14%** of the time (6% at 2.0s); the owner watched 1.2s
and kept it.

This also explains follow-up 2's ceiling. Pose matching was trying to match
arbitrary mid-motion poses across twelve clips with no shared motion manifold,
while the one moment they *all* genuinely share — neutral, which the whole
hub-and-spoke asset constraint exists to create — was not being used at all.

So `_neutral_buckets` makes that moment lookupable, and the page waits for it on
a non-urgent transition, capped, falling through to pose-matched entry only when
the wait times out.

## Consequences

The table is version 3: `match` (pose entry) plus `neutral` (one bool per 0.5s
slice of each clip's own timeline). A stale version-1 or version-2 file
has no `neutral` key at all, which is why the schema change got a version bump
where the direction-penalty retuning did not.

Pose matching stays, narrowed to the 14% tail rather than deleted — a timed-out
wait still has to land somewhere, and it is measurably better there than frame 0.

**Going further means changing the assets, not the code.** Clips that pass
through neutral more often, or several motions rendered as one continuous take
so a real motion manifold exists. From the page's side this is finished.

## What would change our mind

If the clips are ever regenerated as one continuous take per motion group, the
manifold Video Textures assumes would exist and λ's ceiling would move — the
sweep would be worth re-running against those assets, and only those. And if the
owner's eye prefers a 2.0s cap to 1.2s (6% forced cuts instead of 14%, paid for
with up to two seconds of late `listening`/`thinking`), that is a judgement no
measurement here settles.
