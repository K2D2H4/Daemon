"""Which frame to enter a loop clip on, instead of always frame 0.

Task 9's measurement (task-9-brief.md): a loop-to-loop crossfade blends
`outgoing[t]` into `incoming[0]`, and every clip's near-neutral pose sits only at
its own start and end - so `incoming[0]` is usually far from whatever pose
`outgoing` happened to be in when the switch fired. Matching the outgoing pose to
the closest-looking moment in the incoming clip cut the median distance 41% over
all 56 loop-to-loop pairs of the owner's real clips; matching normalised *phase*
instead measured *worse* than today (9.6 vs 6.4), so pose is the only lever.

This is [Video Textures](https://www.researchgate.net/publication/2624304_Video_Textures)
(Schodl et al., SIGGRAPH 2000) applied across clips instead of within one, plus
its own refinement: comparing a small window of frames rather than a single one.
A single frame cannot tell a hand rising through a pose from a hand falling
through the same pose - they can be pixel-identical - and splicing the two plays
the motion backwards. A window catches it: the neighbourhoods disagree even when
the centre frames match exactly.

FOLLOW-UP (still task 9, after shipping): matching *pose* was not enough on its
own. The owner reported a cut landing mid-sigh - a breath cycle is 3-4s, and the
shipped ±200ms window sees about a tenth of one, so it cannot tell inhaling from
exhaling at the same chest position. Measured: 41% of the shipped matches had the
incoming clip moving *opposite* to the outgoing clip at the splice point. Widening
the window to ±500ms alone gets that down to ~28-30% and then plateaus; a direction
term in the cost is what actually fixes it (see LAMBDA_DIRECTION below).

Offline and dependency-light on purpose - no model, no network, and nothing
imported from `daemon/` beyond `face_routes`'s clip vocabulary (`CLIPS`). This
module never runs on the request path; `daemon face-transitions` runs it by hand,
and `daemon/face_routes.py` only ever serves whatever it last wrote.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from daemon.face_routes import CLIPS

FPS = 10
"""Frame rate frames are resampled to before comparison. Coarse enough to keep
the all-pairs distance matrix tiny (a clip is ~5-9s -> ~50-90 rows), fine enough
that the ±WINDOW neighbourhood below still means ±500ms."""

WINDOW = 5
"""±5 frames at FPS=10 is ±500ms (rule 1). Originally ±2 (±200ms), sized only
against "a hand halfway up looks like a hand halfway down" - a single frame
cannot tell the two apart, but a short window can, because the neighbouring
frames move in opposite directions even when the centre frames are pixel-
identical. Widened to ±500ms for a slower motion the ±200ms window could not
see at all: a breath is a 3-4s cycle, so ±200ms samples about a tenth of it -
almost any two points within that tenth look like "the same direction" by
sheer proximity. ±500ms alone does not fix this (see LAMBDA_DIRECTION), but it
is the window the direction term below is measured against, and shrinking it
back down would invalidate that measurement."""

BUCKET = 0.5
"""Seconds per outgoing-time bucket (rule 2) - exactly the clips' keyframe
interval (`-g 12` at 24fps, spec §4.5), so a runtime seek lands on a keyframe."""

THUMB_W, THUMB_H = 32, 48
"""Small enough that a whole clip's frames fit in a few hundred KB and an
all-pairs distance matrix is a trivial matrix product, not a reason to reach for
anything heavier than numpy and Pillow. 32:48 keeps the clips' own 1080:1620
aspect ratio, so the crop is even, not stretched."""

DIRECTION_SPAN = 5
"""±5 frames at FPS=10 is ±500ms: how far apart the two frames are that define a
frame's motion *direction* (see `_directions`). Fixed at 0.5s directly - "the
frames ±0.5s around it" - not derived from WINDOW, even though the two happen to
share a value today: WINDOW governs how much *appearance* neighbourhood gets
averaged into one comparison, this governs how much time a *direction* vector
spans, and there is no reason those should have to move together if either is
retuned later."""

LAMBDA_DIRECTION = 3.0
"""Weight on the direction penalty in `_cost`: cost = appearance + LAMBDA *
(1 - cos(dir_a, dir_b)), so an aligned match (cos=1) pays nothing extra and a
directly opposed one (cos=-1) pays 2*LAMBDA. `appearance` here is *mean*
squared pixel distance (see `_pairwise_mean_sq_dist`), not the raw sum, so its
scale is comparable to LAMBDA's 0..2*3=6 range instead of dwarfing it by three
orders of magnitude (measured: 1536 pixels means the raw sum runs to the tens
of thousands).

Measured across all 56 loop-to-loop pairs of the owner's real clips, at the
±500ms window above:

    lambda | opposed matches | median cosine | appearance cost
    0      | 28%              | 0.08          | 2.52
    1.0    | 15%              | 0.17          | 2.57
    3.0    | 10%              | 0.21          | 2.65 (+5%)
    8.0    | 6%               | 0.24          | 2.74 (+9%)

3.0 is the pick: it cuts opposed motion from 28% to 10% (two thirds) for a 5%
appearance-matching cost, the same kind of trade pose-matching itself made
against frame 0 (41% better pose for a strategy that does nothing for pure
"phase" matching). 8.0 buys another 4 points of opposed-motion reduction for
almost double the appearance cost - not worth it by the same reasoning."""

ONE_SHOTS = frozenset({"amused", "sulky", "curious", "flourish_arms"})
"""Mood one-shots and the idle flourish: each is an arc from the shared neutral
pose and back (design spec §1's "hub-and-spoke"), so entering one mid-arc, at
whatever pose the outgoing loop left off on, would destroy the arc (rule 3).
Hardcoded rather than imported: `daemon/face.py`'s `MOODS` and the flourish list
in `daemon/static/face.html` both know this split, but this module may import
neither - only `face_routes`'s flat `CLIPS` tuple, which does not carry it."""

LOOPS: tuple[str, ...] = tuple(name for name in CLIPS if name not in ONE_SHOTS)
"""Every clip that is a continuous activity loop rather than a one-shot. Building
the table only over these - both as source and destination - is what makes rule
3 automatic instead of a check someone could forget: a one-shot can never appear
in `match`, because it is never in the set the table iterates."""


def build_table(face_dir: Path) -> dict[str, Any]:
    """The pose-match table for every ordered pair of LOOP clips present under
    `face_dir`. A clip that does not exist on disk is simply absent from every
    row and column it would have appeared in (rule 4) - there is no interpolation
    or substitution for a missing clip, only omission.

    `version` is 2 as of the direction-penalty follow-up (still task 9): the
    seek values a version-1 table produced can move mid-sigh, so a stale
    version-1 `transitions.json` left over from before this fix needs to be
    distinguishable from a fresh one, even though the schema shape (fps,
    window, bucket, match) is unchanged.
    """
    present = [name for name in LOOPS if (face_dir / f"{name}.mp4").is_file()]
    frames = {name: _frames(face_dir / f"{name}.mp4") for name in present}

    match: dict[str, dict[str, list[float]]] = {}
    for a in present:
        frames_a = frames[a]
        if frames_a.shape[0] == 0:
            continue
        row: dict[str, list[float]] = {}
        for b in present:
            if b == a:
                continue
            frames_b = frames[b]
            if frames_b.shape[0] == 0:
                continue
            row[b] = _seeks(frames_a, frames_b)
        if row:
            match[a] = row

    return {"version": 2, "fps": FPS, "window": WINDOW, "bucket": BUCKET, "match": match}


def write_table(face_dir: Path) -> Path:
    """Build the table and write it to `<face_dir>/transitions.json` - the path
    `GET /face/transitions` serves from, and the one the page's absence-means-
    frame-0 fallback (rule 4) depends on simply not existing until this has run.
    """
    path = face_dir / "transitions.json"
    path.write_text(json.dumps(build_table(face_dir)), encoding="utf-8")
    return path


def _seeks(frames_a: np.ndarray, frames_b: np.ndarray) -> list[float]:
    """One seek-into-B per `BUCKET`-second slice of A's timeline: the second
    (from B's own start) with the lowest cost - appearance plus the direction
    penalty (`_cost`) - within that slice of A.
    """
    cost = _cost(frames_a, frames_b)
    frames_per_bucket = round(BUCKET * FPS)
    n_buckets = math.ceil(frames_a.shape[0] / frames_per_bucket)
    seeks: list[float] = []
    for k in range(n_buckets):
        lo = k * frames_per_bucket
        hi = min(frames_a.shape[0], lo + frames_per_bucket)
        window = cost[lo:hi]
        _, best_j = np.unravel_index(np.argmin(window), window.shape)
        seeks.append(round(float(best_j) / FPS, 3))
    return seeks


def _cost(frames_a: np.ndarray, frames_b: np.ndarray) -> np.ndarray:
    """Appearance distance plus the direction penalty (see LAMBDA_DIRECTION's
    docstring for the measured table behind the weight). An aligned match pays
    nothing extra; an opposed one pays 2*LAMBDA_DIRECTION.

    Frames within DIRECTION_SPAN of either end of *their own* clip have no
    direction (`_directions`' `has_direction` is False there). The two ends of
    that gap are handled differently, deliberately:

    - As a destination (a column, `frames_b`): excluded from ever being picked
      - forced to +inf - rather than defaulting to "no penalty". A silent
        default would make those frames systematically *cheaper* than a
        well-matched interior frame paying even a small aligned-ish penalty,
        which would quietly pull the table back toward the clip edges rule 4
        was written to get away from. The one exception: if a clip is so short
        that direction is undefined everywhere in it (shorter than
        `2*DIRECTION_SPAN`), excluding every column would leave nothing to
        pick at all, so the exclusion is skipped entirely for that clip and
        appearance alone decides, same as before this feature existed.
    - As a source (a row, `frames_a`): left at zero penalty (appearance-only)
      for the whole row instead. Every bucket of the outgoing clip must
      produce *some* seek - a source row cannot simply be excluded the way a
      destination column can - so "we don't know its direction" has to mean
      "judge this row on appearance alone", not "exclude it" or "guess".
    """
    appearance = _windowed(_pairwise_mean_sq_dist(frames_a, frames_b), WINDOW)
    dir_a, has_dir_a = _directions(frames_a)
    dir_b, has_dir_b = _directions(frames_b)

    # Both are already unit vectors (or exactly zero where direction is
    # undefined), so the plain dot product below already *is* the cosine
    # similarity - no division, so no way for this step to produce a NaN.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cos_sim = dir_a @ dir_b.T
    penalty = LAMBDA_DIRECTION * (1.0 - cos_sim)

    if has_dir_b.any():
        penalty[:, ~has_dir_b] = np.inf
    else:
        penalty[:, :] = 0.0
    # Applied after the column exclusion above so it wins where both would
    # otherwise apply to the same cell: a row with no direction of its own has
    # nothing to compare, regardless of what its columns are.
    penalty[~has_dir_a, :] = 0.0

    return appearance + penalty


def _directions(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame motion direction: the normalised difference between the frame
    DIRECTION_SPAN ahead and DIRECTION_SPAN behind. `has_direction[i]` is False
    - and `directions[i]` is left as zero rather than divided-by-zero - for a
    frame within DIRECTION_SPAN of either end of the clip (no ±DIRECTION_SPAN
    neighbour to difference against) and for a frame whose ±DIRECTION_SPAN
    neighbourhood is perfectly static (a zero difference has no direction
    either). Callers decide what "no direction" means for their purpose
    (`_cost` does not treat it as "aligned"); this function only ever reports
    it, never guesses past it.
    """
    n = frames.shape[0]
    directions = np.zeros_like(frames)
    has_direction = np.zeros(n, dtype=bool)
    span = 2 * DIRECTION_SPAN
    if n > span:
        diffs = frames[span:] - frames[: n - span]
        norms = np.linalg.norm(diffs, axis=1)
        nonzero = norms > 0
        # A basic slice is a view, so writing through it (fancy-indexed by
        # `nonzero`) mutates `directions` in place - no separate write-back.
        directions[DIRECTION_SPAN : n - DIRECTION_SPAN][nonzero] = (
            diffs[nonzero] / norms[nonzero, None]
        )
        has_direction[DIRECTION_SPAN : n - DIRECTION_SPAN] = nonzero
    return directions, has_direction


def _pairwise_mean_sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean (not summed) squared Euclidean distance per pixel between every
    frame of `a` and every frame of `b`. The expansion
    `||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y` turns what would be an (nA x nB)
    Python loop over frame pairs into one matrix product (`a @ b.T`) plus two
    cheap per-row sums.

    Divided by the pixel count (THUMB_W*THUMB_H) so this is directly
    comparable to LAMBDA_DIRECTION's 0..2*LAMBDA_DIRECTION range in `_cost` -
    measured: the raw (unnormalised) sum runs to the tens of thousands on the
    owner's real clips, which would make any LAMBDA_DIRECTION small enough to
    read as a sane weight in its own docstring completely inert in practice.
    Dividing by a positive constant does not change which cell is the minimum,
    so this is a no-op for every appearance-only comparison that existed
    before the direction penalty did.
    """
    a_sq = np.sum(a * a, axis=1, keepdims=True)
    b_sq = np.sum(b * b, axis=1, keepdims=True).T
    # Measured on this project's own dev machine: numpy's Accelerate BLAS backend
    # (Apple Silicon) raises spurious divide-by-zero/overflow/invalid warnings on
    # small `@` matmuls even though the result never actually contains NaN or Inf
    # (checked directly against pixel-range inputs) - a known Accelerate quirk,
    # not a sign anything here is wrong. Suppressed narrowly around the one call
    # that triggers it, so a real problem elsewhere still surfaces normally.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cross = a @ b.T
    # Floating-point noise can push an exact-zero distance a hair negative.
    sq_dist = np.maximum(a_sq + b_sq - 2.0 * cross, 0.0)
    return sq_dist / a.shape[1]


def _windowed(distance: np.ndarray, window: int) -> np.ndarray:
    """Average `distance[i+k, j+k]` over `k` in `[-window, window]` (rule 1).

    Near either clip's edge, fewer than `2*window+1` offsets are in bounds; those
    cells are averaged over however many are, rather than padded with a fabricated
    value that would bias the match toward or away from the edges. `k=0` always
    contributes to every cell, so `count` is never zero anywhere and this never
    divides by it.
    """
    n_a, n_b = distance.shape
    total = np.zeros_like(distance)
    count = np.zeros_like(distance)
    for k in range(-window, window + 1):
        a_lo, a_hi = max(0, -k), min(n_a, n_a - k)
        b_lo, b_hi = max(0, -k), min(n_b, n_b - k)
        # A clip shorter than `window` makes this range empty for some k (more
        # of the window falls off the end than the clip has left). `a_hi+k`
        # would then go negative, and Python's negative-index slicing would
        # silently reinterpret it as "count from the end" instead of "empty" -
        # the mismatch that used to raise a broadcast error here. Skipping the
        # whole k outright, rather than computing a slice from it, sidesteps
        # that reinterpretation instead of working around its consequences.
        if a_hi <= a_lo or b_hi <= b_lo:
            continue
        total[a_lo:a_hi, b_lo:b_hi] += distance[a_lo + k : a_hi + k, b_lo + k : b_hi + k]
        count[a_lo:a_hi, b_lo:b_hi] += 1
    return total / count


def _frames(clip: Path) -> np.ndarray:
    """Every frame of `clip`, resampled to FPS and shrunk to a grey THUMB_W x
    THUMB_H thumbnail: one float64 row per frame, flattened.
    """
    with tempfile.TemporaryDirectory() as td:
        pattern = Path(td) / "f%05d.png"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(clip),
                "-vf",
                f"fps={FPS},scale={THUMB_W}:{THUMB_H}:flags=lanczos,format=gray",
                str(pattern),
            ],
            check=True,
            capture_output=True,
        )
        files = sorted(Path(td).glob("f*.png"))
        if not files:
            return np.empty((0, THUMB_W * THUMB_H), dtype=np.float64)
        return np.stack([np.asarray(Image.open(f), dtype=np.float64).reshape(-1) for f in files])
