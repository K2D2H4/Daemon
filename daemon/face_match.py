"""Where to enter a clip, and when it is a good moment to leave one.

Two lookups, both offline, both written to `<data_dir>/face/transitions.json` by
`daemon face-transitions` and served to the page by `daemon/face_routes.py`:

- `neutral[stem]` - one bool per BUCKET-second slice of `stem`'s own timeline,
  true where some frame in it is near that clip's own frame 0. The page waits
  (capped) for the next such moment before a non-urgent transition. This is the
  main mechanism: every clip starts and ends on the same neutral pose by
  construction (design spec §4.3's hub-and-spoke), and that shared moment is the
  only thing twelve independently generated clips actually have in common.
- `match[a][b]` - for each BUCKET-second slice of `a`, the second to seek to in
  `b` so the incoming frame looks, and moves, most like the outgoing one. This
  covers the measured 14% where the wait above expires and the page has to cut
  anyway. It is [Video
  Textures](https://www.researchgate.net/publication/2624304_Video_Textures)
  (Schodl et al., SIGGRAPH 2000) applied across clips instead of within one, plus
  its refinement of comparing a short window rather than a single frame - a hand
  rising through a pose and a hand falling through it can be pixel-identical, and
  splicing the two plays the motion backwards.

**That order - wait first, match second - is the decision, and it was originally
the other way round.** Why, what the measurements were, and why no amount of
tuning closes the gap pose matching leaves, are in
[ADR 0017](../docs/adr/0017-the-neutral-moment-not-the-matched-pose.md); the
constants below each carry the number that set them.

The numbered rules cited below are Task 9's four in
`docs/superpowers/plans/2026-08-25-face-v1.md`, which is also where the
median-distance table behind pose matching lives.

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

LAMBDA_DIRECTION = 100.0
"""Weight on the direction penalty in `_cost`: cost = appearance + LAMBDA *
(1 - cos(dir_a, dir_b)), so an aligned match (cos=1) pays nothing extra and a
directly opposed one (cos=-1) pays 2*LAMBDA. `appearance` here is *mean*
squared pixel distance (see `_pairwise_mean_sq_dist`), not the raw sum, so its
scale is comparable to LAMBDA's range instead of dwarfing it by orders of
magnitude (measured: 1536 pixels means the raw sum runs to the tens of
thousands).

Measured by sweeping LAMBDA through *this* implementation's own `_directions`
and `_frames` against all 56 loop-to-loop pairs of the owner's real clips -
not against a separate offline script with its own normalisation, which is
exactly how this constant misled once already (a first table here, measured
against a different appearance scale, made 3.0 look like a strong weight when
on this implementation's scale it was barely distinguishable from off):

    lambda | opposed matches | median cosine | appearance | vs lambda=0
    0      | 38%              | 0.04          | 24.3       | --
    3.0    | 34%              | 0.07          | --         | --
    10     | 31%              | 0.08          | --         | --
    30     | 26%              | 0.11          | 26.0       | +7%
    100    | 20%              | 0.13          | 27.4       | +13%
    300    | 16%              | 0.16          | 35.0       | +44%

100 is the knee: opposed matches nearly halve (38% -> 20%) for 13% worse
appearance matching. 300 buys four more points of opposed-motion reduction
for 44% - clearly past the knee. 3.0 (this constant's first value, carried
over from a table measured on a different scale) bought only 4 points and
was barely distinguishable from LAMBDA=0.

**The limit this sweep exposes matters more than the value it picked.** Even
at LAMBDA=300, median cosine only reaches 0.16 - across every weight tried,
matched frames have essentially uncorrelated motion. No larger LAMBDA fixes
this; the ceiling is structural, not a tuning problem (ADR 0017 §2)."""

NEUTRAL_THRESHOLD_FRACTION = 0.20
"""How close a frame must be to its own clip's frame 0 to count as "near
neutral" (see `_neutral_buckets`) - a fraction of that clip's own peak
departure (the largest mean-squared distance any of its frames reaches from
frame 0), not an absolute distance, since clips differ in how far they stray.

Measured at 0.15 and 0.25 of peak departure against all twelve of the owner's
real clips (ADR 0017 §3): the resulting wait times (median 0s,
p90 well under a couple of seconds for every loop clip but flourish_arms)
barely moved between the two, so the choice inside that band is not load-
bearing. 0.20 is the midpoint of the two endpoints actually measured, not a
third, untested value - splitting the difference rather than committing to
either edge those two runs happened to use.

What the flag does *not* say is where a cut lands. A bucket is true when *some*
frame in its BUCKET-second slice qualifies - 12 frames of the played clip - and
the runtime only ever sees the bucket, so the frame a transition actually lands
on need not be one of the qualifying ones. Measured against real joins, that
costs: cutting at a near-neutral moment into a one-shot's frame 0 runs to p90
5.06 and max 8.95 against a 2.14 loop-point worst case, with 34% of samples over
it ([ADR 0020](../docs/adr/0020-lip-sync-makes-a-clip-ambient.md)). The
imprecision is the mechanism's, not the measurement's, which is why this flag is
a heuristic about the clip and not a promise about the join."""

ONE_SHOTS = frozenset({"amused", "sulky", "curious", "flourish_arms"})
"""Mood one-shots and the idle flourish: each is an arc from its own neutral
start pose and back (design spec §1's "hub-and-spoke"), so entering one mid-arc,
at whatever pose the outgoing loop left off on, would destroy the arc (rule 3).
That reason is unchanged, and it is why a one-shot never appears in `match`.

This used to read "an arc from the *shared* neutral pose", which claimed more
than anything here measures. `neutral` is computed against *each clip's own*
frame 0 (see NEUTRAL_THRESHOLD_FRACTION), so a bucket flagged neutral in one
clip says nothing about its distance from a *different* clip's frame 0 - the
pose a one-shot has to be entered at. Measured: `idle2@1.75s` is flagged
neutral and sits 8.94 from `amused`'s frame 0, against a loop-point baseline of
1.14 median and 2.14 worst case
([ADR 0020](../docs/adr/0020-lip-sync-makes-a-clip-ambient.md)). Hub-and-spoke
is how the clips were authored; it is not a distance this module checks.

Hardcoded rather than imported: `daemon/face.py`'s `Mood` literal and the
flourish list in `daemon/static/face.html` both know this split, but this module
may import neither - only `face_routes`'s flat `CLIPS` tuple, which does not
carry it."""

LOOPS: tuple[str, ...] = tuple(name for name in CLIPS if name not in ONE_SHOTS)
"""Every clip that is a continuous activity loop rather than a one-shot. Building
the table only over these - both as source and destination - is what makes rule
3 automatic instead of a check someone could forget: a one-shot can never appear
in `match`, because it is never in the set the table iterates."""


def build_table(face_dir: Path) -> dict[str, Any]:
    """The pose-match and near-neutral table for every LOOP clip present under
    `face_dir`. A clip that does not exist on disk is simply absent from every
    row, column and `neutral` entry it would have appeared in (rule 4) - there
    is no interpolation or substitution for a missing clip, only omission.

    `version` is 3 as of the neutral-wait decision (ADR 0017 §3): earlier
    versions carried `match` alone, this one adds `neutral`, so a stale
    version-1 or version-2 `transitions.json` (no `neutral` key at all) needs
    to be distinguishable from a fresh one - the same reasoning version 2 used
    for the direction-penalty fix, now applied to an actual schema change
    rather than a same-shape retuning.
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

    neutral = {name: _neutral_buckets(frames[name]) for name in present}

    return {
        "version": 3,
        "fps": FPS,
        "window": WINDOW,
        "bucket": BUCKET,
        "match": match,
        "neutral": neutral,
    }


def write_table(face_dir: Path) -> Path:
    """Build the table and write it to `<face_dir>/transitions.json` - the path
    `GET /face/transitions` serves from, and the one the page's absence-means-
    frame-0 fallback (rule 4) depends on simply not existing until this has run.

    Creates the directory. It is `<data_dir>/face/` and nothing else makes it:
    the clips are the owner's and they arrive by being dropped there, so on an
    install where they have not been yet this used to raise `FileNotFoundError`
    from `write_text` - which `daemon.cli._face_transitions` then reported as a
    missing ffmpeg.
    """
    face_dir.mkdir(parents=True, exist_ok=True)
    path = face_dir / "transitions.json"
    path.write_text(json.dumps(build_table(face_dir)), encoding="utf-8")
    return path


def _bucket_ranges(n_frames: int) -> list[tuple[int, int]]:
    """The `[lo, hi)` frame-index range of each `BUCKET`-second slice of a
    clip's own `n_frames`-long timeline (rule 2) - the one place this
    computation lives, so `_seeks` and `_neutral_buckets` cannot drift apart
    on what a "slice" means.
    """
    frames_per_bucket = round(BUCKET * FPS)
    n_buckets = math.ceil(n_frames / frames_per_bucket)
    return [
        (k * frames_per_bucket, min(n_frames, (k + 1) * frames_per_bucket))
        for k in range(n_buckets)
    ]


def _seeks(frames_a: np.ndarray, frames_b: np.ndarray) -> list[float]:
    """One seek-into-B per `BUCKET`-second slice of A's timeline: the second
    (from B's own start) with the lowest cost - appearance plus the direction
    penalty (`_cost`) - within that slice of A.
    """
    cost = _cost(frames_a, frames_b)
    seeks: list[float] = []
    for lo, hi in _bucket_ranges(frames_a.shape[0]):
        window = cost[lo:hi]
        _, best_j = np.unravel_index(np.argmin(window), window.shape)
        seeks.append(round(float(best_j) / FPS, 3))
    return seeks


def _neutral_buckets(frames: np.ndarray) -> list[bool]:
    """Which `BUCKET`-second slices of `frames`' own timeline are near its
    own frame 0 (ADR 0017 §3: waiting for the clip's next such moment beats
    matching pose against an unrelated clip's motion, which twelve
    independently generated clips were never going to share). A slice counts
    as near neutral if any
    frame within it is within `NEUTRAL_THRESHOLD_FRACTION` of the clip's own
    peak departure from frame 0 - bucketed the same way as `match`
    (`_bucket_ranges`), so the page can look this up instead of recomputing
    distances itself.
    """
    if frames.shape[0] == 0:
        return []
    dist_to_first = _pairwise_mean_sq_dist(frames, frames[:1])[:, 0]
    threshold = NEUTRAL_THRESHOLD_FRACTION * float(dist_to_first.max())
    near = dist_to_first <= threshold
    return [bool(near[lo:hi].any()) for lo, hi in _bucket_ranges(frames.shape[0])]


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

    `check=True` means two different failures surface as two different
    exceptions, both left to propagate rather than caught here: a missing
    `ffmpeg` binary raises `FileNotFoundError` (an `OSError`), and a present
    but corrupt or unreadable clip raises `CalledProcessError`. Deliberately
    not this module's problem - it stays offline and side-effect-free
    (module docstring) - so `daemon.cli._face_transitions` (the one caller
    that runs this on an operator's machine) is where both get turned into
    a printed message instead of a raw traceback.
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
