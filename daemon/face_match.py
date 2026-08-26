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
that the ±WINDOW neighbourhood below still means ±200ms."""

WINDOW = 2
"""±2 frames at FPS=10 is ±200ms (rule 1). A hand halfway up and a hand halfway
down are nearly identical single frames; comparing a window instead of one frame
is what tells them apart, because the neighbouring frames move in opposite
directions even though the centre frames can be pixel-identical."""

BUCKET = 0.5
"""Seconds per outgoing-time bucket (rule 2) - exactly the clips' keyframe
interval (`-g 12` at 24fps, spec §4.5), so a runtime seek lands on a keyframe."""

THUMB_W, THUMB_H = 32, 48
"""Small enough that a whole clip's frames fit in a few hundred KB and an
all-pairs distance matrix is a trivial matrix product, not a reason to reach for
anything heavier than numpy and Pillow. 32:48 keeps the clips' own 1080:1620
aspect ratio, so the crop is even, not stretched."""

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

    return {"version": 1, "fps": FPS, "window": WINDOW, "bucket": BUCKET, "match": match}


def write_table(face_dir: Path) -> Path:
    """Build the table and write it to `<face_dir>/transitions.json` - the path
    `GET /face/transitions` serves from, and the one the page's absence-means-
    frame-0 fallback (rule 4) depends on simply not existing until this has run.
    """
    path = face_dir / "transitions.json"
    path.write_text(json.dumps(build_table(face_dir)), encoding="utf-8")
    return path


def _seeks(frames_a: np.ndarray, frames_b: np.ndarray) -> list[float]:
    """One seek-into-B per `BUCKET`-second slice of A's timeline: the second (from
    B's own start) whose ±WINDOW neighbourhood most resembles some frame's
    neighbourhood within that slice of A.
    """
    distance = _windowed(_pairwise_sq_dist(frames_a, frames_b), WINDOW)
    frames_per_bucket = round(BUCKET * FPS)
    n_buckets = math.ceil(frames_a.shape[0] / frames_per_bucket)
    seeks: list[float] = []
    for k in range(n_buckets):
        lo = k * frames_per_bucket
        hi = min(frames_a.shape[0], lo + frames_per_bucket)
        window = distance[lo:hi]
        _, best_j = np.unravel_index(np.argmin(window), window.shape)
        seeks.append(round(float(best_j) / FPS, 3))
    return seeks


def _pairwise_sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance between every frame of `a` and every frame of
    `b`. The expansion `||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y` turns what would be
    an (nA x nB) Python loop over frame pairs into one matrix product (`a @ b.T`)
    plus two cheap per-row sums.
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
    return np.maximum(a_sq + b_sq - 2.0 * cross, 0.0)


def _windowed(distance: np.ndarray, window: int) -> np.ndarray:
    """Average `distance[i+k, j+k]` over `k` in `[-window, window]` (rule 1).

    Near either clip's edge, fewer than `2*window+1` offsets are in bounds; those
    cells are averaged over however many are, rather than padded with a fabricated
    value that would bias the match toward or away from the edges.
    """
    n_a, n_b = distance.shape
    total = np.zeros_like(distance)
    count = np.zeros_like(distance)
    for k in range(-window, window + 1):
        a_lo, a_hi = max(0, -k), min(n_a, n_a - k)
        b_lo, b_hi = max(0, -k), min(n_b, n_b - k)
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
