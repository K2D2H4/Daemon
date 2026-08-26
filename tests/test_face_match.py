"""Tests for daemon/face_match.py - the pose-matched entry builder (Task 9).

Written from task-9-brief.md's four rules, not from the module's own shape: each
test below constructs synthetic clips whose "correct" match is knowable by
construction (disjoint value bands, so only the intended alignment can hit a
distance of zero), then checks the table against that known answer.

Frames are solid grayscale squares - values chosen with generous gaps (>=5 units,
mostly >=10) so H.264's lossy encoding cannot blur one band into another.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from daemon.face_match import build_table, write_table

FPS = 10
"""task-9-brief.md rule 1: "±2 frames at 10fps (±200ms)". Hardcoded here, not
imported from face_match, so these tests exercise the brief's stated contract
rather than whatever face_match.py happens to name its own constant."""

BUCKET = 0.5
"""Rule 2's keyframe-interval bucket, same reasoning as FPS above."""


def _clip(path: Path, values: list[int]) -> None:
    """Write an mp4 at `path` with one solid grayscale frame per value in
    `values`, encoded at FPS so face_match's own `fps={FPS}` resample in
    `_frames()` is a no-op and each value lands on exactly one output row.
    `-crf 0` keeps the encode lossless - these tests depend on exact values
    surviving the round trip through H.264, not approximate ones.
    """
    with tempfile.TemporaryDirectory() as td:
        for i, v in enumerate(values):
            Image.new("L", (64, 64), color=v).save(Path(td) / f"f{i:05d}.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-framerate",
                str(FPS),
                "-i",
                str(Path(td) / "f%05d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "0",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
            capture_output=True,
        )


def test_the_table_matches_the_frame_that_actually_looks_alike(tmp_path):
    """Two synthetic clips whose only similar frames are at known offsets: the
    table must point at those offsets, not at frame 0.

    idle1 carries one 5-frame "signature" ramp (values 100..140) at indices 5-9,
    flanked by junk bands that share no value with the signature. idle2 carries
    the identical ramp at indices 10-14 - a different offset, and deliberately
    not at frame 0 - flanked by its own disjoint junk bands. Because every band
    uses values no other band uses, the only place the two clips can agree
    frame-for-frame is where the two ramps align, so that alignment is
    unambiguously the global minimum distance.
    """
    _clip(
        tmp_path / "idle1.mp4",
        [0, 5, 10, 15, 20] + [100, 110, 120, 130, 140] + [220, 225, 230, 235, 240],
    )
    _clip(
        tmp_path / "idle2.mp4",
        [50, 55, 60, 65, 70] + [150, 155, 160, 165, 170] + [100, 110, 120, 130, 140],
    )

    table = build_table(tmp_path)

    # idle1's ramp is centred on frame 7 -> bucket floor(7 / 5) = 1.
    # idle2's ramp is centred on frame 12 -> the expected seek is 12 / FPS.
    assert table["match"]["idle1"]["idle2"][1] == pytest.approx(12 / FPS)


def test_a_window_is_compared_not_a_single_frame(tmp_path):
    """Two frames identical in isolation but with opposite motion around them
    must NOT be chosen over a pair whose neighbourhood also agrees. This is the
    whole reason the window exists - without it the splice plays the motion
    backwards.

    idle1 carries a rising ramp (60, 70, 80, 90, 100) at indices 5-9, centred on
    frame 7 (value 80). idle2 carries two candidates that both have value 80 at
    their centre frame: a DECOY at indices 5-9 with the ramp reversed (falling:
    100, 90, ..., 60 - opposite motion, but its centre frame is pixel-identical
    to idle1's), and a TRUE MATCH at indices 15-19 with the ramp in the same
    order (rising, like idle1's). A single-frame comparison cannot tell these
    apart - both have centre value 80. A windowed comparison can: the decoy's
    neighbourhood disagrees (distance 800 across the window) while the true
    match's neighbourhood is pixel-identical (distance 0).
    """
    _clip(
        tmp_path / "idle1.mp4",
        [0, 5, 10, 15, 20] + [60, 70, 80, 90, 100] + [150, 155, 160, 165, 170],
    )
    _clip(
        tmp_path / "idle2.mp4",
        [30, 35, 40, 45, 50]
        + [100, 90, 80, 70, 60]  # decoy: same values, reversed (opposite motion)
        + [190, 195, 200, 205, 210]
        + [60, 70, 80, 90, 100]  # true match: same order as idle1's ramp
        + [230, 235, 240, 245, 250],
    )

    table = build_table(tmp_path)

    # idle1's ramp is centred on frame 7 -> bucket floor(7 / 5) = 1.
    seek = table["match"]["idle1"]["idle2"][1]
    # The decoy sits at idle2 frame 7 (7 / FPS = 0.7) - a window-blind matcher
    # would tie on it, or pick it outright since it comes first in scan order.
    assert seek != pytest.approx(0.7), "must not splice onto the reversed-motion decoy"
    # The true match sits at idle2 frame 17.
    assert seek == pytest.approx(17 / FPS)


def test_one_shots_are_not_matched_into(tmp_path):
    """The table has no entries whose destination is a one-shot clip.

    A mood one-shot (amused) is an arc from neutral and back (rule 3); entering
    it mid-arc destroys the arc, so it must never be a destination - or a
    source, since face_match.py's LOOPS is the same set for both axes.
    """
    _clip(tmp_path / "idle1.mp4", [10, 20, 30, 40, 50])
    _clip(tmp_path / "idle2.mp4", [60, 70, 80, 90, 100])
    _clip(tmp_path / "amused.mp4", [110, 120, 130, 140, 150])

    table = build_table(tmp_path)

    assert "amused" not in table["match"], "a one-shot must never be a source"
    for destinations in table["match"].values():
        assert "amused" not in destinations, "a one-shot must never be a destination"


def test_a_missing_clip_is_simply_absent_from_the_table(tmp_path):
    """A clip that does not exist on disk produces no crash and no entry - not
    an interpolated stand-in, not a KeyError, simply nothing referencing it.

    Only idle1 exists here; every other LOOP clip (idle2, listening, thinking,
    working, ...) is absent, so no pair can be formed at all.
    """
    _clip(tmp_path / "idle1.mp4", [10, 20, 30, 40, 50])

    table = build_table(tmp_path)

    assert table["match"] == {}


def test_write_table_writes_transitions_json_at_the_right_path(tmp_path):
    """write_table's own contract: build, then write to <face_dir>/transitions.json,
    and return that path - the one GET /face/transitions serves from."""
    _clip(tmp_path / "idle1.mp4", [10, 20, 30])
    _clip(tmp_path / "idle2.mp4", [40, 50, 60])

    path = write_table(tmp_path)

    assert path == tmp_path / "transitions.json"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written == build_table(tmp_path)
