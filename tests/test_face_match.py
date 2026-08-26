"""Tests for daemon/face_match.py - the pose-matched entry builder (Task 9).

Written from task-9-brief.md's four rules and the direction-penalty follow-up,
not from the module's own shape: each test below constructs synthetic clips
whose "correct" match is knowable by construction (disjoint value bands, so
only the intended alignment can hit a distance of zero, or a hand-derived cost
comparison for the direction tests), then checks the table against that known
answer.

Frames are solid grayscale squares - values chosen with generous gaps so
H.264's lossy encoding cannot blur one band into another. `-crf 0` keeps the
encode lossless on top of that, so exact values survive the round trip.

WINDOW and DIRECTION_SPAN are both 5 frames (±500ms) as of the follow-up, so a
"clean" (edge-effect-free) test point needs 5 frames of clearance on each side
- clips here are correspondingly longer than they needed to be when WINDOW was
±2.
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
"""task-9-brief.md rule 1: "±2 frames at 10fps (±200ms)", later widened to
±500ms by the direction-penalty follow-up. Hardcoded here, not imported from
face_match, so these tests exercise the brief's stated contract rather than
whatever face_match.py happens to name its own constant."""

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

    idle1 carries one 11-frame "signature" ramp (100..150 step 5 - long enough
    that its own centre frame has a full, uncontaminated ±5 window using only
    the ramp's own content) at indices 5-15, flanked by junk bands that share
    no value with the signature. idle2 carries the identical ramp at indices
    8-18 - a different offset, and deliberately not at frame 0 - flanked by
    its own disjoint junk bands. Because every band uses values no other band
    uses, the only place the two clips can agree frame-for-frame across a full
    window is where the two ramps align, so that alignment is unambiguously
    the global minimum cost.
    """
    ramp = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
    _clip(tmp_path / "idle1.mp4", [0, 5, 10, 15, 20] + ramp + [220, 225, 230, 235, 240])
    _clip(
        tmp_path / "idle2.mp4",
        [50, 55, 60, 65, 70, 75, 80, 85] + ramp + [190, 195, 200, 205, 210],
    )

    table = build_table(tmp_path)

    # idle1's ramp is centred on frame 10 -> bucket floor(10 / 5) = 2.
    # idle2's ramp is centred on frame 13 -> the expected seek is 13 / FPS.
    assert table["match"]["idle1"]["idle2"][2] == pytest.approx(13 / FPS)


def test_a_window_is_compared_not_a_single_frame(tmp_path):
    """A decoy that ties the true match on its centre frame AND on its net
    motion direction (so the direction penalty cannot tell them apart either)
    must still lose once the window sees the rest of its shape.

    idle1 carries a straight ramp (0, 10, .., 100) at indices 5-15, centred on
    frame 10 (value 50, net direction 0->100). idle2 carries a DECOY at
    indices 5-15 with the *same* centre value (50) and the *same* endpoints
    (0 and 100, so the same net direction) but a different path between them
    (flat, then a jump) - and a TRUE MATCH at indices 21-31 with the identical
    shape to idle1's ramp. Centre-frame comparison ties them (both 50);
    direction ties them too (both endpoints 0->100). Only comparing the whole
    ±5 window tells them apart: the decoy's intermediate frames diverge from
    idle1's by up to 40, the true match's do not at all.
    """
    ramp = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    decoy = [0, 0, 0, 0, 0, 50, 100, 100, 100, 100, 100]
    _clip(tmp_path / "idle1.mp4", [110, 113, 116, 119, 122] + ramp + [130, 133, 136, 139, 142])
    _clip(
        tmp_path / "idle2.mp4",
        [150, 153, 156, 159, 162]
        + decoy
        + [170, 173, 176, 179, 182]
        + ramp  # true match: identical shape to idle1's ramp
        + [190, 193, 196, 199, 202],
    )

    table = build_table(tmp_path)

    # idle1's ramp is centred on frame 10 -> bucket floor(10 / 5) = 2.
    seek = table["match"]["idle1"]["idle2"][2]
    # The decoy sits at idle2 frame 10 (10 / FPS = 1.0).
    assert seek != pytest.approx(1.0), "must not splice onto the shape-mismatched decoy"
    # The true match sits at idle2 frame 26.
    assert seek == pytest.approx(26 / FPS)


def test_a_direction_penalty_prevents_a_reversed_match(tmp_path):
    """The best *appearance* match can have opposed motion; the table must
    still prefer a slightly-worse appearance match whose motion agrees.

    idle1 carries a gentle ramp (45..55) at indices 5-15, centred on frame 10
    (net direction: rising). idle2 carries an OPPOSED candidate at indices
    5-15 - idle1's own ramp with only its two endpoints swapped (so its centre
    frame and 9 of 11 window taps match idle1 exactly - appearance cost about
    18.2 - but its net direction is falling, opposed to idle1's) - and an
    ALIGNED candidate at indices 21-31, uniformly offset from idle1's ramp by
    +4 or +5 (appearance cost about 20.1 - a worse pointwise match - but
    rising the same way idle1 is).

    Appearance alone would pick the opposed candidate (18.2 < 20.1). With
    LAMBDA_DIRECTION >= ~1.0 (currently 100.0, comfortably above that), the
    opposed candidate pays +2*LAMBDA (fully opposed) and the aligned candidate
    pays +0 (fully aligned), flipping the total cost in the aligned
    candidate's favour - the table must follow that flip regardless of
    exactly how large LAMBDA_DIRECTION is tuned to, as long as it clears the
    ~1.9-point appearance gap this scenario was built around.
    """
    a_ramp = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    opposed = [55, 46, 47, 48, 49, 50, 51, 52, 53, 54, 45]  # endpoints swapped
    aligned = [50, 51, 52, 53, 54, 54, 55, 56, 57, 58, 59]  # uniformly shifted up
    _clip(tmp_path / "idle1.mp4", [0, 5, 10, 15, 20] + a_ramp + [200, 205, 210, 215, 220])
    _clip(
        tmp_path / "idle2.mp4",
        [70, 75, 80, 85, 90]
        + opposed
        + [130, 135, 140, 145, 150]
        + aligned
        + [230, 235, 240, 245, 250],
    )

    table = build_table(tmp_path)

    # idle1's ramp is centred on frame 10 -> bucket floor(10 / 5) = 2.
    seek = table["match"]["idle1"]["idle2"][2]
    # The opposed (appearance-best) candidate sits at idle2 frame 10 (1.0).
    assert seek != pytest.approx(1.0), "must not prefer the appearance-best but opposed candidate"
    # The aligned candidate sits at idle2 frame 26.
    assert seek == pytest.approx(26 / FPS)


def test_a_destination_frame_with_no_direction_is_never_chosen(tmp_path):
    """Rule 3's edge case: a destination frame within DIRECTION_SPAN of its own
    clip's start has no direction at all, and must be excluded from being
    picked even when its raw appearance is the best candidate on offer -
    falling back to "no penalty" for it would make the table systematically
    prefer exactly the edge frames rule 4 already gets away from elsewhere.

    idle1 carries a plain ramp (40..50) at indices 5-15, centred on frame 10
    (net direction: rising). idle2 carries a TENT at indices 5-15 - the same
    rising half as idle1's ramp, but folding back down (endpoints equal, so
    its net direction is undefined) - a better pointwise appearance match
    (cost ~20) than the alternative. idle2 also carries a plain RAMP2 at
    indices 21-31, directed the same way idle1 is, but a worse pointwise
    match (cost ~36). Appearance alone would pick the tent; excluding it for
    having no direction must make the table pick ramp2 instead.
    """
    a_ramp = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    tent = [40, 41, 42, 43, 44, 45, 44, 43, 42, 41, 40]  # symmetric -> no direction
    ramp2 = [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]  # directed, worse appearance
    _clip(tmp_path / "idle1.mp4", [0, 5, 10, 15, 20] + a_ramp + [200, 205, 210, 215, 220])
    _clip(
        tmp_path / "idle2.mp4",
        [70, 75, 80, 85, 90]
        + tent
        + [130, 135, 140, 145, 150]
        + ramp2
        + [230, 235, 240, 245, 250],
    )

    table = build_table(tmp_path)

    # idle1's ramp is centred on frame 10 -> bucket floor(10 / 5) = 2.
    seek = table["match"]["idle1"]["idle2"][2]
    # The tent (better appearance, no direction) sits at idle2 frame 10 (1.0).
    assert seek != pytest.approx(1.0), "a direction-less destination must never be chosen"
    # ramp2 (worse appearance, but directed) sits at idle2 frame 26.
    assert seek == pytest.approx(26 / FPS)


def test_a_source_frame_with_no_direction_falls_back_to_appearance_only(tmp_path):
    """Rule 3's other edge case: a *source* frame within DIRECTION_SPAN of its
    own clip's start has no direction either, but unlike a destination it
    cannot simply be excluded - every bucket of the outgoing clip must produce
    some seek. It must fall back to appearance-only, including reconsidering
    destinations that would otherwise be excluded for lacking direction
    themselves: there is nothing to compare their direction *against*, so
    excluding them here has no justification even though it would elsewhere.

    idle1's frame 2 (near idle1's own start, so no direction) is the near-
    unique match for idle2's frame 2 (also near idle2's own start, so no
    direction either - would be wrongly excluded as a destination without the
    source-side fallback). idle2 also carries a directed ramp elsewhere whose
    values are all far from idle1's frame 2 - a worse match, but the only one
    eligible if the fallback is missing and idle2's frame 2 gets excluded.
    idle1's other frames in the same bucket (0, 1, 3, 4) are given values with
    no good match anywhere in idle2, so they cannot win the bucket instead and
    obscure what frame 2 alone did.
    """
    _clip(tmp_path / "idle1.mp4", [150, 151, 50, 152, 153] + [5] * 15)
    _clip(
        tmp_path / "idle2.mp4",
        [90, 91, 50, 92, 93]
        + [5, 5, 5]
        + [60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80]
        + [5, 5, 5, 5],
    )

    table = build_table(tmp_path)

    # idle1 frame 2 is in bucket floor(2 / 5) = 0; idle2's matching frame 2
    # gives the expected seek of 2 / FPS.
    assert table["match"]["idle1"]["idle2"][0] == pytest.approx(2 / FPS)


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
