"""The clip policy: which clip, and when it is allowed to change.

Two objects with one job each, and the split is the point. `wanted()` is asked every
publish tick and answers *which* clip the current activity wants - a pure function, so
the idle rotation's random pick is injected rather than reached for. `ClipQueue` owns
*when* that answer may be applied, which is only ever at the current clip's own end.

The measurement behind the queue is in `daemon/face_clips.py`'s docstring: end -> next
clip frame 0 is a baseline join (1.41 median against a clip's own loop point at 1.14),
and cutting now is up to 12.82. Every test here that looks like it is about arithmetic
is about that.
"""

from __future__ import annotations

import pytest

from daemon.face_clips import LIPSYNC_CLIPS, ONE_SHOT_CLIPS, ClipQueue, wanted

SECONDS = {
    # Clip lengths in seconds, the way `daemon/app.py` reads them off each prepared
    # cache. Two are the plan's own, because its Task 3 tests are written against
    # them: 8.04 for an idle clip and 5.46 for `amused`. `listening` 6.2 and
    # `thinking` 6.0 follow from the same plan's latency figures (a one-shot queued
    # to the clip end arrives 3.1s behind `listening` and 3.0s behind `thinking`,
    # which is half a clip). The rest are this file's own fixtures and stand for
    # nothing at all - the real numbers come from the caches at assembly time.
    "idle1": 8.04,
    "idle2": 8.04,
    "idle3": 8.04,
    "listening": 6.2,
    "thinking": 6.0,
    "working": 7.0,
    "amused": 5.46,
    "sulky": 5.0,
    "curious": 5.0,
    "flourish_arms": 6.0,
}


def _first(options):
    """The injected `pick`, so the idle rotation is a decision and not a coin."""
    return options[0]


def _queue(current: str = "idle2", ends_at: float = 8.04) -> ClipQueue:
    return ClipQueue(current=current, ends_at=ends_at, lengths=SECONDS)


# --- which clip -------------------------------------------------------------


def test_speaking_does_not_change_the_clip():
    """The owner's report from the live run: "발화할때 바로 클립이 idle 클립으로
    변경돼". With every clip driveable there is no reason to move - the mouth is
    generated for whichever clip is up, so speech begins where the face already is."""
    assert (
        wanted(
            "speaking",
            pending_shot=None,
            current="listening",
            available=LIPSYNC_CLIPS,
            pick=_first,
        )
        == "listening"
    )
    assert (
        wanted(
            "speaking",
            pending_shot=None,
            current="thinking",
            available=LIPSYNC_CLIPS,
            pick=_first,
        )
        == "thinking"
    )


def test_speaking_loud_and_soft_are_not_driveable():
    """They are chosen by loudness, so driving them would swap the clip - and reset
    the mouth's continuity - every time the owner raised their voice. They stay as
    v1 fallback clips with no cache."""
    assert "speaking_loud" not in LIPSYNC_CLIPS
    assert "speaking_soft" not in LIPSYNC_CLIPS


def test_the_idle_rotation_avoids_the_clip_that_will_be_on_screen():
    """Not the clip that is on screen now - the one that will be when this answer is
    applied. `wanted` is asked every tick, and with a one-shot already queued the
    clip after it is a fresh choice: `idle2` is the pending gesture's neighbour, not
    the pose being left, so excluding it would shrink the pool for nothing."""
    assert (
        wanted("idle", pending_shot=None, current="idle1", available=LIPSYNC_CLIPS, pick=_first)
        == "idle2"
    )
    assert (
        wanted(
            "idle", pending_shot="amused", current="idle1", available=LIPSYNC_CLIPS, pick=_first
        )
        == "idle1"
    )


def test_an_activity_whose_clip_was_never_prepared_falls_down_its_own_chain():
    """Rule 4, as `face_match.py` already applies it: omission, never substitution.
    `working` has a chain (working -> thinking -> idle1) and the chain is what a
    partial cache set walks, so a missing cache costs that activity its own clip and
    nothing else."""
    available = frozenset({"idle1", "thinking"})
    assert (
        wanted("working", pending_shot=None, current="idle1", available=available, pick=_first)
        == "thinking"
    )
    assert (
        wanted(
            "working",
            pending_shot=None,
            current="idle1",
            available=frozenset({"idle1"}),
            pick=_first,
        )
        == "idle1"
    )


def test_an_activity_this_module_has_no_chain_for_leaves_the_clip_alone():
    """`daemon/face.py`'s `Activity` is a Literal, but the value arrives over an
    event bus as a string. An unrecognised one must not blank the face or throw on
    the render loop - the face keeps doing what it was doing."""
    assert (
        wanted("dreaming", pending_shot=None, current="idle3", available=LIPSYNC_CLIPS, pick=_first)
        == "idle3"
    )


# --- when it may be applied -------------------------------------------------


def test_a_clip_is_never_cut_mid_way():
    """The whole design. A want registered at 2s into an 8.04s clip takes effect at
    8.04s, not at 2s - measured, end -> frame 0 is a baseline join and any mid-clip
    join is up to ten times worse."""
    q = _queue(current="idle2", ends_at=8.04)
    q.want("listening")
    assert q.due(at=2.0) is None
    assert q.due(at=8.03) is None
    assert q.due(at=8.04) == "listening"


def test_the_last_want_before_the_boundary_wins():
    """Two activity changes inside one clip is ordinary - listening then thinking
    while she works out an answer. The face shows where it ended up, not a queue of
    poses it no longer holds."""
    q = _queue(current="idle2", ends_at=8.04)
    q.want("listening")
    q.want("thinking")
    assert q.due(at=8.04) == "thinking"


def test_a_one_shot_outranks_an_activity_want_at_the_same_boundary():
    """A mood is the daemon saying something; an activity is ambient. If both are
    waiting at the boundary the expression goes first and the activity is still
    pending after it."""
    q = _queue(current="idle2", ends_at=8.04)
    q.want("listening")
    q.want("amused", one_shot=True)
    assert q.due(at=8.04) == "amused"
    assert q.due(at=8.04 + 5.46) == "listening"


def test_a_want_for_the_clip_already_playing_is_dropped():
    """`Renderer.switch()` resets the per-clip continuity - the motion blend and the
    injection weight average - so re-entering the clip already up pays that reset for
    a frame-0 restart nobody asked for. Left alone, the clip crosses its own loop
    point instead, which is the 1.14 median this design is measured against."""
    q = _queue(current="idle2", ends_at=8.04)
    q.want("listening")
    q.want("idle2")
    assert q.due(at=8.04) is None


def test_a_want_for_the_clip_under_a_pending_one_shot_still_queues_the_return():
    """The same rule read at the right moment. `idle2` is on screen, so it looks like
    the case above - but a gesture is queued in front of it, and after the gesture
    `idle2` is where the face has to come back to. Dropping this want is how a mood
    would loop forever."""
    q = _queue(current="idle2", ends_at=8.04)
    q.want("amused", one_shot=True)
    q.want("idle2")
    assert q.due(at=8.04) == "amused"
    assert q.due(at=8.04 + 5.46) == "idle2"


def test_a_want_that_arrives_after_the_clip_looped_waits_for_the_next_loop_point():
    """The clip loops (`Renderer` indexes its frames modulo their count), so a clip
    end that nothing was waiting at is not a boundary the next want may use - it has
    passed, and cutting at 2s into the second pass is the mid-clip join this whole
    module exists to avoid. `due` is what notices the loop, which is why the render
    loop has to call it every tick and not only when it wants something."""
    q = _queue(current="idle2", ends_at=8.04)
    assert q.due(at=8.05) is None
    q.want("listening")
    assert q.due(at=10.0) is None
    assert q.due(at=16.08) == "listening"


def test_a_clip_with_no_prepared_cache_is_omitted_rather_than_substituted():
    """Rule 4 again, on the queue's side: a mood arrives as a name off the face bus
    and nothing promises a cache was ever prepared for it. The expression is simply
    absent - the same thing the page's `playOneShot` does ("no clip, no expression -
    never substitute idle") - and it must not strand the queue or raise on the
    render loop."""
    q = _queue(current="idle2", ends_at=8.04)
    q.want("smug", one_shot=True)
    q.want("listening")
    assert q.pending_shot is None
    assert q.due(at=8.04) == "listening"


def test_the_queue_reports_what_the_next_answer_has_to_be_chosen_against():
    """`wanted` needs `current` and `pending_shot`, and the queue is the only thing
    that knows either after the first switch. Exposed as reads so the render loop
    does not keep a second copy that can disagree."""
    q = _queue(current="idle2", ends_at=8.04)
    assert (q.current, q.pending_shot) == ("idle2", None)
    q.want("amused", one_shot=True)
    assert q.pending_shot == "amused"
    assert q.due(at=8.04) == "amused"
    assert (q.current, q.pending_shot) == ("amused", None)


def test_the_one_shots_are_the_moods_plus_the_flourish():
    """One list, not two: `daemon/face.py` owns the mood names and a fourth one added
    there is a one-shot clip here without anybody remembering to say so."""
    assert ONE_SHOT_CLIPS == frozenset({"amused", "sulky", "curious", "flourish_arms"})
    assert ONE_SHOT_CLIPS <= LIPSYNC_CLIPS


def test_a_queue_that_cannot_know_when_a_clip_ends_is_a_construction_error():
    """The one thing it does is time a boundary. A missing length for the clip it is
    started on would make every want either immediate or never, silently."""
    with pytest.raises(KeyError, match="unprepared"):
        ClipQueue(current="unprepared", ends_at=8.04, lengths=SECONDS)
