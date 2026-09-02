"""Which clip the face plays, and when it is allowed to change.

Pure policy. No I/O, no numpy, and nothing from `daemon/face_lipsync/` (CONTRACTS 4) -
`daemon/app.py` assembles this against the renderer's drivers. It is deliberately not
part of `daemon/face.py` either: that module is the bus, `face_routes.py` imports it,
and a random pick plus a queue do not belong on the event path.

This is the clip chooser that lived in `daemon/static/face.html` (`clipFor`,
`FOR_ACTIVITY`, the idle pool, the flourish interval). It moved server-side because the
renderer composites a mouth onto the frame it *believes* is on screen, and cannot be
told which clip that is a round trip late. The page keeps its own copy as the fallback
for a renderer that latches `failed`, and ADR 0017
(docs/adr/0017-the-neutral-moment-not-the-matched-pose.md) still governs that path,
where the clip carries the mouth.

**A wanted clip is queued and applied at the current clip's own end, never mid-clip.**
That is the whole design and it is one measurement, recorded in ADR 0020
(docs/adr/0020-lip-sync-makes-a-clip-ambient.md): downscaled whole-frame mean absolute
difference across the join, over the owner's ten prepared clips. The baseline is a clip's
own loop point - the join the face has always made every few seconds, which the owner has
never remarked on.

| join | min | median | p90 | max | over baseline max (2.14) |
|---|---|---|---|---|---|
| a clip's own loop point (**baseline**) | 0.88 | 1.14 | - | 2.14 | - |
| clip end -> next clip frame 0 | 1.08 | **1.41** | - | 2.18 | 3 / 90 (3%) |
| a "near-neutral" moment -> one-shot frame 0 | 0.94 | 1.51 | 5.06 | 8.95 | 85 / 252 (34%) |
| any moment -> one-shot frame 0 (cut now) | 1.40 | 7.98 | 12.54 | 12.82 | 76 / 84 (90%) |

Playing every clip to its end is smooth by construction, so there is **no pose-match
lookup and no neutral wait in here**. Both exist to make a *mid-clip* switch survivable,
and this design never makes one - that alone is the reason they are absent. The two
one-shot rows say something narrower and worth not overstating: they measure entry at
frame 0, which is what a one-shot must do, and there a near-neutral wait does not
reliably buy smoothness, because `face_match.py`'s neutral flag is measured against each
clip's own frame 0 rather than a pose shared across clips (`idle2@1.75s` is flagged
neutral and is 8.94 from `amused`'s frame 0). ADR 0017's own path is neutral wait *then*
pose-matched entry, which nothing here measured; its numbers still govern the fallback.

The cost is expression latency and it was accepted knowingly: a one-shot queued to the
clip end arrives after half a clip on average - 4.0s behind idle1/2/3, 3.1s behind
`listening`, 3.0s behind `thinking`. The owner chose that over a picture that snaps one
time in three, having been shown both numbers.

Two further departures from the page's version, both from the live run:

* `wanted("speaking", ...)` returns the clip already up. The page returned the single
  driving clip, which is what the owner saw: *"발화할때 바로 클립이 idle 클립으로
  변경돼"*. With every clip driveable the mouth is generated for whichever clip is up,
  so speech begins where the face already is.
* `speaking_loud` and `speaking_soft` are not driveable. They are chosen by loudness, so
  driving them would swap the clip - and reset the mouth's continuity - every time the
  owner raised their voice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from daemon.face import MOODS

IDLES: tuple[str, ...] = ("idle1", "idle2", "idle3")
"""The idle pool, ported from the page. The rotation is back on: it was pinned to the
one driving clip while only `idle2` could be lip-synced, and that pin is precisely what
made speech read as a different clip starting."""

FLOURISHES: tuple[str, ...] = ("flourish_arms",)

FLOURISH_MIN_SECONDS = 15.0
FLOURISH_MAX_SECONDS = 60.0
"""How long an unbroken idle stretch waits for a flourish. A taste choice and labelled
as one in the page it comes from: 15-60s averages 37s, against the spec's "once a minute
breaks the loop's visible period". The interval was never why the flourish failed to
appear - the page was re-arming the deadline on every idle clip's end, faster than time
advanced. Arming and firing stay with the render loop, as they do with the page's
`tick()`; only the interval is policy."""

LIPSYNC_CLIPS: frozenset[str] = frozenset(
    {
        *IDLES,
        "listening",
        "thinking",
        "working",
        *MOODS,
        *FLOURISHES,
    }
)
"""The ten clips the owner prepared caches for, under `<data_dir>/face/lipsync/`.

All ten have now been judged by the owner with lip-sync on them (the last four -
`thinking`, `sulky`, `curious`, `flourish_arms` - on 2026-09-01: "다 괜찮은듯"). A clip
that turns bad is removed here and nowhere else.

`speaking_loud` and `speaking_soft` are absent deliberately (see the module docstring).
This is what was prepared, not what is loaded: `daemon/app.py` scans for the cache
artefacts and passes the set it actually found as `available`, because rule 4 is omission
- a clip that was never prepared is absent, never interpolated or substituted."""

ONE_SHOT_CLIPS: frozenset[str] = frozenset({*MOODS, *FLOURISHES})
"""Clips that play once and hand back, rather than looping as an activity's clip.

Read off `daemon/face.py`'s `MOODS` rather than restated, so a fourth mood there is a
one-shot here without anyone having to remember this file. Each is an arc from a neutral
pose and back, which is why one only ever enters at frame 0 (`face_match.py:ONE_SHOTS`):
a mid-arc entry destroys the arc."""

FOR_ACTIVITY: Mapping[str, tuple[str, ...]] = {
    "idle": IDLES,
    "listening": ("listening", "idle1"),
    "thinking": ("thinking", "idle1"),
    "working": ("working", "thinking", "idle1"),
    "speaking": ("speaking_soft", "idle1"),
}
"""activity -> clip, with the fallback chain from the face spec's 3.7, ported verbatim.

`speaking_soft` stays in its chain even though no cache exists for it, so that this table
and the page's read identically: `available` is what filters a clip with no cache, and
naming the exception here as well would be the same rule written twice."""


def wanted(
    activity: str,
    *,
    pending_shot: str | None,
    current: str,
    available: frozenset[str],
    pick: Callable[[Sequence[str]], str],
) -> str:
    """The clip `activity` wants, given what will be on screen when it is applied.

    Asked once per publish tick, so it must be cheap and must not surprise: the answer
    goes to `ClipQueue.want`, which drops it when it asks for the clip that will already
    be playing.

    `pending_shot` is what makes "already playing" the right question rather than "on
    screen now". A gesture queued in front of this answer means the clip it follows is a
    fresh choice, so the idle pool is not narrowed by the pose being left behind.

    `pick` is injected rather than `random.choice` reached for at module level, so the
    rotation is a decision a test can pin.

    Returns `current` when nothing better is available - the face keeps doing what it was
    doing. That covers an activity with no chain here (the value arrives over an event bus
    as a string, while `daemon/face.py`'s `Activity` is a Literal) and a cache set too
    partial to answer, and it is why this returns `str` and not `str | None`.
    """
    if activity == "speaking" and current in available:
        return current

    upcoming = pending_shot or current
    if activity == "idle":
        pool = [stem for stem in IDLES if stem in available]
        if not pool:
            return current
        rotated = [stem for stem in pool if stem != upcoming]
        return pick(rotated or pool)

    for stem in FOR_ACTIVITY.get(activity, ()):
        if stem in available:
            return stem
    return current


class ClipQueue:
    """When a wanted clip may be applied: at the current clip's own end, and nowhere else.

    Holds at most two things - one one-shot and one activity clip - because that is all
    the face can be behind on. A second activity want inside one clip replaces the first
    (the face shows where it ended up, not a queue of poses it no longer holds), while a
    one-shot is a separate slot that goes first: a mood is the daemon saying something and
    an activity is ambient.

    Every boundary this reasons about is a clip end, so it has to know how long each clip
    runs. `lengths` is that, in seconds, and `daemon/app.py` reads it off the prepared
    caches - there is no table of clip durations in this file, because the durations are
    the owner's data and not a constant of the design.
    """

    def __init__(self, *, current: str, ends_at: float, lengths: Mapping[str, float]) -> None:
        self._lengths = dict(lengths)
        if current not in self._lengths:
            raise KeyError(
                f"no length for {current!r}, so the queue cannot tell when it ends - "
                "every want would be either immediate or never"
            )
        self._current = current
        self._ends_at = ends_at
        self._shot: str | None = None
        self._clip: str | None = None
        self._after_shot: str | None = None
        """What a gesture interrupted, to return to when its arc is done.

        A mood is an arc and not a state, and nothing on this side fires `ended` - so
        without this a one-shot that became the driving clip simply loops. The owner
        watched `amused` (5.46s) repeat four times through one 20-second answer. It is
        the speaking rule meeting a gesture: `wanted("speaking", ...)` answers with the
        clip already up, because clips must not change mid-utterance, and once a one-shot
        IS that clip the rule reads as "hold this expression"."""

    @property
    def current(self) -> str:
        """The clip being rendered onto, which is `wanted`'s `current`."""
        return self._current

    @property
    def returning_to(self) -> str | None:
        """The clip a gesture in flight will hand back to, or `None`.

        Read by nothing in production yet; it is here because a queue whose state can
        only be inferred from what `due` happens to return is a queue nobody can debug.
        """
        return self._after_shot

    @property
    def pending_shot(self) -> str | None:
        """The gesture waiting at the boundary, which is `wanted`'s `pending_shot`.

        Both are reads rather than something the render loop tracks alongside, so there
        is no second copy of the answer to disagree with this one.
        """
        return self._shot

    def want(self, stem: str, *, one_shot: bool = False) -> None:
        """Ask for `stem` at the next boundary. Nothing happens until `due` says so.

        A want for the clip that will already be playing when it would apply is dropped.
        `Renderer.switch()` resets the per-clip continuity - the motion blend and the
        injection weight average, each a measured defect when it survived a clip change -
        so re-entering the clip already up would pay that reset for a frame-0 restart
        nobody asked for. Left alone the clip crosses its own loop point instead, and that
        loop point is the 1.14 median this design is measured against. Note the *will*: a
        want for the clip on screen underneath a pending gesture is the return the face
        needs after it, and is kept.

        A `stem` with no prepared length is ignored, not raised on. A mood arrives as a
        name off the face bus and nothing promises a cache exists for it; rule 4 is
        omission, the same thing the page's `playOneShot` does - "no clip, no expression,
        never substitute idle" - and raising here would take the render loop down over a
        missing expression.
        """
        if stem not in self._lengths:
            return
        if one_shot:
            self._shot = stem
            return
        self._clip = None if stem == (self._shot or self._current) else stem

    def due(self, *, at: float) -> str | None:
        """The clip to switch to now, if `at` has reached the current clip's end.

        **Call this every tick, not only when something is wanted**, because this is also
        what notices the current clip looping. With nothing waiting at a clip's end the
        clip goes round again (`Renderer` indexes its frames modulo their count) and the
        next boundary is one length further on; a queue that had not been told the loop
        happened would apply the next want wherever it arrived, which is the mid-clip cut
        this class exists to prevent.

        A returned clip is taken to start at `at`, matching the caller setting the new
        `ClipClock`'s epoch to the same instant. So a want is applied at most one tick
        past the clip's own end - that granularity is the render loop's period, and the
        only alternative would be a boundary the caller and this object disagreed about.
        """
        if at < self._ends_at:
            return None

        if self._shot is not None:
            # Only from a loop clip: a gesture interrupted by a second gesture must
            # still return to what the FIRST one interrupted, or the face plays a third
            # expression nobody asked for.
            if self._current not in ONE_SHOT_CLIPS:
                self._after_shot = self._current
            stem, self._shot = self._shot, None
        elif self._clip is not None:
            stem, self._clip = self._clip, None
            # The activity moved on while the gesture played, so the face goes where the
            # conversation is rather than back to a pose that is no longer true.
            self._after_shot = None
        elif self._after_shot is not None:
            # The arc is over. This branch is what keeps a one-shot from ever reaching
            # the loop below - which is the whole defect it exists for.
            stem, self._after_shot = self._after_shot, None
        else:
            length = self._lengths[self._current]
            while self._ends_at <= at:
                self._ends_at += length
            return None

        self._current = stem
        self._ends_at = at + self._lengths[stem]
        return stem
