"""Turn one audio window plus one driving frame into one JPEG."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from daemon.face_lipsync import Cache, LipsyncEngine, composite
from daemon.face_lipsync.audio import CONTEXT_MS, latest_audio_ms
from daemon.face_lipsync.ring import PcmRing

logger = logging.getLogger(__name__)

JPEG_QUALITY = 85

# A published frame is the WHOLE composited frame, and two earlier attempts at
# sending less were both wrong.
#
# The first sent each frame's crop box, on a measurement that was real but did not
# apply: the crop is 3.4x cheaper to send (52KB against 174KB, 0.46ms against 2.45ms)
# - and this page is served over loopback to the same machine, where 47 Mbit/s costs
# nothing. The saving bought nothing and was paid for in the only currency the owner
# is judging.
#
# The second made it worse by sending the union of those boxes, 1.40x the area of any
# one of them, and claimed the extra margin "carries unmodified driving-clip pixels,
# which is what the page has underneath anyway". That is false whenever the page's
# `<video>` is not on the same frame as the render, which is always: the margin lands
# on mismatched pixels and draws a bright rectangle across the head. Measured, the
# model only ever rewrites `mask > 0` inside `box` - 289-314 x 230-274px, a mean of
# 73k pixels against the union's 556k. Seven times more pixels than the model touches,
# every one of them head and chest, to produce a seam that did not have to exist.
#
# The spike's own 1:1 comparison is seamless for one reason: the server composited
# into the same frame, and a gaussian-blurred mask has no boundary to see. Compositing
# in the browser cannot reproduce that - a JPEG has no alpha, so its edge is hard
# however small it is. So the whole frame goes over the wire and the page displays it
# rather than blending anything.

DETAIL_SIGMA = 1.1
"""Gaussian sigma separating "texture" from "structure" in the driving frame.

Small on purpose. The mouth this borrows from is the driving clip's own, usually
closed; at a larger sigma the transfer carries that closed lip *edge* across and
it ghosts through an open mouth. 1.1 was checked on the six most-open generated
frames of `idle2` - single set of teeth, single lip contour, no doubling - and a
global average cannot see that failure, so it has to be checked there.
"""


def restore_detail(
    mouth: np.ndarray,
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    weight: np.ndarray,
    sigma: float = DETAIL_SIGMA,
) -> np.ndarray:
    """Add the driving frame's fine texture back onto a generated mouth.

    The generated mouth is soft: measured on a landmark-derived lip box it keeps
    71% of the driving clip's detail at product size. That softness is what the
    owner saw and called trembling - a blurred patch whose texture shifts frame to
    frame reads as vibration, which is why nine temporal fixes all missed.

    This is not a face enhancer and deliberately hallucinates nothing: the avatar
    is a fixed clip, so the original texture at this exact box is already aligned,
    and only its high-frequency residual is added. Measured 71% -> 97% of the
    driving clip's lip detail for 0.56ms on a 320x394 crop.

    Do not reach for temporal smoothing instead. It lowers frame-to-frame RMSE in
    the lip region, but that number ran *opposite* to the owner's own ranking of
    three renders - it buys the metric by destroying the detail.

    **The texture is weighted down where the two pictures disagree, and that is what
    stopped the mouth ghosting.** The driving frame's mouth is CLOSED. Borrowing its
    high frequencies wholesale stamps a closed lip line onto an open mouth, every
    frame, which reads as an afterimage - and no temporal filter can remove it,
    because it is re-applied after the filter and re-derived from scratch each frame.
    Two arms that mathematically cannot ghost (a temporal median, which never averages
    two poses, and a blend of only the low frequencies, which takes every edge from
    the current frame) both still showed it; turning this off removed it. So the
    injection is scaled by how far the generated mouth has moved from the driving one:
    full strength on cheeks and chin where they agree, off inside an open mouth where
    they do not. The owner ranked it least-ghosting of three arms and read its
    sharpness as unchanged.

    **`weight` is passed in rather than derived here, and that is not tidiness.** A
    version of this recomputed it from the current frame alone, so the strength of the
    borrowed texture pulsed at frame rate - a spatial operation modulated every frame,
    which is a vibration source in its own right. The owner picked the smoothed one
    over it. `Renderer` keeps that average; `detail_weight` computes one frame's.

    `mouth` must already be `box`-sized, as `composite` requires.
    """
    x1, y1, x2, y2 = box
    orig = frame[y1:y2, x1:x2].astype(np.float32)
    high = orig - cv2.GaussianBlur(orig, (0, 0), sigma)
    scaled = cv2.resize(weight, (x2 - x1, y2 - y1))[..., None]
    return np.clip(mouth.astype(np.float32) + high * scaled, 0, 255).astype(np.uint8)


def detail_weight(mouth: np.ndarray, frame: np.ndarray, box: tuple[int, int, int, int]):
    """How much borrowed texture each pixel may take, at the model's own 256.

    At 256 and not at `box` on purpose: the box breathes with the face (608-726px
    across one clip), and a map that changes size every frame cannot be averaged with
    the one before it. This one can, which is what `Renderer` does with it.
    """
    x1, y1, x2, y2 = box
    small = cv2.resize(frame[y1:y2, x1:x2], (256, 256)).astype(np.float32)
    disagreement = np.abs(cv2.resize(mouth, (256, 256)).astype(np.float32) - small)
    return np.clip(1.0 - disagreement.mean(axis=2) / DETAIL_CUTOFF, 0.0, 1.0)


DETAIL_CUTOFF = 40.0
"""Mean per-channel difference at which the borrowed texture is fully suppressed.

Not a tuned constant so much as a scale: below it the generated mouth and the driving
frame are the same picture and the texture belongs; at it they are different pictures
and the texture is the wrong one. An open mouth over a closed one clears this easily -
teeth against lips is most of the 0-255 range - while the cheeks and chin the paste
also covers sit far below it.
"""

BATCH = 2
"""Frames computed per model step, and this is arithmetic rather than taste.

Measured on the assembled engine: at N=1 a frame costs 49.3ms against a 41.67ms
budget - UNet 41.55, TAESD 4.72, convert 0.93, features 2.12 - so 24fps is not
reachable one frame at a time on any of it. At N=2 the UNet drops to 29.29ms/frame
and the whole two-frame step is 72.21ms, i.e. 36.10ms/frame with 13% headroom.

N=3 is not the next step up: the batch cannot start until its last frame's audio has
arrived, so widening costs `(N-1) x 41.67ms` of latency and N=3 misses the 250ms
ceiling by 7.6ms.
"""


class FrameClock:
    """Which frame to render on this tick, or `None` because its audio is not here yet.

    The render loop in `daemon/app.py` ticks at `fps` and asks this each time. Three
    things it exists to get right, none of which are visible from the loop:

    **The batch-fill wait.** A step covers `BATCH` frames, so it cannot start until
    the LAST of them has its audio - `(BATCH - 1) x 41.67ms` past what the first frame
    alone needs. `PcmRing.window` zero-fills what has not arrived instead of erroring,
    so a loop that just renders "the frame for now" gets a mouth conditioned on
    silence and nothing anywhere complains.

    **Re-anchoring.** `PcmRing.origin` moves - a new turn, a long silence, a barge-in
    that rebuilds the clock, and continuously once the ring is full and dropping
    samples. Frame indices are relative to it, so the count restarts whenever it
    jumps. Small forward creep from sample dropping is not a new turn, hence the
    tolerance.

    That tolerance only makes creep harmless; it does not make it correct. A frame
    index means "this far after `origin`", so an `origin` that has moved forward
    points the same index at later audio. **Size the ring to hold a whole utterance**
    and the question does not arise, because nothing is dropped until the turn ends -
    which is what `daemon/app.py` does. A ring sized for the 2.2s window alone is
    dropping samples continuously mid-turn, and the mouth drifts against the sound
    without anything reporting it.

    **One frame per tick, never a skip.** Returning "the newest ready frame" would
    jump the mouth forward after any hitch. Falling behind is corrected by the
    renderer's own held-frame ticks, which cost nothing.

    `now` and `origin` must come from the SAME clock, and in production that clock is
    `loop.time()` - `daemon/voice/conversation.py` stamps audio with it deliberately
    (see its `_playback_until` note). Passing `daemon.clock.now()` here type-checks,
    runs, and puts the mouth an arbitrary offset away from the sound.
    """

    __slots__ = ("_fps", "_frame", "_origin")

    RE_ANCHOR_TOLERANCE = 0.2
    """Seconds of movement **since the previous tick** treated as the ring dropping old
    samples rather than as a new turn. A full ring creeps every feed; a turn boundary
    jumps.

    Since the previous tick, not since the anchor - and the first version got that
    wrong. It compared against the value captured at the last re-anchor, so steady
    creep accumulated against a fixed reference and tripped this threshold every time
    it passed 0.2s: measured 7 false re-anchors in 40 ticks at 40ms of creep each,
    which restarts the utterance from frame 0 five times a second. The test that was
    supposed to cover this fed two 40ms steps and never let them add up.
    """

    def __init__(self, *, fps: float) -> None:
        self._fps = fps
        self._origin: float | None = None
        self._frame = 0

    def due(self, *, now: float, origin: float) -> int | None:
        if self._origin is None or abs(origin - self._origin) > self.RE_ANCHOR_TOLERANCE:
            self._frame = 0
        self._origin = origin
        needed = latest_audio_ms(self._frame + BATCH - 1, self._fps) / 1000.0
        if now - origin < needed:
            return None
        frame = self._frame
        self._frame += 1
        return frame


class ClipClock:
    """Where the driving clip's playhead is, at any moment on the loop's clock.

    **The page never rewinds that clip.** Speech begins on the clip that is already
    playing, at the position it is already at - which is the whole point, because the
    alternative is what the owner saw: idle rotates through `idle1/idle2/idle3` and
    only `idle2` can be lip-synced, so the first word of every reply swapped the head,
    the pose and the framing, and even an `idle2 -> idle2` handover jumped back to
    frame 0. Both halves of that are gone (`daemon/static/face.html`), and what
    replaces them is this: the driving frame on screen is a function of wall time and
    nothing else.

    So both sides need the same function, and it lives here rather than being written
    out twice. `index` is what the renderer composites onto; `position` is what the
    page seeks its `<video>` to, in the `currentTime` units a `<video>` speaks -
    handed over once through `/face/manifest`, after which the page free-runs on its
    own playback and re-reads this only where a seek is invisible (behind the opaque
    overlay, or on the browser-imposed pause `ensurePlaying` already recovers from).

    `at` is `loop.time()`, the same clock `FrameClock` and the audio stamps use, for
    the same reason: `daemon.clock.now()` here type-checks, runs, and puts the pose an
    arbitrary offset from the page.

    `epoch` is arbitrary - any fixed instant defines the same clock - and
    `daemon/app.py` takes it when the renderer is built, so the clip's position is
    small and readable rather than a monotonic seconds-since-boot remainder.
    """

    __slots__ = ("_epoch", "_fps", "_period")

    def __init__(self, *, fps: float, frames: int, epoch: float) -> None:
        self._fps = fps
        self._epoch = epoch
        self._period = frames / fps

    def index(self, at: float) -> int:
        """The driving frame on screen at `at`, unwrapped - `Renderer.encode` takes it
        modulo the clip length, exactly as it already did with the turn-relative one.

        Floored, not rounded, because that is what a `<video>` does with its own
        `currentTime`: it shows the frame the instant falls *inside*. Rounding would
        disagree with the page for half of every frame period. On an exact frame
        boundary the answer is decided by the last bit of the subtraction rather than
        by anything here - a 4e-14s residue is enough to floor one frame low - and
        that is left alone: the pair is only ever as close as `DISPLAY_LEAD` rounds
        5.81 to 6, and no real instant lands on a boundary.
        """
        return math.floor((at - self._epoch) * self._fps)

    def nearest(self, at: float) -> int:
        """The driving frame CLOSEST to `at`, which is a different question.

        `index` answers "what is on screen", so it floors. `Renderer.step` is asking
        something else - which frame this turn's audio lines up with - and it then adds
        whole frame counts to that answer, so a floor there hands every frame of the
        turn the same discarded fraction: `round(x) + k` is the right quantisation of
        `x + k`, and `floor(x) + k` is that same value biased half a frame late on
        average. Arithmetic, not measurement.

        **And it is deliberately not claimed as a measurement, because the measurement
        cannot see it.** This was changed on a reading of +0.4 to +0.6 frames at the
        socket that looked like exactly this bias; re-measuring afterwards moved the
        *other* phase from -0.42 to -0.71, which it could not have if that were the
        cause. What that reading actually is, over three runs, is in
        `evals/face_lipsync_live.py:report_alignment` - per-turn pipeline latency,
        which moves about half a frame on its own. So this stands on being the correct
        rounding and nothing else; half a frame is well under what the socket can
        resolve, and the handover measured clean with the floor in place too.
        """
        return round((at - self._epoch) * self._fps)

    def position(self, at: float) -> float:
        """Seconds into the clip at `at`, `0 <= position < frames / fps`.

        The same instant `index` answers for, in the units a `<video>` reads. The two
        agree by construction because a `<video>` shows the frame containing its
        `currentTime`, and this clip's `duration` is exactly `frames / fps` (measured
        in Chrome: 8.041667s against 193/24, to the last digit either reports).
        """
        return (at - self._epoch) % self._period


WEIGHT_BLEND = 0.25
"""Weight of the current frame in the running average `restore_detail` is handed.

Lower than `MOTION_BLEND` on purpose. This is not smoothing the picture - it is
smoothing how strongly a texture is applied to it, and that strength has no business
changing at frame rate. Where the mouth is, and where it is not, moves slowly.
"""

MOTION_BLEND = 0.55
"""Weight of the new mouth against the one before it, applied before `restore_detail`.

The owner's report was that the mouth "moves too fast" - not that it trembled. Measured
against real talking footage the generated mouth's per-frame motion really is larger:
independent per-frame generation has none of the inertia a face has, so every frame is
free to jump. Three arms were rendered against the same audio and the owner chose this
one over 0.7 and over no blend at all, calling it natural enough.

That is the opposite of an earlier verdict on the same idea, and the difference is
where it sits in the pipeline rather than the number - see `Renderer._blend`.
"""

DISPLAY_LEAD = 6
"""How many driving frames AHEAD of its own audio a mouth is composited onto.

Not a fudge factor - it reconciles two clocks on the same clip.
`daemon/static/face.html` plays the driving clip at 1.0x and never rewinds it, so at
wall time T the page is showing frame `ClipClock.index(T)`. A frame conditioned on the
audio at index k cannot reach
the socket before `k/fps + 242ms`: the model's window ends 80ms past its own frame,
the batch's second frame needs 41.67ms more, and the step and the two JPEGs are
another 89ms. Measured at the socket over a 9-second utterance, the overlay ran a
median **5.81 frames (242ms)** behind the page's playhead - first frame 6.35, last
5.81 (`evals/face_lipsync_live.py:report_alignment`).

That matters for the 180ms crossfade the page brings the overlay in on: for 180ms it
dissolves between two poses of the same avatar a quarter-second apart in its own
motion. **How big that is depends on where in the clip the handover lands, and the
average is the wrong number.** When this was written the page rewound to 0, so the
handover always landed in the clip's first ten frames, and `idle2` is nearly still
there: the 6-frame gap cost 0.34-0.47 mean|diff| over the whole frame at frames 6-10,
against **3.55 averaged across the clip and up to 5.65 mid-clip**. That is why it read
as faint on the shipped avatar, and it was kept anyway on two arguments the mean hides.

The first is that the error is **structured**: without the lead, the dissolve's
difference from the clip is a halo over the forehead, the hairline, the cheeks and the
jaw as well as the mouth; with it, the difference is the mouth and nothing else
(measured 0.27 -> 0.11 mean at the midpoint of the fade, and the residue goes from a
whole-face haze to the lip box alone). A structured error is what an eye reads, not a
mean. The second was **insurance priced at nothing** - that the smallness was a
property of one clip's first ten frames, which nobody chose for it, and that "a turn
that begins where the page has not rewound puts the same dissolve somewhere the gap
costs 3-5.7 instead of 0.4."

That sentence is now the ordinary case rather than the hypothetical one. The page
stopped rewinding, so a turn begins wherever the clip already was, and the insurance
is the main line: uncorrected, this dissolve would cost 3.55 on average and 5.65 at
its worst instead of 0.4.

So the audio index and the driving index are two questions, and this answers the
second: the mouth for audio frame k is generated from, and composited into, the
driving frame the page will be showing when it arrives - `ClipClock.index(origin)`
plus k, plus this. MuseTalk pairs any audio with any reference frame (the latent
carries the pose, the audio carries the phoneme), so the pair is exactly as coherent
as before.

**A constant, and deliberately not `(now - origin) * fps` per step.** The driving
index has to advance evenly or the head judders, and a wall-clock reading jitters by
however late the loop noticed the clock's grant. It rounds 5.81 to 6; a slower
machine runs further behind and this under-corrects, which is still better than not
correcting at all.

Not to be confused with the thing the owner actually reported. The quality drop at
the moment speech started was the **frame rate** - 20.1fps arriving as pairs at 10Hz
- and it is fixed above, in `Renderer`'s split. This is the part of that moment that
was left over once the cadence was even.
"""


@dataclass(frozen=True, slots=True)
class Step:
    """One model step's output, in transit from the model thread to the CPU one.

    The indices travel with the mouths instead of staying on the renderer because two
    steps are deliberately in flight at once: this pair's `encode` runs while the next
    pair's `step` does, so anything remembered on the instance would be the wrong
    pair's by the time it was read.
    """

    indices: list[int]
    """Which DRIVING frame each mouth belongs to, unwrapped - `DISPLAY_LEAD` ahead of
    the audio index it was conditioned on. `encode` takes them modulo the clip
    length."""

    mouths: list[np.ndarray]
    """256x256 BGR, straight out of `LipsyncEngine.mouths`, one per index."""


class Renderer:
    """Two frames per model step, in two halves that run on different threads.

    `step` is the model: audio windows out of the ring, `BATCH` mouths back. `encode`
    is the CPU tail: the detail transfer, the composite, and one JPEG each. Neither
    publishes anything - the loop in `daemon/app.py` owns the `Slot` and puts one
    frame into it per frame interval.

    **The split is the frame rate, and it is arithmetic.** Timed in situ on the
    assembled engine, per pair: `step` 71.86ms (35.93ms/frame - features 2.13, UNet at
    N=2 29.29, TAESD 4.26, convert 0.92) and `encode` 16.87ms (8.43ms/frame - the
    detail transfer, the composite, the JPEG, and reading two 5.2MB driving frames out
    of the memory-mapped clip). In sequence inside one call, as the first build ran
    them, that is 88.73ms a pair - **44.36ms a frame, a 22.5fps ceiling before any
    scheduling at all**, and at the socket it measured 20.1fps. On two threads, this
    pair's JPEGs while the next pair's model step is already running, the ceiling is
    the model's own 71.86ms: 35.93ms a frame, 27.8fps, and 24fps has 14% of headroom.
    Spec section 3 assumed exactly this from the start ("CPU 합성은 GPU와 겹친다", the
    way MuseTalk's own `realtime_inference.py` does it); the first build did not, and
    nothing serial reaches 24fps.

    **`encode` must never run concurrently with itself.** `_buffer` is one frame of
    reusable storage, so the pair's second composite overwrites the first's pixels -
    which is why the JPEG is taken inside the loop rather than after it. One worker
    thread for this half, and `daemon/app.py` gives it exactly one.

    The batch-fill wait the spec names belongs to the caller and `FrameClock` owns
    it: a step needs audio through `frame_index + 1`'s window, 41.67ms past what
    `frame_index` alone needs, and `PcmRing.window` zero-fills what has not arrived
    rather than raising - so a loop that steps too eagerly gets a mouth conditioned
    on silence and no complaint from anywhere.
    """

    def __init__(
        self,
        *,
        engine: LipsyncEngine,
        cache: Cache,
        ring: PcmRing,
        clip: ClipClock,
    ) -> None:
        self._engine = engine
        self._cache = cache
        self._ring = ring
        self._clip = clip
        """Where the page's own playhead is. Required rather than defaulted, because
        the default that would read naturally - "the turn starts at frame 0" - is
        exactly the assumption this stopped being able to make."""
        self._buffer = np.empty_like(cache.frames[0])
        self._previous: np.ndarray | None = None
        """Last mouth encoded, for `MOTION_BLEND`. float32, the model's own 256."""
        self._smoothed: np.ndarray | None = None
        """Running average of the injection weight - see `restore_detail`."""
        self._weight_at = -1
        """Continuity marker for `_smoothed`, separate from `_continues_at` on purpose:
        `_blend` runs first and advances that one, so sharing it made this reset on
        every frame and the average never accumulated at all."""
        self._continues_at = -1
        """The display index the next mouth must carry to count as continuous. A turn
        restarts at 0, so a jump here is a turn boundary and the blend starts over
        rather than mixing in a pose from the previous utterance."""
        self.failed = False
        """Latched on the first failure in either half. The caller drops back to v1
        clips and logs once; retrying per frame would fill the log at 24Hz."""

    def step(self, *, frame_index: int, origin: float, fps: float) -> Step | None:
        """`BATCH` mouths from `frame_index` on, or `None`. Never raises.

        `None` means a latched failure and nothing else. There is no "not yet" here:
        whether the audio has arrived is `FrameClock`'s question, asked before this.
        """
        if self.failed:
            return None
        try:
            n = len(self._cache.boxes)
            audio = [frame_index + offset for offset in range(BATCH)]
            windows = [
                self._ring.window(
                    frame_index=index, fps=fps, origin=origin, context_ms=CONTEXT_MS
                )
                for index in audio
            ]
            # The window is addressed by the audio index and the reference frame by the
            # display index. One number answered both until DISPLAY_LEAD, which is why
            # the page was dissolving between two poses a quarter-second apart.
            #
            # The audio index stays turn-relative because it has to: `PcmRing.window`
            # measures from `origin`, so index 0 IS the first audio of this turn. The
            # display index cannot be, because the clip it indexes never restarted -
            # so it is taken from the clock the page is following, anchored at this
            # turn's origin. `index(origin) + k` and `index(origin + k/fps)` are the
            # same integer, which is why one reading of the clock per step is enough
            # and the pair still advances by exactly one frame.
            began = self._clip.nearest(origin)
            shown = [began + index + DISPLAY_LEAD for index in audio]
            return Step(
                indices=shown,
                mouths=self._engine.mouths(windows, [index % n for index in shown]),
            )
        except Exception:
            logger.exception("face: lip-sync engine failed, falling back to clips")
            self.failed = True
            return None

    def _weight(self, mouth: np.ndarray, clip_index: int, index: int) -> np.ndarray:
        """This frame's injection weight, averaged with the frames before it.

        Averaged because the un-averaged version pulsed: recomputed per frame, the
        borrowed texture's strength moved every frame and the owner read that as a
        vibration. Held at 256 so the average is possible at all - the box the texture
        lands in changes size frame to frame, this does not.

        Restarts with `_blend` on a turn boundary: the average is over one utterance.
        """
        current = detail_weight(
            mouth, self._cache.frames[clip_index], self._cache.boxes[clip_index]
        )
        if self._smoothed is None or index != self._weight_at:
            self._smoothed = current
        else:
            self._smoothed = (
                WEIGHT_BLEND * current + (1.0 - WEIGHT_BLEND) * self._smoothed
            )
        self._weight_at = index + 1
        return self._smoothed

    def _blend(self, mouth: np.ndarray, index: int) -> np.ndarray:
        """Mix `mouth` with the one before it, `MOTION_BLEND` of the new one.

        Before `restore_detail`, and that ordering is the whole reason this is usable.
        Smoothing on its own was ranked "거의 못써먹을 수준" by the owner when it was the
        last thing to touch the pixels: it took the teeth with the judder. Applied
        first, it settles the mouth's *structure*, and the driving frame's own texture
        goes back on top afterwards - so the crispness that the mix removed is restored
        from a source the mix never touched. Same alpha, opposite verdict.
        """
        current = mouth.astype(np.float32)
        if self._previous is not None and index == self._continues_at:
            current = MOTION_BLEND * current + (1.0 - MOTION_BLEND) * self._previous
        self._previous = current
        self._continues_at = index + 1
        return current.astype(np.uint8)

    def encode(self, step: Step) -> list[bytes]:
        """One whole-frame JPEG per mouth, in the pair's own order. Never raises."""
        if self.failed:
            return []
        try:
            encoded: list[bytes] = []
            n = len(self._cache.boxes)
            for index, mouth in zip(step.indices, step.mouths, strict=True):
                i = index % n
                box = self._cache.boxes[i]
                x1, y1, x2, y2 = box
                blended = self._blend(mouth, index)
                sized = cv2.resize(blended, (x2 - x1, y2 - y1))
                sized = restore_detail(
                    sized, self._cache.frames[i], box, self._weight(blended, i, index)
                )
                out = composite(
                    self._cache.frames[i],
                    sized,
                    box,
                    self._cache.crop_boxes[i],
                    self._cache.masks[i],
                    out=self._buffer,
                )
                # Encode inside the loop: `out` is one reusable buffer, so the second
                # composite overwrites the first frame's pixels.
                ok, buf = cv2.imencode(
                    ".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                )
                if ok:
                    encoded.append(buf.tobytes())
            return encoded
        except Exception:
            logger.exception("face: lip-sync compositing failed, falling back to clips")
            self.failed = True
            return []
