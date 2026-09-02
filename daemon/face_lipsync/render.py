"""Turn one audio window plus one driving frame into one JPEG."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from daemon.face_lipsync import Cache, LipsyncEngine, composite
from daemon.face_lipsync.audio import CONTEXT_MS, MS_PER_INDEX, WINDOW, latest_audio_ms
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

MOTION_BLEND = 0.75
"""Weight of the new mouth against the one before it, applied before `restore_detail`.

**Read the mechanism before retuning this.** `_blend` stores the BLENDED result, not the
new mouth, so this is an exponential moving average and old frames leave a geometric
tail. What survives from how long ago, at 24fps:

    a       now    42ms   83ms   125ms   167ms
    0.48   48.0%  25.0%  13.0%   6.7%    3.5%
    0.55   55.0%  24.8%  11.1%   5.0%    2.3%
    0.75   75.0%  18.8%   4.7%   1.2%    0.3%

That tail IS the afterimage, and the reason this knob exists is the opposite complaint:
MuseTalk generates each frame's mouth from its own audio window alone and has none of
the inertia a face has, so every frame is free to jump. Lower is more inertia and a
longer tail; higher is crisper per-frame shapes and more jump. **The owner has now
reported a defect at both ends**, which is what a blunt instrument looks like.

**Three judgements, and the third one is the value.** The order matters more than any
of them alone:

1. 0.55, ranked against 0.7 and no blend at all - "natural enough". The sweep only ever
   went up from there, which is the first thing that was wrong with it.
2. 0.48, ranked against 0.55, 0.40 and 0.25 after "입이 너무 빨리 움직인다" - "이제
   입모양이 확실히 보여".
3. 0.75, ranked against 0.48 after a live conversation produced "잔상처럼 보이는게 좀
   심한데" - "0.75가 나은듯".

**Judgements 1 and 2 were made on a 7-second clip at 800px, where the mouth region is
about 150px across. Judgement 3 was made on the mouth alone, cropped from the native
1620px render, upscaled 3x nearest and played at half speed.** 0.75 had been rendered
and passed over twice before that; at a magnification where the tail is visible it won.
Do not treat 1 and 2 as evidence against this value - treat them as evidence that the
difference was below the resolution they were judged at.

Measured on the native-resolution renders judgement 3 used, over `mask > 0` - the pixels
that actually reach the screen:

    0.48   inter-frame motion 3.000   sharpness 67.4
    0.75   inter-frame motion 3.433   sharpness 74.5

Both rise together, as they did in the earlier sweep at the other scale. Do not tune this
against the sharpness number alone: judgement 2 chose the value that measured *blurrier*
and read as more legible.

Applied before `restore_detail`, and that ordering is why the same alpha got the opposite
verdict when smoothing was the last thing to touch the pixels - see `Renderer._blend`. It
is also the most likely reason the four-arm afterimage A/B (this knob, `DETAIL_CUTOFF`,
and `restore_detail` removed entirely) read as "다 거기서 거기" at 800px: the texture
goes back on top and covers much of what the mix changed."""

RELEASE_FRAMES = 10
"""How many frames the mouth takes to hand itself back to the driving clip.

417ms at 24fps. Ranked by the owner against 0 and 5 on a 2.4x side-by-side of the
falling edge; 10 was "제일 자연스럽네". See `Renderer.release`."""

MOTION_FILTER = "euro"
"""Which filter `_blend` runs the mouth through: `"euro"` or `"ema"`.

Two arms on purpose, for the owner's side-by-side. `"ema"` is `MOTION_BLEND` exactly as
judged three times above. `"euro"` is the One Euro filter (Casiez, Roussel, Vogel 2012)
over the same state: an EMA whose alpha is not fixed but rises with the speed of the
signal, per pixel. The owner has reported a defect at BOTH ends of `MOTION_BLEND` - 0.48
ghosts, 0.75 jumps - and a fixed alpha cannot satisfy both because a still mouth wants
inertia and a moving one wants none. The adaptive one is built to give each what it
asks for. The loser is deleted after the judgement, the way the median and FIR arms were.
"""

EURO_MIN_CUTOFF = 3.0
"""Hz. The cutoff a still mouth is filtered at - its inertia at rest.

At 24fps this is an alpha of 0.44, a little more inertia than the 0.48 the owner chose
when the complaint was "너무 빨리 움직인다". A jittering mouth's speed keeps changing sign,
so the speed estimate averages toward zero and the cutoff sits here."""

EURO_BETA = 0.02
"""Cutoff gained per unit of speed, in Hz per (intensity unit / second).

A lip-edge pixel that moves 100 units between frames is doing 2400/s. Filtered through
`EURO_D_CUTOFF` on its first frame that is ~500/s, +10Hz - an alpha of 0.77, close to
the 0.75 the owner picked when the complaint was ghosting. Sustained over two or three
frames it approaches 2400/s and an alpha above 0.9: a mouth that is actually moving is
barely filtered at all. A one-frame blip gets the 0.75 treatment; real motion gets less."""

EURO_D_CUTOFF = 1.0
"""Hz. How fast the speed estimate itself may change.

The paper's default. Lower makes the filter slower to notice motion has started (and
slower to believe it has stopped); higher lets single-frame noise read as motion."""

QUIET_DBFS = -45.0
"""Below this the model's window is a pause rather than speech.

dBFS of the 200ms tail the model actually reads. Spoken TTS sits around -13 to -25;
the gaps between its sentences are digital zero (-180 by this arithmetic). Generous
room on both sides, and hysteresis comes from `QUIET_FRAMES` rather than from a second
threshold."""

QUIET_FRAMES = 4
"""How many quiet windows in a row before the mouth starts to close - 167ms at 24fps.

A breath between two words must not start closing anything; a sentence boundary
should. TTS gaps between sentences run 400-800ms, and this plus `CLOSE_FRAMES` has to
finish inside the short ones or the closure never lands."""

CLOSE_FRAMES = 6
"""How many frames the paste ramps out over once a pause is confirmed - 250ms.

Shorter than `RELEASE_FRAMES`, which is the end of an utterance and can afford 417ms:
a pause has a sentence coming after it. The mechanism is the one `release` uses - the
paste's `strength` falls and the clip's own sealed mouth shows through - because this
engine cannot render a closed mouth (all 88 conditioning windows measured, none do),
so the only closed mouth available is the artist's."""

OPEN_FRAMES = 2
"""How many frames the paste takes to come back when speech resumes - 83ms.

Closing is slow because a closing mouth should be seen to close. Opening cannot be:
the first voiced window after a pause is the start of a sentence, and playing it on
the clip's sealed mouth is the defect this exists to remove."""


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

    energy: list[float]
    """dBFS of the 200ms the model read for each mouth - what `Renderer._close` decides
    from. Travels with the pair for the same reason `indices` does."""


def _dbfs(samples: np.ndarray) -> float:
    """Level of `samples` (float32, -1..1) in dBFS; digital zero is -180."""
    if samples.size == 0:
        return -180.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * math.log10(rms + 1e-9)


def _euro_alpha(cutoff: float | np.ndarray, rate: float) -> float | np.ndarray:
    """One Euro's smoothing factor for a `cutoff` in Hz at `rate` samples per second."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * rate)


def encode_still(frame: np.ndarray) -> bytes:
    """One driving-clip frame as a JPEG, for the tick that is publishing it.

    The page shows these while nothing is being said, in place of playing the clip in a
    `<video>`, and that is a colour fix rather than a convenience. Measured in Chrome on
    this machine: the browser's decode of the untagged mp4 comes out R +3.0, G +2.1,
    B +1.2 against the same frames arriving as JPEG - which the browser decodes to our
    bytes within 0.2. So the two players disagree, the JPEG is the faithful one, and the
    whole picture used to shift darker and off-hue the moment speech started. The owner
    saw it as "어두워지면서 채도가 좀 올라가는" across the entire frame, background
    included.

    It is not a range problem (that would move all three channels by the same +8.7) and
    not a BT.601/709 matrix choice (neither matches). It is the browser colour-managing
    an untagged video against a P3 display, and there is no way to tell it not to without
    re-encoding the clips - which changes their pixels (mean luma 62.94 -> 59.82) and
    would still be one browser's guess. Removing the second decoder is the only fix that
    does not depend on guessing right.

    **Per frame, and this replaced a whole-clip strip encoded on first use.** That was
    right for one driving clip and turned over at ten. Measured: 9.10ms median here for
    a 1080x1620 frame at quality 85 (p95 10.99), which is 22% of an idle tick where the
    model does not run at all - against a **1051ms** burst for one clip's strip, once per
    clip, ten times a session. The owner felt it as "첫 발화 시작할 때 아주 잠깐 멈추는
    느낌", and it measured at the socket as a **539.8ms** gap where the warm steady state
    is 41.7ms median and 90ms worst.

    The burst is what costs, not the disk. A separate executor did not help and the page
    cache was already warm when it was measured: the strip is a 151-iteration Python loop
    and each iteration copies 5.25MB out of the memory-mapped store with the GIL held, so
    the event loop cannot run through it. One frame per tick holds the GIL for one such
    copy and releases it for the encode.

    The old docstring justified the strip against "2.45ms of CPU per frame forever if
    they were encoded live". That figure was wrong: 2.45ms is this file's own measured
    cost of *sending* a whole frame rather than a crop (see the transport note above),
    not of encoding one. Encoding is 9.10ms, and it is still the cheaper side.

    The speaking half has always encoded per frame - `Renderer.encode` is 8.43ms a frame
    including two of these reads and both JPEGs - so this is idle doing what speech
    already did, not a new risk taken on for it.
    """
    ok, payload = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        raise RuntimeError("cv2 refused to encode a driving frame")
    return payload.tobytes()


@dataclass(frozen=True, slots=True)
class Driver:
    """The clip being rendered onto: its name, its prepared frames, and where its
    playhead is. One object because these three are never independently correct - a
    cache paired with another clip's clock composites the mouth onto the wrong pose,
    and a name that has moved on selects another clip's reference latents.
    """

    name: str
    """Which prepared clip this is. Travels to `LipsyncEngine.mouths` as `clip`, and
    is what `daemon/app.py` tells the page it is showing."""

    cache: Cache
    """That clip's own prepared frames, boxes and masks."""

    clip: ClipClock
    """Where the page's own playhead is. Required rather than defaulted, because the
    default that would read naturally - "the turn starts at frame 0" - is exactly the
    assumption this stopped being able to make."""


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
        driver: Driver,
        ring: PcmRing,
    ) -> None:
        self._engine = engine
        self._driver = driver
        """The clip being rendered onto, and the only thing a clip change moves - see
        `switch`. One attribute rather than three, so a half-applied change (new
        frames, old name) has nowhere to exist."""
        self._ring = ring
        self._buffer = np.empty_like(driver.cache.frames[0])
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
        self._speed: np.ndarray | None = None
        """One Euro's filtered speed estimate, per pixel. Reset with `_previous`."""
        self._fps = 24.0
        """The frame rate the last `step` was asked for - the One Euro filter's clock."""
        self._quiet = 0
        """Consecutive quiet windows so far - see `_close`."""
        self._closure = 1.0
        """How much of the paste is showing, 0..1. Falls through a pause, comes back
        when speech does, and is where `release` starts its ramp from."""
        self.failed = False
        """Latched on the first failure in either half. The caller drops back to v1
        clips and logs once; retrying per frame would fill the log at 24Hz."""

    def switch(self, driver: Driver) -> None:
        """Render onto a different clip from here on.

        No crossfade, and that is a measurement rather than an omission: the caller
        only ever switches at a clip's own end, and end -> the next clip's frame 0
        measures 1.41 median downscaled whole-frame mean absolute difference against
        the 1.14 of a clip's own loop point - the join the face has always made every
        few seconds and that the owner has never remarked on. Over the ten prepared
        clips, 3 of the 90 ordered pairs exceed that baseline's own worst case (2.14),
        by 0.01-0.04. There is nothing for a fade to hide. (A cut in the middle of a
        clip is a different question and up to ten times worse - 7.98 median - which
        is why the caller only ever calls this at a clip's end.)

        Everything reset here is per-clip continuity, and each was a measured defect
        when it survived a turn boundary: `_previous` mixes the last mouth into the
        next (`_blend`), `_smoothed` averages the injection weight (`_weight`). Their
        two markers go with them, or the first frame of the new clip counts as
        continuous with the last frame of the old one and blends against a mouth this
        head never made.
        """
        if driver.cache.frames[0].shape != self._driver.cache.frames[0].shape:
            raise ValueError(
                f"geometry changed: {driver.name} is {driver.cache.frames[0].shape} "
                f"against {self._driver.cache.frames[0].shape}"
            )
        self._driver = driver
        self._previous = None
        self._smoothed = None
        self._weight_at = -1
        self._continues_at = -1
        self._speed = None
        self._quiet = 0
        self._closure = 1.0

    def step(self, *, frame_index: int, origin: float, fps: float) -> Step | None:
        """`BATCH` mouths from `frame_index` on, or `None`. Never raises.

        `None` means a latched failure and nothing else. There is no "not yet" here:
        whether the audio has arrived is `FrameClock`'s question, asked before this.
        """
        if self.failed:
            return None
        try:
            n = len(self._driver.cache.boxes)
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
            began = self._driver.clip.nearest(origin)
            shown = [began + index + DISPLAY_LEAD for index in audio]
            # The level of the 200ms the model reads, which is the TAIL of each window
            # (`PcmRing.window`): the context before it is whisper's, not the frame's.
            tail = int(self._ring.sample_rate * WINDOW * MS_PER_INDEX / 1000.0)
            self._fps = fps
            return Step(
                indices=shown,
                mouths=self._engine.mouths(
                    windows,
                    [index % n for index in shown],
                    clip=self._driver.name,
                ),
                energy=[_dbfs(window[-tail:]) for window in windows],
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
            mouth, self._driver.cache.frames[clip_index], self._driver.cache.boxes[clip_index]
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
            if MOTION_FILTER == "euro":
                current = self._euro(current)
            else:
                current = MOTION_BLEND * current + (1.0 - MOTION_BLEND) * self._previous
        else:
            # A turn boundary: no momentum and no pause carried across it. The paste
            # opens in full on the first frame, as it always has.
            self._speed = None
            self._quiet = 0
            self._closure = 1.0
        self._previous = current
        self._continues_at = index + 1
        return current.astype(np.uint8)

    def _euro(self, current: np.ndarray) -> np.ndarray:
        """One Euro over `current` against `_previous`, per pixel - see `MOTION_FILTER`.

        `_previous` is the last filtered output, exactly as the EMA leaves it, so the
        two arms share their state and swap cleanly. `_speed` is the paper's dx-hat: the
        raw per-pixel speed, itself low-passed at `EURO_D_CUTOFF` so a single noisy
        frame does not read as motion. The cutoff, and so the alpha, is then per pixel:
        the lips get one alpha and the cheek beside them another.
        """
        rate = self._fps
        speed = (current - self._previous) * rate
        if self._speed is None:
            self._speed = np.zeros_like(speed)
        a_d = _euro_alpha(EURO_D_CUTOFF, rate)
        self._speed = a_d * speed + (1.0 - a_d) * self._speed
        alpha = _euro_alpha(EURO_MIN_CUTOFF + EURO_BETA * np.abs(self._speed), rate)
        return alpha * current + (1.0 - alpha) * self._previous

    def _close(self, dbfs: float) -> float:
        """This frame's paste `strength`, after hearing how loud its window was.

        `QUIET_FRAMES` quiet windows in a row start the paste ramping out over
        `CLOSE_FRAMES`; the first loud one brings it back over `OPEN_FRAMES`. The
        asymmetry is the point (see both constants), and the count is the hysteresis:
        one quiet window between two words moves nothing.
        """
        if dbfs < QUIET_DBFS:
            self._quiet += 1
            if self._quiet >= QUIET_FRAMES:
                self._closure = max(0.0, self._closure - 1.0 / CLOSE_FRAMES)
        else:
            self._quiet = 0
            self._closure = min(1.0, self._closure + 1.0 / OPEN_FRAMES)
        return self._closure

    def release(
        self, *, index: int, step: int, count: int = RELEASE_FRAMES
    ) -> bytes | None:
        """One frame of an utterance's tail: the last mouth dissolving into the artist's.

        Speech stops and the frame after it is the driving clip untouched, so a
        generated mouth is replaced by a real one between two frames. The owner saw it
        as the mouth "갑자기 확 닫히는" snap, and it is structural rather than a timing
        fault: `evals/face_lipsync_idle_spike.py` measured all 88 conditioning windows,
        digital zero included, and **not one renders this avatar's resting mouth
        closed** - every one leaves the lips parted with a sliver of teeth, where the
        clip's own mouth is cleanly sealed. So the last generated frame is always at
        least slightly open and the next one is shut, and no better audio alignment
        removes that step.

        Ramping `composite`'s `strength` to zero removes it, and the dissolve *is* the
        closing motion, because a closed mouth is what shows through underneath.
        Measured on the mouth region, mean step from one frame to the next: the
        handover was **5.97px against a 2.46px median during speech**, and at count 10
        it is **2.26px** - no longer distinguishable from an ordinary speaking frame.
        (The 5-6px steps three to six frames later are the clip's own motion; they are
        just as large with no ramp at all, which is how they were ruled out.)

        **No model step pays for it.** The alternative - keep the engine running on the
        trailing silence and fade that - costs `count` UNet steps to animate a mouth
        that the same 88-window measurement says will sit open with its teeth showing,
        so it buys motion that is wrong rather than motion that is late.

        **One frame per call, on purpose.** Rendering all `count` at the falling edge
        measured ~84ms, which is two publish ticks holding one frame at exactly the
        moment this exists to smooth. This costs one composite, the same as a speaking
        frame's `encode`.

        `index` is the driving frame to composite onto - the caller passes the page's
        own playhead each tick rather than counting up from a captured start, so a late
        tick still lands on the frame the page is showing. `step` is 1..`count` and
        drives the strength alone.

        The mouth is `_previous`, held: the last thing `_blend` produced. It is stale by
        up to `count` frames of head motion, which is why the ramp is short and why
        staleness costs less every frame - the alpha it is multiplied by is already on
        its way to nothing. The injection weight is `_smoothed`, held for the same
        reason `_weight` averages it in the first place: recomputing it per frame is
        itself a vibration source.

        Returns a whole-frame JPEG like `encode`, and shares `_buffer` with it, so the
        same rule applies - one thread, never concurrent with `encode`. `daemon/app.py`
        submits both to the same single-worker executor, which is what serialises them.
        """
        if self.failed or self._previous is None or self._smoothed is None:
            return None
        if step < 1 or step > count:
            return None
        try:
            i = index % len(self._driver.cache.boxes)
            box = self._driver.cache.boxes[i]
            x1, y1, x2, y2 = box
            sized = restore_detail(
                cv2.resize(np.clip(self._previous, 0, 255).astype(np.uint8), (x2 - x1, y2 - y1)),
                self._driver.cache.frames[i],
                box,
                self._smoothed,
            )
            out = composite(
                self._driver.cache.frames[i],
                sized,
                box,
                self._driver.cache.crop_boxes[i],
                self._driver.cache.masks[i],
                out=self._buffer,
                # Ends one step short of zero, because the frame after the last of
                # these is the clip itself - which is strength 0 already. Starts from
                # wherever `_close` left the paste: a pause may already have taken it
                # to 0, and ramping from 1 would paint the mouth back for a tenth of a
                # second before closing it again.
                strength=self._closure * (1.0 - step / (count + 1)),
            )
            ok, buf = cv2.imencode(
                ".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if step == count:
                # The tail is done: drop the held mouth so a new turn cannot blend
                # against it and so a repeated call cannot render a stale frame.
                self._previous = None
                self._smoothed = None
                self._speed = None
                self._quiet = 0
                self._closure = 1.0
            return buf.tobytes() if ok else None
        except Exception:
            logger.exception("face: lip-sync release failed, falling back to clips")
            self.failed = True
            return None

    def encode(self, step: Step) -> list[bytes]:
        """One whole-frame JPEG per mouth, in the pair's own order. Never raises."""
        if self.failed:
            return []
        try:
            encoded: list[bytes] = []
            n = len(self._driver.cache.boxes)
            for index, mouth, loud in zip(step.indices, step.mouths, step.energy, strict=True):
                i = index % n
                box = self._driver.cache.boxes[i]
                x1, y1, x2, y2 = box
                blended = self._blend(mouth, index)
                # After `_blend`: a turn boundary resets the closure there, and this
                # frame's own window then has the first say.
                strength = self._close(loud)
                sized = cv2.resize(blended, (x2 - x1, y2 - y1))
                sized = restore_detail(
                    sized, self._driver.cache.frames[i], box, self._weight(blended, i, index)
                )
                out = composite(
                    self._driver.cache.frames[i],
                    sized,
                    box,
                    self._driver.cache.crop_boxes[i],
                    self._driver.cache.masks[i],
                    out=self._buffer,
                    strength=strength,
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
