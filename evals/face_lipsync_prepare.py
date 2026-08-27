"""Turn one driving clip into the cache `daemon/face_lipsync` reads at runtime.

By hand, once per clip. **Never in CI** and never from the daemon: nothing here is on
a latency path, and that separation is the whole reason the runtime needs no face
detector and no torch.

    python3 -m evals.face_lipsync_prepare /Users/me/Daemon/data/face/idle1.mp4 \
        --out data/face/lipsync/idle1 \
        --musetalk ~/spikes/musetalk-stage1/MuseTalk \
        --weights ~/spikes/musetalk-stage1

## What it writes

  * `frames.npy` - the clip, uint8 BGR `(n, h, w, 3)`, for `np.load(mmap_mode="r")`.
  * `latents.safetensors` - one key, `latents`, `(n, 32, 32, 8)` **NHWC** fp16. That
    is what `engine.load` does `mx.load(path)["latents"]` on.
  * `masks.npz` - the blurred BiSeNet blend mask per frame, each sized to its own
    crop box.
  * `boxes.json` - `boxes` (MuseTalk's face box), `crop_boxes` (the blend region),
    and the clip's `fps` and `size`.
  * `bbox_overlay.png` - frame 0 with the landmarks and both boxes drawn on it, so
    the box gets looked at rather than trusted.

`fps` is in `boxes.json` because `Renderer.render` takes it as an argument and the
point of this cache is that the runtime never opens the mp4 again.

## What the operator must have, and where

None of this is a daemon dependency and none of it may become one - the runtime is all
MLX (docs/superpowers/specs/2026-08-26-face-lipsync-design.md section 4). Put it in a
separate interpreter and run this tool with that one:

  * `pyobjc-framework-Vision` - macOS landmarks, no model download, no torch.
  * `torch`, `diffusers`, `transformers`, `opencv-python`, `numpy`, `safetensors`.
  * `scipy` - one `savgol_filter` call, for `--smooth`; see `smooth_boxes`.
  * a **MuseTalk checkout** - `--musetalk` - for `musetalk.models.vae`,
    `musetalk.utils.blending` and `musetalk.utils.face_parsing`.
  * its **weights**, under `<--weights>/models/` as `download_weights.sh` lays them
    out: `sd-vae/`, `face-parse-bisent/` (both files), and a checkout at v1.5.
    `FaceParsing` and `VAE` resolve `./models/...` against the process cwd, which is
    why `--weights` is a directory to run from rather than a path to pass down.

`--weights` defaults to `--musetalk`, which is upstream's own layout. The spike keeps
them apart, hence the flag.

**No `mlx` import, on purpose.** The latents are written with `safetensors.numpy`,
which `mx.load` reads back identically (verified: the engine loaded a file written this
way and reported `(193, 32, 32, 8) mlx.core.float16`). Section 7 of the spec forbids
running MLX and torch in one process - the spike's cache grew until drift hit +41.2%
and the dev machine stalled once - and importing mlx purely to serialise an array was
the one place this tool broke that rule for no gain.

## Landmarks: Apple Vision, and the 12px this substitution costs

**Corrected from the previous revision, which used `face_alignment` (FAN).** The brief
that produced it asked for FAN; that instruction was wrong. Section 4-1 of the spec
approved macOS **Vision** for exactly this, and the spike used FAN only for measurement
convenience. So this is Vision, and the mapping onto MuseTalk's box formula is measured
rather than asserted:

  * **`faceContour` has 17 points** where iBUG-68's jawline has 17, and its extremes
    are the same anatomy. Over `idle1`'s 193 frames, against FAN: `max x` differs
    **+2.0 +- 1.7px**, `max y` (the chin, which sets the box bottom) **+1.1 +- 1.4px**.
  * **`lm[29]` is `noseCrest[2]`.** Not a guess: FAN puts iBUG-29 at fraction 0.697 of
    the 27->30 bridge and Vision's four crest points sit at 0, 0.331, 0.671, 1.0, so
    index 2 is the same vertebra of the same bridge. Drawn and looked at - the two
    points land on top of each other on the nose.
  * **The two places it does not agree.** `min x`: Vision's contour stops **+11.9 +-
    1.9px** inside FAN's, so the box is ~10px narrower. `noseCrest[2]` sits **-12.1px**
    above FAN's `lm[29]`, and the formula doubles that (`upper_bond = 2*half_y - chin`),
    so the box top edge is **~24px higher** and the box **5.6% taller**.

That last one is a real cost and it is not corrected here. Measured over 7 clips and
175 frames the bias is stable (-11.6 to -13.3px, and FAN's `lm[29]` sits at fraction
0.809-0.842 of Vision's crest), so a constant would remove it - but all 7 clips are the
**same avatar**, so that constant would be fitted on one face and would go silently
wrong on another, which is the exact failure section 4-1 warns about. A stated 12px bias
beats a fitted constant nobody can re-derive.

What the bias buys and costs, on the assembled engine: `2 * 12px = 24px` of top edge is
`bbox_shift = -12`, inside the +-20 the spike swept when it found the whole documented
range moves lip openness by under 10%. Rendering the same 60 frames both ways, the
Vision box moves the mouth **1.48x** one frame of its own natural motion and reads
**122%** of the driving frame's lip detail against FAN's 135% - the taller box spends
6% more of its 256 pixels on forehead. Visibly the same mouth; measurably a softer one.

## The VAE the spec never named

Section 4 lists UNet / whisper / TAESD, but the reference latents are 8 channels - a
half-masked encode concatenated with a full one - and **producing them needs a VAE
encoder that is in no list and no open question.** Named here, with the measurement
that chose it:

**It is `sd-vae-ft-mse`'s encoder**, through MuseTalk's own `VAE.get_latents_for_unet`,
so the half-masking is not reimplemented. Correct by construction: it is what MuseTalk
trained the UNet against.

**TAESD's encoder was the tempting alternative and it is rejected.** It is already in
the weights file this repo downloads (67 encoder keys in the same
`diffusion_pytorch_model.safetensors`), shares the 4-channel latent space, and costs
1.9s against sd-vae's 34.6s. It also **degrades the mouth visibly.** Built both ways off
the same Vision boxes and rendered through the real engine on 60 frames:

| | lip saturation | lip contrast (gray sd) |
|---|---|---|
| driving frame | 90.0 | 37.4 |
| sd-vae encoder | **91.1** | **32.3** |
| TAESD encoder | 85.9 | 29.5 |

**-5.7% saturation and -8.6% contrast, in the lip region, every one of the 60 frames**
(delta -5.20 +- 0.64, worst -6.81). Looked at: the vermilion border goes soft and the
lips go pale. That is the same softness `render.restore_detail` exists to buy back, so
paying 8.6% of it for encoder speed that is not on any latency path is a bad trade.

Two numbers that make that reading safe. sd-vae's `latent_dist.sample()` is stochastic,
so the honest noise floor is sd-vae against itself on a second seed: **0.0001 mean abs
on the latents and 0.00 +- 0.01 on lip saturation** - the encoder is effectively
deterministic, and the -5.20 is not sampling noise. And the latent ranges say why TAESD
loses: sd-vae reaches `[-7.26, 5.89]` where TAESD is clamped into `[-3.11, 2.89]`, so
the UNet gets a compressed reference it was never trained on. Whole-latent cosine 0.980
looked survivable and was not, which is why this was decided on rendered mouths.

**What this does NOT settle: torch.** Section 4-1's stated goal is no torch in the
build, and this tool does not reach it - but the VAE is not what is standing in the way.
**The BiSeNet blend mask is.** `musetalk.utils.face_parsing.FaceParsing` is a torch
model plus a torchvision transform, and nothing in this repo replaces it. So porting
sd-vae's encoder to MLX (`mlx-examples/stable_diffusion` ships the `vae.py` this repo
vendored only the UNet half of) would remove `diffusers` and a 335MB download and leave
torch exactly where it is. **Whether preprocessing should stay on torch is the owner's
call, not this tool's** - what is settled here is that TAESD's encoder cannot buy the
answer, and that the next thing to cost is BiSeNet, not the VAE.

## A margin, and a knob that is not one

**`extra_margin = 10` is applied here**, `y2 = min(y2 + 10, frame_height)`, before the
crop. That is MuseTalk v1.5's own default; it is fidelity to upstream, not a fix -
measured, it made no difference to output quality. (Correcting the brief on a second
point: `~/spikes/musetalk-stage1/throwaway_prep.py` already applies it, line 119, so
this is not a divergence the spike had.)

**`bbox_shift` stays 0 and is not exposed.** Its whole documented range was swept and
moved lip openness by under 10%, so a knob for it would be a knob for nothing.

## Divergences from the spec's file list

The spec section 4 names `frames.raw` and `masks.npz`. `frames.npy` instead, because
`Cache.frames` is an `np.ndarray` and `np.load(mmap_mode="r")` gives one for free where
a headerless `.raw` needs the shape carried somewhere else.

`masks.npz` keeps its name and is a **pile of differently-shaped arrays, not a stack.**
MuseTalk's `get_crop_box` derives the blend region from each frame's own face box, so
the region breathes with the face: measured over `idle1.mp4`'s 193 frames, the crop box
ranges 624 to 632 px square. `Cache.masks` is annotated `np.ndarray`; whatever loads
this has to hand it a *sequence* of per-frame masks instead, because no single array
holds them. Flagged rather than worked around - that annotation is in a file this tool
is not allowed to edit.

## What one run cost, and what came out

`idle1.mp4`, 193 frames of 1080x1620 at 24fps, on an M-series Mac (MPS): **36.2s** of
model work - Vision landmarks 13.1s, sd-vae latents 11.5s, BiSeNet masks 11.6s - and
then a minute or so writing the frame store. Written: `frames.npy` 966MB, `masks.npz`
73MB, `latents.safetensors` 3.0MB, `boxes.json` 16KB.

Vision is also the faster landmarker, which was not the reason for the swap: 13.1s
against FAN's 29-32s on the same 193 frames, and no model download.

The output was then driven through `daemon.face_lipsync`: `mx.load` gives
`(193, 32, 32, 8)` fp16, the real engine on 2.2s windows through `PcmRing` returns
256x256 BGR mouths that track the audio, and `composite` on these boxes, crop boxes and
masks pastes with no seam and no offset - looked at, at the blend boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

EXTRA_MARGIN = 10
PARSING_MODE = "jaw"
CHEEK_WIDTH = 90
CROP = 256
"""MuseTalk v1.5 defaults, from `scripts/realtime_inference.py`'s argparse. Not knobs -
see the module docstring on `bbox_shift`."""

LOWER_NOSE_BRIDGE = 2
"""Index into Vision's `noseCrest` that is iBUG-68's `lm[29]`. Measured, not chosen -
see the module docstring."""

REGIONS = (
    "faceContour",
    "leftEye",
    "rightEye",
    "leftEyebrow",
    "rightEyebrow",
    "nose",
    "noseCrest",
    "medianLine",
    "outerLips",
    "innerLips",
    "leftPupil",
    "rightPupil",
)
"""Every landmark region Vision returns. All of them, because MuseTalk's box formula
takes its min and max over all 68 landmarks rather than over the jawline - on `idle1`
the two give the same three extremes to the pixel, and taking the union means that
agreement does not have to hold for the box to stay faithful to upstream."""

MISSING = """\
the by-hand preprocessing stack is not importable here:

  {exc}

It is deliberately not a daemon dependency (the runtime is all MLX and must never
import torch), so a plain checkout does not have it. Install it into its own
interpreter and run this tool with that one:

  uv venv ~/.venvs/face-prepare --python 3.11
  VIRTUAL_ENV=~/.venvs/face-prepare uv pip install \\
      torch diffusers transformers opencv-python numpy safetensors scipy \\
      pyobjc-framework-Vision
  ~/.venvs/face-prepare/bin/python -m evals.face_lipsync_prepare ...

If the missing name is `musetalk`, `--musetalk` is not pointing at a MuseTalk
checkout. If it is a weight file, `--weights` is not pointing at the directory that
holds `models/`. `Vision` is macOS-only and needs no download.
"""


def read_frames(path: Path) -> tuple[np.ndarray, float]:
    """The whole clip as one uint8 BGR array, plus its frame rate."""
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    clip = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        clip.append(frame)
    capture.release()
    if not clip:
        raise SystemExit(f"no frames decoded from {path} - is it a video this cv2 reads?")
    frames = np.stack(clip)
    del clip
    return frames, fps


def musetalk_bbox(
    points: np.ndarray, half_y: float, height: int
) -> tuple[int, int, int, int]:
    """`musetalk/utils/preprocessing.py:get_landmark_and_bbox`, plus v1.5's margin.

    The upper bound is a reflection, not a fraction: the distance from the lower nose
    bridge down to the chin, mirrored above it. Copied rather than reasoned about,
    because the weights were trained against whatever this produces - which is also
    why `half_y` arrives as an argument. Deciding *which* landmark is the lower nose
    bridge is Vision's problem, not this formula's.
    """
    y2 = int(points[:, 1].max())
    upper_bond = max(0, int(half_y) - (y2 - int(half_y)))
    return (
        int(points[:, 0].min()),
        upper_bond,
        int(points[:, 0].max()),
        min(y2 + EXTRA_MARGIN, height),
    )


SMOOTH_WINDOW = 9
"""Default box-smoothing window. Measured; see `smooth_boxes`."""


def smooth_boxes(
    boxes: list[tuple[int, int, int, int]], window: int
) -> list[tuple[int, int, int, int]]:
    """Savitzky-Golay the box series, wrapping at the loop point. 0 or 1 disables it.

    Vision's per-frame boxes carry detector noise on top of the real head motion, and
    the second difference tells them apart: over idle2's 193 frames the per-frame edge
    shift is **1.08px** but the *jerk* is **1.42px**. Real motion has a smooth
    trajectory, so its jerk is small - a jerk larger than the shift is noise. Every
    frame therefore resamples its reference crop from a slightly different rectangle,
    **0.89px/frame at the model's 256**, and a different reference latent generates a
    different mouth. That is a vibration source, and the owner sees it as residual
    tremor after the jitter causes inside the render were exhausted.

    It costs nothing to remove, because the avatar is a **fixed clip**: the whole series
    is known here, so the filter is centred (no runtime lag, no runtime work at all) and
    it wraps - idle2 loops seamlessly, its wrap-pair frame difference 1.33 against 0.82
    for a typical adjacent pair.

    Sweep on idle2, jerk against how far the smoothed box departs from the detected one:

        w=5   shift 0.84  jerk 0.57  departure max 2.1px
        w=9   shift 0.72  jerk 0.28  departure max 3.7px
        w=15  shift 0.67  jerk 0.17  departure max 5.0px
        w=31  shift 0.60  jerk 0.09  departure max 6.1px

    The jerk falls 5x by w=9 while the shift only falls a third - noise leaving, motion
    staying. Past that the departure grows faster than the jerk gain. w=9's 3.7px
    maximum is inside Vision's own per-frame scatter (the box formula's landmarks
    disagree with FAN's by +-1.4 to +-1.7px), so the box is no less faithful to the face
    than the raw one; it is only less noisy.

    Everything downstream inherits this - the reference latents, the BiSeNet masks and
    the crop boxes are all derived from `boxes` - which is why this is the one place to
    do it and why it cannot drift out of step with the paste-back.
    """
    if window < 3:
        return boxes
    # Inside the function, like every other heavy import here: CI imports this module
    # for its reachability check and must not need the operator's interpreter.
    from scipy.signal import savgol_filter

    series = np.asarray(boxes, dtype=float)
    if window > len(series):
        raise SystemExit(f"--smooth {window} exceeds the clip's {len(series)} frames")
    smoothed = savgol_filter(series, window, 2, axis=0, mode="wrap")
    return [tuple(int(round(v)) for v in box) for box in smoothed]


def check_geometry(
    boxes: list[tuple[int, int, int, int]],
    crop_boxes: list[tuple[int, int, int, int]],
    masks: list[np.ndarray],
    size: tuple[int, int],
) -> None:
    """Refuse to write a cache `composite` would paste crooked.

    MuseTalk crops through PIL, which zero-pads outside the image; `composite` slices
    numpy, which clips instead. So a crop box hanging off the frame edge is not a
    cosmetic difference between the two - it makes the slice smaller than the mask and
    shifts the paste, which is exactly the visible seam this whole file exists to get
    right the first time.
    """
    width, height = size
    for i, (box, crop, mask) in enumerate(zip(boxes, crop_boxes, masks, strict=True)):
        x1, y1, x2, y2 = box
        xs, ys, xe, ye = crop
        if xs < 0 or ys < 0 or xe > width or ye > height:
            raise SystemExit(
                f"frame {i}: crop box {crop} leaves the {width}x{height} frame. The "
                "blend region is the face box expanded 1.5x, so the face sits too "
                "close to an edge in this clip - reframe it and re-shoot rather than "
                "clamping, which would silently mis-align the paste."
            )
        if not (xs <= x1 < x2 <= xe and ys <= y1 < y2 <= ye):
            raise SystemExit(f"frame {i}: face box {box} is not inside crop box {crop}")
        if mask.shape != (ye - ys, xe - xs):
            raise SystemExit(
                f"frame {i}: mask is {mask.shape}, crop box {crop} wants "
                f"{(ye - ys, xe - xs)}"
            )


def landmarks(frame: np.ndarray, size: tuple[int, int]) -> dict[str, np.ndarray] | None:
    """Vision's landmark regions for the biggest face in `frame`, in image pixels.

    `None` if Vision finds no face. Three things in here are not stylistic:

    **PNG bytes rather than a temp file.** `VNImageRequestHandler` takes NSData
    directly, so nothing touches the disk for 193 frames.

    **`normalizedPoints()` returns an `objc.varlist`, which has no length.** It is a
    bare C pointer with Python indexing bolted on, so reading past `pointCount()`
    walks off the buffer - it does not raise, it takes the interpreter down with
    SIGBUS. `as_tuple(count)` is the bounded read. (`pointsInImageOfSize_` is used
    below instead and agrees with the bounding-box arithmetic to 0.0px, measured; it
    is the same varlist and the same rule applies.)

    **y is flipped.** Vision's origin is bottom-left and every other coordinate in
    this pipeline is top-left, which is a bug that would look like a face detected
    upside down rather than like an error.
    """
    import Quartz
    import Vision
    from Foundation import NSData

    width, height = size
    ok, png = cv2.imencode(".png", frame)
    if not ok:
        raise SystemExit("cv2 could not encode a frame as PNG")
    data = NSData.dataWithBytes_length_(png.tobytes(), png.size)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
    request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
    request.setRevision_(Vision.VNDetectFaceLandmarksRequestRevision3)
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise SystemExit(f"Vision request failed: {error}")
    faces = request.results()
    if not faces:
        return None
    face = max(faces, key=lambda f: f.boundingBox().size.width * f.boundingBox().size.height)
    found = face.landmarks()
    regions: dict[str, np.ndarray] = {}
    for name in REGIONS:
        region = getattr(found, name)()
        if region is None:
            continue
        count = region.pointCount()
        points = region.pointsInImageOfSize_(Quartz.CGSizeMake(width, height))
        regions[name] = np.array(
            [[p.x, height - p.y] for p in points.as_tuple(count)], dtype=np.float64
        )
    return regions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare one driving clip's lip-sync cache. By hand; see the module docstring.",
    )
    parser.add_argument("clip", type=Path, help="the driving mp4")
    parser.add_argument("--out", type=Path, required=True, help="cache directory to write")
    parser.add_argument(
        "--musetalk", type=Path, required=True, help="a MuseTalk v1.5 checkout"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="the directory holding models/ (default: --musetalk, upstream's layout)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=SMOOTH_WINDOW,
        help=f"Savitzky-Golay window over the box series (default {SMOOTH_WINDOW}, "
        "0 disables); see smooth_boxes for the measurement",
    )
    args = parser.parse_args()

    clip = args.clip.expanduser().resolve()
    out = args.out.expanduser().resolve()
    musetalk = args.musetalk.expanduser().resolve()
    weights = (args.weights or args.musetalk).expanduser().resolve()
    for path in (clip, musetalk, weights):
        if not path.exists():
            raise SystemExit(f"not found: {path}")
    out.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    sys.path.insert(0, str(musetalk))
    # `FaceParsing` and `VAE` resolve './models/...' against the process cwd, so this
    # is a chdir and not an argument. Every path above is already absolute.
    os.chdir(weights)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(MISSING.format(exc=exc)) from None
    # torch>=2.6 defaults torch.load to weights_only=True, which rejects both the
    # legacy-.tar resnet18 that BiSeNet loads and MuseTalk's own checkpoints. These
    # are files the operator fetched from the official publishers by hand and this
    # tool never runs unattended, so restore the old default rather than rewrite them.
    _torch_load = torch.load
    torch.load = lambda *a, **kw: _torch_load(*a, **{"weights_only": False, **kw})
    try:
        import Vision  # noqa: F401  - imported here so a missing pyobjc says so once
        from musetalk.models.vae import VAE
        from musetalk.utils.blending import get_image_prepare_material
        from musetalk.utils.face_parsing import FaceParsing
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise SystemExit(MISSING.format(exc=exc)) from None

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    frames, fps = read_frames(clip)
    n, height, width = frames.shape[:3]
    print(f"clip {clip.name}: {n} frames at {width}x{height}, {fps:.3f}fps, device {device}")

    start = time.perf_counter()
    boxes, first_regions = [], None
    for i in range(n):
        regions = landmarks(frames[i], (width, height))
        if regions is None:
            raise SystemExit(
                f"Vision found no face in frame {i}. Every frame needs one - the cache "
                "is per-frame and there is no detector at runtime to fill a gap."
            )
        if i == 0:
            first_regions = regions
        every = np.concatenate(list(regions.values()))
        half_y = regions["noseCrest"][LOWER_NOSE_BRIDGE][1]
        boxes.append(musetalk_bbox(every, half_y, height))
    print(f"Vision landmarks + boxes: {time.perf_counter() - start:.1f}s")

    raw_jerk = float(np.abs(np.diff(np.asarray(boxes, dtype=float), axis=0, n=2)).mean())
    boxes = smooth_boxes(boxes, args.smooth)
    kept_jerk = float(np.abs(np.diff(np.asarray(boxes, dtype=float), axis=0, n=2)).mean())
    print(
        f"box smoothing: window {args.smooth}, jerk {raw_jerk:.2f} -> {kept_jerk:.2f}px"
        if args.smooth >= 3
        else f"box smoothing: off, jerk {raw_jerk:.2f}px"
    )

    vae = VAE(model_path="models/sd-vae")
    vae.vae.to(device)
    vae.device = device
    start = time.perf_counter()
    latents = np.empty((n, 32, 32, 8), dtype=np.float16)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = cv2.resize(
            frames[i][y1:y2, x1:x2], (CROP, CROP), interpolation=cv2.INTER_LANCZOS4
        )
        # (1, 8, 32, 32) NCHW from MuseTalk; the MLX UNet reads NHWC.
        latents[i] = vae.get_latents_for_unet(crop).detach().cpu().numpy()[0].transpose(1, 2, 0)
    print(f"sd-vae latents: {time.perf_counter() - start:.1f}s")

    parsing = FaceParsing(left_cheek_width=CHEEK_WIDTH, right_cheek_width=CHEEK_WIDTH)
    start = time.perf_counter()
    masks, crop_boxes = [], []
    for i, box in enumerate(boxes):
        # upper_boundary_ratio=0.5 and expand=1.5 are this function's own defaults,
        # which is what MuseTalk's realtime path uses; passed by name anyway because
        # they are the two numbers that decide where the blend starts.
        mask, crop_box = get_image_prepare_material(
            frames[i], box, upper_boundary_ratio=0.5, expand=1.5, fp=parsing, mode=PARSING_MODE
        )
        masks.append(mask)
        crop_boxes.append(tuple(int(v) for v in crop_box))
    print(f"BiSeNet masks: {time.perf_counter() - start:.1f}s")

    check_geometry(boxes, crop_boxes, masks, (width, height))

    np.save(out / "frames.npy", frames)
    save_file({"latents": latents}, str(out / "latents.safetensors"))
    np.savez(out / "masks.npz", **{f"{i:06d}": mask for i, mask in enumerate(masks)})
    (out / "boxes.json").write_text(
        json.dumps(
            {
                "fps": fps,
                "size": [width, height],
                "boxes": [list(box) for box in boxes],
                "smooth": args.smooth,
                "crop_boxes": [list(box) for box in crop_boxes],
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    overlay = frames[0].copy()
    x1, y1, x2, y2 = boxes[0]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    xs, ys, xe, ye = crop_boxes[0]
    cv2.rectangle(overlay, (xs, ys), (xe, ye), (0, 200, 255), 2)
    cv2.line(overlay, (xs, (ys + ye) // 2), (xe, (ys + ye) // 2), (0, 0, 255), 2)
    assert first_regions is not None
    for points in first_regions.values():
        for px, py in points.astype(int):
            cv2.circle(overlay, (px, py), 3, (255, 0, 255), -1)
    px, py = first_regions["noseCrest"][LOWER_NOSE_BRIDGE].astype(int)
    cv2.circle(overlay, (px, py), 9, (255, 255, 0), -1)
    cv2.imwrite(str(out / "bbox_overlay.png"), overlay)

    sides = sorted({box[2] - box[0] for box in crop_boxes})
    print(f"\nwrote {out}")
    print(f"  frames.npy          {frames.shape} {frames.dtype}")
    print(f"  latents.safetensors {latents.shape} {latents.dtype} NHWC")
    print(f"  masks.npz           {n} masks, crop box {sides[0]}-{sides[-1]}px square")
    print(f"  boxes.json          {n} boxes + {n} crop boxes, fps {fps:.3f}")
    print("  bbox_overlay.png    green = model input box, orange = blend region,")
    print("                      red = upper_boundary_ratio=0.5 cut (only below is blended),")
    print("                      cyan = the noseCrest point standing in for iBUG-68 lm[29]")
    print("\nLook at bbox_overlay.png. The cyan point sits ~12px above where FAN puts")
    print("lm[29]; that bias is measured and documented, not corrected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
