"""Turn one driving clip into the cache `daemon/face_lipsync` reads at runtime.

By hand, once per clip, on an interpreter that has torch. **Never in CI** and never
from the daemon: nothing here is on a latency path, and that separation is the whole
reason the runtime needs no face detector and no torch.

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
  * `bbox_overlay.png` - frame 0 with the boxes drawn on it, so the box gets looked
    at rather than trusted.

`fps` is in `boxes.json` because `Renderer.render` takes it as an argument and the
point of this cache is that the runtime never opens the mp4 again.

## What the operator must have, and where

None of this is a daemon dependency and none of it may become one - the runtime is
all MLX (docs/superpowers/specs/2026-08-26-face-lipsync-design.md section 4). Put it
in a separate interpreter and run this tool with that one:

  * `torch`, `diffusers`, `transformers`, `face-alignment`, `opencv-python`, `numpy`,
    and `mlx` (only to write the safetensors).
  * a **MuseTalk checkout** - `--musetalk` - for `musetalk.models.vae`,
    `musetalk.utils.blending` and `musetalk.utils.face_parsing`. The latents come
    from MuseTalk's own `VAE.get_latents_for_unet`, which concatenates a half-masked
    encode with a full one into 8 channels; reimplementing that masking is how the
    conditioning silently stops matching the weights.
  * its **weights**, under `<--weights>/models/` as `download_weights.sh` lays them
    out: `sd-vae/`, `face-parse-bisent/` (both files), and for `--musetalk` to be
    useful at all a checkout at v1.5. `FaceParsing` and `VAE` resolve `./models/...`
    against the process cwd, which is why `--weights` is a directory to run from
    rather than a path to pass down.

`--weights` defaults to `--musetalk`, which is upstream's own layout. The spike keeps
them apart, hence the flag.

## Two substitutions and a margin, stated rather than buried

**Landmarks come from `face_alignment` (FAN, 2DFAN4), not from DWPose.** MuseTalk's
`get_landmark_and_bbox` reads DWPose's COCO-wholebody `keypoints[23:91]`, which follow
the same iBUG-68 scheme FAN emits, so the box formula below indexes the same anatomy.
That is an argument, not a measurement: **the box FAN produces has never been compared
against DWPose's**, because DWPose needs mmpose and mmcv, which have no usable macOS
arm64 wheel and do not build on this machine. The spec's own plan (section 4-1) is
macOS Vision for this, for the same "no torch" reason; that is also not what this does.
So the overlay PNG exists, and the divergence stays written down until somebody
measures it.

**`extra_margin = 10` is applied here**, `y2 = min(y2 + 10, frame_height)`, before the
crop. That is MuseTalk v1.5's own default and the spike's throwaway prep is where it
came from; it is fidelity to upstream, not a fix for anything - measured, it made no
difference to output quality.

**`bbox_shift` stays 0 and is not exposed.** Its whole documented range was swept and
moved lip openness by under 10%, so a knob for it would be a knob for nothing.

## Divergences from the spec's file list

The spec section 4 names `frames.raw` and `masks.npz`. `frames.npy` instead, because
`Cache.frames` is an `np.ndarray` and `np.load(mmap_mode="r")` gives one for free where
a headerless `.raw` needs the shape carried somewhere else.

`masks.npz` keeps its name and is a **pile of differently-shaped arrays, not a stack.**
MuseTalk's `get_crop_box` derives the blend region from each frame's own face box, so
the region breathes with the face: measured over `idle1.mp4`'s 193 frames, the crop box
ranges 572 to 608 px square. `Cache.masks` is annotated `np.ndarray`; whatever loads
this has to hand it a *sequence* of per-frame masks instead, because no single array
holds them. Flagged rather than worked around - that annotation is in a file this tool
is not allowed to edit.

## What one run cost, and what came out

`idle1.mp4`, 193 frames of 1080x1620 at 24fps, on an M-series Mac (MPS): **54.0s** -
landmarks 30.7s, VAE latents 11.0s, BiSeNet masks 12.3s. Written: `frames.npy` 1.01GB,
`masks.npz` 67MB, `latents.safetensors` 3.2MB, `boxes.json` 15KB.

The output was then driven through `daemon.face_lipsync`: `composite` on these boxes,
crop boxes and masks is **bit-identical** (max px difference 0) to MuseTalk's own
`get_image_blending` fed the same four arrays, and a crop box moved 12px changes the
composited frame by up to 66/255 - so that agreement is a measurement and not a
tautology.
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

MISSING = """\
the by-hand preprocessing stack is not importable here:

  {exc}

It is deliberately not a daemon dependency (the runtime is all MLX and must never
import torch), so a plain checkout does not have it. Install it into its own
interpreter and run this tool with that one:

  uv venv ~/.venvs/face-prepare --python 3.11
  VIRTUAL_ENV=~/.venvs/face-prepare uv pip install \\
      torch diffusers transformers face-alignment opencv-python numpy mlx
  ~/.venvs/face-prepare/bin/python -m evals.face_lipsync_prepare ...

If the missing name is `musetalk`, `--musetalk` is not pointing at a MuseTalk
checkout. If it is a weight file, `--weights` is not pointing at the directory that
holds `models/`.
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


def musetalk_bbox(landmarks: np.ndarray, height: int) -> tuple[int, int, int, int]:
    """`musetalk/utils/preprocessing.py:get_landmark_and_bbox`, plus v1.5's margin.

    The upper bound is a reflection, not a fraction: the distance from the lower nose
    bridge down to the chin, mirrored above it. Copied rather than reasoned about,
    because the weights were trained against whatever this produces.
    """
    lm = landmarks.astype(np.int32)
    half = lm[29]                                       # lower nose bridge
    y2 = int(lm[:, 1].max())
    upper_bond = max(0, int(half[1]) - (y2 - int(half[1])))
    return (
        int(lm[:, 0].min()),
        upper_bond,
        int(lm[:, 0].max()),
        min(y2 + EXTRA_MARGIN, height),
    )


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
    args = parser.parse_args()

    clip = args.clip.expanduser().resolve()
    out = args.out.expanduser().resolve()
    musetalk = args.musetalk.expanduser().resolve()
    weights = (args.weights or args.musetalk).expanduser().resolve()
    for path in (clip, musetalk, weights):
        if not path.exists():
            raise SystemExit(f"not found: {path}")
    out.mkdir(parents=True, exist_ok=True)

    # Both of these must happen before the first torch import. face_alignment calls
    # torch.compile, and inductor's CPU backend cannot build on a host with only
    # Command Line Tools installed - no C++ stdlib headers - which its own try/except
    # misses because the failure lands on the first forward, not on the compile() call.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
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
        import face_alignment
        import mlx.core as mx
        from musetalk.models.vae import VAE
        from musetalk.utils.blending import get_image_prepare_material
        from musetalk.utils.face_parsing import FaceParsing
    except ImportError as exc:
        raise SystemExit(MISSING.format(exc=exc)) from None

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    frames, fps = read_frames(clip)
    n, height, width = frames.shape[:3]
    print(f"clip {clip.name}: {n} frames at {width}x{height}, {fps:.3f}fps, device {device}")

    landmark_type = getattr(
        face_alignment.LandmarksType, "TWO_D", None
    ) or face_alignment.LandmarksType._2D
    fa = face_alignment.FaceAlignment(landmark_type, flip_input=False, device=str(device))
    start = time.perf_counter()
    boxes, first_landmarks = [], None
    for i in range(n):
        found = fa.get_landmarks_from_image(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        if not found:
            raise SystemExit(
                f"no face found in frame {i}. Every frame needs one - the cache is "
                "per-frame and there is no detector at runtime to fill a gap."
            )
        if i == 0:
            first_landmarks = found[0]
        boxes.append(musetalk_bbox(found[0], height))
    print(f"landmarks + boxes: {time.perf_counter() - start:.1f}s")

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
        latents[i] = vae.get_latents_for_unet(crop).cpu().numpy()[0].transpose(1, 2, 0)
    print(f"VAE latents: {time.perf_counter() - start:.1f}s")

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
    mx.save_safetensors(str(out / "latents.safetensors"), {"latents": mx.array(latents)})
    np.savez(out / "masks.npz", **{f"{i:06d}": mask for i, mask in enumerate(masks)})
    (out / "boxes.json").write_text(
        json.dumps(
            {
                "fps": fps,
                "size": [width, height],
                "boxes": [list(box) for box in boxes],
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
    for px, py in first_landmarks.astype(int):
        cv2.circle(overlay, (px, py), 3, (255, 0, 255), -1)
    cv2.imwrite(str(out / "bbox_overlay.png"), overlay)

    sides = sorted({box[2] - box[0] for box in crop_boxes})
    print(f"\nwrote {out}")
    print(f"  frames.npy          {frames.shape} {frames.dtype}")
    print(f"  latents.safetensors {latents.shape} {latents.dtype} NHWC")
    print(f"  masks.npz           {n} masks, crop box {sides[0]}-{sides[-1]}px square")
    print(f"  boxes.json          {n} boxes + {n} crop boxes, fps {fps:.3f}")
    print("  bbox_overlay.png    green = model input box, orange = blend region,")
    print("                      red = upper_boundary_ratio=0.5 cut (only below is blended)")
    print("\nLook at bbox_overlay.png. The FAN-for-DWPose substitution is unverified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
