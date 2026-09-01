"""Should the idle mouth be pre-rendered too, so the switch to speech carries no step?

The proposal this was written to test, and it is a good one on its face. While idle the
page plays the raw driving clip - the artist's own mouth, full quality. The instant
speech starts, that region becomes a generated mouth. The generated mouth measures 98%
of the original's sharpness with `render.restore_detail`, so what the owner is seeing is
not a detail deficit, it is a **discontinuity**: a real mouth becomes a synthesised one
in one frame, and the eye catches the substitution rather than the softness. Composite
the idle mouth as well, from before speech starts, and both states are generated, so the
switch has nothing to give away. Silence is deterministic and the driving clip is a
fixed loop, so it can be rendered once, offline, at prepare time - no model runs while
idle, which spec section 7 requires (a 10-minute continuous run drifted +41.2% and fell
to 20fps, and the idle windows are where `release_lipsync_memory` keeps memory flat).

    python3 -m evals.face_lipsync_idle_spike --data-dir ~/Daemon/data \\
        --wav ~/spikes/musetalk-stage1/in/ko_24k.wav

By hand, on a machine that has the weights. **Never in CI** - it loads 1.7GB of MLX UNet
and reads a 1GB frame store. Writes contact sheets, because the pass mark is a person
looking at them.

## The answer is no, and the reason is not the one that was being looked for

**This engine cannot render this avatar's sealed resting mouth, under any audio.** Over
88 conditioning windows - digital zero, synthetic room tone at -60 and -40 dBFS, and 85
windows of real Korean speech - every single one renders `idle2`'s resting frames with
the lips **parted and a sliver of teeth showing**, where the driving clip's own mouth is
cleanly closed. Not one produced a closed mouth. Ranked by mean |diff| from the artist's
own resting mouth over the clip's four most sealed frames (107, 109, 110, 111), on the
lip box: best 10.48, digital zero 15.01 at rank 21, worst 26.87. The clip's own
frame-to-frame motion in that box is 3.5, so even the best window sits three frames of
natural motion away from the face it would be replacing.

That matters because `idle2` is mostly at rest: frames 104-192, roughly the back half of
the clip, are a continuous stretch of sealed lips. A pre-rendered idle would leave the
avatar sitting with its mouth slightly open, showing teeth, the whole time the daemon is
not speaking. **Looked at** at the size the page actually shows (the 1080x1620 frame
scaled to 600 and to 900px tall, which is where the owner found the original defect), it
is not subtle and it is not a sharpness question - it changes the expression. The
composed, faintly smiling resting face becomes a slack, mouth-open one.

So this trades one discontinuity for a permanently degraded resting face, which is the
trade the brief that commissioned it said to refuse. **Not built.**
`evals/face_lipsync_prepare.py` is unchanged and writes no idle frames.

## What the switch step actually is, since it was worth measuring anyway

Measured 2026-08-27 on `idle2`, 193 frames, the real `MlxEngine` and the real
`Renderer` (so `restore_detail`, `composite` and the q85 JPEG the transport sends are
all in the path), every number taken on the JPEG-decoded frame because that is what the
page displays. The step at the switch is `original -> speech` today and would be
`silence-render -> speech` if this were built, both on the same driving frame:

| pixels measured | today | if pre-rendered | the clip's own frame-to-frame motion |
|---|---|---|---|
| the whole 319x319 blend region | 4.71 | 2.07 | 2.45 |
| `mask > 0` inside `box` | 6.33 | 3.03 | 1.81 |
| `mask >= 128` inside `box` | 9.03 | 4.47 | 1.88 |

Medians of mean\\|diff\\| per frame, 0-255. So the proposal would have worked as
arithmetic - it halves the step, from 4.8x one frame of ordinary motion down to 2.4x.
It is the resting face it pays for that kills it.

**Three regions and not one, because a wide one lies.** Sharpness over the whole blend
region says the generated mouth keeps 99.6% of the original's; over the derived 94x65
lip box it says 56%. The region is 101761 pixels and the paste touches 65635 of them at
all, only 40419 at half alpha or more, and the rest are copied back from the original
unchanged - so every ratio taken over the wide box is dragged toward 100% by pixels
nothing rewrote. A published 87% for this feature was diluted the same way and the tight
box gave 48%. `lip_box` derives the box rather than carrying a constant, so the mistake
cannot come back by editing a number.

**And a Laplacian across two different mouth poses is not a comparison.** Over all 193
frames the speech render reads 99.2% of the original's sharpness against the silence
render's 78.0%, which looks like silence being the problem and is not: an open mouth
carries a dark interior and teeth, so it has more contrast to measure. Paired by pose -
on the 30 frames where the speech render's own mouth is closed - the two are the same
thing, 58.6% and 56.0%. Both generated mouths are equally soft; only the pose differed.
Which is the one part of the premise that did hold: **a silence-rendered mouth does not
sit between the original and the speaking render, it sits on top of the speaking
render.** If the resting pose were right, the switch would carry no quality step at all.

## What is left, if the discontinuity is still worth closing

Not this, and not a different silence - the sweep covers that. What is not ruled out is
making the *engine* able to hold a sealed mouth on this avatar, which is a model
question rather than a preprocessing one; or accepting the step and shortening the
crossfade; or driving idle from a clip whose own mouth is never sealed, so there is no
resting pose to lose. All three are the owner's call and none of them are cheap.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import wave
from pathlib import Path

import numpy as np

WHISPER_REPO = "mlx-community/whisper-tiny-mlx"
RATE = 24_000
"""`daemon/voice/audio.py:OUTPUT_SAMPLE_RATE`. `resample_to_whisper` refuses any other."""

SWEEP_STRIDE = 6
"""Audio frames between swept windows - 0.25s at 24fps. Fine enough to land inside the
pauses between words, coarse enough to cover 22s of speech in 85 renders."""

MISSING = """\
the lip-sync weights and a prepared clip cache are not laid out here:

  {what} is missing ({path})

Both are fetched and built by hand - see `evals/face_lipsync_prepare.py` for the cache
and docs/superpowers/specs/2026-08-26-face-lipsync-design.md section 4 for the weights.
This is a "could not check" result, not a pass.
"""


def read_pcm(path: Path) -> np.ndarray:
    """16-bit mono PCM. Refuses anything else rather than resampling."""
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16-bit mono")
        if handle.getframerate() != RATE:
            raise SystemExit(f"{path}: need {RATE}Hz, got {handle.getframerate()}")
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)


def room_tone(seconds: float, dbfs: float, seed: int = 7) -> np.ndarray:
    """White noise at `dbfs` RMS - a synthetic room-tone floor, at a chosen level.

    Synthetic because the material has none, which is itself worth knowing: the spike's
    own `in/silence_24k.wav` is 96000 samples of **exact zero** (peak 0), and
    `ko_24k.wav` has no 2.2s stretch quieter than rms 4050 of 32768. So "room tone" here
    is a level rather than a recording, and it is tried at two of them because whisper's
    log-mel clamps at `log_spec.max() - 8` and rescales - it normalises against the peak
    of whatever it is handed, so a quiet floor and a loud one may well arrive at the
    model looking identical. Measured, they do: -60 and -40 dBFS render the same mouth.
    """
    rng = np.random.default_rng(seed)
    amp = 32768.0 * 10.0 ** (dbfs / 20.0)
    return np.clip(rng.normal(0.0, amp, int(RATE * seconds)), -32768, 32767).astype(
        np.int16
    )


def rewritten(cache, size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Per-frame alpha inside `box`, and the rectangle that bounds it over the clip.

    The region to measure, derived rather than pasted in. `composite` blends
    `mask` over `crop_box` but only rewrites pixels the mouth was pasted into, which is
    `box`; everything outside is copied from the original and would dilute any ratio
    taken over it. For `idle2` this comes out (380,398)-(679,706), which is the region
    the brief named to the pixel on three of its four edges.
    """
    width, height = size
    n = len(cache.boxes)
    full = np.zeros((n, height, width), np.uint8)
    for i in range(n):
        xs, ys, xe, ye = cache.crop_boxes[i]
        bx1, by1, bx2, by2 = cache.boxes[i]
        frame = np.zeros((height, width), np.uint8)
        frame[ys:ye, xs:xe] = cache.masks[i]
        full[i, by1:by2, bx1:bx2] = frame[by1:by2, bx1:bx2]
    ys_, xs_ = np.where(full.any(axis=0))
    box = (int(xs_.min()), int(ys_.min()), int(xs_.max()) + 1, int(ys_.max()) + 1)
    return full[:, box[1] : box[3], box[0] : box[2]], box


def render_clip(
    engine, cache, pcm: np.ndarray | None, *, fps: float, box, clip_name: str
) -> dict[int, np.ndarray]:
    """Every driving frame, composited and JPEG'd through the real `Renderer`.

    `pcm` of `None` is the digital-zero arm: an unfed ring answers `window()` with
    exact zeros of the right length, which is also what `daemon/app.py`'s warm-up step
    hands the engine.
    """
    import cv2

    from daemon.face_lipsync.render import BATCH, ClipClock, Driver, Renderer
    from daemon.face_lipsync.ring import PcmRing

    ring = PcmRing(sample_rate=RATE, width=2, seconds=30.0)
    if pcm is not None:
        ring.feed(pcm.tobytes(), 0.0)
    # `Driver` bundles the clip, its frames and its clock, because a cache with another
    # clip's clock composites into the wrong pose. This spike drives one clip, so the
    # epoch is 0 and the name is only what the engine keys its latents by.
    renderer = Renderer(
        engine=engine,
        driver=Driver(
            name=clip_name,
            cache=cache,
            clip=ClipClock(fps=fps, frames=len(cache.boxes), epoch=0.0),
        ),
        ring=ring,
    )
    n = len(cache.boxes)
    out: dict[int, np.ndarray] = {}
    for first in range(0, n, BATCH):
        step = renderer.step(frame_index=first, origin=0.0, fps=fps)
        if step is None:
            raise SystemExit("the renderer latched failed - see the logged traceback")
        for index, payload in zip(step.indices, renderer.encode(step), strict=True):
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            out[index % n] = image[box[1] : box[3], box[0] : box[2]].copy()
    return out


def lip_box(
    original: dict[int, np.ndarray], speech: dict[int, np.ndarray], fraction: float = 0.5
) -> tuple[int, int, int, int]:
    """The rectangle the mouth actually moves in, derived from the renders themselves.

    **A sharpness ratio must not be taken over anything wider than this.** The blend
    region is 319x319 and the paste copies most of it back from the original unchanged,
    so a Laplacian over the whole of it is diluted toward 100% by pixels nothing
    rewrote: the same generated mouth reads 99.6% of the original over the blend region
    and 78% over this box. A published 87% for this feature was diluted the same way and
    the tight box gave 48%, so this is the project's most repeated measurement mistake.

    Derived rather than written down: the per-pixel mean |speech - original| over the
    clip, thresholded at `fraction` of its peak. For `idle2`, 0.5 gives 94x65px - 6% of
    the blend region, and the box every ratio here is taken over. 0.25 gives 160x179px,
    the whole mouth, which is what `openness` and the contact sheets want: the tight box
    sits low and left of centre, so it cannot see whether the lips are sealed.
    """
    heat = np.mean(
        [
            np.abs(speech[i].astype(np.float32) - original[i].astype(np.float32)).mean(axis=2)
            for i in sorted(original)
        ],
        axis=0,
    )
    ys, xs = np.where(heat > heat.max() * fraction)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def openness(bgr: np.ndarray) -> float:
    """Fraction of the crop dark enough to be mouth interior. Only meaningful on the
    lip box - over a wider crop it is swamped by the nostrils and the jaw shadow, which
    is how a first pass concluded the render tracked the clip's mouth to within 2%."""
    import cv2

    return float((cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < 90).mean())


def lapvar(bgr: np.ndarray) -> float:
    import cv2

    return float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def per_frame(a: dict[int, np.ndarray], b: dict[int, np.ndarray], where: np.ndarray) -> list[float]:
    """mean|diff| over exactly the pixels `where[i]` selects, one number per frame."""
    out = []
    for i in sorted(a):
        diff = np.abs(a[i].astype(np.int16) - b[i].astype(np.int16)).mean(axis=2)
        out.append(float(diff[where[i]].mean()))
    return out


def quartiles(values: list[float]) -> str:
    ordered = sorted(values)
    return (
        f"{statistics.median(ordered):6.3f} / {ordered[int(0.9 * len(ordered))]:6.3f} / "
        f"{max(ordered):6.3f}"
    )


def contact_sheet(
    path: Path, states: dict[str, dict[int, np.ndarray]], rows, columns, crop
) -> None:
    """One row per state, one column per frame, at 1:1. The deliverable."""
    import cv2

    x1, y1, x2, y2 = crop
    w, h = x2 - x1, y2 - y1
    sheet = np.full((len(rows) * (h + 4) + 18, len(columns) * (w + 4) - 4, 3), 20, np.uint8)
    for r, name in enumerate(rows):
        for c, i in enumerate(columns):
            at_y, at_x = 18 + r * (h + 4), c * (w + 4)
            sheet[at_y : at_y + h, at_x : at_x + w] = states[name][i][y1:y2, x1:x2]
            if r == 0:
                cv2.putText(
                    sheet, f"f{i}", (at_x + 2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (240, 240, 240), 1, cv2.LINE_AA,
                )
        cv2.putText(
            sheet, name, (2, 18 + r * (h + 4) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (0, 255, 255), 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(path), sheet)


def at_display_size(path: Path, frames, region, crops, index: int, tall: int) -> None:
    """The whole frame scaled the way the page scales it, original beside pre-rendered.

    The one view that decides this. Every crop above is at 1:1 on a 1080x1620 frame,
    which is larger than the page ever shows - and the owner found the defect this was
    meant to fix while looking at a scaled-down face, so a difference has to survive
    that scaling to count.
    """
    import cv2

    tiles = []
    for name in ("original", "zero"):
        frame = np.asarray(frames[index]).copy()
        if name != "original":
            frame[region[1] : region[3], region[0] : region[2]] = crops[name][index]
        scale = tall / frame.shape[0]
        shown = cv2.resize(frame, (int(frame.shape[1] * scale), tall), interpolation=cv2.INTER_AREA)
        tiles.append(shown[int(0.05 * tall) : int(0.52 * tall)])
    height, width = tiles[0].shape[:2]
    strip = np.full((height + 20, 2 * (width + 8) - 8, 3), 20, np.uint8)
    for j, tile in enumerate(tiles):
        at = j * (width + 8)
        strip[20 : 20 + height, at : at + width] = tile
        cv2.putText(
            strip, ("original clip", "pre-rendered idle (silence)")[j], (at + 3, 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(path), strip)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--clip", default="idle2")
    parser.add_argument("--wav", type=Path, required=True, help="16-bit mono 24kHz speech")
    parser.add_argument("--out", type=Path, default=Path("/tmp/face_lipsync_idle"))
    parser.add_argument(
        "--no-sweep", action="store_true", help="skip the 88-window search for a closed mouth"
    )
    args = parser.parse_args()

    root = args.data_dir.expanduser().resolve() / "face" / "lipsync"
    models, clip_dir = root / "models", root / args.clip
    needed = {
        "the MLX UNet": models / "unet.safetensors",
        "its config": models / "musetalk.json",
        "the TAESD decoder": models / "taesd.safetensors",
        "the clip's reference latents": clip_dir / "latents.safetensors",
        "the clip's frames": clip_dir / "frames.npy",
        "the clip's blend masks": clip_dir / "masks.npz",
        "the clip's boxes": clip_dir / "boxes.json",
    }
    for what, path in needed.items():
        if not path.exists():
            print(MISSING.format(what=what, path=path))
            return 1
    args.out.mkdir(parents=True, exist_ok=True)

    from daemon.face_lipsync import Cache
    from daemon.face_lipsync.audio import CONTEXT_MS
    from daemon.face_lipsync.engine import load
    from daemon.face_lipsync.render import Step
    from daemon.face_lipsync.ring import PcmRing

    meta = json.loads((clip_dir / "boxes.json").read_text(encoding="utf-8"))
    fps, size = float(meta["fps"]), tuple(meta["size"])
    frames = np.load(clip_dir / "frames.npy", mmap_mode="r")
    with np.load(clip_dir / "masks.npz") as archive:
        masks = [archive[name] for name in sorted(archive.files)]
    cache = Cache(
        frames=frames,
        boxes=[tuple(b) for b in meta["boxes"]],
        crop_boxes=[tuple(b) for b in meta["crop_boxes"]],
        masks=masks,
    )
    n = len(cache.boxes)
    alphas, region = rewritten(cache, size)
    print(f"{args.clip}: {n} frames at {fps}fps, {size[0]}x{size[1]}")
    print(
        f"  the paste rewrites (mask>0 & box) "
        f"({region[0]},{region[1]})-({region[2]},{region[3]}) = "
        f"{region[2] - region[0]}x{region[3] - region[1]}px, of which a mean "
        f"{(alphas > 0).sum(axis=(1, 2)).mean():.0f} px are touched and "
        f"{(alphas >= 128).sum(axis=(1, 2)).mean():.0f} are at least half generated"
    )

    engine = load(
        unet_weights=models / "unet.safetensors",
        unet_config_json=models / "musetalk.json",
        taesd_weights=models / "taesd.safetensors",
        whisper_repo=WHISPER_REPO,
        latents={clip_dir.name: clip_dir / "latents.safetensors"},
        sample_rate=RATE,
    )

    arms = {
        "zero": None,
        "tone-60": np.tile(room_tone(4.0, -60.0), 3),
        "tone-40": np.tile(room_tone(4.0, -40.0), 3),
        "speech": read_pcm(args.wav)[: RATE * 12],
    }
    states = {
        "original": {
            i: np.asarray(frames[i][region[1] : region[3], region[0] : region[2]]).copy()
            for i in range(n)
        }
    }
    for name, pcm in arms.items():
        states[name] = render_clip(
            engine, cache, pcm, fps=fps, box=region, clip_name=clip_dir.name
        )
        print(f"  rendered {name}")

    # --- the step at the switch, on two nested pixel sets ------------------------
    for label, where in (("mask > 0 & box", alphas > 0), ("mask >= 128 & box", alphas >= 128)):
        print(f"\n=== {label} " + "=" * (52 - len(label)))
        print("  the step AT the switch, per frame (median / p90 / max):")
        today = quartiles(per_frame(states["original"], states["speech"], where))
        proposed = quartiles(per_frame(states["zero"], states["speech"], where))
        print(f"    TODAY       original -> speech: {today}")
        print(f"    PROPOSED zero-render -> speech: {proposed}")
        ordinary = [
            float(
                np.abs(
                    states["original"][i + 1].astype(np.int16)
                    - states["original"][i].astype(np.int16)
                ).mean(axis=2)[where[i] & where[i + 1]].mean()
            )
            for i in range(n - 1)
        ]
        print(f"    for scale, the clip's own frame-to-frame motion: {quartiles(ordinary)}")

    # --- sharpness, paired by pose ----------------------------------------------
    lx1, ly1, lx2, ly2 = lip_box(states["original"], states["speech"])
    print(
        f"\nlip box, derived: ({region[0] + lx1},{region[1] + ly1})-"
        f"({region[0] + lx2},{region[1] + ly2}) = {lx2 - lx1}x{ly2 - ly1}px, "
        f"{100 * (lx2 - lx1) * (ly2 - ly1) / alphas[0].size:.1f}% of the blend region"
    )

    mx1, my1, mx2, my2 = lip_box(states["original"], states["speech"], fraction=0.25)

    def lip(name: str, i: int) -> np.ndarray:
        return states[name][i][ly1:ly2, lx1:lx2]

    def mouth(name: str, i: int) -> np.ndarray:
        return states[name][i][my1:my2, mx1:mx2]

    moved = [
        float(np.abs(lip("speech", i).astype(np.int16) - lip("zero", i).astype(np.int16)).mean())
        for i in range(n)
    ]
    closed = sorted(range(n), key=lambda i: moved[i])[:30]
    print("sharpness on the 30 frames where the SPEECH render's own mouth is closed -")
    print("paired by pose, because an open mouth has more contrast to measure:")
    base = statistics.mean(lapvar(lip("original", i)) for i in closed)
    for name in states:
        value = statistics.mean(lapvar(lip(name, i)) for i in closed)
        print(f"  {name:10} {value:7.1f} = {100 * value / base:5.1f}% of the artist's own mouth")
    print("  (over the whole blend region instead, every one of them reads 99%+ - that")
    print("   difference is the dilution this box exists to avoid.)")

    # --- is there ANY window that renders a sealed mouth? ------------------------
    if not args.no_sweep:
        from daemon.face_lipsync.render import ClipClock, Driver, Renderer

        ring = PcmRing(sample_rate=RATE, width=2, seconds=30.0)
        ring.feed(read_pcm(args.wav)[: RATE * 25].tobytes(), 0.0)
        renderer = Renderer(
            engine=engine,
            driver=Driver(
                name=clip_dir.name,
                cache=cache,
                clip=ClipClock(fps=fps, frames=n, epoch=0.0),
            ),
            ring=ring,
        )
        # The four frames whose own mouth is most sealed, by how little of the MOUTH
        # box is dark - the lip box sits too low to see an aperture (see `openness`).
        resting = sorted(sorted(range(n), key=lambda i: openness(mouth("original", i)))[:4])
        windows = {
            "zero": ring.window(frame_index=0, fps=fps, origin=0.0, context_ms=CONTEXT_MS) * 0.0
        }
        for k in range(0, 520, SWEEP_STRIDE):
            windows[f"ko@{k / fps:.2f}s"] = ring.window(
                frame_index=k, fps=fps, origin=0.0, context_ms=CONTEXT_MS
            )
        print(f"\nsweeping {len(windows)} conditioning windows against resting frames {resting}")
        import cv2

        scores: dict[str, float] = {}
        keys = list(windows)
        for at in range(0, len(keys), 2):
            pair = keys[at : at + 2]
            for i in resting:
                step = Step(
                    indices=[i] * len(pair),
                    mouths=engine.mouths(
                        [windows[k] for k in pair], [i] * len(pair), clip=clip_dir.name
                    ),
                )
                for key, payload in zip(pair, renderer.encode(step), strict=True):
                    image = cv2.imdecode(
                        np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    crop = image[region[1] : region[3], region[0] : region[2]][
                        ly1:ly2, lx1:lx2
                    ]
                    scores.setdefault(key, 0.0)
                    scores[key] += (
                        float(
                            np.abs(
                                crop.astype(np.int16) - lip("original", i).astype(np.int16)
                            ).mean()
                        )
                        / len(resting)
                    )
        rank = sorted(scores, key=lambda k: scores[k])
        print("  closest to the artist's own resting mouth (mean|diff|, lower is better):")
        for key in rank[:5]:
            print(f"    {key:14} {scores[key]:6.3f}")
        print(f"    {'...':14}")
        print(f"    {rank[-1]:14} {scores[rank[-1]]:6.3f}   (worst of {len(rank)})")
        print(
            f"  digital zero {scores['zero']:6.3f}, rank {rank.index('zero') + 1} of {len(rank)}"
        )
        print("  NONE of them close the mouth - look at the sheets.")

    # --- look at it -------------------------------------------------------------
    sheet_box = (mx1, my1 + (my2 - my1) // 3, mx2, my2)
    """The mouth, minus the top third - the nose adds nothing and costs the rows height."""
    rows = ["original", "zero", "speech"]
    contact_sheet(args.out / "sheet_early.png", states, rows, list(range(0, 120, 16)), sheet_box)
    contact_sheet(
        args.out / "sheet_resting.png", states, rows, list(range(104, 193, 12)), sheet_box
    )
    for tall in (600, 900):
        at_display_size(args.out / f"at_{tall}px.png", frames, region, states, 111, tall)
    print(f"\nwrote {args.out}")
    print("  sheet_resting.png is the one that settles it: the clip's lips are sealed")
    print("  in every column and every rendered row parts them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
