"""Does the product loader keep the published MLX weights in MLX layout?

`daemon/face_lipsync/loader.py`'s `rename` and `needs_split` are str -> str and
str -> bool: they never receive a tensor, so `tests/test_face_lipsync_loader.py`
proves they rename and split correctly and proves nothing about layout. That
"never touches layout" claim holds by the functions' signatures, not by any
assertion in that file - the actual hazard is one level up, in how a caller
applies them to real data. The published MLX weights are diffusers-keyed but
MLX-laid-out already: `conv_in.weight` is (320, 3, 3, 8), PyTorch's is
(320, 8, 3, 3). A loader that transposes on top of that runs fine and returns
nonsense - quietly, because a wrong-but-plausible shape looks like nothing until
a rendered mouth does. CI cannot hold the weights that would catch this
(GB-scale, not bundled), so this is the one check only a real run, by hand, ever
performs.

    python3 -m evals.face_lipsync_numerics

Needs the real weights under `<data_dir>/face/lipsync/models/` - `musetalk.json`
and `unet.safetensors`, fetched by hand per
docs/superpowers/specs/2026-08-26-face-lipsync-design.md section 4 (not bundled,
and not present on a fresh checkout). When they are missing, this says so
plainly and exits non-zero rather than raising - a "could not check" result,
never reported as a pass.

What actually gets checked: the rename/split loop is run against every real
tensor (686 -> 718 is what an earlier PyTorch-vs-MLX spike recorded for this
conversion, docs/superpowers/specs/2026-08-25-face-design.md section 6 - print
and compare by eye), `conv_in.weight` must land at (320, 3, 3, 8), and every
`proj_in.weight` / `proj_out.weight` / `conv_shortcut.weight` must be rank 2.
`conv_in` is a
meaningful canary and not an arbitrary pick: its input-channel count (8) differs
from its kernel size (3), so a correct layout and a double-transposed one are
actually distinguishable tuples here - unlike a tensor whose dimensions happen to
coincide, where shape alone could not catch the same mistake. The rank check
exists because `conv_in` cannot see everything: upstream mlx-examples'
`map_unet_weights` does five things (rename, split, squeeze conv_shortcut,
squeeze 4-D proj_in/proj_out, transpose) and this loader does three - rename,
split, squeeze - but never transposes.

**That "three" used to read "two", and the difference is what this eval was for.**
It said the squeeze was an open question: "correct if the published weights truly
need no squeeze, silently wrong if they do". Running the assembled engine settled
it. They need it - `proj_in` arrives as (320, 1, 1, 320) and `mx.addmm` rejects it
outright, which is the loud version of the failure rather than the quiet one. So
this check has changed job: it no longer probes an open premise, it asserts that
`needs_squeeze` is still doing its work. 46 tensors depend on it.

The mapping loop below deliberately calls the same `loader` helpers the engine
does, rather than reimplementing them. An earlier version wrote its own rename and
split inline, which meant it could pass while `daemon/face_lipsync/engine.py`
failed - and it did: the split there collided on `.bias` keys, and mlx's
`tree_unflatten` answers a duplicate key with unbounded recursion rather than an
error naming the key.

**The second half does run the model, once, for a question CI equally cannot ask.**
`MlxEngine` now holds one latent table per prepared clip and `mouths` selects with a
`clip` keyword, because the 1.6GB of UNet/TAESD/whisper weights say nothing about
which clip is on screen and the tables are 1.3-3.0MB each. Two ways for that to be
silently wrong: a second clip's table perturbs the first clip's mouth (it must not -
the arrays are independent, so the same request must be bit-identical to what a
single-clip engine returned), or the keyword is accepted and ignored, in which case
every clip renders `idle2`'s mouth and nothing raises. The first needs two builds to
compare; the second needs two clips in one build. Both are here, and neither is
reachable from CI, which has no weights and no mlx wheel.

It does not recompute the spike's cosine similarity against PyTorch. Never run in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from daemon.face_lipsync.loader import (
    needs_split,
    needs_squeeze,
    rename,
    split_names,
    unet_config,
)


def _second_clip_on_a_worker_thread(mx, root: Path, models: Path) -> int:
    """Whether rendering a clip the warm-up never touched survives off the loading thread.

    This is the one failure in this engine that **no unit test can reach**: every test
    under `tests/` injects a fake engine, and the real one aborts the PROCESS rather than
    raising - `libc++abi: terminating due to uncaught exception of type
    std::runtime_error: There is no Stream(cpu, 1) in current thread`, which no `except`
    sees. It killed the assembled daemon on 2026-08-31 on the first clip change of the
    first spoken turn, with the whole suite green.

    The shape reproduced here is `daemon/app.py`'s exactly: warm up ONE clip on the
    loading thread, then ask a worker for a different one, which is what
    `_lipsync_loop`'s model executor does after `ClipQueue` reaches a boundary. The fix
    it guards is `load()` evaluating every clip's latents on the loading thread; remove
    that and this returns non-zero by dying.
    """
    from concurrent.futures import ThreadPoolExecutor

    from daemon.face_lipsync.audio import CONTEXT_MS
    from daemon.face_lipsync.engine import load
    from daemon.face_lipsync.ring import PcmRing

    clips = [name for name in ("idle2", "listening", "amused") if (root / name).is_dir()]
    if len(clips) < 2:
        print("second clip on a worker thread: needs two prepared caches - skipped")
        return 0
    print("a clip the warm-up never touched, rendered off the loading thread")
    engine = load(
        unet_weights=models / "unet.safetensors",
        unet_config_json=models / "musetalk.json",
        taesd_weights=models / "taesd.safetensors",
        whisper_repo="mlx-community/whisper-tiny-mlx",
        latents={name: root / name / "latents.safetensors" for name in clips},
        # 24kHz because that is what the ring holds - `daemon/voice/audio.py`'s
        # OUTPUT_SAMPLE_RATE, the same value `_clip_selection` above uses.
        sample_rate=24_000,
    )
    ring = PcmRing(sample_rate=24_000, width=2, seconds=30.0)
    window = ring.window(frame_index=0, fps=24.0, origin=0.0, context_ms=CONTEXT_MS)
    engine.mouths([window, window], [0, 0], clip=clips[0])
    print(f"  warmed up {clips[0]} on this thread, as daemon/app.py does")
    with ThreadPoolExecutor(max_workers=1) as worker:
        for name in clips[1:]:
            got = worker.submit(
                engine.mouths, [window, window], [0, 0], clip=name
            ).result()
            print(f"  clip={name} on a worker thread: ok {got[0].shape}")
    return 0


def _clip_selection(mx, root: Path, models: Path) -> int:
    """Build the engine twice and ask whether a second clip changed the first's mouth.

    Sequential rather than side by side: each build holds 1.7GB of UNet, so the first
    engine is dropped before the second is made. The comparison is on the returned
    uint8 mouths, where equality IS bit equality - there is no tolerance to choose,
    because selecting a row out of a dict of arrays cannot be approximately right.

    Deterministic audio, generated here: a 220Hz tone of exactly `CONTEXT_MS` plus one
    200ms window, which is what `PcmRing.window` hands over in production. Two windows
    and two frame indices, because `BATCH` is 2 and a batched call is the only one the
    render loop ever makes.
    """
    from daemon.face_lipsync.audio import CONTEXT_MS
    from daemon.face_lipsync.engine import load

    others = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and d.name != "idle2" and (d / "latents.safetensors").exists()
    )
    idle2 = root / "idle2" / "latents.safetensors"
    if not idle2.exists() or not others:
        print("MISSING CLIP CACHES - the selection check did not run, this is not a pass:")
        print(f"  idle2 latents: {idle2} {'found' if idle2.exists() else 'NOT FOUND'}")
        print(f"  other prepared clips with latents: {others or 'none'}")
        print(
            "Each is built by hand, once per clip: "
            "`python3 -m evals.face_lipsync_prepare`. Two clips are the minimum this "
            "check needs - one to select, one to prove the selection is a choice."
        )
        return 1
    other = others[0]
    # Named rather than counted as "the ten prepared clips": this finds every
    # directory under the cache root that holds a latents file, and a machine that
    # has done this work also has backups of one (`idle2-raw-backup`, on the owner's).
    print(f"  driving: idle2, second table: {other}")
    print(f"  latents found under {root}: {', '.join(['idle2'] + others)}")

    # 24kHz because that is what the ring holds - `daemon/voice/audio.py`'s
    # OUTPUT_SAMPLE_RATE, not whisper's 16kHz, which `resample_to_whisper` reaches.
    rate = 24_000
    samples = int(rate * (CONTEXT_MS + 200.0) / 1000.0)
    time = np.arange(samples, dtype=np.float32) / rate
    windows = [
        (0.25 * np.sin(2.0 * np.pi * 220.0 * (time + shift))).astype(np.float32)
        for shift in (0.0, 0.02)
    ]
    indices = [7, 8]

    def build(latents: dict[str, Path]):
        return load(
            unet_weights=models / "unet.safetensors",
            unet_config_json=models / "musetalk.json",
            taesd_weights=models / "taesd.safetensors",
            whisper_repo="mlx-community/whisper-tiny-mlx",
            latents=latents,
            sample_rate=rate,
        )

    print("  building the single-clip engine", flush=True)
    engine = build({"idle2": idle2})
    single = engine.mouths(windows, indices, clip="idle2")
    del engine
    mx.clear_cache()

    print("  building the two-clip engine", flush=True)
    engine = build({"idle2": idle2, other: root / other / "latents.safetensors"})
    both = engine.mouths(windows, indices, clip="idle2")
    elsewhere = engine.mouths(windows, indices, clip=other)

    for i, (one, two) in enumerate(zip(single, both, strict=True)):
        if not np.array_equal(one, two):
            differing = int(np.count_nonzero(one != two))
            print(
                f"FAIL frame {indices[i]} of idle2 changed when a second clip's "
                f"latents were loaded: {differing} of {one.size} bytes differ, "
                f"max |delta| {int(np.abs(one.astype(int) - two.astype(int)).max())}"
            )
            return 1
    print(f"  idle2 bit-identical with {other} loaded ({len(both)} frames)")

    # The other direction: if `clip` were accepted and ignored, every frame above
    # would still match and every clip would render idle2's mouth.
    if all(np.array_equal(a, b) for a, b in zip(both, elsewhere, strict=True)):
        print(
            f"FAIL clip={other} returned idle2's mouths exactly - the keyword is "
            "being ignored, or both tables hold the same latents"
        )
        return 1
    print(f"  clip={other} selects its own latents (mouths differ, as they must)")
    return 0


def main() -> int:
    try:
        import mlx.core as mx
    except ImportError as exc:
        print("MISSING MLX - nothing was checked, this is not a pass:")
        print(f"  {exc}")
        print(
            "mlx is a macOS/Apple-silicon runtime dependency this eval needs to load "
            "the weights. It is deliberately not declared in pyproject.toml or any "
            "extra - this file is never run in CI (see the module docstring) - so a "
            "fresh checkout does not have it. Install it by hand and re-run:\n"
            "  pip install mlx"
        )
        return 1

    root = Path("data/face/lipsync")
    models = root / "models"
    config_path = models / "musetalk.json"
    weights_path = models / "unet.safetensors"

    print(f"weights root: {models}")
    missing = [path for path in (config_path, weights_path) if not path.exists()]
    if missing:
        print("MISSING WEIGHTS - nothing was checked, this is not a pass:")
        for path in missing:
            print(f"  not found: {path}")
        print(
            "These are not bundled (docs/superpowers/specs/2026-08-26-face-lipsync-design.md "
            "section 4) and are not present on this machine. unet.safetensors comes from "
            "mlx-community/MuseTalk-1.5-fp16. Fetch both into the path above and re-run."
        )
        return 1

    print("loading", flush=True)
    with open(config_path) as handle:
        config = unet_config(json.load(handle))
    print(
        f"  in_channels={config['in_channels']} "
        f"cross_attention_dim={config['cross_attention_dim'][0]} "
        f"heads={config['num_attention_heads']}"
    )
    weights = mx.load(weights_path)
    mapped = []
    for key, value in weights.items():
        target = rename(key)
        if needs_squeeze(target, value.ndim):
            value = value.squeeze()
        if needs_split(key):
            a, b = mx.split(value, 2)
            first, second = split_names(target)
            mapped.append((first, a))
            mapped.append((second, b))
        else:
            mapped.append((target, value))
    print(f"  {len(weights)} tensors -> {len(mapped)} arrays")

    # A duplicate destination is not a shape error and not an exception: mlx's
    # tree_unflatten only stops recursing when a leaf holds exactly one entry, so it
    # answers a collision with RecursionError from inside its own dict comprehension.
    # Checking here names the key instead.
    seen: set[str] = set()
    duplicates = sorted({k for k, _ in mapped if k in seen or seen.add(k)})
    if duplicates:
        print("FAIL two weights mapped to the same destination:")
        for key in duplicates:
            print(f"  {key}")
        return 1
    print(f"  destinations unique ({len(mapped)})")

    # The check that matters: shape must stay NHWC. A double transpose shows up
    # here before it shows up as a blurry mouth.
    conv_in = dict(mapped)["conv_in.weight"]
    if tuple(conv_in.shape) != (320, 3, 3, 8):
        print(f"FAIL conv_in.weight is {tuple(conv_in.shape)}, expected (320, 3, 3, 8)")
        return 1
    print("  layout OK (NHWC, not transposed)")

    # The narrower check conv_in cannot do: mlx-examples declares proj_in/proj_out
    # and conv_shortcut as nn.Linear, which needs a 2-D (out, in) weight, and the
    # published weights arrive as 1x1 convolutions. `needs_squeeze` drops the
    # singleton axes; this confirms it still did. proj_out is in the list now - it is
    # squeezed for the same reason as the other two and was simply missing before.
    linear_like = [
        (key, value)
        for key, value in mapped
        if key.endswith(("proj_in.weight", "proj_out.weight", "conv_shortcut.weight"))
    ]
    bad_rank = [(key, tuple(value.shape)) for key, value in linear_like if value.ndim != 2]
    if bad_rank:
        print("FAIL expected 2-D (nn.Linear) weight, got:")
        for key, shape in bad_rank:
            print(f"  {key}: {shape}")
        return 1
    print(f"  proj_in/proj_out/conv_shortcut rank OK ({len(linear_like)} tensors)")

    print("one engine, latents per clip", flush=True)
    rc = _clip_selection(mx, root, models)
    if rc:
        return rc
    return _second_clip_on_a_worker_thread(mx, root, models)


if __name__ == "__main__":
    sys.exit(main())
