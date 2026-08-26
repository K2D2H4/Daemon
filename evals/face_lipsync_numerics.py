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

This does not run the model or recompute the spike's cosine similarity; it checks
structural facts a running model would depend on. Never run in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from daemon.face_lipsync.loader import (
    needs_split,
    needs_squeeze,
    rename,
    split_names,
    unet_config,
)


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

    root = Path("data/face/lipsync/models")
    config_path = root / "musetalk.json"
    weights_path = root / "unet.safetensors"

    print(f"weights root: {root}")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
