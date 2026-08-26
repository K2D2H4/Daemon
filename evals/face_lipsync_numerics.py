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
and compare by eye), and `conv_in.weight` must land at (320, 3, 3, 8). That
tensor is a meaningful canary and not an arbitrary pick: its input-channel count
(8) differs from its kernel size (3), so a correct layout and a double-transposed
one are actually distinguishable tuples here - unlike a tensor whose dimensions
happen to coincide, where shape alone could not catch the same mistake. This does
not run the model or recompute the spike's cosine similarity; it checks the one
structural fact that numeric pass depended on. Never run in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx

from daemon.face_lipsync.loader import needs_split, rename, unet_config


def main() -> int:
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
        if needs_split(key):
            a, b = mx.split(value, 2)
            mapped.append((rename(key).replace("ff.net.0.proj", "linear1"), a))
            mapped.append((rename(key).replace("ff.net.0.proj", "linear2"), b))
        else:
            mapped.append((rename(key), value))
    print(f"  {len(weights)} tensors -> {len(mapped)} arrays")

    # The check that matters: shape must stay NHWC. A double transpose shows up
    # here before it shows up as a blurry mouth.
    conv_in = dict(mapped)["conv_in.weight"]
    if tuple(conv_in.shape) != (320, 3, 3, 8):
        print(f"FAIL conv_in.weight is {tuple(conv_in.shape)}, expected (320, 3, 3, 8)")
        return 1
    print("  layout OK (NHWC, not transposed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
