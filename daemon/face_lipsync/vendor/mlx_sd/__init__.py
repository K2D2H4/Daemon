"""Apple's MLX Stable Diffusion UNet, vendored from `ml-explore/mlx-examples`.

MIT licensed (see `LICENSE`), Copyright (c) 2023 Apple Inc.

Vendored rather than depended on because `mlx-examples` is a repository of examples
and publishes nothing to PyPI, and a git dependency in a self-hosted app is a
liability every user pays at install time. Two files, 525 lines, importing only
`typing` and each other - so this is the smallest form the dependency can take.

Why this file at all: MuseTalk's UNet *is* Stable Diffusion's, differing in exactly
three config fields (`in_channels` 8, `cross_attention_dim` 384, `attention_head_dim`
8), and `UNetConfig` is fully parameterised. `loader.unet_config` does that
translation; nothing here knows about MuseTalk.

Provenance, checked before committing:
    stable_diffusion/stable_diffusion/unet.py    identical to upstream main
    stable_diffusion/stable_diffusion/config.py  identical to upstream main

Do not edit these two files. Re-vendor from upstream and diff instead.
"""

from daemon.face_lipsync.vendor.mlx_sd.config import UNetConfig
from daemon.face_lipsync.vendor.mlx_sd.unet import UNetModel

__all__ = ["UNetConfig", "UNetModel"]
