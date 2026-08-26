"""Load MuseTalk's UNet into mlx-examples' Stable Diffusion UNet.

Two facts make this small. mlx-examples' `UNetConfig` is fully parameterised, and
MuseTalk's config differs from Stable Diffusion's in exactly three fields
(`in_channels` 8, `cross_attention_dim` 384, `attention_head_dim` 8). And the
published MLX weights are already in MLX layout, so unlike mlx-examples' own
`map_unet_weights` this renames and splits but never transposes.
"""

from __future__ import annotations

from typing import Any

_RENAMES = (
    ("downsamplers.0.conv", "downsample"),
    ("upsamplers.0.conv", "upsample"),
    ("mid_block.resnets.0", "mid_blocks.0"),
    ("mid_block.attentions.0", "mid_blocks.1"),
    ("mid_block.resnets.1", "mid_blocks.2"),
    ("to_k", "key_proj"),
    ("to_out.0", "out_proj"),
    ("to_q", "query_proj"),
    ("to_v", "value_proj"),
    ("ff.net.2", "linear3"),
)

SPLIT_KEY = "ff.net.0.proj"
"""diffusers keeps GEGLU's two projections in one tensor; MLX wants them apart."""


def rename(key: str) -> str:
    """diffusers parameter path -> mlx-examples parameter path."""
    for old, new in _RENAMES:
        if old in key:
            key = key.replace(old, new)
    return key


def needs_split(key: str) -> bool:
    return SPLIT_KEY in key


def unet_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """MuseTalk's diffusers config -> mlx-examples `UNetConfig` kwargs."""
    n = len(cfg["block_out_channels"])
    head_dim = cfg["attention_head_dim"]
    return {
        "in_channels": cfg["in_channels"],
        "out_channels": cfg["out_channels"],
        "block_out_channels": cfg["block_out_channels"],
        "layers_per_block": [cfg["layers_per_block"]] * n,
        "transformer_layers_per_block": cfg.get("transformer_layers_per_block", (1,) * n),
        "num_attention_heads": [head_dim] * n if isinstance(head_dim, int) else head_dim,
        "cross_attention_dim": [cfg["cross_attention_dim"]] * n,
        "norm_num_groups": cfg["norm_num_groups"],
        "down_block_types": cfg["down_block_types"],
        "up_block_types": cfg["up_block_types"][::-1],
    }
