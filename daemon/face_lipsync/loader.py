"""Load MuseTalk's UNet, and TAESD's decoder, into their MLX modules.

Both mappings live here on purpose. They answer the transpose question in OPPOSITE
directions, and each is silent when wrong - a mis-laid-out convolution still runs and
still returns an image of the right shape. Keeping them side by side is the only place
a reader sees the contrast:

    UNet  (`rename`)         published MLX weights, ALREADY in MLX layout - never transpose
    TAESD (`taesd_rename`)   upstream PyTorch weights, (out, in, kH, kW) - must transpose


Two facts make the UNet side small. mlx-examples' `UNetConfig` is fully parameterised, and
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


def split_names(key: str) -> tuple[str, str]:
    """The two destination paths a GEGLU tensor splits into.

    Substituting the whole `ff.net.0.proj` segment, rather than editing the trailing
    `weight`, is the point: the projection has a bias as well, and a rule written
    around the word "weight" leaves a `.bias` key untouched so BOTH halves land on the
    same path. mlx's `tree_unflatten` does not report that as a collision - it only
    stops recursing when a leaf has exactly one entry, so a duplicate key sends it
    into infinite recursion and reports `RecursionError` from inside a dict
    comprehension, which says nothing about the key that caused it.
    """
    return key.replace(SPLIT_KEY, "linear1"), key.replace(SPLIT_KEY, "linear2")


def rename(key: str) -> str:
    """diffusers parameter path -> mlx-examples parameter path."""
    for old, new in _RENAMES:
        if old in key:
            key = key.replace(old, new)
    return key


def needs_split(key: str) -> bool:
    return SPLIT_KEY in key


ONE_BY_ONE = ("proj_in.weight", "proj_out.weight", "conv_shortcut.weight")
"""The three weights mlx-examples declares as `nn.Linear` and diffusers stores as 1x1
convolutions. Measured in the published file: 16 + 16 + 14 tensors, every one rank 4
with (1, 1) spatial extents."""


def needs_squeeze(key: str, ndim: int) -> bool:
    """Does this tensor have to lose its singleton spatial axes?

    `nn.Linear` wants a 2-D `(out, in)` weight and the published MLX weights keep the
    4-D convolution form, so the loader has to drop it. This was an open question
    rather than a guess - `evals/face_lipsync_numerics.py` was written to flag exactly
    this, on the premise that pre-converted weights might need no squeeze at all.
    Running the assembled engine settled it: `proj_in` arrives as (320, 1, 1, 320) and
    `mx.addmm` rejects it outright, which is the good case. A silent version of this
    mistake is what the eval was guarding against.

    Guarded on rank so an already-squeezed weight is left alone rather than collapsed
    further. Kept as `str, int -> bool` so this module still never touches a tensor,
    which is what lets it be tested without weights or MLX.
    """
    return ndim == 4 and key.endswith(ONE_BY_ONE)


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


_BLOCK_CONV = {"conv.0": "conv0", "conv.2": "conv2", "conv.4": "conv4"}


def taesd_rename(key: str) -> str | None:
    """diffusers TAESD key -> `Decoder` attribute path. `None` means not the decoder.

    The encoder half ships in the same file and is dead weight here, so it is dropped
    rather than mapped: loading it would double the resident cost of a model chosen
    for being small.
    """
    if not key.startswith("decoder.layers."):
        return None
    rest = key[len("decoder.layers.") :]
    index, _, tail = rest.partition(".")
    for old, new in _BLOCK_CONV.items():
        if tail.startswith(old):
            return f"layer_{index}.{new}.{tail[len(old) + 1 :]}"
    return f"layer_{index}.{tail}"


def taesd_to_mlx(key: str, value: Any) -> Any:
    """Transpose a TAESD convolution kernel into MLX layout. Biases pass through.

    Note the contrast with `rename` above, which must NOT transpose: these weights come
    from upstream PyTorch, not from a pre-converted MLX repo.

    PyTorch stores `(out, in, kH, kW)`; MLX wants `(out, kH, kW, in)`. Deciding on
    `ndim == 4` rather than on the name is deliberate - every 4-D tensor in this
    decoder is a conv kernel, and matching on "weight" would also catch the biases if
    the naming ever changed.
    """
    return value.transpose(0, 2, 3, 1) if getattr(value, "ndim", 0) == 4 else value
