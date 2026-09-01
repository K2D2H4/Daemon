"""Key mapping for the pre-converted MLX weights.

The published file is diffusers-keyed but MLX-laid-out: `conv_in.weight` is
(320, 3, 3, 8) where torch's is (320, 8, 3, 3). Transposing it again produces a
model that runs and returns nonsense. `rename` and `needs_split` below are
str -> str and str -> bool - they never receive a tensor, so the mapping never
touches layout by construction, not because anything in this file asserts it:
these tests check renaming and splitting only. The layout invariant itself is
checked by `evals/face_lipsync_numerics.py`, against the real weights, by hand,
outside CI.
"""

import inspect

import numpy as np

from daemon.face_lipsync.loader import (
    needs_split,
    needs_squeeze,
    rename,
    split_names,
    taesd_rename,
    taesd_to_mlx,
    unet_config,
)


def test_the_engine_is_told_which_clip_s_latents_to_use():
    """The UNet, TAESD and whisper weights are 1.6GB and clip-independent; the
    latents are 1.3-3.0MB (measured over the ten prepared caches) and are the only
    per-clip tensor. One engine, ten latent sets - not ten engines.

    Checked on the protocol rather than on `MlxEngine`, because `mlx` has no Linux
    wheel and CI is ubuntu: importing the engine here would fail on the runner.
    """
    from daemon.face_lipsync import LipsyncEngine

    sig = inspect.signature(LipsyncEngine.mouths)
    assert "clip" in sig.parameters
    assert sig.parameters["clip"].kind is inspect.Parameter.KEYWORD_ONLY


def test_attention_projections_are_renamed():
    q_key = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
    assert rename(q_key).endswith("query_proj.weight")
    out_key = "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.weight"
    assert rename(out_key).endswith("out_proj.weight")


def test_the_mid_block_becomes_an_indexed_list():
    assert rename("mid_block.resnets.0.norm1.weight").startswith("mid_blocks.0.")
    assert rename("mid_block.attentions.0.proj_in.weight").startswith("mid_blocks.1.")
    assert rename("mid_block.resnets.1.norm1.weight").startswith("mid_blocks.2.")


def test_the_geglu_split_gives_two_distinct_paths_for_weight_AND_bias():
    """The bias is the regression. A rule written around the word "weight" leaves
    `.bias` untouched, so both halves land on one path - and mlx's `tree_unflatten`
    answers a duplicate key with an unbounded recursion whose traceback names a dict
    comprehension inside mlx, not the key."""
    for suffix in ("weight", "bias"):
        key = f"down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.{suffix}"
        first, second = split_names(key)
        assert first != second
        assert first == f"down_blocks.0.attentions.0.transformer_blocks.0.linear1.{suffix}"
        assert second == f"down_blocks.0.attentions.0.transformer_blocks.0.linear2.{suffix}"


def test_the_split_replaces_the_segment_rather_than_deleting_it():
    """Removing `ff.net.0.proj.` and re-appending a name happens to work for a weight
    and silently collides for a bias, which is exactly how this broke."""
    first, _ = split_names("x.ff.net.0.proj.bias")
    assert first == "x.linear1.bias"
    assert "ff.net" not in first


def test_the_feedforward_projection_is_the_only_split():
    assert needs_split("down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.weight")
    assert not needs_split("down_blocks.0.attentions.0.transformer_blocks.0.ff.net.2.weight")
    assert not needs_split("conv_in.weight")


def test_musetalk_config_differs_from_stable_diffusion_in_exactly_three_fields():
    musetalk = {
        "in_channels": 8, "out_channels": 4,
        "block_out_channels": [320, 640, 1280, 1280],
        "layers_per_block": 2, "attention_head_dim": 8,
        "cross_attention_dim": 384, "norm_num_groups": 32,
        "down_block_types": ["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"],
        "up_block_types": ["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3,
    }
    got = unet_config(musetalk)
    assert got["in_channels"] == 8
    assert got["cross_attention_dim"] == [384] * 4
    assert got["num_attention_heads"] == [8] * 4
    # up_block_types is reversed, as mlx-examples' own loader does
    assert got["up_block_types"][0] == "CrossAttnUpBlock2D"
    assert got["up_block_types"][-1] == "UpBlock2D"


# --- TAESD, which answers the transpose question the other way ------------------
#
# The UNet's weights are pre-converted and must be left alone; TAESD's come from
# upstream PyTorch and must be transposed. Both failure modes are silent - the model
# runs and returns an image of the right shape either way - so these are the only
# place in CI where the distinction is checked. Numerics against the real decoder are
# in `evals/face_lipsync_numerics.py`.

# The real decoder's structure, from diffusers' own module tree: 19 slots, of which
# these 13 carry parameters. Three convolutions are bias-free.
_DECODER_SLOTS = (0, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18)
_BLOCK_SLOTS = (2, 3, 4, 7, 8, 9, 12, 13, 14, 17)


def test_a_blocks_three_convolutions_keep_their_identity():
    """diffusers indexes them 0/2/4 inside a Sequential whose 1 and 3 are ReLUs.
    Renumbering them 0/1/2 would load the second convolution's weights into the
    third and still run."""
    assert taesd_rename("decoder.layers.2.conv.0.weight") == "layer_2.conv0.weight"
    assert taesd_rename("decoder.layers.2.conv.2.bias") == "layer_2.conv2.bias"
    assert taesd_rename("decoder.layers.2.conv.4.weight") == "layer_2.conv4.weight"


def test_plain_convolutions_map_straight_through():
    assert taesd_rename("decoder.layers.0.weight") == "layer_0.weight"
    assert taesd_rename("decoder.layers.6.weight") == "layer_6.weight"
    assert taesd_rename("decoder.layers.18.bias") == "layer_18.bias"


def test_the_encoder_half_is_dropped_not_mapped():
    """It ships in the same file and is dead weight - loading it would double the
    resident cost of a model chosen for being small."""
    assert taesd_rename("encoder.layers.0.weight") is None
    assert taesd_rename("encoder.layers.4.conv.0.weight") is None


def test_every_real_decoder_key_maps_somewhere_distinct():
    """A collision would leave one parameter filled twice and another never filled,
    which `Decoder.update` does not complain about."""
    keys = []
    for i in _DECODER_SLOTS:
        if i in _BLOCK_SLOTS:
            keys += [
                f"decoder.layers.{i}.conv.{c}.{p}"
                for c in (0, 2, 4)
                for p in ("weight", "bias")
            ]
        else:
            keys += [f"decoder.layers.{i}.weight", f"decoder.layers.{i}.bias"]
    mapped = [taesd_rename(k) for k in keys]
    assert all(m is not None for m in mapped)
    assert len(set(mapped)) == len(mapped)


def test_the_transpose_is_out_height_width_in_and_not_its_mirror():
    """(0, 2, 3, 1), not (0, 3, 2, 1). Both produce a 4-D array MLX accepts and a
    square kernel makes them indistinguishable by shape, so this uses distinct
    extents on every axis."""
    w = np.zeros((5, 4, 3, 2), np.float32)          # out, in, kH, kW
    assert taesd_to_mlx("decoder.layers.0.weight", w).shape == (5, 3, 2, 4)


def test_biases_are_not_transposed():
    b = np.zeros((64,), np.float32)
    assert taesd_to_mlx("decoder.layers.0.bias", b).shape == (64,)


def test_the_transpose_decision_is_by_rank_not_by_name():
    """Matching on "weight" in the key would also catch a bias if diffusers ever
    renamed one, and a 1-D transpose(0, 2, 3, 1) raises rather than degrading."""
    odd = np.zeros((7, 6, 5, 4), np.float32)
    assert taesd_to_mlx("anything.at.all", odd).shape == (7, 5, 4, 6)
    assert taesd_to_mlx("decoder.layers.0.weight", np.zeros((3,), np.float32)).shape == (3,)


# --- the squeeze the eval was written to catch ----------------------------------
#
# mlx-examples declares proj_in / proj_out / conv_shortcut as nn.Linear, which needs a
# 2-D (out, in) weight, while diffusers stores them as 1x1 convolutions. The eval
# flagged this as an open question - "correct if the published weights truly need no
# squeeze, silently wrong otherwise" - and running the assembled engine answered it:
# they are all rank 4 with (1, 1) spatial extents, 46 tensors of them.


def test_the_three_one_by_one_weights_are_squeezed():
    for name in ("proj_in", "proj_out", "conv_shortcut"):
        assert needs_squeeze(f"down_blocks.0.attentions.0.{name}.weight", 4)


def test_an_already_squeezed_weight_is_left_alone():
    """Squeezing twice would collapse a (1, N) weight to (N,). Guarding on rank rather
    than trusting the caller keeps that impossible."""
    assert not needs_squeeze("down_blocks.0.attentions.0.proj_in.weight", 2)


def test_ordinary_convolutions_keep_their_rank():
    """conv_in is the one to protect: it is genuinely 4-D and squeezing it would be
    silent, since (320, 3, 3, 8) has no singleton axis to lose."""
    assert not needs_squeeze("conv_in.weight", 4)
    assert not needs_squeeze("down_blocks.0.resnets.0.conv1.weight", 4)


def test_biases_are_never_squeezed():
    assert not needs_squeeze("down_blocks.0.attentions.0.proj_in.bias", 4)
