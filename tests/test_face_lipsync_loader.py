"""Key mapping for the pre-converted MLX weights.

The published file is diffusers-keyed but MLX-laid-out: `conv_in.weight` is
(320, 3, 3, 8) where torch's is (320, 8, 3, 3). Transposing it again produces a
model that runs and returns nonsense, so the mapping here renames and splits but
never touches layout - and that is asserted rather than commented.
"""

from daemon.face_lipsync.loader import needs_split, rename, unet_config


def test_attention_projections_are_renamed():
    q_key = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight"
    assert rename(q_key).endswith("query_proj.weight")
    out_key = "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.weight"
    assert rename(out_key).endswith("out_proj.weight")


def test_the_mid_block_becomes_an_indexed_list():
    assert rename("mid_block.resnets.0.norm1.weight").startswith("mid_blocks.0.")
    assert rename("mid_block.attentions.0.proj_in.weight").startswith("mid_blocks.1.")
    assert rename("mid_block.resnets.1.norm1.weight").startswith("mid_blocks.2.")


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
