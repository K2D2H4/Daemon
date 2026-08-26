"""TAESD's decoder in MLX, and the weight mapping that feeds it.

MuseTalk decodes with Stable Diffusion's ordinary `sd-vae-ft-mse`, and that one term
is 55% of a batch-1 frame - larger on its own than the whole 41.67ms budget for 24fps.
`madebyollin/taesd` is a drop-in for the same 4-channel latent space at 5.4ms against
64.0ms measured, which is what makes the frame rate arithmetically possible at all.

It costs detail: on the 256 output TAESD keeps 39% of the driving clip's lip detail
where sd-vae keeps 63%. That is paid back by `render.restore_detail`, not by reverting
the decoder - at the product's display size the two decoders differ by 7 points, and
buying those back costs 58.6ms/frame.

The key mapping and the transpose live in `loader.py`, deliberately next to the
UNet's - because the two have OPPOSITE answers on transposing and confusing them
produces a decoder that runs and returns garbage. See `loader.taesd_to_mlx`.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

LATENT_CHANNELS = 4
WIDTH = 64
"""TAESD is 64 channels throughout - there is no channel ladder to configure."""


class Block(nn.Module):
    """`AutoencoderTinyBlock`: three convolutions, then a ReLU on the residual sum.

    diffusers builds this as `Sequential(conv, relu, conv, relu, conv)` with an
    `Identity` skip and a `ReLU` fuse, so the skip needs no parameters and the
    attribute names below match its `conv.0` / `conv.2` / `conv.4` indices.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.conv2 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)
        self.conv4 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv4(nn.relu(self.conv2(nn.relu(self.conv0(x)))))
        return nn.relu(h + x)


def _upsample(x: mx.array) -> mx.array:
    """Nearest-neighbour x2 on an NHWC array, matching `nn.Upsample(mode='nearest')`."""
    n, h, w, c = x.shape
    return mx.broadcast_to(x[:, :, None, :, None, :], (n, h, 2, w, 2, c)).reshape(
        n, h * 2, w * 2, c
    )


class Decoder(nn.Module):
    """The decoder half of `AutoencoderTiny`, 32x32x4 -> 256x256x3.

    Attributes are named for diffusers' own `decoder.layers.N` indices rather than
    packed into a list. It is more lines, but it makes the weight mapping a single
    string substitution and a mis-mapped layer impossible to miss when reading.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer_0 = nn.Conv2d(LATENT_CHANNELS, WIDTH, 3, padding=1)
        self.layer_2, self.layer_3, self.layer_4 = Block(), Block(), Block()
        self.layer_6 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1, bias=False)
        self.layer_7, self.layer_8, self.layer_9 = Block(), Block(), Block()
        self.layer_11 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1, bias=False)
        self.layer_12, self.layer_13, self.layer_14 = Block(), Block(), Block()
        self.layer_16 = nn.Conv2d(WIDTH, WIDTH, 3, padding=1, bias=False)
        self.layer_17 = Block()
        self.layer_18 = nn.Conv2d(WIDTH, 3, 3, padding=1)

    def __call__(self, latent: mx.array) -> mx.array:
        """`latent` is NHWC. Returns NHWC RGB in -1..1, diffusers' convention."""
        x = mx.tanh(latent / 3.0) * 3.0        # DecoderTiny's own clamp, not an extra
        x = nn.relu(self.layer_0(x))
        for blk in (self.layer_2, self.layer_3, self.layer_4):
            x = blk(x)
        x = self.layer_6(_upsample(x))
        for blk in (self.layer_7, self.layer_8, self.layer_9):
            x = blk(x)
        x = self.layer_11(_upsample(x))
        for blk in (self.layer_12, self.layer_13, self.layer_14):
            x = blk(x)
        x = self.layer_16(_upsample(x))
        x = self.layer_18(self.layer_17(x))
        return x * 2.0 - 1.0                   # [0,1] -> [-1,1]
