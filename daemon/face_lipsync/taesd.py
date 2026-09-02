"""TAESD's decoder in MLX, and the weight mapping that feeds it.

MuseTalk decodes with Stable Diffusion's ordinary `sd-vae-ft-mse`, and that one term
is 55% of a batch-1 frame - larger on its own than the whole 41.67ms budget for 24fps.
`madebyollin/taesd` is a drop-in for the same 4-channel latent space at 5.4ms against
64.0ms measured, which is what makes the frame rate arithmetically possible at all.

It costs detail: on the 256 output TAESD keeps 39% of the driving clip's lip detail
where sd-vae keeps 63%. At the product's display size the two differ by 7 points.

**The swap is closed, and here is the number that closes it.** The 58.6ms/frame this
file used to cite was diffusers on MPS, and MLX is where this engine lives precisely
because MLX was faster for the UNet - so "sd-vae in MLX has never been timed" stayed an
open question for as long as the file said 58.6. Measured 2026-08-31, both decoders in
one process on the same `(2, 32, 32, 4)` latents at this module's own dtype:

    TAESD    4.21ms/frame   (median 8.42ms per BATCH=2 pair, p95 8.58)
    sd-vae  35.87ms/frame   (median 71.73ms per pair, p95 73.13)

MLX does help - **+31.66ms against the +58.6ms that first rejected it, nearly half** -
and it is still far outside the budget. The model half is 35.93ms/frame with TAESD in it;
swapping puts it at **67.59ms/frame against 41.67ms**, a 62% overrun that takes 24fps to
14.8fps. Halving it again would not fit. Do not re-cost this on the strength of a faster
machine without re-deriving the whole budget, which is what the earlier note asked for
and what these numbers now answer.

Two obstacles cost an hour to rediscover, so they are written down rather than left for
the next attempt. `mlx-examples`' own `Autoencoder` passes `layers_per_block + 1` to the
decoder - a diffusers VAE decoder has one more resnet per block than the encoder - so
building its `Decoder` by hand fails with `List index 2 is out of bounds`. And MuseTalk's
`sd-vae` checkpoint is diffusers **0.4.2**, which predates the `to_q`/`to_k`/`to_v`/
`to_out.0` names `map_vae_weights` knows: its attention keys are `query`/`key`/`value`/
`proj_attn`, 16 of them across both halves, and they must be renamed by exact path
segment - matching "key" as a substring rewrites unrelated paths. The measurement itself
was throwaway; it needs the spike's vendored `mlxsd/` and its 334MB `sd-vae` weights,
neither of which is in this repo.

**The other place to run it is closed too.** The GPU is shared with the UNet, so the
obvious escape was the Neural Engine: convert the sd-vae decoder to CoreML, run it on
the ANE *concurrently* with the MLX UNet, and hide its cost behind the UNet's 71.86ms
per pair. Measured 2026-09-01 on this M4 Max (coremltools 9.0, fp16 mlprogram, the
`torch.export` frontend - `torch.jit.trace` fails on diffusers' tensor-valued
`height*width`; warm-up 10, 50 timed runs, `(2, 4, 32, 32)` in, `(2, 3, 256, 256)` out):

    CPU_AND_NE   median  98.46ms/pair   p95 102.12   (re-runs 95.70, 94.76)
    CPU_AND_GPU  median 180.84ms/pair   p95 307.73
    CPU_ONLY     median 235.07ms/pair   p95 294.61

The ANE genuinely ran - `MLComputePlan` placed 404 of 404 ops on it, and the timing is
2.4x off CPU_ONLY, so no silent fallback - and the overlap holds: under a GPU load shaped
like the UNet's duty cycle its latency did not move (96.07 / p95 97.08). Numerics are fine
(mean abs error 0.003, worst pixel 7/255, PSNR 47dB against fp32). **It is simply too
slow: ~96ms per pair against the ~72ms it had to hide behind**, so the pipeline would
run at max(72, 96) = 48ms/frame, 20.8fps, over the 41.67ms budget by 15%. None of the
cheap levers move it more than 3%: a newer opset (fused SDPA) 94.80, fp16 I/O 95.36, a
batch-1 model exactly half (48.15 - compute-bound, not per-call overhead), two concurrent
requests 81.4 effective with unfair latency. Closing a 33% gap is a different decoder
(an ANE-shaped layout, or a smaller network), not a tuning of this one. The machine was
busy during the runs (the resident with MLX loaded, and renders), which taints the
CPU and GPU rows - the ANE row was stable to +-2ms over five runs and nothing else on
the machine uses the ANE, so it is the one to trust. The spike's scripts (`convert2.py`,
`bench.py`, `compute_plan.py`, `ane_under_gpu.py`) were throwaway, like the MLX one.

The detail gap is what `render.restore_detail` was meant to pay back - though see its own
docstring for how little of that it is measured to actually do.

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
