"""`MlxEngine` - the only place a model runs. Everything here was measured, not chosen.

The chain is audio -> whisper features -> UNet -> TAESD -> a 256x256 BGR mouth, and
`daemon/face_lipsync/render.py` owns everything on either side of it. Nothing in this
module knows what a driving clip or a blend mask is.

Four decisions in here look arbitrary and are not:

  * whisper runs on MLX, not transformers. It reproduces transformers' own output
    (mel cosine 1.000000, encoder hidden states 1.000000 / 0.999997 for the
    post-layer-norm one), which is what lets the daemon skip a 2.5GB torch dependency
    for an 8M-parameter encoder.
  * the encoder attends over ~110 positions, not whisper's 1500. The rest of a padded
    mel is silence, so this is bit-identical - cosine 1.0000 and the mouth unchanged
    pixel-for-pixel - at 2.25ms instead of 8.17ms. The spike report claimed this was
    impossible because "HF WhisperEncoder hard-checks 3000 mel frames"; that was a
    property of the wrapper, not the model.
  * the ten hidden-state indices are read from the END of `audio`, not the start. The
    array carries `CONTEXT_MS` of lead-in that whisper's normalisation needs.
  * the mel keeps its 30s padding. Dropping it saves 0.36ms and moves the features
    (cosine 0.986 against the padded path), which is a bad trade for a term that is
    already under a millisecond.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten
from mlx_whisper import audio as whisper_audio
from mlx_whisper.load_models import load_model as load_whisper

from daemon.face_lipsync.audio import (
    WHISPER_RATE,
    WINDOW,
    encoder_positions,
    resample_to_whisper,
)
from daemon.face_lipsync.loader import (
    needs_split,
    needs_squeeze,
    rename,
    split_names,
    taesd_rename,
    taesd_to_mlx,
    unet_config,
)
from daemon.face_lipsync.taesd import Decoder
from daemon.face_lipsync.vendor.mlx_sd import UNetConfig, UNetModel

DTYPE = mx.float16
"""One dtype for every array crossing into a model, and it is load-bearing.

The published UNet weights are fp16. Feeding them an fp32 latent or fp32 conditioning
does not error - MLX promotes, silently, and the whole UNet then runs in fp32. Measured
on the assembled engine: 71.86ms/frame that way against a 41.67ms budget, where the
spike's fp16 numbers predicted about half that. Anything built here in numpy (the
latents, the audio positional encoding, the mel) starts out fp32, so each one has to
be cast at the boundary or it drags the graph up with it.
"""

MEL_BINS = 80
HIDDEN_STATES = 5
"""whisper-tiny has 4 encoder layers, and MuseTalk conditions on 5 states: the input
to each layer, plus `layer_norm` applied to the last layer's output. That final entry
is the only one that is normalised - transformers collects the others *before* each
layer, and treating all five alike gets the shapes right and the values wrong."""

FEATURE_DIM = 384
MEL_PAD_SAMPLES = 30 * WHISPER_RATE
"""whisper's front-end works in 30s frames. Kept - see the module docstring."""


def _audio_positional_encoding(seq_len: int, d_model: int) -> mx.array:
    """MuseTalk's `PositionalEncoding`, which has no weights to load.

    Reproduced rather than mapped because it is a buffer, not a parameter: the
    published checkpoint does not contain it, and an engine that skipped it would run
    and produce a mouth that ignores where in the window each feature sat.
    """
    position = np.arange(seq_len, dtype=np.float32)[:, None]
    div = np.exp(
        np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model)
    )
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div)
    pe[:, 1::2] = np.cos(position * div)
    return mx.array(pe[None]).astype(DTYPE)


class MlxEngine:
    """Satisfies `LipsyncEngine`. Construct it with `load`; `daemon/app.py` injects it."""

    def __init__(
        self,
        *,
        unet: UNetModel,
        taesd: Decoder,
        whisper_encoder: nn.Module,
        latents: mx.array,
        sample_rate: int,
    ) -> None:
        self._unet = unet
        self._taesd = taesd
        self._enc = whisper_encoder
        self._latents = latents.astype(DTYPE)
        """(n, 32, 32, 8) in MLX layout, one per driving frame, prepared offline."""
        self._rate = sample_rate
        self._timestep = mx.array([0])
        """MuseTalk is single-step inpainting: there is no schedule to walk."""
        self._audio_pe = _audio_positional_encoding(
            WINDOW * HIDDEN_STATES, FEATURE_DIM
        )

    def _features(self, audio: np.ndarray) -> mx.array:
        """One frame's (1, 50, 384) conditioning, from the tail of `audio`."""
        pcm = resample_to_whisper(audio.astype(np.float32), rate=self._rate)
        positions = encoder_positions(pcm.size)
        mel = np.array(
            whisper_audio.log_mel_spectrogram(
                pcm, n_mels=MEL_BINS, padding=max(0, MEL_PAD_SAMPLES - pcm.size)
            )
        ).T
        x = mx.array(mel.T[None]).astype(DTYPE)
        x = nn.gelu(self._enc.conv1(x))
        x = nn.gelu(self._enc.conv2(x))
        # Trim to the positions that hold real audio. Self-attention makes a shorter
        # window a different answer, so this cuts padding only - measured
        # bit-identical (cosine 1.0000, mouth unchanged) at 2.25ms instead of 8.17ms.
        # Cutting into real audio is not free: at 60 positions the features fall to
        # cosine 0.91 and the mouth moves a fifth of one frame's own motion.
        x = x[:, :positions] + self._enc._positional_embedding[:positions]
        states = []
        for block in self._enc.blocks:
            states.append(x[:, -WINDOW:])
            x, _, _ = block(x)
        states.append(self._enc.ln_post(x)[:, -WINDOW:])
        stacked = mx.stack(states, axis=2)          # (1, WINDOW, 5, 384)
        return stacked.reshape(1, WINDOW * HIDDEN_STATES, FEATURE_DIM) + self._audio_pe

    def mouths(
        self, windows: Sequence[np.ndarray], frame_indices: Sequence[int]
    ) -> list[np.ndarray]:
        """Batched on purpose - see `BATCH` in `render.py` for the arithmetic.

        Everything downstream of the features runs as one batch: the UNet at N=2 is
        29.29ms/frame against 41.55ms alone, which is the whole reason the protocol
        takes a sequence. The decoder and the trip back to numpy batch too, though
        they barely care (4.26 vs 4.76ms, 0.92 vs 0.93ms).
        """
        if len(windows) != len(frame_indices):
            raise ValueError(
                f"{len(windows)} windows for {len(frame_indices)} frames - one each"
            )
        features = mx.concatenate([self._features(w) for w in windows])
        latents = mx.concatenate(
            [self._latents[i % self._latents.shape[0]][None] for i in frame_indices]
        )
        images = self._taesd(self._unet(latents, self._timestep, features))
        rgb = np.array(mx.clip(images / 2.0 + 0.5, 0.0, 1.0) * 255.0)
        bgr = rgb.round().astype(np.uint8)[..., ::-1]
        return [bgr[i] for i in range(bgr.shape[0])]


def _reject_duplicates(mapped: list[tuple[str, mx.array]]) -> None:
    """Fail loudly on a repeated destination path.

    `tree_unflatten` responds to one by recursing until Python gives up, and the
    traceback points at a dict comprehension in mlx rather than at the key. This
    turned a one-word mapping bug into a long hunt once already.
    """
    seen: set[str] = set()
    clashes = {path for path, _ in mapped if path in seen or seen.add(path)}
    if clashes:
        raise ValueError(f"weight mapping produced duplicate paths: {sorted(clashes)}")


def load(
    *,
    unet_weights: Path,
    unet_config_json: Path,
    taesd_weights: Path,
    whisper_repo: str,
    latents: Path,
    sample_rate: int,
) -> MlxEngine:
    """Build an engine from files on disk. Never called at import time."""
    config = unet_config(json.loads(unet_config_json.read_text(encoding="utf-8")))
    unet = UNetModel(UNetConfig(**config))
    mapped: list[tuple[str, mx.array]] = []
    for key, value in mx.load(str(unet_weights)).items():
        value = value.astype(DTYPE)
        target = rename(key)
        if needs_squeeze(target, value.ndim):
            value = value.squeeze()
        if needs_split(key):
            half = value.shape[0] // 2
            first, second = split_names(target)
            mapped.append((first, value[:half]))
            mapped.append((second, value[half:]))
        else:
            mapped.append((target, value))
    _reject_duplicates(mapped)
    unet.update(tree_unflatten(mapped))

    taesd = Decoder()
    decoder_weights = [
        (path, taesd_to_mlx(key, value).astype(DTYPE))
        for key, value in mx.load(str(taesd_weights)).items()
        if (path := taesd_rename(key)) is not None
    ]
    taesd.update(tree_unflatten(decoder_weights))

    engine = MlxEngine(
        unet=unet,
        taesd=taesd,
        whisper_encoder=load_whisper(whisper_repo, dtype=mx.float16).encoder,
        latents=mx.load(str(latents))["latents"],
        sample_rate=sample_rate,
    )
    mx.eval(unet.parameters(), taesd.parameters())
    return engine
