"""Silero VAD - the `VoiceActivityDetector` half of the wake gate in
daemon/voice/base.py.

Why a vendored .onnx and not `pip install silero-vad`: the published package
imports torchaudio, and on this project's target machine torchaudio's wheel is
ABI-mismatched against the installed torch, so `import silero_vad` fails outright
before any audio is touched. The graph itself is 2.3 MB and needs nothing but
numpy and onnxruntime, so the model is committed next to this file and the
package is not a dependency at all. The arithmetic below was checked against
silero's own API before that: max|diff| = 0.00e+00 over four fixtures.

`onnxruntime` is imported lazily, never at module scope, for the same reason
`daemon/voice/audio.py` defers `sounddevice`: voice dependencies live in the
`voice` extra, and `import daemon.voice.vad` must still succeed in a text-only
install.

Two things about the exported graph that are quiet if you get them wrong:

  * A call does not take a bare 512-sample frame. The model expects its own
    64-sample lookback prepended, so the tensor is 576 wide, and the lookback is
    the *previous* frame's tail. Feed bare frames and it still runs and still
    returns numbers - about 0.001 for real Korean speech where the correct carry
    gives 0.9. Nothing raises.
  * The frame size is not enforced by the graph either. 512 is what it was
    exported for, but widths from 256 to 600 run and return nonsense, and only
    values further out raise. So the length check in `probability` is what makes
    a wrong frame size a failure instead of a silently dead gate.

Both are why the protocol says frames must arrive in order and `reset` must be
called between unrelated streams.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from daemon.voice.audio import AudioUnavailable

MODEL_PATH = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"
"""Resolved against this file, not the cwd: the daemon runs as a LaunchAgent whose
working directory is not the repo."""

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 512
"""32 ms at 16 kHz. Fixed by the export, not chosen here - see the module
docstring."""

CONTEXT_SAMPLES = 64
"""Samples of the previous frame the model wants in front of this one."""

STATE_SHAPE = (2, 1, 128)
"""The LSTM state the graph hands back and takes again. Batch of exactly one; a
wider batch is rejected."""

FRAME_BYTES = FRAME_SAMPLES * 2
"""16-bit PCM, so two bytes a sample. What `probability` requires."""

_INT16_FULL_SCALE = 32768.0
"""Divisor for int16 -> [-1, 1). 32768 rather than 32767 because that is what
silero's own front end uses, and matching it is what makes the outputs identical."""

_INSTALL_HINT = "install with: pip install -e '.[voice]'"


def _onnxruntime() -> Any:
    """Import onnxruntime on first construction, with an error worth reading."""
    try:
        import onnxruntime
    except ImportError:
        raise AudioUnavailable(f"onnxruntime is not installed; {_INSTALL_HINT}") from None
    return onnxruntime


class SileroVad:
    """Implements the `VoiceActivityDetector` protocol in daemon/voice/base.py.

    Stateful and not thread-safe: one instance belongs to one stream of frames.
    """

    def __init__(self) -> None:
        self.frame_samples = FRAME_SAMPLES
        self.sample_rate = SAMPLE_RATE
        if not MODEL_PATH.exists():
            raise AudioUnavailable(
                f"the Silero VAD model is missing from {MODEL_PATH}; it ships with "
                "the package, so a checkout without it is incomplete"
            )
        ort = _onnxruntime()
        options = ort.SessionOptions()
        # One thread each, deliberately. A frame is 32 ms of audio and ~0.1 ms of
        # work, so the thread pool would cost more than it saves - and this runs
        # forever in the background of a machine someone is using, where the
        # measurement that justifies the gate at all is 0.49% of one core.
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        # Loaded here rather than on the first frame: building the session is
        # 40 ms warm and 340 ms with onnxruntime cold, and the first frame arrives
        # on the audio path where that would read as a hang. CPU only - the graph
        # is tiny and a GPU provider would add a dependency this has no use for.
        self._session = ort.InferenceSession(
            str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
        )
        # Built once. A fresh np.array per frame is not free at 31 frames a second
        # for the life of the process.
        self._sample_rate_input = np.array(SAMPLE_RATE, dtype=np.int64)
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def probability(self, frame: bytes) -> float:
        """Speech likelihood in [0, 1] for one frame of 16-bit little-endian PCM."""
        if len(frame) != FRAME_BYTES:
            # The graph would accept several wrong sizes and answer nonsense, so
            # this is the only place a wrong frame can be caught at all.
            raise ValueError(
                f"expected {FRAME_BYTES} bytes ({FRAME_SAMPLES} samples of 16-bit PCM), "
                f"got {len(frame)}"
            )
        # Explicit "<i2": the wire format is little-endian regardless of the host.
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / _INT16_FULL_SCALE
        window = np.concatenate([self._context, samples.reshape(1, -1)], axis=1)
        output, self._state = self._session.run(
            None,
            {
                "input": window,
                "state": self._state,
                "sr": self._sample_rate_input,
            },
        )
        # The tail of what we just fed, which includes this frame's last 64
        # samples - so the next call sees a continuous window.
        self._context = window[:, -CONTEXT_SAMPLES:]
        return float(output[0][0])

    def reset(self) -> None:
        """Forget the stream.

        Both halves, because either one alone leaks: measured on
        tests/fixtures/wake/wake-and-question.wav, the first frame of pure silence
        after that utterance reads 0.96 with the carry intact and 0.002 after this.
        """
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
