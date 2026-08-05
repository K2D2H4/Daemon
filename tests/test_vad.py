"""Wake-gate VAD tests: the real vendored model, real Korean speech, no hardware.

The model is 2.3 MB of committed ONNX and the fixtures are committed WAVs, so this
file needs no network, no microphone and no API key - it runs the actual detector
rather than a stand-in, because the defect this module invites is arithmetic that
looks plausible and is wrong. A version that feeds the model bare 512-sample
frames returns ~0.001 for the same Korean speech the correct one calls 0.94, and
nothing raises. So the numbers below are measured, and the tolerances are tight
enough that the naive version fails them.

Every expectation here was measured on this exact model and these exact fixtures:

  wake-alone.wav          23 frames, mean 0.944
  wake-and-question.wav   69 frames, mean 0.900
  no-wake-word.wav        52 frames, mean 0.991   (speech, just not the phrase)
  silence                 every frame < 0.01
  440 Hz tone, noise      every frame < 0.05
  chord with vibrato      46.8% of frames over 0.5 - a false positive, asserted
                          as one, because the gate is not allowed to trust the
                          VAD alone (daemon/voice/base.py)
"""

from __future__ import annotations

import builtins
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from daemon.voice import vad
from daemon.voice.audio import AudioUnavailable
from daemon.voice.base import VoiceActivityDetector
from daemon.voice.vad import CONTEXT_SAMPLES, FRAME_BYTES, FRAME_SAMPLES, SAMPLE_RATE, SileroVad

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "wake"

SPEECH = ("wake-alone.wav", "wake-and-question.wav", "no-wake-word.wav")


@pytest.fixture
def detector() -> SileroVad:
    """A fresh detector per test. Not shared: this class carries LSTM state
    between calls, so a shared one would let one test's audio decide another's
    result - which is the exact failure the class is written to avoid."""
    return SileroVad()


def _fixture(name: str) -> bytes:
    with wave.open(str(FIXTURES / name)) as wav:
        return wav.readframes(wav.getnframes())


def _frames(pcm: bytes) -> list[bytes]:
    """Whole frames only; a trailing partial frame is not a frame."""
    return [pcm[at : at + FRAME_BYTES] for at in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES)]


def _probabilities(detector: SileroVad, pcm: bytes) -> list[float]:
    return [detector.probability(frame) for frame in _frames(pcm)]


def _pcm(signal: np.ndarray) -> bytes:
    """Float samples in [-1, 1] as the 16-bit little-endian PCM the protocol takes -
    the same conversion `AudioIO.record` would have already done."""
    return (np.clip(signal, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _seconds(count: float) -> np.ndarray:
    return np.arange(int(SAMPLE_RATE * count), dtype=np.float64) / SAMPLE_RATE


def _tone(frequency: float, seconds: float = 2.0) -> np.ndarray:
    return 0.5 * np.sin(2 * np.pi * frequency * _seconds(seconds))


def _white_noise(seconds: float = 2.0) -> np.ndarray:
    # Seeded: at low amplitudes an unlucky draw peaks over 0.1 on a single frame,
    # so an unseeded version would fail a few runs in a hundred and look like a
    # regression in the model.
    return np.random.default_rng(0).normal(0, 0.1, int(SAMPLE_RATE * seconds))


def _chord_with_vibrato(depth: float = 0.14, seconds: float = 2.0) -> np.ndarray:
    """Three notes at once, each swept +/-14% at 6.5 Hz, through a 700 Hz resonance.

    Deliberately not a synth beep: the vibrato and the formant-like resonance are
    what make it read as a voice, and `depth` is a parameter only so a test can
    show that removing the vibrato removes the false positive.
    """
    signal = np.zeros(int(SAMPLE_RATE * seconds))
    time_s = _seconds(seconds)
    for ratio in (1.0, 1.26, 1.5):  # a major chord, near G#4
        instantaneous = 415.3 * ratio * (1 + depth * np.sin(2 * np.pi * 6.5 * time_s))
        signal += np.sin(2 * np.pi * np.cumsum(instantaneous) / SAMPLE_RATE)
    # A 2-pole resonance as its impulse response, so the filter is one convolution
    # rather than a sample loop.
    decay = np.arange(int(SAMPLE_RATE * 0.06)) / SAMPLE_RATE
    resonance = np.exp(-np.pi * 700.0 * decay / 10.0) * np.sin(2 * np.pi * 700.0 * decay)
    shaped = np.convolve(signal, resonance)[: len(signal)]
    return 0.7 * shaped / np.max(np.abs(shaped))


# --- the contract -------------------------------------------------------------


def test_it_satisfies_the_voice_activity_detector_protocol(detector: SileroVad) -> None:
    assert isinstance(detector, VoiceActivityDetector)
    assert (detector.frame_samples, detector.sample_rate) == (FRAME_SAMPLES, SAMPLE_RATE)


def test_the_frame_and_context_sizes_are_the_ones_the_export_requires() -> None:
    """Pinned, not chosen. 512 samples of frame plus 64 of carried context is the
    576-wide window the graph was exported for; the next test shows what happens
    to a caller who picks its own numbers."""
    assert (FRAME_SAMPLES, CONTEXT_SAMPLES, FRAME_BYTES) == (512, 64, 1024)
    assert SAMPLE_RATE == 16_000


def test_the_model_wants_the_previous_frames_tail_in_front_of_this_one(
    detector: SileroVad,
) -> None:
    """The trap, made falsifiable. Feeding bare frames is not an error the graph
    reports - it runs, and answers 0.001 for speech it otherwise calls 0.94."""
    bare = []
    for frame in _frames(_fixture("wake-alone.wav")):
        # Reproduces the naive implementation one frame at a time: empty the
        # context the previous call saved, leaving a 512-wide window.
        detector._context = np.zeros((1, 0), dtype=np.float32)
        bare.append(detector.probability(frame))

    assert max(bare) < 0.01, "bare frames must not look like speech, or the trap is gone"


# --- real speech --------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "frames", "mean"),
    [
        ("wake-alone.wav", 23, 0.944),
        ("wake-and-question.wav", 69, 0.900),
        # No wake phrase, and still speech - the VAD's job is "someone is
        # talking", so 0.99 here is correct, not a false positive.
        ("no-wake-word.wav", 52, 0.991),
    ],
)
def test_korean_speech_reads_as_speech(
    detector: SileroVad, name: str, frames: int, mean: float
) -> None:
    probabilities = _probabilities(detector, _fixture(name))

    assert len(probabilities) == frames
    # abs=0.02 is loose enough for a different onnxruntime build and far too
    # tight for a broken context or state carry, which lands near 0.001.
    assert float(np.mean(probabilities)) == pytest.approx(mean, abs=0.02)


@pytest.mark.parametrize("name", SPEECH)
def test_the_fixtures_are_what_the_microphone_would_capture(name: str) -> None:
    """16 kHz mono 16-bit, so a test can feed them in with no resampling. If a
    fixture is ever re-recorded at another rate every number above changes."""
    with wave.open(str(FIXTURES / name)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16_000, 1, 2)


# --- everything that is not speech --------------------------------------------


def test_silence_is_not_speech(detector: SileroVad) -> None:
    probabilities = _probabilities(detector, bytes(FRAME_BYTES * 31))

    assert probabilities and max(probabilities) < 0.01


@pytest.mark.parametrize("kind", ["tone", "noise"])
def test_a_tone_and_white_noise_are_not_speech(detector: SileroVad, kind: str) -> None:
    """The two things a room actually contains when nobody is talking: something
    humming and the noise floor."""
    signal = _tone(440.0) if kind == "tone" else _white_noise()
    probabilities = _probabilities(detector, _pcm(signal))

    assert probabilities and max(probabilities) < 0.05


def test_music_is_a_false_positive_and_the_vad_is_therefore_not_the_whole_gate(
    detector: SileroVad,
) -> None:
    """46.8% of frames over 0.5 for a chord nobody said. This is asserted rather
    than tolerated because it is the reason `SpeechRecognizer` exists in
    daemon/voice/base.py: a real `daemon voice` run against ambient music recorded
    a lyric as though the owner had said it.

    Two-sided on purpose. If the rate drops, the VAD stopped being fooled - good
    news, but the second stage was justified by this number, so re-measure it
    instead of loosening this.
    """
    probabilities = _probabilities(detector, _pcm(_chord_with_vibrato()))
    over = sum(p > 0.5 for p in probabilities) / len(probabilities)

    assert over == pytest.approx(0.468, abs=0.08)


def test_the_same_chord_without_vibrato_does_not_fool_it(detector: SileroVad) -> None:
    """Which is what makes the test above about music rather than about loudness:
    identical notes, identical level, no vibrato, no false positive."""
    probabilities = _probabilities(detector, _pcm(_chord_with_vibrato(depth=0.0)))

    assert max(probabilities) < 0.05


# --- state, which is where this goes quietly wrong ----------------------------


def test_reset_forgets_the_previous_stream(detector: SileroVad) -> None:
    """Without it the tail of an utterance is still speaking into the next
    segment: measured, the first frame of pure silence after a Korean sentence
    reads 0.96."""
    speech = _fixture("wake-and-question.wav")
    silence = bytes(FRAME_BYTES * 10)

    _probabilities(detector, speech)
    leaked = _probabilities(detector, silence)
    detector.reset()
    forgotten = _probabilities(detector, silence)

    assert max(leaked) > 0.5, "the leak this guards against must be real"
    assert max(forgotten) < 0.01


def test_reset_restores_the_starting_state_exactly(detector: SileroVad) -> None:
    """Not approximately. Half a reset - state cleared, context kept, or the other
    way round - would still shift the head of every segment."""
    first = _probabilities(detector, _fixture("wake-alone.wav"))
    detector.reset()
    second = _probabilities(detector, _fixture("wake-alone.wav"))

    assert first == second


def test_a_wrong_frame_length_raises(detector: SileroVad) -> None:
    """The protocol says so, and the graph will not: it accepts several wrong
    widths and answers nonsense, so this check is the only thing standing between
    a caller's arithmetic mistake and a wake gate that silently never fires."""
    for length in (0, FRAME_BYTES - 1, FRAME_BYTES + 1, FRAME_BYTES * 2):
        with pytest.raises(ValueError, match=str(FRAME_BYTES)):
            detector.probability(bytes(length))


def test_it_is_cheap_enough_to_run_forever(detector: SileroVad) -> None:
    """daemon/voice/base.py justifies the gate with 0.155 ms per 32 ms frame. The
    bound here is 30x that and still 6x under real time, because a slow CI box
    must not fail this - but a version that woke a thread pool per frame would."""
    frames = _frames(_fixture("wake-and-question.wav"))
    for frame in frames:  # warm: the first call pays for lazily built kernels
        detector.probability(frame)
    detector.reset()

    started = time.perf_counter()
    for frame in frames:
        detector.probability(frame)
    per_frame_ms = (time.perf_counter() - started) * 1_000 / len(frames)

    assert per_frame_ms < 5.0


# --- a text-only install ------------------------------------------------------


BLOCK_ONNXRUNTIME = """
import sys
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name == "onnxruntime":
            raise ImportError("No module named 'onnxruntime'")
        return None
sys.meta_path.insert(0, Blocker())
import daemon.voice.vad as vad
print(vad.FRAME_SAMPLES)
"""


def test_importing_the_module_does_not_need_onnxruntime() -> None:
    """Voice dependencies live in the `voice` extra, so a text-only install has no
    onnxruntime - and this module is importable from code that runs whether or not
    voice is on. Subprocess with the import blocked, because faking it in-process
    would keep passing if someone moved the import to the top of the module."""
    done = subprocess.run(
        [sys.executable, "-c", BLOCK_ONNXRUNTIME],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(FRAME_SAMPLES)


def test_a_missing_onnxruntime_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "onnxruntime":
            raise ImportError("No module named 'onnxruntime'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(AudioUnavailable, match="pip install -e") as caught:
        vad._onnxruntime()

    assert ".[voice]" in str(caught.value)


def test_a_missing_model_file_is_reported_as_an_incomplete_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The model is committed, so this is not a runtime condition - it is what a
    packaging mistake looks like, and it has to name the path it looked at."""
    monkeypatch.setattr(vad, "MODEL_PATH", tmp_path / "silero_vad.onnx")
    with pytest.raises(AudioUnavailable, match="silero_vad.onnx"):
        SileroVad()
