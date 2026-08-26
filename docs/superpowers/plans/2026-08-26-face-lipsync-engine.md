# 립싱크 엔진과 렌더러 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MuseTalk 립싱크의 모델 레이어와 렌더 루프를 만든다 — 오디오와 구동 클립에서 완성 프레임이 나오는 데까지.

**Architecture:** 모델은 `LipsyncEngine` 프로토콜 뒤에만 산다. 링 관리·프레임 인덱스·합성은 그 밖에 있어 모델 없이 테스트된다. 전처리는 오프라인 스텝이고 런타임에 얼굴 검출기가 없다.

**Tech Stack:** MLX (UNet·TAESD), mlx-whisper, macOS Vision (전처리만), numpy, opencv

**Spec:** [docs/superpowers/specs/2026-08-26-face-lipsync-design.md](../specs/2026-08-26-face-lipsync-design.md)

## Global Constraints

- **런타임은 전부 MLX.** `torch`, `diffusers`, `transformers`를 런타임 경로에서 import하지 않는다. 전처리는 `pyobjc` Vision을 쓴다.
- **`daemon/face_lipsync*.py`는 daemon 구현을 import하지 않는다** (비협상 4). `app.py`만 조립한다.
- **`daemon/voice/` 아래를 건드리지 않는다.** 한 줄도.
- **CI에 GB급 가중치가 들어가지 않는다.** 모델을 만지는 검증은 `evals/`에 두고 손으로 돌린다.
- **프로즌 파일 변경 없음**: `daemon/tasks.py`, `daemon/memory/schema.sql`, 프로토콜 파일.
- **드라이브 클립은 24fps, 1080×1620.** 프레임 예산 41.67ms.
- **오디오 윈도는 10 인덱스 = 200ms** (과거 **최소** 120ms / 미래 **최대** 80ms), 인덱스당 20ms.
  둘 다 주기적이다(주기 12프레임) — 미래는 61.67~80.0ms를, 과거는 120.0~138.33ms를 순환하고
  (200ms 총량에서 서로를 뺀 나머지라 미래가 최대일 때 과거가 최소다), 지연 예산에는 미래
  쪽 최대치를 쓴다 — 과거 오디오는 이미 도착해 있어 대기 시간에 들어가지 않는다.
- 코드와 주석은 영어, 설계 문서는 한국어 (docs/CONTRACTS.md, Style).

---

## File Structure

| 파일 | 책임 |
|---|---|
| `daemon/face_lipsync/__init__.py` | `LipsyncEngine` 프로토콜, `Cache` 데이터클래스. 여기만 밖에서 import한다 |
| `daemon/face_lipsync/loader.py` | safetensors → MLX UNet. 키 매핑, transpose 금지 |
| `daemon/face_lipsync/taesd.py` | TAESD 디코더 (MLX) |
| `daemon/face_lipsync/audio.py` | mel → whisper 인코더 → 프레임별 (50, 384) 청크. 인덱스 산술 |
| `daemon/face_lipsync/engine.py` | 셋을 묶은 `MlxEngine(LipsyncEngine)` |
| `daemon/face_lipsync/ring.py` | PCM 링과 프레임 인덱스. 모델 없음, 순수 로직 |
| `daemon/face_lipsync/render.py` | 렌더 루프, 합성, JPEG 슬롯 |
| `scripts/face_lipsync_prepare.py` | 오프라인 전처리 (Vision 랜드마크 → 캐시) |
| `tests/test_face_lipsync_audio.py` | 인덱스 산술 — 룩어헤드가 구조적이라 여기가 가장 중요하다 |
| `tests/test_face_lipsync_ring.py` | 링과 프레임 인덱스 |
| `tests/test_face_lipsync_render.py` | 가짜 엔진으로 렌더러 |
| `tests/test_face_lipsync_loader.py` | 키 매핑 (가중치 없이, 순수 함수) |
| `evals/face_lipsync_numerics.py` | 실제 가중치로 수치 검증. 손으로 |

---

### Task 1: 오디오 인덱스 산술

립싱크의 유일한 구조적 상수가 여기 있다. 룩어헤드 80ms는 GPU를 바꿔도 줄지 않으므로,
이 함수가 틀리면 지연 예산 전체가 틀린다. 모델 없이 완전히 테스트된다.

**Files:**
- Create: `daemon/face_lipsync/audio.py`
- Test: `tests/test_face_lipsync_audio.py`

**Interfaces:**
- Consumes: 없음
- Produces: `window_for(frame_index: int, fps: float) -> tuple[int, int]` — 미패딩 whisper 인덱스 `[first, last]` (양끝 포함). `latest_audio_ms(frame_index, fps) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_face_lipsync_audio.py
"""The audio window, which is where the latency budget actually comes from.

MuseTalk v1.5 reads 10 whisper indices per frame at 20ms each - 120ms of past and
80ms of future. That 80ms is the one number a faster GPU cannot reduce, so it is
pinned here rather than left to the model code to imply.
"""

import pytest

from daemon.face_lipsync.audio import latest_audio_ms, window_for


def test_window_is_ten_indices():
    first, last = window_for(600, 24.0)
    assert last - first + 1 == 10


@pytest.mark.parametrize(
    "frame,expected_lookahead",
    [(24, 80.0), (600, 80.0), (0, 80.0)],
)
def test_steady_state_lookahead_is_80ms(frame, expected_lookahead):
    """Audio needed beyond the frame's own start time."""
    ahead = latest_audio_ms(frame, 24.0) - frame * 1000.0 / 24.0
    assert ahead == pytest.approx(expected_lookahead, abs=7.0)


def test_the_first_frames_ask_for_audio_before_zero():
    """At a turn's start there is no past, and the caller must clamp rather than
    index negatively - the model repeats the edge feature, which is why the first
    ~200ms degrades to a neutral mouth instead of crashing."""
    first, _ = window_for(0, 24.0)
    assert first < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_lipsync_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'daemon.face_lipsync'`

- [ ] **Step 3: Write minimal implementation**

```python
# daemon/face_lipsync/audio.py
"""Whisper feature windowing, copied from MuseTalk v1.5's own arithmetic.

`musetalk/utils/audio_processor.py:get_whisper_chunk` left-pads the feature array
by `ceil(50/fps) * padding_left` and then slices 2*(left+right+1) indices from
`floor(frame * 50/fps)`. Reproducing the padding here rather than the padded array
keeps every index in one coordinate system - the unpadded one, where negative means
"before the audio started" and the caller clamps.
"""

from __future__ import annotations

import math

AUDIO_FPS = 50
"""Whisper encoder frames per second. One index is 20ms."""

MS_PER_INDEX = 1000.0 / AUDIO_FPS

PADDING_LEFT = 2
PADDING_RIGHT = 2
"""MuseTalk v15 defaults (`--audio_padding_length_left/right`)."""

WINDOW = 2 * (PADDING_LEFT + PADDING_RIGHT + 1)
"""10 indices = 200ms."""


def window_for(frame_index: int, fps: float) -> tuple[int, int]:
    """Inclusive `[first, last]` unpadded whisper indices for one video frame.

    Negative `first` is normal at a turn's start and means the caller must clamp.
    """
    multiplier = AUDIO_FPS / fps
    left_pad = math.ceil(multiplier) * PADDING_LEFT
    start = math.floor(frame_index * multiplier) - left_pad
    return start, start + WINDOW - 1


def latest_audio_ms(frame_index: int, fps: float) -> float:
    """The newest audio timestamp this frame needs, in ms from the turn's start."""
    _, last = window_for(frame_index, fps)
    return (last + 1) * MS_PER_INDEX
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_lipsync_audio.py -v`
Expected: PASS (3 tests, one parametrised 3 ways)

- [ ] **Step 5: Commit**

```bash
git add daemon/face_lipsync/ tests/test_face_lipsync_audio.py
git commit -m "face: the audio window, where the 80ms lookahead actually comes from"
```

---

### Task 2: UNet 로더 — transpose 함정

`mlx-community/MuseTalk-1.5-fp16`은 키 이름이 diffusers 스타일이지만 레이아웃은 이미
MLX(NHWC)다. `mlx-examples`의 `map_unet_weights`를 그대로 쓰면 transpose가 두 번 들어가
**조용히 망가진다.** 키 매핑을 순수 함수로 떼어 가중치 없이 테스트한다.

**Files:**
- Create: `daemon/face_lipsync/loader.py`
- Test: `tests/test_face_lipsync_loader.py`

**Interfaces:**
- Consumes: 없음
- Produces: `rename(key: str) -> str`, `needs_split(key: str) -> bool`, `unet_config(cfg: dict) -> dict` — mlx-examples `UNetConfig` 생성자 인자.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_face_lipsync_loader.py
"""Key mapping for the pre-converted MLX weights.

The published file is diffusers-keyed but MLX-laid-out: `conv_in.weight` is
(320, 3, 3, 8) where torch's is (320, 8, 3, 3). Transposing it again produces a
model that runs and returns nonsense, so the mapping here renames and splits but
never touches layout - and that is asserted rather than commented.
"""

from daemon.face_lipsync.loader import needs_split, rename, unet_config


def test_attention_projections_are_renamed():
    assert rename("down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight").endswith("query_proj.weight")
    assert rename("mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.weight").endswith("out_proj.weight")


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_lipsync_loader.py -v`
Expected: FAIL with `ImportError: cannot import name 'rename'`

- [ ] **Step 3: Write minimal implementation**

```python
# daemon/face_lipsync/loader.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_lipsync_loader.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add daemon/face_lipsync/loader.py tests/test_face_lipsync_loader.py
git commit -m "face: map MuseTalk's UNet onto MLX without transposing it twice"
```

---

### Task 3: PCM 링과 프레임 인덱스

렌더 루프에서 모델을 뺀 나머지 전부. 순수 로직이라 완전히 테스트되고, 슬롯이 큐가 아니라
덮어쓰기라는 성질이 여기서 고정된다.

**Files:**
- Create: `daemon/face_lipsync/ring.py`
- Test: `tests/test_face_lipsync_ring.py`

**Interfaces:**
- Consumes: `daemon.face_lipsync.audio.window_for`
- Produces: `PcmRing(sample_rate, width, seconds)` with `.feed(chunk, audible_at)`, `.window(frame_index, fps, origin) -> np.ndarray`; `Slot()` with `.put(bytes)`, `.get() -> bytes | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_face_lipsync_ring.py
"""The audio ring and the frame slot.

Both encode a rule from the design. The ring clamps rather than indexes negatively,
because a turn's first frames legitimately ask for audio from before the turn began.
The slot overwrites rather than queues, for the same reason `level` does on the bus:
a window that stopped consuming should resume at the current mouth, not replay a
backlog of stale ones.
"""

import numpy as np

from daemon.face_lipsync.ring import PcmRing, Slot


def _silence(ms: int, rate: int = 24_000) -> bytes:
    return b"\x00\x00" * int(rate * ms / 1000)


def test_a_window_before_the_start_is_clamped_not_negative():
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    ring.feed(_silence(500), audible_at=0.0)
    got = ring.window(frame_index=0, fps=24.0, origin=0.0)
    assert len(got) > 0
    assert not np.isnan(got).any()


def test_the_window_grows_no_longer_than_ten_indices():
    ring = PcmRing(sample_rate=24_000, width=2, seconds=4.0)
    ring.feed(_silence(2000), audible_at=0.0)
    got = ring.window(frame_index=48, fps=24.0, origin=0.0)
    assert len(got) == int(24_000 * 0.200)


def test_the_slot_keeps_only_the_latest_frame():
    slot = Slot()
    slot.put(b"first")
    slot.put(b"second")
    assert slot.get() == b"second"
    assert slot.get() == b"second"


def test_an_empty_slot_reports_nothing_rather_than_blocking():
    assert Slot().get() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_lipsync_ring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'daemon.face_lipsync.ring'`

- [ ] **Step 3: Write minimal implementation**

```python
# daemon/face_lipsync/ring.py
"""A rolling PCM buffer and a latest-frame slot.

`SpeechClock` already stamps every chunk with the moment it becomes audible, so
this stores audio on that timeline rather than on arrival time - which is what lets
a frame index be turned into a sample offset without knowing anything about queues.
"""

from __future__ import annotations

import threading

import numpy as np

from daemon.face_lipsync.audio import MS_PER_INDEX, WINDOW, window_for


class PcmRing:
    """Recent PCM, addressed by the time it is heard rather than when it arrived."""

    def __init__(self, *, sample_rate: int, width: int, seconds: float) -> None:
        self._rate = sample_rate
        self._width = width
        self._max = int(sample_rate * seconds)
        self._samples = np.zeros(0, dtype=np.int16)
        self._start = 0.0
        """Audible time of `self._samples[0]`."""

    def feed(self, chunk: bytes, audible_at: float) -> None:
        block = np.frombuffer(chunk, dtype=np.int16)
        if self._samples.size == 0:
            self._start = audible_at
        self._samples = np.concatenate([self._samples, block])
        if self._samples.size > self._max:
            drop = self._samples.size - self._max
            self._samples = self._samples[drop:]
            self._start += drop / self._rate

    def window(self, *, frame_index: int, fps: float, origin: float) -> np.ndarray:
        """The 200ms the model reads for `frame_index`, as float32 in -1..1.

        Clamped at both ends: a turn's first frames ask for audio from before it
        began, and the newest frames may outrun what has arrived.
        """
        first, _ = window_for(frame_index, fps)
        begin_s = origin + first * MS_PER_INDEX / 1000.0
        need = int(self._rate * WINDOW * MS_PER_INDEX / 1000.0)
        offset = int((begin_s - self._start) * self._rate)
        lo = max(0, offset)
        hi = min(self._samples.size, offset + need)
        got = self._samples[lo:hi] if hi > lo else self._samples[:0]
        out = np.zeros(need, dtype=np.float32)
        if got.size:
            at = max(0, -offset)
            out[at : at + got.size] = got.astype(np.float32) / 32768.0
        return out


class Slot:
    """One frame, latest wins. Never queues - see the module docstring."""

    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: bytes | None = None

    def put(self, frame: bytes) -> None:
        with self._lock:
            self._value = frame

    def get(self) -> bytes | None:
        with self._lock:
            return self._value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_lipsync_ring.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add daemon/face_lipsync/ring.py tests/test_face_lipsync_ring.py
git commit -m "face: the pcm ring clamps, and the frame slot overwrites"
```

---

### Task 4: 엔진 프로토콜과 렌더러

모델이 사는 유일한 자리를 프로토콜로 고정하고, 렌더러를 가짜 엔진으로 테스트한다.
합성은 스파이크가 픽셀 동일함을 확인한 numpy 크롭박스 방식이다.

**Files:**
- Create: `daemon/face_lipsync/__init__.py`, `daemon/face_lipsync/render.py`
- Test: `tests/test_face_lipsync_render.py`

**Interfaces:**
- Consumes: `PcmRing`, `Slot`
- Produces: `LipsyncEngine` Protocol with `mouths(audio, frame_indices) -> list[np.ndarray]`; `Cache` dataclass with `frames, boxes, crop_boxes, masks`; `composite(frame, mouth, box, crop_box, mask, out) -> np.ndarray`; `Renderer(engine, cache, ring, slot)` with `.render(frame_index, origin, fps)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_face_lipsync_render.py
"""The renderer, with no model in sight.

Everything that decides how the face looks - which driving frame, which audio, how
the mouth is blended in - lives outside the engine protocol, so a fake that returns
a flat colour exercises all of it. That is the whole reason the protocol exists:
CI must never touch a gigabyte of weights.
"""

import numpy as np

from daemon.face_lipsync import Cache, composite
from daemon.face_lipsync.render import Renderer
from daemon.face_lipsync.ring import PcmRing, Slot


class FakeEngine:
    """Returns a solid colour per requested frame, and records what it was asked."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def mouths(self, audio, frame_indices):
        self.calls.append(list(frame_indices))
        return [np.full((256, 256, 3), 200, np.uint8) for _ in frame_indices]


def _cache(n=4):
    # The box must sit inside the crop box - MuseTalk expands the face box by 1.5x
    # to get the blend region, and the mask is the size of that larger box.
    frames = np.zeros((n, 200, 160, 3), np.uint8)
    masks = np.full((n, 80, 80), 255, np.uint8)
    return Cache(
        frames=frames,
        boxes=[(40, 40, 100, 100)] * n,        # 60x60
        crop_boxes=[(30, 30, 110, 110)] * n,   # 80x80
        masks=masks,
    )


def test_composite_only_touches_the_crop_box():
    cache = _cache()
    frame = cache.frames[0].copy()
    mouth = np.full((60, 60, 3), 200, np.uint8)
    out = composite(frame, mouth, cache.boxes[0], cache.crop_boxes[0], cache.masks[0])
    assert out[0, 0].tolist() == [0, 0, 0]          # outside the crop box
    assert out[60, 60].tolist() == [200, 200, 200]  # inside it


def test_rendering_publishes_one_frame_to_the_slot():
    slot = Slot()
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    Renderer(engine=engine, cache=_cache(), ring=ring, slot=slot).render(
        frame_index=0, origin=0.0, fps=24.0
    )
    assert slot.get() is not None


def test_a_failing_engine_does_not_take_the_renderer_down():
    """The design says a mid-stream failure falls back to v1 clips and logs once -
    which it can only do if the frame that failed does not propagate."""

    class Broken:
        def mouths(self, audio, frame_indices):
            raise RuntimeError("weights went away")

    slot = Slot()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    r = Renderer(engine=Broken(), cache=_cache(), ring=ring, slot=slot)
    r.render(frame_index=0, origin=0.0, fps=24.0)   # must not raise
    assert slot.get() is None
    assert r.failed is True


def test_the_driving_clip_cycles_rather_than_running_out():
    slot = Slot()
    engine = FakeEngine()
    ring = PcmRing(sample_rate=24_000, width=2, seconds=2.0)
    ring.feed(b"\x00\x00" * 24_000, audible_at=0.0)
    r = Renderer(engine=engine, cache=_cache(n=4), ring=ring, slot=slot)
    r.render(frame_index=9, origin=0.0, fps=24.0)
    assert engine.calls[-1][0] < 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_face_lipsync_render.py -v`
Expected: FAIL with `ImportError: cannot import name 'Cache'`

- [ ] **Step 3: Write minimal implementation**

```python
# daemon/face_lipsync/__init__.py
"""The lip-sync render path: the model boundary, the clip cache, and compositing.

Nothing here imports anything else in `daemon/` (CONTRACTS 4). `daemon/app.py`
builds a renderer and injects it; no other module knows this package exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class LipsyncEngine(Protocol):
    """The only place a model lives.

    Everything else - which driving frame, which audio, how the mouth is blended -
    sits outside this, so the renderer is testable without weights.
    """

    def mouths(
        self, audio: np.ndarray, frame_indices: Sequence[int]
    ) -> list[np.ndarray]:
        """256x256 BGR mouths, one per index, in the same order."""
        ...


@dataclass(frozen=True, slots=True)
class Cache:
    """One driving clip, prepared offline. `frames` is memory-mapped in production."""

    frames: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    crop_boxes: list[tuple[int, int, int, int]]
    masks: np.ndarray


def composite(
    frame: np.ndarray,
    mouth: np.ndarray,
    box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    mask: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Blend one mouth into one driving frame, touching only the crop box.

    MuseTalk's own `get_image_blending` round-trips the whole frame through PIL
    twice for the same pixels; this was measured bit-identical at a fraction of the
    cost, which matters because the per-frame budget is 41.67ms.
    """
    x1, y1, x2, y2 = box
    xs, ys, xe, ye = crop_box
    if out is None:
        out = frame.copy()
    elif out is not frame:
        np.copyto(out, frame)
    orig = frame[ys:ye, xs:xe].astype(np.float32)
    pasted = frame[ys:ye, xs:xe].copy()
    pasted[y1 - ys : y2 - ys, x1 - xs : x2 - xs] = mouth
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    out[ys:ye, xs:xe] = (
        pasted.astype(np.float32) * alpha + orig * (1.0 - alpha)
    ).round().astype(np.uint8)
    return out
```

```python
# daemon/face_lipsync/render.py
"""Turn one audio window plus one driving frame into one JPEG."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from daemon.face_lipsync import Cache, LipsyncEngine, composite
from daemon.face_lipsync.ring import PcmRing, Slot

logger = logging.getLogger(__name__)

JPEG_QUALITY = 85


class Renderer:
    """One frame at a time. The loop that calls this lives in `daemon/app.py`."""

    def __init__(
        self,
        *,
        engine: LipsyncEngine,
        cache: Cache,
        ring: PcmRing,
        slot: Slot,
    ) -> None:
        self._engine = engine
        self._cache = cache
        self._ring = ring
        self._slot = slot
        self._buffer = np.empty_like(cache.frames[0])
        self.failed = False
        """Latched on the first engine failure. The caller drops back to v1 clips
        and logs once; retrying per frame would fill the log at 24Hz."""

    def render(self, *, frame_index: int, origin: float, fps: float) -> None:
        """Render `frame_index` and publish it. Never raises."""
        if self.failed:
            return
        try:
            self._render(frame_index=frame_index, origin=origin, fps=fps)
        except Exception:
            logger.exception("face: lip-sync engine failed, falling back to clips")
            self.failed = True

    def _render(self, *, frame_index: int, origin: float, fps: float) -> None:
        n = len(self._cache.boxes)
        i = frame_index % n
        audio = self._ring.window(frame_index=frame_index, fps=fps, origin=origin)
        mouth = self._engine.mouths(audio, [i])[0]
        x1, y1, x2, y2 = self._cache.boxes[i]
        sized = cv2.resize(mouth, (x2 - x1, y2 - y1))
        out = composite(
            self._cache.frames[i],
            sized,
            self._cache.boxes[i],
            self._cache.crop_boxes[i],
            self._cache.masks[i],
            out=self._buffer,
        )
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            self._slot.put(buf.tobytes())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_face_lipsync_render.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add daemon/face_lipsync/ tests/test_face_lipsync_render.py
git commit -m "face: a renderer with the model behind a protocol, so tests need no weights"
```

---

### Task 5: 실제 가중치 수치 검증 (eval, 손으로)

CI가 만질 수 없는 유일한 검증. 스파이크가 코사인 0.999999를 얻은 것과 같은 확인을
제품 로더로 반복한다 — 로더가 transpose를 잘못 걸면 여기서 잡힌다.

**Files:**
- Create: `evals/face_lipsync_numerics.py`
- Modify: `evals/CLAUDE.md` (표에 한 줄)

**Interfaces:**
- Consumes: `daemon.face_lipsync.loader.unet_config`, `rename`, `needs_split`
- Produces: 없음 (손으로 읽는 출력)

- [ ] **Step 1: Write the eval**

```python
# evals/face_lipsync_numerics.py
"""Does the product loader produce the model the spike measured?

The published MLX weights are diffusers-keyed but MLX-laid-out, so a loader that
transposes them computes nonsense quickly and looks like a pass. This asserts the
one thing CI cannot: that every parameter finds a home and the output matches a
reference recorded from the verified PyTorch path.

    python3 -m evals.face_lipsync_numerics

Needs the weights under `<data_dir>/face/lipsync/models/`. Never run in CI.
"""

import sys

import json
import sys

import mlx.core as mx

from daemon.face_lipsync.loader import needs_split, rename, unet_config


def main() -> int:
    root = "data/face/lipsync/models"
    print("loading", flush=True)
    with open(f"{root}/musetalk.json") as handle:
        config = unet_config(json.load(handle))
    print(f"  in_channels={config['in_channels']} "
          f"cross_attention_dim={config['cross_attention_dim'][0]} "
          f"heads={config['num_attention_heads']}")
    weights = mx.load(f"{root}/unet.safetensors")
    mapped = []
    for key, value in weights.items():
        if needs_split(key):
            a, b = mx.split(value, 2)
            mapped.append((rename(key).replace("ff.net.0.proj", "linear1"), a))
            mapped.append((rename(key).replace("ff.net.0.proj", "linear2"), b))
        else:
            mapped.append((rename(key), value))
    print(f"  {len(weights)} tensors -> {len(mapped)} arrays")

    # The check that matters: shapes must stay NHWC. A double transpose shows up
    # here before it shows up as a blurry mouth.
    conv_in = dict(mapped)["conv_in.weight"]
    if tuple(conv_in.shape) != (320, 3, 3, 8):
        print(f"FAIL conv_in.weight is {tuple(conv_in.shape)}, expected (320, 3, 3, 8)")
        return 1
    print("  layout OK (NHWC, not transposed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it and read the output**

Run: `python3 -m evals.face_lipsync_numerics`
Expected: the tensor counts (686 -> 718) and `layout OK`. A non-zero exit means the loader is wrong.

- [ ] **Step 3: Record the run in `evals/CLAUDE.md`**

Add one row to the table near the top:

```markdown
| `face_lipsync_numerics.py` | whether the product loader builds the model the spike measured - the published weights are diffusers-keyed but MLX-laid-out, and a second transpose is silent |
```

- [ ] **Step 4: Commit**

```bash
git add evals/face_lipsync_numerics.py evals/CLAUDE.md
git commit -m "face: an eval for the one thing CI cannot check about the loader"
```

---

## 이 계획에 없는 것

스펙의 구현 순서 4~6단계 — **`FaceBus`의 PCM sink**, `app.py` 배선, `/face/frames` 전송, 매니페스트, 페이지 전환,
설정 플래그, admin 토글, 열화 경로 — 는 **두 번째 계획**이다. 1~3단계가 끝나야 알 수 있는
것 위에 서 있기 때문이다: TAESD를 MLX로 옮겼을 때 프레임당 여유가 2.5%에서 얼마로
돌아오는지, 그리고 Vision 랜드마크가 MuseTalk의 박스 공식에 실제로 대응되는지.

또한 **TAESD MLX 구현**과 **Vision 전처리 스크립트**는 이 계획에 코드가
없다. 둘 다 참조 구현을 보면서 써야 하는 이식 작업이라, 바이트-사이즈 TDD 단계로 미리
적으면 계획이 거짓이 된다. Task 1~5가 끝나면 그 둘을 각각 하나의 계획으로 다시 쓴다.
