# Multi-clip lip-sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lip-sync renders onto every clip the face plays, every clip plays from frame 0
to its end without interruption, and the server owns which clip is up.

**Architecture:** The engine's only clip-dependent state is `MlxEngine._latents` (~2-3MB
per clip), so one engine serves all ten and switching is a tensor selection rather than
a reload. A `Driver` bundles (name, `Cache`, `ClipClock`); `Renderer.switch()` swaps it
and resets the per-clip continuity that a turn boundary already resets. Clip choice
moves out of `daemon/static/face.html` into a pure server module assembled by
`daemon/app.py` (CONTRACTS 4), because the renderer composites onto the frame it
believes is on screen and cannot be told that a round trip late. **A wanted clip is
queued and applied at the current clip's own end** — never mid-clip. The page becomes a
viewer of `/face/frames`; its v1 policy stays untouched as the fallback for a renderer
that latches `failed`.

**Tech Stack:** MLX (UNet + TAESD + whisper encoder), numpy/cv2 on the CPU tail,
FastAPI + `multipart/x-mixed-replace`.

**Spec:** [docs/superpowers/specs/2026-08-26-face-lipsync-design.md](../specs/2026-08-26-face-lipsync-design.md)
and [docs/superpowers/specs/2026-08-25-face-design.md](../specs/2026-08-25-face-design.md).
[ADR 0017](../../adr/0017-the-neutral-moment-not-the-matched-pose.md) governs the v1
fallback path and is **not** reopened for it; Task 6 records why its mechanisms do not
apply once lip-sync is driving.

## Global Constraints

- **CONTRACTS 4.** `daemon/face_lipsync/` imports nothing from `daemon/`; only
  `daemon/app.py` imports an implementation. The new policy module must be importable
  by `app.py` and must not import `face_lipsync`.
- **CI is ubuntu.** `mlx` has no Linux wheel; no module CI imports may import it at
  module level. `opencv-python-headless>=4.10,<5` — the `<5` bound is load-bearing.
- **Frame budget 41.67ms.** The model half is 35.93ms/frame and is the ceiling; the CPU
  tail is 8.43ms/frame. Nothing in this plan adds to either.
- **One-shots enter at frame 0, always** (`face_match.py:ONE_SHOTS`): each is an arc
  from a neutral pose and back, and a mid-arc entry destroys the arc.
- **Judgement is the owner's eye.** Interframe RMSE in the lip region and Laplacian
  sharpness have both run *opposite* to their own ranking. Report interval medians
  against 41.67ms, never one run's fps.
- **Ten caches exist** under `<data_dir>/face/lipsync/`: idle1, idle2, idle3, listening,
  thinking, working, amused, sulky, curious, flourish_arms. `speaking_loud` and
  `speaking_soft` have none, deliberately — Task 3.

---

## The measurement this plan is built on

Downscaled whole-frame mean absolute difference across the join, over the owner's real
ten prepared clips. **The baseline is a clip's own loop point** — the join the face has
always made, every few seconds, that the owner has never remarked on.

| join | min | median | p90 | max | over baseline max (2.14) |
|---|---|---|---|---|---|
| a clip's own loop point (**baseline**) | 0.88 | 1.14 | — | 2.14 | — |
| **clip end → next clip frame 0** | 1.08 | **1.41** | — | 2.18 | **3 / 90 (3%)** |
| a "near-neutral" moment → one-shot frame 0 | 0.94 | 1.51 | 5.06 | 8.95 | 85 / 252 (**34%**) |
| any moment → one-shot frame 0 (cut now) | 1.40 | 7.98 | 12.54 | 12.82 | 76 / 84 (90%) |

Three things follow, and they are why this plan looks the way it does.

**Playing every clip to its end is smooth by construction.** The end→0 join sits on top
of the baseline, and only three of ninety pairs exceed the baseline's own worst case, by
0.01–0.04. No crossfade, no pose-match table and no neutral wait are needed to achieve
it — those exist to make a *mid-clip* switch survivable, and this design has none.

**Waiting for a "near-neutral moment" does not reliably buy smoothness *for a one-shot*.**
`face_match.py`'s neutral flag is measured against *each clip's own frame 0*, not a pose
shared across clips, so "neutral for `idle2`" does not imply "close to `amused`'s frame
0" — `idle2@1.75s` is flagged neutral and is 8.94 from it. The buckets are 0.5s (12
frames) and the flag means *some* frame in the slice is near neutral, not the frame the
cut lands on; the runtime only knows the bucket, so that imprecision is the mechanism's
and not the measurement's. `face_match.py`'s hub-and-spoke docstring reads stronger than
what the data supports. **Scope, because the table is narrower than that
heading:** both one-shot rows measure entry at frame 0, which is what a one-shot must
do. ADR 0017's loop-to-loop path is neutral-wait *then pose-matched* entry, and nothing
here measures it — 0017's own numbers still govern that half, and this plan does not
touch it.

**The cost is expression latency, and it was accepted knowingly.** A one-shot queued to
the clip end arrives after half a clip on average: 4.0s behind idle1/2/3, 3.1s behind
`listening`, 3.0s behind `thinking`. The owner chose that over a picture that snaps one
time in three, having been shown both numbers.

---

### Task 1: One engine, latents per clip

**Files:**
- Modify: `daemon/face_lipsync/engine.py` (`MlxEngine.__init__`, `mouths`, `load`)
- Modify: `daemon/face_lipsync/__init__.py` (the `LipsyncEngine` protocol)
- Test: `tests/test_face_lipsync_loader.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `MlxEngine.mouths(windows, frame_indices, *, clip: str)` and
  `load(..., latents: dict[str, Path], ...)`. `LipsyncEngine.mouths` gains the same
  keyword-only `clip`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_engine_is_told_which_clip_s_latents_to_use():
    """The UNet, TAESD and whisper weights are 1.6GB and clip-independent; the
    latents are 2-3MB and are the only per-clip tensor. One engine, ten latent
    sets - not ten engines."""
    import inspect
    from daemon.face_lipsync import LipsyncEngine

    sig = inspect.signature(LipsyncEngine.mouths)
    assert "clip" in sig.parameters
    assert sig.parameters["clip"].kind is inspect.Parameter.KEYWORD_ONLY
```

- [ ] **Step 2: Run it failing**

Run: `perl -e 'alarm 300; exec @ARGV' python3 -m pytest tests/test_face_lipsync_loader.py -k latents -q`
Expected: FAIL — `clip` is not a parameter.

- [ ] **Step 3: Implement**

`LipsyncEngine.mouths` gains `*, clip: str`. `MlxEngine.__init__` takes
`latents: dict[str, mx.array]`; `mouths` selects `self._latents[clip]`:

```python
        table = self._latents[clip]
        latents = mx.concatenate(
            [table[i % table.shape[0]][None] for i in frame_indices]
        )
```

`load()` takes `latents: dict[str, Path]` and `mx.load`s each. Keep the `DTYPE` cast at
the boundary — a fp32 array here drags the whole graph up (engine.py's own note).

- [ ] **Step 4: Tests pass**

Run: `perl -e 'alarm 300; exec @ARGV' python3 -m pytest tests/test_face_lipsync_loader.py tests/test_face_lipsync_render.py -q`

- [ ] **Step 5: Numeric check by hand, never in CI**

Extend `evals/face_lipsync_numerics.py` to load two clips' latents and assert a step
against `idle2` is bit-identical to the single-clip build's. Real weights, by hand.

- [ ] **Step 6: Commit**

```bash
git add daemon/face_lipsync/engine.py daemon/face_lipsync/__init__.py tests/test_face_lipsync_loader.py evals/face_lipsync_numerics.py
git commit -m "face: the engine's only per-clip tensor was the latents, so one engine serves ten clips"
```

---

### Task 2: `Driver`, and a renderer that can change clips

**Files:**
- Modify: `daemon/face_lipsync/render.py` (new `Driver`, `Renderer.__init__`, `switch`)
- Test: `tests/test_face_lipsync_render.py`

**Interfaces:**
- Consumes: Task 1's `mouths(..., clip=...)`.
- Produces: frozen `Driver(name: str, cache: Cache, clip: ClipClock)` and
  `Renderer.switch(driver: Driver) -> None`. `Renderer.__init__` takes `driver=` in
  place of `cache=` and `clip=`.

- [ ] **Step 1: Write the failing tests**

```python
def test_switching_the_driving_clip_restarts_the_motion_blend():
    """`_previous` is the last mouth encoded and `_blend` mixes it into the next
    one. Carried across a clip change it mixes a pose from a different head - the
    same discontinuity a turn boundary already resets for."""
    r = _renderer(engine=_Mouths([0, 200]), driver=_driver("idle2"))
    _encoded_mouth(r, r._driver.cache, None)
    assert r._previous is not None
    r.switch(_driver("listening"))
    assert r._previous is None
    assert r._smoothed is None, "the injection weight average belongs to one clip too"


def test_switching_refuses_a_cache_of_a_different_geometry():
    """`_buffer` is `np.empty_like(cache.frames[0])` and is reused. All ten prepared
    clips are 1080x1620 (measured); a cache that is not must fail loudly here
    rather than corrupt a composite."""
    r = _renderer(engine=_Mouths([0]), driver=_driver("idle2"))
    before = r._buffer.shape
    r.switch(_driver("listening"))
    assert r._buffer.shape == before
    with pytest.raises(ValueError, match="geometry"):
        r.switch(_driver("odd", height=720))
```

- [ ] **Step 2: Run them failing**

Run: `perl -e 'alarm 300; exec @ARGV' python3 -m pytest tests/test_face_lipsync_render.py -k switch -q`
Expected: FAIL — `Renderer` has no `switch`.

- [ ] **Step 3: Implement**

```python
@dataclass(frozen=True)
class Driver:
    """The clip being rendered onto: its name, its prepared frames, and where its
    playhead is. One object because these three are never independently correct - a
    cache with another clip's clock composites a mouth into the wrong pose."""

    name: str
    cache: Cache
    clip: ClipClock


    def switch(self, driver: Driver) -> None:
        """Render onto a different clip from here on.

        No crossfade, and that is a measurement rather than an omission: the caller
        only ever switches at a clip's own end, and end -> next clip's frame 0
        measures 1.41 median against the 1.14 of a clip's own loop point (see the
        plan's table). There is nothing for a fade to hide.

        Everything reset here is per-clip continuity, and each was a measured defect
        when it survived a turn boundary: `_previous` mixes the last mouth into the
        next (`_blend`), `_smoothed` averages the injection weight (`_weight`).
        """
        if driver.cache.frames[0].shape != self._driver.cache.frames[0].shape:
            raise ValueError(
                f"geometry changed: {driver.name} is {driver.cache.frames[0].shape} "
                f"against {self._driver.cache.frames[0].shape}"
            )
        self._driver = driver
        self._previous = None
        self._smoothed = None
        self._weight_at = -1
        self._continues_at = -1
```

- [ ] **Step 4: Tests pass**

Run: `perl -e 'alarm 300; exec @ARGV' python3 -m pytest tests/test_face_lipsync_render.py -q`

- [ ] **Step 5: Commit**

```bash
git add daemon/face_lipsync/render.py tests/test_face_lipsync_render.py
git commit -m "face: a clip change is a discontinuity, and the renderer had no way to say so"
```

---

### Task 3: The clip policy — queue it, apply it at the end

**Files:**
- Create: `daemon/face_clips.py`
- Test: `tests/test_face_clips.py` (create)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces:
  ```python
  LIPSYNC_CLIPS: frozenset[str]      # the ten with prepared caches
  ONE_SHOT_CLIPS: frozenset[str]     # amused, sulky, curious, flourish_arms
  def wanted(activity: str, *, pending_shot: str | None, current: str,
             available: frozenset[str], pick: Callable[[Sequence[str]], str]) -> str
  class ClipQueue:                   # .want(stem) ; .due(at) -> str | None
  ```

**Why a new module:** `face.py` is the bus and is imported by `face_routes.py`; policy
that owns a random pick and a queue does not belong on the event path. `app.py`
assembles both (CONTRACTS 4).

- [ ] **Step 1: Write the failing tests**

```python
def test_speaking_does_not_change_the_clip():
    """The owner's report from the live run: "발화할때 바로 클립이 idle 클립으로
    변경돼". With every clip driveable there is no reason to move - the mouth is
    generated for whichever clip is up, so speech begins where the face already is."""
    assert wanted("speaking", pending_shot=None, current="listening",
                  available=LIPSYNC_CLIPS, pick=_first) == "listening"
    assert wanted("speaking", pending_shot=None, current="thinking",
                  available=LIPSYNC_CLIPS, pick=_first) == "thinking"


def test_a_clip_is_never_cut_mid_way():
    """The whole design. A want registered at 2s into an 8.04s clip takes effect at
    8.04s, not at 2s - measured, end -> frame 0 is a baseline join and any mid-clip
    join is up to ten times worse."""
    q = ClipQueue(current="idle2", ends_at=8.04)
    q.want("listening")
    assert q.due(at=2.0) is None
    assert q.due(at=8.03) is None
    assert q.due(at=8.04) == "listening"


def test_the_last_want_before_the_boundary_wins():
    """Two activity changes inside one clip is ordinary - listening then thinking
    while she works out an answer. The face shows where it ended up, not a queue of
    poses it no longer holds."""
    q = ClipQueue(current="idle2", ends_at=8.04)
    q.want("listening")
    q.want("thinking")
    assert q.due(at=8.04) == "thinking"


def test_a_one_shot_outranks_an_activity_want_at_the_same_boundary():
    """A mood is the daemon saying something; an activity is ambient. If both are
    waiting at the boundary the expression goes first and the activity is still
    pending after it."""
    q = ClipQueue(current="idle2", ends_at=8.04)
    q.want("listening")
    q.want("amused", one_shot=True)
    assert q.due(at=8.04) == "amused"
    assert q.due(at=8.04 + 5.46) == "listening"


def test_speaking_loud_and_soft_are_not_driveable():
    """They are chosen by loudness, so driving them would swap the clip - and reset
    the mouth's continuity - every time the owner raised their voice. They stay as
    v1 fallback clips with no cache."""
    assert "speaking_loud" not in LIPSYNC_CLIPS
    assert "speaking_soft" not in LIPSYNC_CLIPS
```

- [ ] **Step 2: Run them failing**

Run: `perl -e 'alarm 300; exec @ARGV' python3 -m pytest tests/test_face_clips.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'daemon.face_clips'`.

- [ ] **Step 3: Implement**

Port `FOR_ACTIVITY`, the idle pool and the flourish interval out of
`daemon/static/face.html` verbatim where they are correct. **Three changes, all from the
live run and the table above:**

1. `wanted("speaking", ...)` returns `current` when it is driveable. The page returned
   the single driving clip, which is what made every utterance switch clips.
2. `LIPSYNC_CLIPS` excludes `speaking_loud`/`speaking_soft`.
3. No pose-match lookup and no neutral wait. Both exist to survive a mid-clip switch,
   and there are none.

`pick` is injected so the idle rotation is deterministic in tests.

- [ ] **Step 4: Tests pass**

Run: `perl -e 'alarm 300; exec @ARGV' python3 -m pytest tests/test_face_clips.py -q`

- [ ] **Step 5: Declare it reachable**

Add the module's classes and functions to `tests/test_reachable.py` so a policy built
and constructed by nothing fails the suite rather than sitting dead.

- [ ] **Step 6: Commit**

```bash
git add daemon/face_clips.py tests/test_face_clips.py tests/test_reachable.py
git commit -m "face: a clip runs to its own end, and what wants to follow it waits"
```

---

### Task 4: Assembly — every prepared clip, loaded lazily

**Files:**
- Modify: `daemon/app.py` (`_build_lipsync`, `_load_lipsync_cache`, `_lipsync_loop`,
  `_LipsyncFrames`, and `LIPSYNC_CLIP`, which goes)
- Test: `tests/test_face_lipsync_wiring.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: `_LipsyncFrames.clip` becomes a live read of the current driver's name;
  `app.state.face_frames` unchanged in shape.

- [ ] **Step 1: Write the failing tests**

```python
async def test_every_prepared_clip_is_offered_and_a_missing_one_is_simply_absent():
    """rule 4, as face_match.py already applies it: no interpolation and no
    substitution for a clip that was never prepared - only omission."""
    built = await _build(caches=["idle2", "listening"])
    assert set(built.drivers) == {"idle2", "listening"}


async def test_the_idle_jpegs_are_encoded_when_a_clip_is_first_driven():
    """Ten clips is 7.8GB of frames and ~240MB of pre-encoded JPEGs. The frames are
    `np.load(mmap_mode="r")` so they cost address space, not RSS; encoding every
    clip's JPEGs at boot would put seconds and 240MB into startup for clips a
    session may never reach."""
```

- [ ] **Step 2: Run them failing**

Run: `perl -e 'alarm 400; exec @ARGV' python3 -m pytest tests/test_face_lipsync_wiring.py -q`

- [ ] **Step 3: Implement**

`_build_lipsync` scans `<data_dir>/face/lipsync/*/` for the four cache artefacts, builds
one `Driver` per complete cache, and hands `{name: latents_path}` to `load()`. Delete
`LIPSYNC_CLIP`; the "one clip, not a choice per activity" reasoning in its docstring is
exactly what this plan reverses, so replace it with a constant naming where the caches
live. Keep the missing-files log line — it is the only thing that tells a human why the
mouth is absent — and have it name how many clips were found.

`_lipsync_loop` asks `face_clips` for the wanted clip once per publish tick and calls
`renderer.switch(driver)` only when `ClipQueue.due()` returns one, setting the new
`ClipClock`'s epoch to `now` so the clip starts at frame 0.

- [ ] **Step 4: The whole suite**

Run: `perl -e 'alarm 900; exec @ARGV' python3 -m pytest -q`
Expect one pre-existing failure: `tests/test_companion.py`'s expired-commitment test,
which is clock-dependent and reproduces on `origin/main`.

- [ ] **Step 5: Commit**

```bash
git add daemon/app.py tests/test_face_lipsync_wiring.py
git commit -m "face: ten prepared clips, one engine, and a loop that only moves at a boundary"
```

---

### Task 5: The page becomes a viewer

**Files:**
- Modify: `daemon/static/face.html`, `daemon/face_routes.py` (a `clip` event on
  `/face/stream`)
- Test: `tests/test_face_page.py`, `tests/test_face_routes.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_canvas_is_on_for_every_activity_while_lipsync_is_live():
    """The colour shift the owner saw return is a decoder change: Chrome's decode of
    the untagged mp4 against ours of the JPEGs, measured R+3.0 G+2.1 B+1.2.
    `d6eba3c` closed it for idle<->speaking by publishing idle as frames too; a real
    conversation crosses listening->speaking, where the canvas was off. Every clip
    being driveable is what closes the rest."""
    body = PAGE.read_text(encoding="utf-8")
    assert 'activity === "speaking" || activity === "idle"' not in body
    assert "mouthLive()" in body


def test_the_page_follows_the_server_s_clip_and_keeps_its_own_chain_as_fallback():
    body = PAGE.read_text(encoding="utf-8")
    assert '"clip"' in body, "the page has to be told which clip the frames are of"
    assert "FOR_ACTIVITY" in body, "and must still pick for itself when frames stop"
```

- [ ] **Step 2-4:** run failing, implement, run passing.

`clipFor`/`FOR_ACTIVITY`/`scheduleFlourish`/`poseMatchTime`/`neutralWaitMs` stay
**untouched** as the `mouthDead` fallback — ADR 0017 still governs that path, where the
clip carries the mouth. What changes is that `mouthReady()` routes clip choice to the
server's `clip` event and `refreshMouth()` drops its activity gate.

- [ ] **Step 5: Look at it in a browser, not only in text**

`tests/test_face_page.py` greps the body and cannot see a canvas that never paints. Run
`evals/face_lipsync_live.py` and look. **Check `pgrep -f face_lipsync_live` is 0 first**
— two MLX engines split the GPU and drop frames, which has happened twice.

- [ ] **Step 6: Commit**

---

### Task 6: The ADR

**Files:**
- Create: `docs/adr/0020-lip-sync-makes-a-clip-ambient.md`
- Modify: `docs/adr/README.md`, `daemon/face_match.py` (docstring correction only)

- [ ] **Step 1: Write it**

The decision: **lip-sync changes what a clip is for.** Without it the clip carries the
mouth, so it must be reactive — speech means switching to a speaking clip, and
[ADR 0017](0017-the-neutral-moment-not-the-matched-pose.md)'s pose matching and neutral
wait exist to make that mid-clip switch survivable. With lip-sync the clip is ambient
body motion, so it can be allowed to finish, and a boundary join needs neither
mechanism. Carry the plan's measurement table in verbatim; it is the whole argument.

Record the correction too, because a reader of `face_match.py` will otherwise trust it:
its `ONE_SHOTS` docstring says each one-shot is "an arc from the **shared** neutral pose
and back", but its `neutral` flag is computed against **each clip's own frame 0**. The
two are not the same claim, and the data separates them — `idle2@1.75s` is flagged
neutral and is 8.94 from `amused`'s frame 0, against a 2.14 baseline. ADR 0017's
conclusion for the fallback path is untouched; only this over-strong sentence is.

- [ ] **Step 2: `python3 scripts/check_docs.py`, then commit**

---

### Task 7: The live check, and what it may conclude

- [ ] **Step 1: Drive a real voice conversation**

Back the sqlite up with `sqlite3 .backup` first — `cp` loses the WAL. Stop the resident
(`launchctl bootout gui/$(id -u)/ai.daemon.default`), run the worktree build through
`Daemon.app`'s launcher so the TCC microphone grant applies, with `DAEMON_DATA_DIR`
pointing at the real data dir and `DAEMON_FACE_LIPSYNC_ENABLED=true`. The managed
interpreter needs the `face` extra; the framework `python3` has `mlx` but is the one
that steals the microphone grant. Restore the resident afterwards and **confirm
`wake_gate` frames are climbing** — a stale PortAudio device has left that gate
permanently deaf before.

- [ ] **Step 2: Ask the owner four things**

1. Does any transition read as a cut?
2. Does the picture still change brightness or saturation when she starts speaking?
3. Does the mouth still match on clips other than `idle2`?
4. Is an expression arriving 3-4s late actually acceptable in conversation? The number
   was accepted on paper; this is where it is judged in use.

- [ ] **Step 3: Report intervals, not fps**

Median gap against 41.67ms over the whole session. One run's fps is not a number: the
same code measured 23.2fps at load 3.5 and 17.7fps at 7.4, and max-gap swings 87-297ms
between runs.

---

## Self-review notes

- **Removed against the first draft of this plan:** the server-side crossfade
  (measured unnecessary — end→0 is a baseline join), the pose-match lookup and the
  neutral wait (both exist for mid-clip switches, which no longer happen). That is one
  task and two mechanisms deleted by one measurement.
- **What this plan does not touch:** `MOTION_BLEND`, `RELEASE_FRAMES`, the detail
  transfer, the audio window and the batching. All were ranked by eye on cv2 4.11, and
  moving one under this change would put two things under one judgement.
- **Known risk.** Nothing has rendered a mouth onto `sulky`, `curious` or
  `flourish_arms` in any path. `amused` was rendered offline and judged good by the
  owner (2026-08-31). Task 7 question 3 is where the other three are answered, and a NO
  there costs Task 3's `LIPSYNC_CLIPS` set, not the architecture.
- **Deferred, by the owner's own words ("나중에 고도화"):** choosing the cut moment by
  measuring the join at runtime rather than trusting a bucket flag. It would buy back
  the 3-4s expression latency, and it needs one-shots in the match table, which
  `face_match.py` excludes today.
