# tests/test_face_page.py
"""Guards on the page's load-bearing details.

These are not a substitute for looking at it - "does it feel alive" is the actual
requirement and no assertion reaches it (Task 8). They exist because three specific
mistakes were made while building this by hand, each invisible in code review and
obvious on screen: no decoder priming flashes black on the first switch, fading both
elements at once dips to the page background, and driving animation from rAF stalls in
a background window.
"""

import re
from pathlib import Path

import pytest

PAGE = Path("daemon/static/face.html").read_text(encoding="utf-8")


def _without_line_comments(js):
    """Drop `// ...` line comments so a guard answers to the code, not to prose
    that happens to mention the same identifier the guard is checking for."""
    return "\n".join(line.split("//", 1)[0] for line in js.splitlines())


def test_every_clip_element_is_muted_and_inline():
    # An unmuted <video> cannot autoplay, and without playsinline iOS Safari takes
    # the page over full-screen.
    assert PAGE.count("playsinline") >= 1
    assert PAGE.count("muted") >= 1


def test_clips_are_primed_to_a_decoded_frame():
    assert "PRIME" in PAGE or "prime" in PAGE
    # Scoped to prime()'s own body: show() also calls .pause() as part of the
    # crossfade, so checking the whole page would let that unrelated call stand in
    # for the priming-specific one this test is actually about.
    start = PAGE.index("async function prime(stem) {")
    end = PAGE.index("\n}\n", start)
    body = PAGE[start:end]
    assert "pause()" in body, "priming is play-then-pause; without the pause it plays"


def test_the_crossfade_is_single_sided():
    # The outgoing element must be hidden AFTER the fade, never faded alongside it.
    # >= 2, not >= 1: the constant's declaration alone would satisfy a plain
    # membership check even if nothing ever used it to delay the hide.
    assert PAGE.count("FADE_MS") >= 2, (
        "FADE_MS must be declared AND used to delay hiding the outgoing element"
    )
    assert "zIndex" in PAGE, "the incoming element has to be on top to fade in over"


def test_the_page_never_requests_anything_off_host():
    for bad in ("http://", "https://", "//cdn", "fonts.googleapis"):
        assert bad not in PAGE, f"{bad!r} would make the face need the internet"


def test_reduced_motion_is_honoured():
    assert "prefers-reduced-motion" in PAGE


@pytest.mark.parametrize("stem", ["idle1", "idle2", "idle3", "flourish_arms"])
def test_the_page_knows_the_clip_vocabulary(stem):
    assert stem in PAGE


def test_speaking_falls_back_to_idle_when_both_speaking_clips_are_missing():
    # Static-text guard, not a behavioural harness: this suite is pure Python and CI
    # installs no JS runtime (.github/workflows/ci.yml has no node step), so actually
    # running clipFor() with a missing-clip manifest is out of reach without adding a
    # dependency the project does not otherwise carry (see task-3-report.md). This
    # pins the code shape instead - spec 3.7's chain is speaking_loud -> speaking_soft
    # -> idle, and the speaking branch must reach the same idle fallback every other
    # activity already uses (FOR_ACTIVITY[act]) rather than returning null and leaving
    # whatever clip was already on screen.
    start = PAGE.index('if (act === "speaking")')
    end = PAGE.index("\n  }\n", start)
    branch = _without_line_comments(PAGE[start:end])
    assert "FOR_ACTIVITY[act]" in branch, (
        "speaking must fall through to FOR_ACTIVITY[act] (-> idle1) when neither "
        "speaking clip exists, per spec 3.7's ...-> idle"
    )


def test_the_wait_is_bounded_and_generalised_beyond_idle():
    # spec 3.2: idle/listening/thinking/working may wait - up to WAIT_MS - for the
    # outgoing clip to reach its own loop end (its neutral pose) instead of cutting
    # it mid-gesture. Scoped to toActivity()'s own body, comments stripped, so a
    # mention of WAIT_MS elsewhere (its own top-level declaration) or in prose can't
    # stand in for the mechanism actually living here.
    start = PAGE.index("function toActivity(act) {")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert "WAIT_ACTIVITIES" in body, (
        "the wait must be conditioned on the activity being wait-eligible, not "
        "hardcoded to idle alone"
    )
    assert "WAIT_MS" in body, "the wait must be bounded, not open-ended"
    assert "showing.loop = false" in body, (
        "letting the outgoing clip reach its own end relies on disabling its loop "
        "so it fires `ended` instead of seeking back to 0"
    )

    # The pool itself, scoped to its own declaration, must actually name the three
    # activities the wait was extended to - not just idle, which is where it began.
    start = PAGE.index("WAIT_ACTIVITIES = new Set([")
    end = PAGE.index("]", start)
    pool = _without_line_comments(PAGE[start:end])
    for act in ("listening", "thinking", "working"):
        assert f'"{act}"' in pool, f"{act} should be wait-eligible per the generalised spec 3.2"


def test_a_one_shot_can_only_be_cut_by_speaking():
    # spec 3.6: an in-flight one-shot runs to completion unless the activity
    # changes to speaking (the mouth cannot lag the audio) - idle, listening,
    # thinking and working have no such claim and must be held, not dropped or
    # applied mid-gesture. Scoped to toActivity()'s own body, comments stripped.
    start = PAGE.index("function toActivity(act) {")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert "oneShotUntil" in body and 'act !== "speaking"' in body, (
        "blocking a switch must be conditioned on a one-shot being in flight AND "
        "the target activity not being speaking"
    )
    assert "queuedActivity = act" in body, (
        "a blocked activity change must be remembered, not silently dropped"
    )

    # And advance() must actually apply the queued activity once the one-shot
    # finishes, not just leave the state declared and unread.
    start = PAGE.index("function advance() {")
    end = PAGE.index("\n}\n", start)
    adv_body = _without_line_comments(PAGE[start:end])
    assert "queuedActivity" in adv_body and "toActivity(" in adv_body, (
        "advance() must apply a queued activity once the one-shot actually ends"
    )


def test_the_fade_is_split_into_a_fast_and_a_slow_duration():
    # spec 3.2: speaking's mouth-snap and the mood one-shots keep the original fast
    # fade, but a mismatched pose (listening/thinking/working, and the idle return)
    # needs longer to read as deliberate rather than a cut.
    fast = re.search(r"const FADE_MS = (\d+)", PAGE)
    slow = re.search(r"const FADE_SLOW_MS = (\d+)", PAGE)
    assert fast and slow, "both a fast and a slow fade constant must be declared"
    assert int(slow.group(1)) > int(fast.group(1)), "the slow fade must actually be slower"

    # The activities that get it, scoped to the pool's own declaration so a stray
    # mention of e.g. "thinking" elsewhere on the page can't satisfy this.
    start = PAGE.index("SLOW_FADE_ACTIVITIES = new Set([")
    end = PAGE.index("]", start)
    pool = _without_line_comments(PAGE[start:end])
    for act in ("listening", "thinking", "working", "idle"):
        assert f'"{act}"' in pool, f"{act} should get the slow fade per spec 3.2"

    # And toActivity must actually consult that pool and the slow constant, not
    # just declare them elsewhere unused.
    start = PAGE.index("function toActivity(act) {")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert "FADE_SLOW_MS" in body and "SLOW_FADE_ACTIVITIES" in body, (
        "toActivity must select FADE_SLOW_MS via SLOW_FADE_ACTIVITIES, not just declare them"
    )


def test_playback_recovers_from_an_external_pause_or_a_rejected_switch():
    # Static-text guard only, and it stops there: this suite has no JS runtime (no
    # node step in .github/workflows/ci.yml), so it cannot actually pause a clip or
    # reject a play() and observe recovery - that would need a DOM/JS harness this
    # repo does not carry. This pins that the recovery machinery exists and is wired
    # to the two triggers that were measured live to matter, not that it works.
    assert "ensurePlaying" in PAGE

    start = PAGE.index("async function prime(stem) {")
    end = PAGE.index("\n}\n", start)
    prime_body = _without_line_comments(PAGE[start:end])
    assert '"pause"' in prime_body and "ensurePlaying" in prime_body, (
        "an unexpected pause on the showing clip must trigger recovery"
    )

    start = PAGE.index('addEventListener("visibilitychange"')
    end = PAGE.index("});", start)
    visibility_body = _without_line_comments(PAGE[start:end])
    assert "ensurePlaying" in visibility_body, "becoming visible again must trigger recovery"

    start = PAGE.index("function ensurePlaying() {")
    end = PAGE.index("\n}\n", start)
    heal_body = _without_line_comments(PAGE[start:end])
    assert ".play()" in heal_body, "recovery must actually attempt to resume playback"


# --- the lip-synced mouth ---------------------------------------------------
# Static-text guards, and they stop there for the reason the file already says: this
# suite has no JS runtime (no node step in .github/workflows/ci.yml), so it can pin
# the shape of the mechanism and not that it works. What it works or fails at is a
# window with a face in it, which is where §8 of the lip-sync design puts the pass
# mark anyway.


def _body(signature):
    """A named function's own body, comments stripped, so a guard answers to the code
    rather than to prose that happens to mention the same identifier."""
    start = PAGE.index(signature)
    return _without_line_comments(PAGE[start : PAGE.index("\n}\n", start)])


def test_the_mouth_is_an_overlay_that_is_allowed_to_die():
    # The failure this exists for: the renderer can latch `failed` mid-sentence and
    # then publish nothing. An <img> fed a stream that stopped keeps showing its last
    # frame and fires no event, so a page that treated lip-sync as a mode it had
    # switched into would hold a frozen mouth over a face that is still talking.
    assert "function mouthReady()" in PAGE

    # The per-frame staleness clock this used to require is gone with the transport.
    # An <img> fed multipart/x-mixed-replace gives the page no per-frame event to time
    # from - Chrome fires `load` once for the whole stream - so there is nothing to age.
    # What replaces it is narrower and does not need one: the overlay is gated on
    # `activity === "speaking"` from the activity stream, and the renderer only draws
    # while speaking.
    stream = _body("function mouthStream() {")
    assert "onerror" in stream, "the end of the stream is what falls back"
    assert "mouthDead" in stream, "giving up has to be recorded, or nothing falls back"
    assert 'mouth.src = "/face/frames"' in stream, (
        "the browser has to be the one streaming it; a page assigning data: URIs per "
        "frame is the base64 path this replaced"
    )


def test_a_dead_mouth_falls_back_to_the_v1_speaking_clip():
    # clipFor() is the one place the fallback can live: tick() already re-asks it every
    # frame while speaking, so flipping mouthDead crossfades to speaking_soft through
    # the existing single-sided crossfade rather than through a second code path.
    start = PAGE.index('if (act === "speaking")')
    branch = _without_line_comments(PAGE[start : PAGE.index("\n  }\n", start)])
    assert "mouthReady()" in branch and "mouthClip" in branch, (
        "the driving clip has to be the speaking clip while lip-sync is alive - the "
        "crops are that clip's own pixels, so speaking_soft under them is a different head"
    )
    assert "speaking_soft" in branch, (
        "and the v1 chain must still be there underneath, unreachable only while the "
        "mouth is alive"
    )


def test_the_driving_clip_is_never_rate_modulated():
    # v1's mouth IS playbackRate (spec 3.4). Left on while lip-sync drives the clip it
    # slides the pose under the crop away from the pose the renderer composited into
    # it, which is a seam at the crop border rather than a mouth.
    body = _body("function tick() {")
    assert "mouthReady()" in body and "playbackRate" in body, (
        "tick() must pin the driving clip to 1.0x while the lip-synced mouth is the "
        "speaking path, not modulate it as v1 does"
    )


def test_the_crop_is_not_taken_off_the_screen_by_rAF():
    # Measured, by getting it wrong first: with the visibility toggle inside tick(),
    # killing the renderer left the overlay stuck mid-fade at opacity 0.13 and it never
    # came off - rAF does not throttle in an occluded window, it stops. Header note 3
    # tolerates that for playbackRate because being late there is cosmetic; being late
    # here is a frozen mouth over a talking face, which is the whole failure the
    # fallback exists for. So the crop's visibility must be event-driven.
    assert "refreshMouth" not in _body("function tick() {"), (
        "the overlay's visibility must not depend on rAF running"
    )
    # No timer here any more: with the browser streaming the image there is no
    # per-frame callback to re-arm one from, and nothing to age out. Everything that
    # can change the answer still has to say so, which is what the loop below checks -
    # and that set now carries the whole job rather than sharing it with a watchdog.
    # And every other place the answer can change has to say so, or the crop outlives
    # the state that justified it: a mood one-shot replacing the driving clip, the
    # activity leaving speaking, and becoming visible after rAF was stopped.
    for where in ("function toActivity(act) {", "function playOneShot(stem) {",
                  "function advance() {"):
        assert "refreshMouth" in _body(where), f"{where} must refresh the overlay"
    start = PAGE.index('addEventListener("visibilitychange"')
    handler = _without_line_comments(PAGE[start : PAGE.index("});", start)])
    assert "refreshMouth" in handler, (
        "becoming visible is when a stale overlay has to be corrected"
    )


def test_the_frame_stream_is_only_opened_when_the_daemon_offers_one():
    # /face/manifest answers `false` for all three ways there is no mouth (switch off,
    # no renderer, latched failed), so the page never has to infer it from a failed
    # request - and must not open a connection that is going to 503.
    body = _body("async function boot() {")
    assert "manifest.lipsync" in body, "the switch is read from the manifest, not guessed"
    assert "mouthStream()" in body and "videoWidth" in body, (
        "the stream opens only behind the manifest AND a driving clip that actually "
        "decoded - there is nothing to lay a crop over otherwise"
    )


def test_the_rendered_frame_needs_no_placing_at_all():
    """This replaced a test that checked the crop was scaled through object-fit's own
    letterboxing, which was the right check for the wrong design.

    Two versions of this page laid a crop of the mouth over the playing clip, and both
    drew a visible rectangle across the head: a JPEG has no alpha, so its edge is hard
    however small the crop is, and the clip underneath is never on the frame the
    renderer drew. The payload is the whole composited frame now, so it takes the same
    geometry as the clips it replaces and there is nothing left to line up - which is
    the property to pin, because re-introducing any positioning maths would mean the
    crop is back.
    """
    assert "layoutMouth" not in PAGE, "positioning the overlay means it is a crop again"
    assert "mouthBox" not in PAGE, "and so does carrying a box to position it with"
    start = PAGE.index("#mouth {")
    css = PAGE[start : PAGE.index("}", start)]
    assert "inset: 0" in css and "object-fit: contain" in css, (
        "the frame has to be drawn exactly where the clips are, by the same fit"
    )


def test_the_mouth_overlay_cannot_fall_behind_a_clip():
    # show() hands each incoming clip ++z, so a fixed z-index on a sibling of #stage
    # would eventually lose. #stage carrying its own z-index makes it a stacking
    # context, which contains that counter for good.
    assert "#stage { position: fixed; inset: 0; z-index: 0; }" in PAGE
    assert "#mouth" in PAGE and "z-index: 1" in PAGE
