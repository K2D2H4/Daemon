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
    # dependency the project does not otherwise carry. This pins the code shape
    # instead - spec 3.7's chain is speaking_loud -> speaking_soft
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


def test_the_wait_is_for_the_next_neutral_moment_and_is_bounded():
    # Task 9's 3rd follow-up: idle/listening/thinking/working may wait for the
    # outgoing clip's next near-neutral moment (not only its own end) before
    # cutting, capped so a clip that never gets there does not stall the page.
    # Scoped to toActivity()'s own body, comments stripped, so a mention of
    # these names elsewhere (their own top-level declarations) can't stand in
    # for the mechanism actually living here.
    start = PAGE.index("function toActivity(act) {")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert "WAIT_ACTIVITIES" in body, (
        "the wait must be conditioned on the activity being wait-eligible, not "
        "hardcoded to idle alone"
    )
    assert "neutralWaitMs(" in body, (
        "the wait duration must come from the neutral-moment lookup, not a fixed "
        "or hardcoded delay"
    )
    assert "setTimeout(" in body, (
        "waiting for a moment that is not necessarily the clip's own end cannot "
        "rely on the native `ended` event - it needs its own timer"
    )

    # The bound itself lives in neutralWaitMs(), scoped to its own body: it must
    # actually be used to cap the returned wait, not just declared unused.
    start = PAGE.index("function neutralWaitMs(video) {")
    end = PAGE.index("\n}\n", start)
    wait_body = _without_line_comments(PAGE[start:end])
    assert "NEUTRAL_WAIT_CAP_MS" in wait_body, "the wait must be bounded, not open-ended"
    assert "Math.min(" in wait_body, (
        "NEUTRAL_WAIT_CAP_MS must actually cap the computed wait, not just be "
        "declared nearby and left unused"
    )
    assert "transitions.neutral" in wait_body, (
        "the wait must be driven by the neutral-moment lookup the table provides, "
        "not guessed or hardcoded"
    )

    # The pool itself, scoped to its own declaration, must actually name the three
    # activities the wait was extended to - not just idle, which is where it began.
    start = PAGE.index("WAIT_ACTIVITIES = new Set([")
    end = PAGE.index("]", start)
    pool = _without_line_comments(PAGE[start:end])
    for act in ("listening", "thinking", "working"):
        assert f'"{act}"' in pool, f"{act} should be wait-eligible per the generalised spec 3.2"


def test_a_mood_can_only_be_cut_by_speaking_but_a_flourish_yields_to_anything():
    # The spec contradicted itself here - 3.2 said an in-flight one-shot is cut
    # only by speaking, 3.6 said any activity change cuts a flourish - and this
    # page cited 3.6 for the opposite of what 3.6 says. Settled as a split, and
    # both sections now say it: a MOOD is the daemon saying something and only
    # speaking (whose mouth cannot lag the audio) may cut it; a FLOURISH is
    # decoration with no trigger and no meaning, so holding `listening` behind
    # six seconds of it is wrong. Scoped to toActivity()'s own body, comments
    # stripped.
    start = PAGE.index("function toActivity(act) {")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert "oneShotUntil" in body and 'act !== "speaking"' in body, (
        "blocking a switch must be conditioned on a one-shot being in flight AND "
        "the target activity not being speaking"
    )
    assert "!oneShotIsFlourish" in body, (
        "a flourish must not block an activity change the way a mood does"
    )
    assert "queuedActivity = act" in body, (
        "a blocked activity change must be remembered, not silently dropped"
    )

    # And the flourish has to be *marked* as one where it is started, or the
    # exemption above can never fire.
    start = PAGE.index("function tick(now) {")
    end = PAGE.index("\n}\n", start)
    tick_body = _without_line_comments(PAGE[start:end])
    assert "{ flourish: true }" in tick_body, (
        "the random idle flourish must tell playOneShot() what it is"
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


def test_ensure_playing_defers_to_a_pending_neutral_wait():
    # ensurePlaying() must not pre-empt a pending neutral-moment wait
    # (pendingWaitTimer). toActivity() sets `activity` to the wait's target
    # *before* scheduling its timer, so ensurePlaying()'s own clipFor(activity)
    # lookup already agrees with what the wait is waiting for - without this
    # guard it would force an immediate switch out from under the wait, at the
    # fast fade instead of the wait's own chosen one, and skip the neutral
    # moment entirely. Live, not theoretical: visibilitychange calls this
    # every time an ambient window is occluded and shown again.
    start = PAGE.index("function ensurePlaying() {")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert "!pendingWaitTimer" in body, (
        "the forced-switch branch must be skipped while a wait is pending"
    )


def test_a_loop_entry_is_looked_up_not_hardcoded_to_frame_zero():
    # Task 9: show() must seek to the pose-matched entry time instead of always
    # frame 0. Scoped to show()'s own body so a stray "currentTime = 0" elsewhere
    # on the page (prime()'s own decoder-priming reset, which must stay frame 0)
    # cannot stand in for this.
    start = PAGE.index("function show(stem")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert "currentTime = 0" not in body, (
        "show() must not hardcode frame 0 - poseMatchTime() is what falls back "
        "to 0 when there is no table or no matching entry"
    )
    assert "poseMatchTime(" in body, "show() must call the pose-match lookup"

    # And prime()'s own reset - a different concern (decode one frame, hold it)
    # - must be untouched: the brief says change one thing in show(), not this.
    start = PAGE.index("async function prime(stem) {")
    end = PAGE.index("\n}\n", start)
    prime_body = _without_line_comments(PAGE[start:end])
    assert "currentTime = 0" in prime_body, "priming must still hold at frame 0"


def test_the_transitions_table_is_fetched_alongside_the_manifest_at_boot():
    start = PAGE.index("async function boot() {")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert "/face/transitions" in body, (
        "the pose-match table must be fetched at boot, same as the manifest"
    )
    assert "transitions =" in body, "the fetched table must actually be kept, not discarded"


def test_pose_match_lookup_falls_back_to_frame_zero_when_absent():
    # Rule 4: no table, or a clip missing from it, must silently produce 0 -
    # never throw and never leave currentTime unset. Scoped to the lookup
    # function's own body.
    start = PAGE.index("function poseMatchTime(")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert "!transitions" in body, "a missing table must be checked for explicitly"
    assert "return 0" in body, "the fallback must be 0, not undefined or a thrown error"

    # And it must actually key the lookup off both the outgoing and incoming
    # clip's names, using the table's own bucket size (rule 2) - not a bare
    # constant, since face_match.py's BUCKET is what the table was built with.
    assert "prev.dataset.stem" in body and "stem" in body, (
        "the lookup must be keyed on both the outgoing and incoming clip"
    )
    assert "transitions.bucket" in body, (
        "bucketing must use the table's own bucket size, not a hardcoded one"
    )


# --- whole-branch review: what the page tests could not see -------------------
#
# All of the guards below are code-shape checks, scoped to one function body with
# comments stripped, and that is a real limit rather than a preference: this suite
# is pure Python and CI installs no JS runtime (.github/workflows/ci.yml has no
# node step), so none of them can observe the behaviour they are about. Each was
# checked by hand against a stubbed-DOM harness under node - every one of these
# seven reproduced as a real defect there and stopped reproducing after the fix -
# and each assertion below goes red if the single line it names is deleted. That
# is the most they do.


def test_the_deferred_hide_checks_the_element_is_still_outgoing():
    # show() schedules the outgoing element's hide at fadeMs. Switch away and back
    # inside that window and the stale timer hides the element that is now
    # `showing`, leaving BOTH layers transparent - a flat #19191B page until
    # something switches to a different element again. With tools at the default
    # mode=full a read_file round finishes well inside 400ms, so
    # thinking -> working -> thinking hits it.
    start = PAGE.index("function show(stem")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert "prev === showing" in body, (
        "the deferred hide must check `prev` is still the outgoing element before "
        "hiding it"
    )


def test_the_self_heal_does_not_restart_a_clip_that_has_ended():
    # `pause` fires before `ended` per the HTML spec, so a clip reaching its own
    # end arrives at the pause listener looking exactly like a browser-imposed
    # pause - and play() at currentTime == duration replays it from 0, underneath
    # the crossfade `ended` is about to start.
    start = PAGE.index("async function prime(stem) {")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert "!v.ended" in body, (
        "the pause listener must exclude a clip that ended, not just one that was "
        "paused"
    )


def test_advance_clears_a_pending_wait_and_rearms_the_flourish():
    # Two separate defects in one function, both about state that outlives the
    # clip advance() is replacing. The wait is scheduled against the OUTGOING
    # clip's timeline and idle clips are shown with loop:false, so it survives
    # that clip ending and fires up to NEUTRAL_WAIT_CAP_MS later over whatever
    # replaced it. And scheduleFlourish() used to be reachable only from
    # toActivity(), so entering idle armed exactly one flourish ever: tick()
    # zeroes flourishAt when it fires and nothing set it again - against spec 1's
    # one every 40-120s, which is what breaks an 8s loop's period.
    start = PAGE.index("function advance() {")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert "clearPendingWait()" in body, (
        "a wait timed against the outgoing clip must not outlive it"
    )
    assert "scheduleFlourish()" in body, (
        "returning to an idle loop must arm the next flourish, or idle gets one "
        "flourish per entry rather than one every 40-120s"
    )


def _onmessage_body():
    """The SSE handler's own body, comments stripped."""
    start = PAGE.index("es.onmessage = (m) => {")
    end = PAGE.index("\n  };", start)
    return _without_line_comments(PAGE[start:end])


def test_a_queued_activity_cannot_go_stale():
    # toActivity() does not update `activity` when it queues a change behind a
    # mood, so comparing an incoming event against `activity` alone let a change
    # back and forth during the one-shot apply the stale half of it afterwards.
    assert "queuedActivity ?? activity" in _onmessage_body(), (
        "the incoming activity must be compared against the queued one when there "
        "is a queued one"
    )


# Envelope measured off the running resident's /face/stream, one live 8-turn voice
# session, this page's own ATTACK/RELEASE replayed over the captured levels. The
# thresholds below are checked against these, not against taste.
ENV_MEDIAN, ENV_P90, ENV_MAX = 0.233, 0.272, 0.315


def _const(name):
    """The numeric value of a top-level `const NAME = <number>` in the page."""
    m = re.search(rf"\b{name}\s*=\s*([0-9.]+)", _without_line_comments(PAGE))
    assert m, f"{name} is gone from the page"
    return float(m.group(1))


def test_the_loud_threshold_is_reachable_by_the_signal_it_is_compared_against():
    """`speaking_loud` shipped unreachable: LOUD_ENTER was 0.62 against an envelope
    whose measured maximum is 0.315, so a rendered clip could never play once. The
    guard is the measurement, not a literal - any threshold above what the signal
    reaches is the same bug again, however it got there.
    """
    loud = _const("LOUD_ENTER")
    assert loud <= ENV_MAX, (
        f"LOUD_ENTER={loud} is above the measured envelope maximum {ENV_MAX} - "
        "speaking_loud can never engage"
    )
    assert loud > ENV_MEDIAN, (
        f"LOUD_ENTER={loud} is at or below the median {ENV_MEDIAN} - speaking_loud "
        "would be the normal case rather than the emphatic one"
    )


def test_an_idle_clip_ending_does_not_push_the_flourish_deadline_away():
    """Why `flourish_arms` never appeared, and it was not the interval.

    `advance()` runs on every idle clip's `ended`, and an idle clip is about 8s while
    the flourish wait is tens of seconds. Re-arming unconditionally there moved the
    deadline further out than the clock advanced, so it could never be reached -
    measured in a live page, flourishAt went 37.3s -> 73.1s while performance.now()
    went 7.1s -> 35.7s. Re-arm only when disarmed: entering idle arms it, and tick()
    zeroes it after one fires.
    """
    body = _without_line_comments(PAGE)
    m = re.search(r'if \(activity === "idle"([^)]*)\)\s*scheduleFlourish\(\)', body)
    assert m, "advance() no longer re-arms the flourish at all - tick() zeroes it once"
    assert "!flourishAt" in m.group(1), (
        "advance() re-arms the flourish unconditionally; an 8s idle clip ending then "
        "pushes the deadline past where the clock can reach it"
    )
    lo, hi = _const("FLOURISH_MIN_MS") / 1000, _const("FLOURISH_MAX_MS") / 1000
    assert lo < hi, "the flourish window has to be a window"


def test_the_loud_switch_reads_the_envelope_and_the_envelope_is_recorded():
    # LOUD_ENTER gated on the raw per-tick level, which spec 3.4 itself calls too
    # jittery to use directly - one sub-threshold 40ms tick reset the hold - so it
    # reads the smoothed envelope now. The threshold itself is now a measurement;
    # the two tests above hold it to that. logEnvelope() stays because one session
    # on one voice is what those numbers are, and the next move needs more.
    body = _onmessage_body()
    assert "env >= LOUD_ENTER" in body, (
        "the loud switch must read the smoothed envelope, not the raw level"
    )
    assert "logEnvelope()" in body, (
        "the envelope distribution must be recorded somewhere, or open question 3 "
        "stays unanswerable"
    )
    assert "envSamples.push(env)" in body, "a summary of nothing is not data"


def test_a_partial_set_with_no_idle_clip_still_explains_itself():
    # Spec 3.7 promises a partial set works. The hint only appeared when the set
    # was completely empty, so clips present but no idle clip rendered a flat
    # background with no explanation at all - boot's own toActivity("idle") has
    # nothing to show.
    start = PAGE.index("async function boot() {")
    end = PAGE.index("\n}\n", start)
    body = _without_line_comments(PAGE[start:end])
    assert 'clipFor("idle")' in body, (
        "the hint must also cover a set that has clips but no idle clip"
    )


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
    # What replaces it is narrower and does not need one: the overlay is gated on
    # `activity === "speaking"` from the activity stream, and the renderer only draws
    # while speaking.
    #
    # This used to check for the <img>'s `onerror`, which was the right check for a
    # transport the page no longer uses: presentation now has to be timed separately
    # from arrival (see test_the_frame_is_presented_on_a_clock_of_its_own), and an
    # <img> draws its own parts the moment they land. The property being pinned is
    # unchanged - the response ENDING is what falls back - so what moved is only where
    # the signal comes from: a rejected or exhausted fetch instead of an `error` event.
    stream = _body("function mouthStream() {")
    assert "finally" in stream, (
        "the response ending is the fallback signal whether it ended by failing or by "
        "running out, so the record of it cannot hang off the failure path alone"
    )
    assert "mouthDead" in stream, "giving up has to be recorded, or nothing falls back"
    assert "close()" in stream, (
        "and the frames still queued have to be released - an ImageBitmap is ~7MB and "
        "nothing will ever come to draw them"
    )
    assert "fetch" in stream or 'fetch("/face/frames")' in _body("async function readFrames() {"), (
        "the page has to be reading the stream itself now; it is the only way to "
        "decide WHEN each frame is drawn rather than let the transport decide"
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


def test_idle_does_not_rotate_while_lip_sync_is_alive():
    """The owner's report, in one line of code.

    `clipFor("idle")` picked at random out of idle1/idle2/idle3 and `clipFor("speaking")`
    could only ever answer `mouthClip` (idle2), so four times in five the first word of
    a reply crossfaded to a different head, a different pose and a different framing.
    "그냥 새로운 클립이 재생되는 느낌" was not a feeling about the render - a new clip
    was being played.

    Pinned inside the idle branch, comments stripped, so the prose above the code
    cannot satisfy it. And the branch has to keep its rotation underneath: a renderer
    that dies mid-session leaves `mouthReady()` false, and idle going back to three
    clips is the v1 face rather than a degraded one.
    """
    start = PAGE.index('if (act === "idle")')
    branch = _without_line_comments(PAGE[start : PAGE.index("\n  }\n", start)])
    assert "mouthReady()" in branch and "return mouthClip" in branch, (
        "while lip-sync drives a clip, idle has to BE that clip - anything else means "
        "speech begins by swapping the head"
    )
    assert "IDLES" in branch and "Math.random" in branch, (
        "and the rotation must still be there for a dead or absent renderer"
    )

    # The rotation used to be what ended an idle clip: show() left `loop` off for idle
    # so `ended` fired and advance() picked another. With nothing on the other side of
    # that `ended`, leaving it off parks the face on the last frame for good.
    body = _without_line_comments(_body("function loops(act) {"))
    assert 'act !== "idle"' in body and "mouthReady()" in body


def test_speech_begins_on_the_clip_that_is_already_playing():
    """The second half of the same report, and the one a matching clip still had.

    `toActivity` used to rewind the driving clip to frame 0 on `speaking`, because the
    renderer indexed that clip from 0 at every turn. So even idle2 -> idle2 - the one
    case where nothing needed to change - threw the playhead back to the top, which is
    a pose jump on its own. The renderer follows the page's free-running playhead now
    (`render.py:ClipClock`), so there is nothing left to move and `show()`'s own
    `next === showing` guard makes the whole handover a no-op.
    """
    body = _body("function toActivity(act) {")
    assert not re.search(r"currentTime\s*=[^=]", body), (
        "moving a playhead here is the pose jump this was reported as; the driving "
        "clip's is the daemon's, not this function's. Reading one is fine and the "
        "WAIT_MS window does - what must not come back is an assignment"
    )

    # show() still rewinds a clip it is genuinely switching TO - that one has no
    # position of its own - but must not rewind the driving clip, which does.
    body = _body("function show(stem, { loop = true, fadeMs = FADE_MS } = {}) {")
    assert "isDriver(stem) ? driverAt() : poseMatchTime(prev, stem)" in body, (
        "an ordinary clip starts at 0; the driving clip starts wherever the daemon's "
        "clock says it is, or the pose under the overlay is not the pose it was drawn "
        "for"
    )


def test_the_page_follows_the_daemons_clip_clock_rather_than_its_own():
    """Which frame the page is showing has one answer and the daemon owns it.

    It has to: the renderer needs that answer for a quarter-second in the FUTURE
    (`render.py:DISPLAY_LEAD`), which only a clock can give, and this page has no
    channel back to be asked on. So `/face/manifest` hands the position over once and
    the page anchors it to its own `performance.now()`.

    The stamp has to be taken when the manifest ARRIVES. Priming a dozen clips takes
    hundreds of milliseconds, and anchoring after them would put the page that many
    frames away from every pose the renderer draws into.
    """
    body = _body("async function boot() {")
    manifest_at = body.index("manifestAt = performance.now()")
    assert manifest_at < body.index("prime"), (
        "the position is as of the response, so it has to be stamped before priming, "
        "not after it"
    )
    assert "driverEpoch = manifestAt - (ls.position || 0)" in body, (
        "the anchor is the manifest's position against the page's own clock"
    )

    # And the one thing measured to move the <video> off that clock has to put it
    # back: between pauses a playing clip holds to performance.now() within 0.002
    # frames over 8s, and every pause costs 0.2-0.4 frames that never return.
    heal = _body("function ensurePlaying() {")
    assert "isDriver" in heal and "driverAt()" in heal, (
        "resuming a clip that is a quarter-second behind the pose the renderer is "
        "drawing into is a backgrounded tab coming back with the mouth on the wrong face"
    )


def test_the_overlay_waits_for_a_frame_belonging_to_this_turn():
    """`speaking` arrives at once; this turn's first rendered frame cannot for ~250ms.

    Whatever is on the canvas in between belongs to the LAST utterance - a pose from
    wherever that sentence ended - and fading it in is the same "a different clip
    started" by another route. So the overlay is gated on a frame having arrived, and
    the queue is dropped at the boundary so the frame that lifts the gate is this
    turn's rather than the last one's leftovers.
    """
    assert "function mouthLive() { return mouthReady() && mouthFresh; }" in PAGE

    turn = _body("function toActivity(act) {")
    assert "mouthFresh = false" in turn and "mouthQueue.shift().close()" in turn, (
        "a turn boundary has to reset the gate AND drop the last turn's frames, or "
        "the gate is lifted by a mouth for a sentence that has finished"
    )

    # Lifted from the decode callback and not from presentMouth(), because rAF stops
    # outright in an occluded window and a microtask does not - the same split
    # test_the_crop_is_not_taken_off_the_screen_by_rAF pins from the other side.
    decode = _body("function decodeFrame(bytes) {")
    assert "mouthFresh = true" in decode and "refreshMouth()" in decode


def test_the_driving_clip_is_never_rate_modulated():
    # v1's mouth IS playbackRate (spec 3.4). Left on while lip-sync drives the clip it
    # slides the pose under the crop away from the pose the renderer composited into
    # it, which is a seam at the crop border rather than a mouth.
    body = _body("function tick(now) {")
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
    #
    # tick() does now DRAW the lip-synced frames, and that is the opposite case rather
    # than a loosening of this one: a frame drawn late is a cosmetic lag in a window
    # nobody is looking at, and rAF stopping leaves the frame that was already up. The
    # two must stay separate functions, which is what the pair of assertions below pins.
    body = _body("function tick(now) {")
    assert "refreshMouth" not in body, (
        "the overlay's visibility must not depend on rAF running"
    )
    assert "presentMouth" in body, (
        "and the frames themselves must, or presentation is back on the socket's clock"
    )
    # No timer here any more: with the browser streaming the image there is no
    # per-frame callback to re-arm one from, and nothing to age out. Everything that
    # can change the answer still has to say so, which is what the loop below checks -
    # and that set now carries the whole job rather than sharing it with a watchdog.
    # And every other place the answer can change has to say so, or the crop outlives
    # the state that justified it: a mood one-shot replacing the driving clip, the
    # activity leaving speaking, and becoming visible after rAF was stopped.
    for where in ("function toActivity(act) {",
                  "function playOneShot(stem, { flourish = false } = {}) {",
                  "function advance() {"):
        assert "refreshMouth" in _body(where), f"{where} must refresh the overlay"
    start = PAGE.index('addEventListener("visibilitychange"')
    handler = _without_line_comments(PAGE[start : PAGE.index("});", start)])
    assert "refreshMouth" in handler, (
        "becoming visible is when a stale overlay has to be corrected"
    )


def test_the_frame_is_presented_on_a_clock_of_its_own():
    """Measured in a real browser against the assembled stack, and the reason the
    overlay is a <canvas> rather than an <img>.

    Fed straight into an <img>, the frames were repainted whenever a part ARRIVED, and
    arrival is a socket event: median gap 43.1ms but p90 61.4ms, a quarter of the gaps
    under 25ms and a quarter over 60ms. The rate was fine - 23.8fps on screen - and the
    cadence was not: 37.4% of frames changed duration by a whole 17ms from the frame
    before, and 21% stayed up longer than a source frame lasts, to 83ms. The <video>
    the page plays at idle never exceeded 41.7ms once. Same rate, different regularity,
    and that is what got reported as the frame rate dropping when speech starts.

    So arrival must not be allowed to decide presentation. Three things carry that and
    none of them is inferable from the others: something to draw into on the page's own
    schedule, a queue between the socket and the draw, and a due time that the draw
    waits for.

    Re-measuring this needs a real browser and there is no eval for it - Playwright is
    not a dependency of this project and adding one for a page guard was not thought
    worth it. Two traps if you write one anyway. Fingerprinting the presented frame by
    drawImage()-ing the element into a scratch canvas is accurate for an <img> and NOT
    for a <canvas> or a <video>: those are GPU-backed, and the copy invented 1939
    transitions out of 968 frames. Read a canvas through its OWN context's
    getImageData, take a video's cadence from requestVideoFrameCallback, and check
    every method against the arrival count, because nothing can be presented more
    often than it arrived. And check the renderer's own fps first: under machine load
    it fell to 15fps here, which looks exactly like a page that got worse.
    """
    assert "<canvas id=\"mouth\">" in PAGE, (
        "an <img> repaints itself when a part lands; only a canvas lets the page pick "
        "the moment"
    )
    body = _body("function presentMouth(now) {")
    assert "mouthQueue" in body, "arrival has to be able to run ahead of presentation"
    assert "mouthDueAt" in body and "MOUTH_PERIOD_MS" in body, (
        "and presentation has to be paced by the renderer's frame period rather than "
        "by whenever the socket happened to deliver"
    )
    assert "drawImage" in body, "something has to actually reach the screen"
    # Nothing may be discarded on the way: the complaint is about cadence, and a fix
    # that evened it out by throwing frames away would be a real drop in frame rate
    # rather than an apparent one.
    assert "mouthQueue.shift()" in body and "MOUTH_CATCH_UP" in body, (
        "a queue past MOUTH_CATCH_UP means this clock is behind the renderer's, and "
        "the way back has to be catching up, never dropping"
    )
    # Off the main thread. The thread that draws is the thread that runs the clip
    # crossfades, and a 180KB JPEG decoded on it 24 times a second is the cost the
    # multipart <img> was chosen over SSE to avoid in the first place.
    assert "createImageBitmap" in _body("function decodeFrame(bytes) {"), (
        "decoding has to stay off the thread that composites"
    )
    assert "data:image" not in PAGE, "and the base64 path must not come back"


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
