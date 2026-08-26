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
    start = PAGE.index("function tick() {")
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
LONGEST_IDLE_GAP_S = 36.0


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


def test_the_flourish_window_can_open_in_a_gap_that_actually_occurs():
    """`flourish_arms` was equally unreachable, for the mirror-image reason: the
    minimum wait was 40s and the longest uninterrupted idle stretch in a live
    session was 36s, so the window never opened at all.
    """
    lo, hi = _const("FLOURISH_MIN_MS") / 1000, _const("FLOURISH_MAX_MS") / 1000
    assert lo <= LONGEST_IDLE_GAP_S, (
        f"FLOURISH_MIN_MS={lo}s exceeds the longest measured idle stretch "
        f"{LONGEST_IDLE_GAP_S}s - flourish_arms can never fire"
    )
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
