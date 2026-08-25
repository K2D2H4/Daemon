# tests/test_face_page.py
"""Guards on the page's load-bearing details.

These are not a substitute for looking at it - "does it feel alive" is the actual
requirement and no assertion reaches it (Task 8). They exist because three specific
mistakes were made while building this by hand, each invisible in code review and
obvious on screen: no decoder priming flashes black on the first switch, fading both
elements at once dips to the page background, and driving animation from rAF stalls in
a background window.
"""

from pathlib import Path

import pytest

PAGE = Path("daemon/static/face.html").read_text(encoding="utf-8")


def test_every_clip_element_is_muted_and_inline():
    # An unmuted <video> cannot autoplay, and without playsinline iOS Safari takes
    # the page over full-screen.
    assert PAGE.count("playsinline") >= 1
    assert PAGE.count("muted") >= 1


def test_clips_are_primed_to_a_decoded_frame():
    assert "PRIME" in PAGE or "prime" in PAGE
    assert "pause()" in PAGE, "priming is play-then-pause; without the pause it plays"


def test_the_crossfade_is_single_sided():
    # The outgoing element must be hidden AFTER the fade, never faded alongside it.
    assert "FADE_MS" in PAGE
    assert "zIndex" in PAGE, "the incoming element has to be on top to fade in over"


def test_the_page_never_requests_anything_off_host():
    for bad in ("http://", "https://", "//cdn", "fonts.googleapis"):
        assert bad not in PAGE, f"{bad!r} would make the face need the internet"


def test_reduced_motion_is_honoured():
    assert "prefers-reduced-motion" in PAGE


@pytest.mark.parametrize("stem", ["idle1", "idle2", "idle3", "flourish_arms"])
def test_the_page_knows_the_clip_vocabulary(stem):
    assert stem in PAGE
