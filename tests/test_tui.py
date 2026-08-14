"""The presentation layer, checked as strings.

Two of these tests are load-bearing beyond this file.

`test_nothing_coloured_*` protects every other suite that captures stdout: the
wizard's tests read its output as text, so one escape sequence leaking out of a
`Theme` built for a pipe would turn their assertions into assertions about ANSI.

The width tests protect Korean. `len()` and terminal columns agree for ASCII and
disagree for every Korean syllable, so a box drawn with `len()` looks correct in
this file's own English examples and visibly crooked the first time a Korean word
goes inside it. Every box here is compared column by column, and the Korean cases
are compared against the exact expected string.
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import pytest

from daemon import tui
from daemon.tui import Choice, Row, Theme, display_width

KOREAN = "한국어만 있는 줄"
MIXED = "한국어와 English가 섞인 줄"
EMOJI = "달 🌙 두 칸"
COMBINING = "cafe\u0301 결합 문자"
"""Spelled with an explicit U+0301 so this file cannot silently hold the
precomposed form instead - the whole point of the case is the two-code-point
`e`."""


class FakeTty(io.StringIO):
    """A stream that claims to be a terminal, so colour can be tested without
    one. `io.StringIO.isatty()` answers False, which is the whole point of it in
    the other direction."""

    def isatty(self) -> bool:
        return True


def sample_choices() -> tuple[Choice, ...]:
    return (
        Choice(
            "offline",
            "Everything on this machine. No keys, no accounts.",
            (
                "Conversation, reflection and the decision to speak first all run "
                "here through Ollama. Nothing leaves the machine.",
                "Voice is not available: native audio means a hosted model is both "
                "the brain and the voice, and leaving that out is what makes the "
                "privacy promise true instead of aspirational.",
            ),
        ),
        Choice(
            "balanced",
            "Claude for talking and reflecting, local for the rest.",
            ("The five-minute proactive check stays local so it costs nothing.",),
        ),
        Choice("quality", "Everything hosted. Best answers, highest bill."),
    )


def sample_rows() -> tuple[Row, ...]:
    return (
        Row("DAEMON_PROVIDER", "anthropic", "was ollama"),
        Row("DAEMON_VOICE_ENABLED", "false"),
        Row("ANTHROPIC_API_KEY", "...9999", "was (empty)"),
        Row("한국어_설정", "켜짐", "was 꺼짐"),
    )


def whole_screen(theme: Theme) -> str:
    """Everything this module can draw, in one string.

    A single renderer that forgot to go through `Theme` would be invisible to a
    test that only rendered the others, so the escape-sequence and overflow
    checks run over the lot.
    """
    return "\n".join(
        [
            tui.wordmark(theme),
            tui.rule(theme),
            tui.heading(theme, "Anthropic API key", step=(3, 4)),
            tui.heading(theme, MIXED),
            tui.choices(theme, sample_choices()),
            tui.choices(theme, sample_choices(), expanded=True),
            tui.table(theme, sample_rows()),
            tui.box(theme, [KOREAN, MIXED, EMOJI, COMBINING, ""], title="한국어 정렬"),
            tui.box(theme, ["plain body"]),
            *(tui.status(theme, kind, MIXED) for kind in ("ok", "warn", "missing", "fail")),
        ]
    )


# --- plain text unless the terminal asked for colour -------------------------


def test_nothing_coloured_when_stdout_is_not_a_terminal() -> None:
    theme = Theme.detect(io.StringIO(), env={"TERM": "xterm-256color"})
    assert theme.color is False
    assert "\033" not in whole_screen(theme)


def test_nothing_coloured_when_no_color_is_set() -> None:
    # no-color.org: the variable's presence is the signal. An empty value, or the
    # string "0", still means no colour.
    for value in ("", "0", "1"):
        theme = Theme.detect(FakeTty(), env={"TERM": "xterm-256color", "NO_COLOR": value})
        assert theme.color is False, value
        assert "\033" not in whole_screen(theme)


def test_nothing_coloured_on_a_dumb_terminal() -> None:
    for term in ("dumb", ""):
        theme = Theme.detect(FakeTty(), env={"TERM": term})
        assert theme.color is False, term
        assert "\033" not in whole_screen(theme)


def test_missing_term_variable_is_treated_as_dumb() -> None:
    assert Theme.detect(FakeTty(), env={}).color is False


def test_a_real_terminal_gets_colour() -> None:
    theme = Theme.detect(FakeTty(), env={"TERM": "xterm-256color"})
    assert theme.color is True
    assert "\033[1m" in tui.heading(theme, "Voice", step=(2, 4))
    assert "\033[32m" in tui.status(theme, "ok", "key works")
    assert "\033[31m" in tui.status(theme, "fail", "Google refused the key")
    assert "\033[2m" in tui.rule(theme)


def test_colour_is_reset_after_every_span() -> None:
    theme = Theme(color=True)
    screen = whole_screen(theme)
    assert screen.count("\033[0m") == screen.count("\033[") - screen.count("\033[0m")


def test_empty_text_is_not_wrapped_in_escapes() -> None:
    # `dim("")` happens on any line whose divider fill came out zero-length; an
    # escape pair around nothing would put colour on whatever printed next.
    assert Theme(color=True).dim("") == ""


def test_a_piped_stream_ignores_the_environment_width() -> None:
    # Otherwise the same command piped from two different shells would produce
    # two different files.
    theme = Theme.detect(io.StringIO(), env={"COLUMNS": "200"})
    assert theme.width == tui.DEFAULT_WIDTH


def test_detected_width_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    for reported, expected in ((0, tui.DEFAULT_WIDTH), (12, tui.MIN_WIDTH), (400, tui.MAX_WIDTH)):
        monkeypatch.setattr(shutil, "get_terminal_size", lambda *_, r=reported: os_size(r))
        assert tui.terminal_width() == expected


def os_size(columns: int) -> object:
    import os

    return os.terminal_size((columns, 24))


# --- display width -----------------------------------------------------------


def test_korean_syllables_are_two_columns() -> None:
    assert display_width("한국어") == 6
    assert len("한국어") == 3  # what a naive box would have used


def test_mixed_korean_and_latin() -> None:
    assert display_width("한국어 English") == 6 + 1 + 7


def test_emoji_is_two_columns() -> None:
    assert display_width("🌙") == 2
    # The variation selector that turns some glyphs into emoji prints nowhere.
    assert display_width("\u2757\ufe0f") == 2  # emoji + variation selector


def test_combining_marks_take_no_column() -> None:
    assert display_width("cafe\u0301") == 4  # e + combining acute
    assert display_width("caf\u00e9") == 4  # precomposed, same columns
    assert display_width("\u200b") == 0  # zero width space
    assert display_width("\u200d") == 0  # zero width joiner


def test_width_ignores_colour_already_applied() -> None:
    theme = Theme(color=True)
    assert display_width(theme.bold(KOREAN)) == display_width(KOREAN)


def test_fullwidth_forms_are_two_columns() -> None:
    assert display_width("ＡＢ") == 4


# --- padding, truncation, wrapping -------------------------------------------


def test_pad_uses_columns_not_characters() -> None:
    assert tui.pad("한국", 8) == "한국    "
    assert tui.pad("abcd", 8) == "abcd    "
    assert display_width(tui.pad(MIXED, 40)) == 40


def test_truncate_never_exceeds_the_budget() -> None:
    for width in range(1, 12):
        assert display_width(tui.truncate("한국어입니다", width)) <= width
        assert display_width(tui.truncate("abcdefghijk", width)) <= width


def test_truncate_leaves_short_text_alone() -> None:
    assert tui.truncate(KOREAN, 40) == KOREAN


def test_wrap_respects_columns_for_korean() -> None:
    for line in tui.wrap(MIXED * 4, 20):
        assert display_width(line) <= 20


def test_wrap_breaks_inside_an_unspaced_run() -> None:
    # Chinese and Japanese have no spaces to break at, and Korean allows long
    # unspaced runs; a space-only wrapper would return one overflowing line.
    lines = tui.wrap("한국어" * 20, 10)
    assert len(lines) > 1
    assert all(display_width(line) <= 10 for line in lines)
    assert "".join(lines) == "한국어" * 20  # nothing dropped


def test_wrap_keeps_every_word() -> None:
    text = "Nothing is written until the end, and nothing anywhere but ./.env."
    assert " ".join(tui.wrap(text, 24)) == text


# --- wordmark ----------------------------------------------------------------


def test_wide_wordmark_is_the_mark_with_the_tagline_beside_it() -> None:
    rendered = tui.wordmark(Theme(width=80))
    lines = rendered.splitlines()
    assert len(lines) == 3
    assert "┌┬┐" in lines[0]
    assert rendered.count("─") > 6  # it is drawn, not spelled
    for part in tui.TAGLINE:
        assert part in rendered


def test_narrow_wordmark_collapses_to_one_line() -> None:
    rendered = tui.wordmark(Theme(width=40))
    assert rendered == tui.NAME
    assert "\n" not in rendered
    assert display_width(rendered) <= 40


def test_wordmark_collapses_one_column_below_the_lockup() -> None:
    # The boundary, because either side of it is a different renderer.
    assert "\n" in tui.wordmark(Theme(width=tui.LOCKUP_WIDTH))
    assert tui.wordmark(Theme(width=tui.LOCKUP_WIDTH - 1)) == tui.NAME


def test_wordmark_never_overflows() -> None:
    for width in range(20, 101):
        for line in tui.wordmark(Theme(width=width)).splitlines():
            assert display_width(line) <= width, width


# --- headings ----------------------------------------------------------------


def test_heading_fills_the_width_and_shows_the_step() -> None:
    rendered = tui.heading(Theme(width=50), "Voice", step=(2, 4))
    assert rendered == "── 2/4 ─ Voice " + "─" * 35
    assert display_width(rendered) == 50


def test_heading_without_a_step() -> None:
    assert tui.heading(Theme(width=20), "Voice") == "── Voice " + "─" * 11


def test_heading_truncates_a_title_that_cannot_fit() -> None:
    theme = Theme(width=30)
    rendered = tui.heading(theme, MIXED * 3, step=(10, 10))
    assert display_width(rendered) <= 30
    assert "…" in rendered


# --- boxes -------------------------------------------------------------------


def test_box_borders_align_for_korean() -> None:
    theme = Theme(width=32)
    rendered = tui.box(theme, [KOREAN, MIXED, EMOJI, COMBINING], title="확인")
    assert rendered == (
        "╭─ 확인 ───────────────────────╮\n"
        "│ 한국어만 있는 줄             │\n"
        "│ 한국어와 English가 섞인 줄   │\n"
        "│ 달 🌙 두 칸                  │\n"
        "│ cafe\u0301 결합 문자               │\n"
        "╰──────────────────────────────╯"
    )


def test_every_box_line_is_the_same_width() -> None:
    for width in (20, 24, 40, 60, 80, 100):
        theme = Theme(width=width)
        for title in ("", "제목 title"):
            rendered = tui.box(theme, [KOREAN, MIXED, EMOJI, COMBINING, "", "x"], title=title)
            widths = {display_width(line) for line in rendered.splitlines()}
            assert widths == {width}, (width, title, widths)


def test_box_wraps_a_body_line_that_does_not_fit() -> None:
    rendered = tui.box(Theme(width=30), [KOREAN * 4])
    assert len(rendered.splitlines()) > 3
    assert all(display_width(line) == 30 for line in rendered.splitlines())


def test_box_will_not_exceed_the_terminal() -> None:
    rendered = tui.box(Theme(width=40), ["x"], width=200)
    assert all(display_width(line) == 40 for line in rendered.splitlines())


# --- choices -----------------------------------------------------------------


def test_expanded_choices_contain_everything_the_folded_form_does() -> None:
    theme = Theme(width=80)
    items = sample_choices()
    folded = tui.choices(theme, items)
    expanded = tui.choices(theme, items, expanded=True)

    folded_lines = [line for line in folded.splitlines() if line.strip()]
    expanded_lines = expanded.splitlines()
    for line in folded_lines:
        assert line in expanded_lines, line
    assert len(expanded_lines) > len(folded_lines)


def test_folding_hides_the_detail_and_expanding_shows_all_of_it() -> None:
    theme = Theme(width=80)
    items = sample_choices()
    folded = tui.choices(theme, items)
    expanded = tui.choices(theme, items, expanded=True)

    for choice in items:
        assert choice.summary in _unwrapped(folded)
        for paragraph in choice.detail:
            assert paragraph not in _unwrapped(folded)
            # The whole paragraph survives wrapping, which is the point of
            # folding rather than shortening.
            assert paragraph in _unwrapped(expanded)


def _unwrapped(rendered: str) -> str:
    """Rendered text with the layout taken back out, so a wrapped sentence can be
    compared against the sentence that was passed in."""
    return " ".join(rendered.split())


def test_choices_align_summaries_on_the_longest_name() -> None:
    items = sample_choices()
    rendered = tui.choices(Theme(width=80), items).splitlines()
    assert len(rendered) == len(items)  # every summary fitted on its own line
    columns = {
        line.index(choice.summary) for line, choice in zip(rendered, items, strict=True)
    }
    assert columns == {len("  1) balanced") + 2}


def test_choices_stack_the_summary_when_the_terminal_is_narrow() -> None:
    theme = Theme(width=30)
    rendered = tui.choices(theme, sample_choices())
    assert "1) offline" in rendered.splitlines()[0]
    assert all(display_width(line) <= 30 for line in rendered.splitlines())
    assert sample_choices()[0].summary in _unwrapped(rendered)


def test_choices_survive_korean_names_and_summaries() -> None:
    items = (
        Choice("오프라인", "이 기계에서만 돌아갑니다. 키도 계정도 필요 없습니다."),
        Choice("balanced", "대화와 성찰은 Claude, 나머지는 로컬."),
    )
    for width in (40, 60, 80):
        rendered = tui.choices(Theme(width=width), items)
        assert all(display_width(line) <= width for line in rendered.splitlines()), width


def test_no_choices_renders_nothing() -> None:
    assert tui.choices(Theme(), ()) == ""


# --- tables ------------------------------------------------------------------


def test_table_aligns_values_including_korean_keys() -> None:
    rendered = tui.table(Theme(width=80), sample_rows())
    assert rendered.splitlines() == [
        "  DAEMON_PROVIDER       anthropic  was ollama",
        "  DAEMON_VOICE_ENABLED  false",
        "  ANTHROPIC_API_KEY     ...9999    was (empty)",
        "  한국어_설정           켜짐       was 꺼짐",
    ]


def test_table_note_drops_to_its_own_line_rather_than_overflowing() -> None:
    rows = (Row("TELEGRAM_BOT_TOKEN", "...ABCD", "was ...7f21, connected to @my_daemon_bot"),)
    for width in (40, 56, 80):
        rendered = tui.table(Theme(width=width), rows)
        assert all(display_width(line) <= width for line in rendered.splitlines()), width
        assert _unwrapped(rendered).endswith("@my_daemon_bot")


def test_table_stacks_when_a_key_leaves_no_room_for_its_value() -> None:
    rows = (Row("DAEMON_A_VERY_LONG_CONFIGURATION_KEY", "value", "was other"),)
    rendered = tui.table(Theme(width=40), rows)
    assert len(rendered.splitlines()) == 3
    assert all(display_width(line) <= 40 for line in rendered.splitlines())


def test_no_rows_renders_nothing() -> None:
    assert tui.table(Theme(), ()) == ""


# --- status lines ------------------------------------------------------------


def test_status_prefixes_line_up() -> None:
    theme = Theme(width=80)
    rendered = [tui.status(theme, kind, "detail") for kind in ("ok", "warn", "missing", "fail")]
    assert rendered == [
        "ok:      detail",
        "warn:    detail",
        "missing: detail",
        "fail:    detail",
    ]


def test_status_wraps_under_its_own_prefix() -> None:
    theme = Theme(width=40)
    rendered = tui.status(theme, "warn", "'claude-4-opus' is not in your model list, so set it")
    lines = rendered.splitlines()
    assert len(lines) > 1
    assert all(display_width(line) <= 40 for line in lines)
    assert all(line.startswith(" " * 9) for line in lines[1:])


def test_unknown_status_kind_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown status kind"):
        tui.status(Theme(), "explode", "detail")  # type: ignore[arg-type]


# --- the whole screen --------------------------------------------------------


def test_nothing_overflows_at_any_supported_width() -> None:
    for width in (40, 48, 56, 60, 72, 80, 100):
        theme = Theme(width=width)
        for line in whole_screen(theme).splitlines():
            assert display_width(line) <= width, (width, line)


def test_nothing_overflows_with_colour_either() -> None:
    # Colour must not change the layout: if it did, the terminal a user sees and
    # the pipe a test reads would be different screens.
    for width in (40, 60, 80):
        plain = _shape(whole_screen(Theme(color=False, width=width)))
        coloured = _shape(whole_screen(Theme(color=True, width=width)))
        assert plain == coloured, width


def _shape(rendered: str) -> Sequence[int]:
    return [display_width(line) for line in rendered.splitlines()]
