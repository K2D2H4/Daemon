"""Terminal presentation: the layer that decides how `daemon setup` reads.

Onboarding is the first sentence the product says, and until now it said
everything in one voice - title, explanation and question all at the same weight,
in the same colour, with no sense of how much was left. This module is the
vocabulary for saying it with structure instead: a wordmark, headings that carry
a step count, folded detail, aligned tables, status prefixes.

It is presentation only. Nothing here knows what a preset is or which questions
come in which order; a renderer takes text and gives back text. That split is
what lets the wizard's flow be tested as a flow and the layout be tested as
strings.

Three constraints shape everything below.

**Plain text is the default, not the degraded mode.** ANSI is emitted only when
the destination is a real terminal that has not asked to be left alone -
`NO_COLOR` (no-color.org: presence is the signal, the value means nothing) and
`TERM=dumb` both turn it off. The suite captures stdout through a pipe, so a
single leaked escape would turn readable assertions into unreadable ones; the
`Theme` a piped stream produces cannot emit colour at all, which is why the
guarantee is structural rather than a rule to remember.

**Width is measured, never counted.** A Korean syllable occupies two terminal
columns and `len()` says one, so `len()` draws a box that is visibly crooked the
moment a Korean word appears inside it - and the design documents are Korean, so
that moment is the first one. `display_width` asks `unicodedata` instead, and
every border, pad and wrap in this file goes through it.

**Folding is not omission.** The prose the wizard used to print is the answer to
"why does this preset exist", which is the one thing a person choosing between
three presets actually needs. The collapsed renderer is an entrance to it, so
`choices(expanded=True)` is a superset of `choices()` - never a rewrite of it.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TextIO

NAME = "Daemon"

TAGLINE: tuple[str, ...] = ("an outside soul,", "a resident process")
"""docs/PLAN.md 1: the name means both an external soul that grows to resemble
its owner, and a process that stays up. Two lines because it sits beside the
wordmark like a lockup; joined with a space it is also the one-line form."""

DEFAULT_WIDTH = 80
"""What a stream that is not a terminal gets. Deliberately not `COLUMNS`: piped
output that changed shape with the shell that happened to run it would make both
diffs and test assertions unstable."""

MIN_WIDTH = 40
MAX_WIDTH = 100
"""Layout is capped even on a very wide terminal. A 200-column line of prose is
harder to read than a 100-column one, and the box borders stop looking like a
frame and start looking like a table."""


# --- display width -----------------------------------------------------------

_ANSI = re.compile(r"\033\[[0-9;]*m")

_WIDE = frozenset("WF")
"""East Asian Width classes that take two columns. `A` (ambiguous) is left at one
- it covers Greek, Cyrillic and typographic punctuation, which a modern UTF-8
terminal renders narrow unless it has been put in a legacy CJK mode."""

_ZERO = frozenset(("Mn", "Me", "Cf"))
"""Nonspacing and enclosing marks, and format characters: combining accents, the
emoji variation selector U+FE0F, zero-width joiners and spaces. They print into
the previous cell, or into none."""


def char_width(char: str) -> int:
    """Terminal columns one character occupies."""
    if unicodedata.combining(char) or unicodedata.category(char) in _ZERO:
        return 0
    return 2 if unicodedata.east_asian_width(char) in _WIDE else 1


def display_width(text: str) -> int:
    """Terminal columns `text` occupies, ignoring any colour it already carries.

    Escape sequences are stripped first so that a value which has been through
    `Theme.bold` can still be padded into a column - otherwise every coloured
    cell in a table would be nine columns too wide.

    One case stays wrong on purpose: an emoji built from a joiner sequence
    (person + ZWJ + laptop) measures as two emoji here, and terminals disagree
    with each other about whether it is one glyph or two. Nothing in this file
    emits one, and guessing would trade a rare misalignment for a common one.
    """
    return sum(char_width(char) for char in _ANSI.sub("", text))


def pad(text: str, width: int) -> str:
    """Left-aligned in a field `width` columns wide, by display width."""
    return text + " " * max(0, width - display_width(text))


def truncate(text: str, width: int, *, marker: str = "…") -> str:
    """`text` shortened to at most `width` columns, never more.

    Cuts on column boundaries, so a two-column character is dropped whole rather
    than leaving the terminal to guess at half of one.
    """
    if display_width(text) <= width:
        return text
    budget = width - display_width(marker)
    if budget <= 0:
        return _split(text, width)[0]
    return _split(text, budget)[0] + marker


def wrap(text: str, width: int) -> list[str]:
    """Word wrap by display width, breaking inside a word when it cannot fit.

    The fallback matters more than it looks: Chinese and Japanese are written
    without spaces, and Korean allows long unspaced runs, so a wrapper that only
    ever breaks at spaces overflows the line instead of wrapping it.
    """
    limit = max(1, width)
    lines: list[str] = []
    current = ""
    for token in text.split():
        candidate = f"{current} {token}" if current else token
        if display_width(candidate) <= limit:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        while display_width(token) > limit:
            head, token = _split(token, limit)
            if not head:  # a single character wider than the whole line
                break
            lines.append(head)
        current = token
    if current:
        lines.append(current)
    return lines or [""]


def _split(text: str, width: int) -> tuple[str, str]:
    """`text` cut at the last column boundary that fits in `width`."""
    used = 0
    for index, char in enumerate(text):
        step = char_width(char)
        if used + step > width:
            return text[:index], text[index:]
        used += step
    return text, ""


# --- theme -------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"

_NO_COLOR_TERMS = frozenset(("", "dumb"))


@dataclass(frozen=True, slots=True)
class Theme:
    """Whether colour may be used, and how wide the screen is.

    Constructed once per command from the stream it will print to, then passed to
    every renderer. Callers never write an escape sequence themselves, so "does
    this output have colour in it" is answered in one place instead of at every
    print site.
    """

    color: bool = False
    width: int = DEFAULT_WIDTH

    @classmethod
    def detect(
        cls, stream: TextIO | None = None, *, env: Mapping[str, str] | None = None
    ) -> Theme:
        out = sys.stdout if stream is None else stream
        environ = os.environ if env is None else env
        if not bool(getattr(out, "isatty", _not_a_tty)()):
            # A pipe, a file, a log, or a test's StringIO. None of them want ANSI,
            # and none of them have a width worth asking about.
            return cls(color=False, width=DEFAULT_WIDTH)
        color = "NO_COLOR" not in environ and environ.get("TERM", "") not in _NO_COLOR_TERMS
        return cls(color=color, width=terminal_width())

    def bold(self, text: str) -> str:
        return self._paint(text, _BOLD)

    def dim(self, text: str) -> str:
        return self._paint(text, _DIM)

    def ok(self, text: str) -> str:
        return self._paint(text, _GREEN)

    def warn(self, text: str) -> str:
        return self._paint(text, _YELLOW)

    def bad(self, text: str) -> str:
        return self._paint(text, _RED)

    def _paint(self, text: str, code: str) -> str:
        if not self.color or not text:
            return text
        return f"{code}{text}{_RESET}"


def _not_a_tty() -> bool:
    return False


def terminal_width() -> int:
    """The terminal's width, clamped into a range that can actually be laid out.

    `shutil.get_terminal_size` already falls back to 80, but it reports whatever
    the terminal claims - including 0 from a few emulators during a resize, which
    would turn every `"-" * (width - 2)` in this file into a crash or an empty
    line.
    """
    columns = shutil.get_terminal_size((DEFAULT_WIDTH, 24)).columns
    if columns <= 0:
        columns = DEFAULT_WIDTH
    return max(MIN_WIDTH, min(columns, MAX_WIDTH))


# --- wordmark ----------------------------------------------------------------

_MARK: tuple[str, ...] = (
    "┌┬┐ ┌─┐ ┌─┐ ┌┬┐ ┌─┐ ┌┐┌",
    " ││ ├─┤ ├┤  │││ │ │ │││",
    "─┴┘ ┴ ┴ └─┘ ┴ ┴ └─┘ ┘└┘",
)
"""DAEMON in light box drawing. Three rows rather than the five or six a block
font needs: a wordmark that fills a third of the screen on every run stops being
identity and becomes something to scroll past."""

_MARK_WIDTH = 23
_GAP = 3
LOCKUP_WIDTH = _MARK_WIDTH + _GAP + 18
"""Below this the lockup does not fit and `wordmark` returns one line."""


def wordmark(theme: Theme) -> str:
    """The product's name, with the tagline set beside it.

    Under `LOCKUP_WIDTH` columns it collapses to the name on one line - a mark
    that wraps is worse than no mark, and a narrow terminal is usually narrow for
    a reason (a phone over ssh, a split pane) rather than a mistake to be drawn
    through. The tagline goes with it rather than being kept on its own line:
    `Daemon · an outside soul, a resident process` needs 44 columns, the same as
    the lockup, so there is no width where one fits and the other does not.
    """
    if theme.width < LOCKUP_WIDTH:
        return theme.bold(NAME)

    beside = ("", *TAGLINE)
    lines = []
    for row, side in zip(_MARK, beside, strict=True):
        line = theme.bold(row)
        if side:
            line += " " * _GAP + theme.dim(side)
        lines.append(line)
    return "\n".join(lines)


# --- rules and headings ------------------------------------------------------


def rule(theme: Theme) -> str:
    """A full-width divider."""
    return theme.dim("─" * theme.width)


def heading(theme: Theme, title: str, *, step: tuple[int, int] | None = None) -> str:
    """A titled divider, optionally carrying `1/4`.

    The step count goes to the left of the title, where the eye starts a line:
    the question a person has partway through a wizard is "how much is left",
    and answering it after the title means finding it first.
    """
    marker = f"{step[0]}/{step[1]}" if step else ""
    lead = "──" if not marker else f"── {marker} ─"
    shown = truncate(title, max(1, theme.width - display_width(lead) - 2))
    used = display_width(lead) + display_width(shown) + 2
    fill = "─" * max(0, theme.width - used)
    return f"{theme.dim(lead)} {theme.bold(shown)} {theme.dim(fill)}".rstrip()


# --- boxes -------------------------------------------------------------------

_BOX_OVERHEAD = 4
"""Two borders and the space of padding inside each."""

MIN_BOX_WIDTH = 20


def box(
    theme: Theme, lines: Sequence[str], *, title: str = "", width: int | None = None
) -> str:
    """Body text inside a light frame, with every border aligned.

    Long lines are wrapped to fit, which means the body should be plain text; a
    line that already carries colour is padded correctly (`display_width` ignores
    escapes) but would be cut through the middle of a sequence if it also needed
    wrapping.
    """
    total = max(MIN_BOX_WIDTH, theme.width if width is None else min(width, theme.width))
    inner = total - _BOX_OVERHEAD

    body: list[str] = []
    for line in lines:
        body.extend(wrap(line, inner) if display_width(line) > inner else [line])

    out = [_box_top(theme, title, total, inner)]
    for line in body:
        out.append(theme.dim("│") + " " + pad(line, inner) + " " + theme.dim("│"))
    out.append(theme.dim("╰" + "─" * (total - 2) + "╯"))
    return "\n".join(out)


def _box_top(theme: Theme, title: str, total: int, inner: int) -> str:
    if not title:
        return theme.dim("╭" + "─" * (total - 2) + "╮")
    shown = truncate(title, inner)
    fill = "─" * max(0, total - display_width(shown) - 5)
    return (
        theme.dim("╭─")
        + " "
        + theme.bold(shown)
        + " "
        + theme.dim(fill + "╮")
    )


# --- numbered choices --------------------------------------------------------

_INDENT = 2
_DETAIL_INDENT = _INDENT + 4
_MIN_SUMMARY = 24
"""Below this the summary stops fitting beside the name and goes under it."""


@dataclass(frozen=True, slots=True)
class Choice:
    """One numbered option: a line to choose by, and the reasons behind it.

    `summary` is what a person reads while deciding; `detail` is what they read
    when the summary is not enough.

    Each element of `detail` is one paragraph: it is rewrapped to the terminal
    and separated from the next by a blank line. Pass prose, not lines someone
    has already broken by hand - those would come out ragged, and every hand
    break would become a paragraph.
    """

    name: str
    summary: str
    detail: tuple[str, ...] = field(default_factory=tuple)


def choices(theme: Theme, items: Sequence[Choice], *, expanded: bool = False) -> str:
    """The list, folded by default.

    `expanded=True` adds every paragraph of `detail` and changes nothing else, so
    the folded form is an entrance to the full text rather than a replacement for
    it. Whether and when to expand belongs to whoever is reading the keyboard.
    """
    if not items:
        return ""

    labels = [f"{index}) {choice.name}" for index, choice in enumerate(items, start=1)]
    column = max(display_width(label) for label in labels)
    summary_column = _INDENT + column + 2
    beside = theme.width - summary_column >= _MIN_SUMMARY

    lines: list[str] = []
    for label, choice in zip(labels, items, strict=True):
        number, _, name = label.partition(") ")
        painted = f"{number}) {theme.bold(name)}"
        if beside:
            wrapped = wrap(choice.summary, theme.width - summary_column)
            lines.append(" " * _INDENT + pad(painted, column) + "  " + wrapped[0])
            lines.extend(" " * summary_column + line for line in wrapped[1:])
        else:
            lines.append(" " * _INDENT + painted)
            lines.extend(
                " " * (_INDENT + 2) + line
                for line in wrap(choice.summary, theme.width - _INDENT - 2)
            )
        if expanded and choice.detail:
            for paragraph in choice.detail:
                lines.append("")
                lines.extend(
                    " " * _DETAIL_INDENT + theme.dim(line)
                    for line in wrap(paragraph, theme.width - _DETAIL_INDENT)
                )
            lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


# --- key/value tables --------------------------------------------------------

_MIN_VALUE = 8
_NOTE_GAP = 2


@dataclass(frozen=True, slots=True)
class Row:
    """One line of a summary table. `note` is dimmed and may be dropped to its
    own line, so it must be commentary ("was ...9f21") and never the value."""

    key: str
    value: str
    note: str = ""


def table(theme: Theme, rows: Sequence[Row]) -> str:
    """Aligned key/value lines - what changes in `.env`, what a check found.

    Alignment is by display width, so a Korean value does not push the column
    that follows it out of line.
    """
    if not rows:
        return ""

    key_column = max(display_width(row.key) for row in rows)
    room = theme.width - _INDENT - key_column - 2
    if room < _MIN_VALUE:
        return "\n".join(_stacked(theme, rows))

    values = [truncate(row.value, room) for row in rows]
    value_column = max(display_width(value) for value in values)
    lines: list[str] = []
    for row, value in zip(rows, values, strict=True):
        head = (
            " " * _INDENT
            + pad(row.key, key_column)
            + "  "
            + pad(theme.bold(value), value_column)
        )
        if not row.note:
            lines.append(head.rstrip())
        elif display_width(head) + _NOTE_GAP + display_width(row.note) <= theme.width:
            lines.append(head + " " * _NOTE_GAP + theme.dim(row.note))
        else:
            # The note is commentary; losing its column is better than wrapping
            # the row it comments on. It still wraps rather than truncating -
            # `was ...7f21` says which value is about to be replaced, and half of
            # that sentence is worse than a second line.
            indent = " " * (_INDENT + key_column + 2)
            lines.append(head.rstrip())
            lines.extend(
                indent + theme.dim(line)
                for line in wrap(row.note, max(1, theme.width - len(indent)))
            )
    return "\n".join(lines)


def _stacked(theme: Theme, rows: Sequence[Row]) -> list[str]:
    """One row over two or three lines, for a terminal too narrow for columns."""
    room = theme.width - _INDENT - 2
    lines: list[str] = []
    for row in rows:
        lines.append(" " * _INDENT + truncate(row.key, theme.width - _INDENT))
        lines.append(" " * (_INDENT + 2) + theme.bold(truncate(row.value, room)))
        if row.note:
            lines.append(" " * (_INDENT + 2) + theme.dim(truncate(row.note, room)))
    return lines


# --- status lines ------------------------------------------------------------

Kind = Literal["ok", "warn", "missing", "fail"]

_LABEL_WIDTH = 9
"""`missing:` plus a space, so the text of every status line starts in the same
column and a run of them can be read down rather than across."""


def status(theme: Theme, kind: Kind, text: str) -> str:
    """`ok: ...`, `warn: ...`, `missing: ...`, `fail: ...`.

    Colour is the second signal and the word is the first, because the word
    survives a pipe, a log file and colour blindness.
    """
    painters = {"ok": theme.ok, "warn": theme.warn, "missing": theme.warn, "fail": theme.bad}
    if kind not in painters:
        raise ValueError(f"unknown status kind {kind!r}")
    label = f"{kind}:"
    gap = " " * max(1, _LABEL_WIDTH - display_width(label))
    wrapped = wrap(text, max(1, theme.width - _LABEL_WIDTH))
    first = (painters[kind](label) + gap + wrapped[0]).rstrip()
    return "\n".join([first, *(" " * _LABEL_WIDTH + line for line in wrapped[1:])])
