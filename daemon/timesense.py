"""How the daemon speaks about time.

Everything else in this repo reads the clock in UTC and stores it that way
(`daemon/clock.py`). This module is the other half: turning an instant into
something a person - and a model that cannot do date arithmetic - reads correctly.

It exists because the conversation path had no time sense at all. `Companion.context`
put persona, tool rules and recall in front of the model and nothing else, and
`ConversationLoop._assemble` carried twenty messages as bare role+content, so a
Friday thread and a Tuesday greeting arrived as one unbroken conversation. The daemon
answered a bare "벨라" on Tuesday morning by reporting that it was standing by for a
reminder whose meeting had happened the previous Friday.

**Every string here is built from computed values and fixed lexicon words.** No
substring of any message passes through, which is why these blocks carry no nonce -
there is nothing in them an old message could have authored. `commitments` follows the
same discipline `daemon/proactivity/candidates.py` states for its `reason` field.

`now` is always a parameter, never read here. Two reasons: the callers already hold
one and two reads of the clock inside one turn can straddle a minute, and it is what
makes every case in `tests/test_timesense.py` pinnable.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Protocol

from daemon import clock

WEEKDAYS: tuple[str, ...] = ("월", "화", "수", "목", "금", "토", "일")
"""Monday-first, matching `datetime.weekday()`."""

_DAY_PARTS: tuple[tuple[int, str], ...] = (
    (6, "새벽"),
    (9, "아침"),
    (12, "오전"),
    (14, "점심"),
    (18, "오후"),
    (22, "저녁"),
    (24, "밤"),
)
"""Exclusive upper bounds on the local hour. The buckets are coarse on purpose: the
question they answer is "is it the middle of their night", not what o'clock it is."""


class Timed(Protocol):
    """The shape both `LoggedMessage` and `RecalledItem` already have.

    Structural rather than a shared base class: this module must not care whether a
    line came from the recent window or from a recall hit, and neither dataclass
    should have to learn about this module.
    """

    ts: datetime
    content: str
    role: str
    origin: str


def now_block(now: datetime) -> str:
    """The current instant, as the model's first fact about the world."""
    here = clock.local(now)
    kind = "주말" if here.weekday() >= 5 else "평일"
    return (
        f"[현재 시각] 지금은 {here.year}년 {here.month}월 {here.day}일 "
        f"{WEEKDAYS[here.weekday()]}요일 {_clock_face(here)}입니다. "
        f"{kind} {_day_part(here.hour)}입니다."
    )


def relative(ts: datetime, now: datetime, *, with_gap: bool = True) -> str:
    """`ts` as a person would say it, relative to `now`.

    Absolute date *and* relative phrase, because relative alone is ambiguous the
    moment two items land on the same weekday - and "지난주 금요일" with no date is
    exactly the kind of near-miss the model would then reason from confidently.

    `with_gap=False` drops the "(N일 전)" tail for use mid-sentence, where
    "...(4일 전)에 말한" does not read as Korean.
    """
    here, then = clock.local(now), clock.local(ts)
    days = (here.date() - then.date()).days
    face = _clock_face(then)
    # These four say which day by themselves; a "(N일 전)" after them is noise.
    if days == 0:
        return f"오늘 {face}"
    if days == 1:
        return f"어제 {face}"
    if days == 2:
        return f"그저께 {face}"
    if days == -1:
        return f"내일 {face}"
    if days == -2:
        return f"모레 {face}"

    weekday = f"{WEEKDAYS[then.weekday()]}요일"
    # Monday-difference rather than `isocalendar()`: ISO weeks renumber at the year
    # edge, so 12월 29일 and the following 1월 5일 would compare as far apart.
    weeks = (_monday(here.date()) - _monday(then.date())).days // 7
    if weeks == 0:
        phrase = f"이번주 {weekday} {face}"
    elif weeks == 1:
        phrase = f"지난주 {weekday} {face}"
    else:
        phrase = f"{then.month}월 {then.day}일 {weekday} {face}"
    if not with_gap:
        return phrase
    gap = f"{days}일 전" if days > 0 else f"{-days}일 후"
    return f"{phrase} ({gap})"


def _day_part(hour: int) -> str:
    for bound, name in _DAY_PARTS:
        if hour < bound:
            return name
    return "밤"


def _clock_face(here: datetime) -> str:
    """Local wall time in Korean. The minute is dropped when it is zero, because
    "오전 10시 0분" is not something anyone says."""
    meridiem = "오전" if here.hour < 12 else "오후"
    twelve = here.hour % 12 or 12
    if here.minute == 0:
        return f"{meridiem} {twelve}시"
    return f"{meridiem} {twelve}시 {here.minute}분"


def _monday(day: date) -> date:
    return day - timedelta(days=day.weekday())
