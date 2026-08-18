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

import re
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
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


BREAK_MIN_HOURS = 6.0
"""A gap this long *and* a change of local date is a new conversation.

Both conditions, because either alone is wrong in a way the owner would notice: a
five-hour afternoon gap is still one thread, and 23:30 to 01:30 changes the date
without anyone having slept. The `voice` path's 120-minute freshness cutoff answers a
different question - whether to hand a tail over at all - and is left alone.
"""


def session_breaks(history: Sequence[Timed], now: datetime) -> list[tuple[int, str]]:
    """Where the conversation window broke, as `(index, line)` in ascending order.

    The line belongs **before** `history[index]`, and `_assemble` splices it in at
    that position - which is still worth doing: on `ollama` and `openai_compatible`
    the line stays exactly there, and the model reads the break with no counting.
    But `gemini`, `anthropic` and `openai` hoist every `role="system"` message out
    of the turn array and concatenate it into a top-level field before the model
    ever sees it (verified against each provider's payload builder, pinned in
    `tests/test_providers.py`), so on those three the position is gone by the time
    the line is read. `_break_line`'s wording does not depend on surviving that -
    see its own docstring for why.
    """
    breaks: list[tuple[int, str]] = []
    for index in range(1, len(history)):
        before = clock.local(history[index - 1].ts)
        after = clock.local(history[index].ts)
        if before.date() == after.date():
            continue
        if (after - before).total_seconds() / 3600.0 < BREAK_MIN_HOURS:
            continue
        breaks.append((index, _break_line(before, after, now)))
    return breaks


def _break_line(before: datetime, after: datetime, now: datetime) -> str:
    """The `[대화 단절]` line, worded to be true whether or not its position survives.

    An earlier version read "위는 ..., 아래는 ... 위쪽은 이미 끝난 대화입니다" -
    correct only at its spliced position, and false the moment a hosted provider
    (`gemini`, `anthropic`, `openai`) hoists it into a top-level system field
    alongside the persona and tool rules, where there is no 위 and no 아래. So this
    names both dates and the gap between them directly, and says outright which one
    is finished, instead of pointing at where it sits in a list the model may not
    even see as a list. True read inline, at the boundary, and true read hoisted,
    next to every other system line.
    """
    gap = (after.date() - before.date()).days
    today = "오늘 " if after.date() == clock.local(now).date() else ""
    before_day = _day_face(before)
    after_day = f"{today}{_day_face(after)}"
    return (
        "[대화 단절] "
        f"{before_day} 대화와 {gap}일 뒤인 {after_day} 대화 사이에 시간이 비었습니다. "
        f"{before_day} 대화는 이미 끝난 대화입니다."
    )


def _day_face(here: datetime) -> str:
    return f"{here.month}월 {here.day}일 {WEEKDAYS[here.weekday()]}요일"


# --- what the owner said would happen ----------------------------------------
#
# These came from `daemon/proactivity/candidates.py`, which still imports them. Two
# callers now ask *different questions of the same primitives*: proactivity asks "is
# `now` inside [due, due + TTL]" to decide whether to raise something, and
# `commitments` asks "is `due` in the past" to decide whether a thing the owner can
# see is still alive. They must not drift apart - the defect that started this was
# the conversation path not being able to see a judgement proactivity had already
# made correctly.

FOLLOWUP_HOUR = 20
"""Local hour an open loop with no stated time becomes due. PLAN 6.1's own example
is "내일 발표" -> the next evening: late enough that the thing has happened."""

EVENT_LAG_HOURS = 2
"""How long after a *stated* time to ask how it went. Without this, "내일 밤 10시
발표" gets asked about at 20:00 - two hours before it happens."""

DAY_OFFSETS: tuple[tuple[str, int], ...] = (
    # Longest first: "내일모레" contains "내일", and means the day after tomorrow.
    ("내일모레", 2),
    ("모레", 2),
    ("내일", 1),
    ("오늘", 0),
    ("이따", 0),
)
"""Only markers whose futurity is *lexical*. "다음주"/"주말"/"금요일" were left out
on purpose: resolving them to a date is a guess (is 금요일 said on Friday today or
in seven days?), and a wrong due time makes the daemon ask how something went
before it happened, which reads as broken rather than early."""

TENSE_NEUTRAL = frozenset({"오늘", "이따"})
"""These say *when* without saying *not yet*, so "오늘 발표 했어" would otherwise
become a candidate to ask about a presentation already reported on. They alone get
the past-tense guard; 내일/모레 do not need it and applying it there would throw
away the common "내일 발표 있어, 오늘 준비 많이 했어"."""

PAST_MARKERS = (
    # Korean past tense contracts into a syllable rather than adding one, so this
    # is a list of contracted forms and not a suffix rule: 하였어->했어, 되었어->
    # 됐어, 보았어->봤어. Not exhaustive, and it does not need to be - a missed
    # past tense costs one unnecessary candidate, and the gate still has to pass it.
    "했", "었", "았", "였", "됐", "갔", "왔", "봤",
)

EVENTS = (
    # Things with a moment and an outcome, so that "how did it go" is a real
    # question afterwards. Deliberately excludes words whose commonest use is not
    # an event: 경기 is most often the economy, 시간 is never an appointment.
    "발표", "시험", "면접", "인터뷰", "회의", "미팅", "세미나", "워크샵", "강연",
    "병원", "진료", "검진", "검사", "수술", "치과", "상담",
    "약속", "소개팅", "데이트", "결혼식", "장례",
    "마감", "데드라인", "제출", "과제", "제안서", "계약", "등록",
    "이사", "출장", "여행", "이직",
    "리허설", "오디션", "촬영", "공연", "시합", "대회", "면허",
    "발표회", "상견례", "면허시험", "자격증", "건강검진", "예방접종",
    "견적", "심사", "발표평가", "학회", "졸업식", "입학식", "회식", "정기점검",
)

EVENT_CANCELLED = (
    # Said in the same message, these mean the thing is not happening. A missed
    # cancellation is the expensive false positive here: asking how a cancelled
    # exam went says plainly that nobody was listening.
    "취소", "연기", "미뤄", "없어졌", "안 하", "안 가", "없어", "없다", "없음",
)

HOUR_RE = re.compile(r"(오전|오후|아침|저녁|밤)?\s*(\d{1,2})\s*시(?!간)")
"""`(?!간)` because "2시간 뒤에" is a duration, and matching `2시` inside it would
schedule the follow-up at 02:00."""


def contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def first_of(text: str, needles: Sequence[str]) -> str | None:
    """The first of `needles` present, in `needles` order rather than text order,
    so the caller controls precedence."""
    for needle in needles:
        if needle in text:
            return needle
    return None


def day_marker(text: str) -> tuple[str, int] | None:
    for surface, offset in DAY_OFFSETS:
        if surface in text:
            return surface, offset
    return None


def stated_hour(text: str) -> int | None:
    """The hour the user named, in 24h local, or `None` if they named none."""
    match = HOUR_RE.search(text)
    if match is None:
        return None
    meridiem, digits = match.group(1), int(match.group(2))
    if digits > 23:
        return None
    if meridiem in ("오후", "저녁", "밤"):
        return digits + 12 if digits < 12 else digits
    if meridiem in ("오전", "아침"):
        return digits
    # No meridiem: a bare single-digit hour before 8 is the afternoon one. "3시
    # 발표" is 15:00, and reading it as 03:00 would put the follow-up before dawn.
    return digits + 12 if digits < 8 else digits


def due_at(said_at: datetime, day_offset: int, stated_hour: int | None) -> datetime:
    """When the follow-up becomes due, as UTC.

    Built in local time and converted, because the offset is a number of *days* on
    the user's calendar: "내일" said at 23:30 KST is a different UTC date already.
    A naive datetime's `.astimezone()` resolves it with the offset in force on that
    date, which is the only form of this that stays correct across a DST boundary.
    """
    target = clock.local(said_at).date() + timedelta(days=day_offset)
    hour = FOLLOWUP_HOUR
    if stated_hour is not None:
        hour = max(FOLLOWUP_HOUR, min(23, stated_hour + EVENT_LAG_HOURS))
    return datetime.combine(target, time(hour=hour)).astimezone().astimezone(UTC)


def commitments(messages: Sequence[Timed], now: datetime) -> str:
    """Which commitments in the visible context are still alive, and which are past.

    **Scans only what is already in front of the model** - the recent window plus the
    recalled items - and never the database. Two consequences, both wanted: the block
    can only annotate text the model can actually see, and the turn costs no extra
    query. A live commitment older than the prompt window is therefore invisible
    here, which is correct: raising one *on time* is proactivity's job and it already
    does it.

    Passive by decision. The block says what is alive and what is past; it does not
    ask the daemon to bring anything up. `OPEN_LOOP_TTL_HOURS`' comment - "사흘 늦은
    건 백로그를 처리하는 기계다" - is a measured decision that stands.

    Nothing from the message reaches the output: `surface` and `event` are lexicon
    entries and every time is computed, the same discipline
    `daemon/proactivity/candidates.py` states for its `reason` field.
    """
    lines: list[str] = []
    seen: set[tuple[str, str, datetime]] = set()
    for item in messages:
        # Only the owner's own words. `origin` is a column precisely so this is
        # decidable: relayed text is not the owner saying they have a meeting.
        if item.role != "user" or item.origin != "owner":
            continue
        text = item.content
        if contains_any(text, EVENT_CANCELLED):
            continue
        marker = day_marker(text)
        event = first_of(text, EVENTS)
        if marker is None or event is None:
            continue
        surface, offset = marker
        if surface in TENSE_NEUTRAL and contains_any(text, PAST_MARKERS):
            continue
        due = due_at(item.ts, offset, stated_hour(text))
        key = (surface, event, due)
        if key in seen:
            continue
        seen.add(key)
        said = relative(item.ts, now, with_gap=False)
        when = relative(due, now, with_gap=False)
        state = (
            "이미 지났습니다. 대기 중인 일이 아닙니다."
            if due <= now
            else "아직 오지 않았습니다."
        )
        lines.append(f"- {said}에 말한 '{surface} {event}'의 시각({when})은 {state}")
    if not lines:
        return ""
    return "\n".join(
        [
            "[약속 상태] 대화에 나온 약속들이 지금 어떤 상태인지입니다. "
            "이미 지난 것을 아직 대기 중인 일처럼 말하지 마십시오.",
            "",
            *lines,
        ]
    )
