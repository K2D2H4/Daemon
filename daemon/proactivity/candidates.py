"""Stage 1 of docs/PLAN.md 6.1: reasons it might be worth speaking. No model calls.

Non-negotiable 7 makes this deterministic, and that is the hard part of the
problem rather than a restriction to work around: the tick has to notice "내일
발표" or "요즘 너무 힘들다" in Korean with nothing but string matching. What makes
it possible at all is that **Korean inflects by suffix**, so a stem is a prefix
of every form it takes and plain `in` matches all of them:

    "발표" in "내일 발표가 있어"   -> True
    "발표" in "발표를 해야 해"     -> True
    "발표" in "발표 준비 중"       -> True

This is exactly the case docs/PLAN.md 4.3 records FTS5 failing at - `unicode61`
matches whole tokens, so `발표가` and `발표` are unrelated - and the reason that
finding does *not* transfer here. Nothing in this module goes near FTS5.

Where suffixing is not enough it is because the stem itself changes (the ㅂ/ㅡ/ㄹ
irregulars: 슬프다 -> 슬퍼, 무섭다 -> 무서워, 힘들다 -> 힘드네), so the lexicons
below list surface variants rather than dictionary forms. A morphological
analyser would do this properly and was declined for the same reason PLAN 4.3
declined it for recall: too heavy a dependency for a self-hoster.

## Five generators

Type E (association) was **deliberately not implemented** until now, because
`MemoryRecall.search` - the recall index's only entry point at the time - was
wrong here twice over:

  * Its score multiplies by recency decay with a 30-day half-life, so a
    three-month-old memory arrives at 0.125x weight. E wants precisely the items
    that decay is built to bury; ranking through it means the old memory almost
    never reaches `limit`.
  * It calls `Store.mark_recalled` on every hit. From a background tick that sets
    `messages.recalled = 1` for rows **nothing actually showed anyone**, and
    docs/PLAN.md 4.2 rule 2 then permanently excludes them from reflection. A
    generator that quietly starves the reflection pass is a worse outcome than an
    absent generator.

Both were fixable, neither was fixable from this file, and the fix belonged to
whoever owns `recall.py`: `MemoryRecall.associate()` - no decay, `min_age_days` as
a floor instead, and no `mark_recalled` call. It existed unused until this file
grew `association_candidates` to call it: PLAN 6.1 names type E as the one that
gives the *Her*-like feeling, precisely because it has no business to transact,
and the other four generators only ever produce a reminder.

`association_candidates` is `async` and lives **outside** `generate_candidates`,
which is synchronous. `associate()` awaits the embedder - a local model, Lane 1's
existing allowance under non-negotiable 7, not a call that thinks - and a
synchronous function cannot await it. `tick.run()` is async and awaits the two
separately, then merges their output.

Type D's guards mean it produces nothing for the first two weeks of history. That
is correct, not a bug: "the hour this person usually talks" is not a fact yet.

## What keeps 288 ticks a day from becoming 288 candidates

Every candidate carries a deterministic `payload["dedup"]` key, and the tick
drops any key `proactive_candidates` has ever held. The keys are built so that
the natural unit of the reason is what gets deduplicated:

    open_loop     the source message's id     - one follow-up per thing said
    emotional     the source message's id     - one follow-up per feeling
    silence       the last message's timestamp - one per silence *episode*, since
                  a reply moves the timestamp and starts a new episode
    pattern_time  the local date              - one per day

The `silence` key has a consequence worth stating: a silence that was generated
and then blocked all the way through its expiry window never comes back, because
the key is already spent. That is deliberate. "It has been quiet too long" is one
observation, and a version of it that retries is an alarm clock.

The key lives in `payload` because `daemon/memory/schema.sql` is frozen and has no
column for it; a real column plus a unique index would be the better shape.

`cooldown_secs` is left at the schema default of a day and is deliberately *not*
wired to `proactive_cooldown_minutes`. There are two cooldowns and they are not the
same thing: this column is per-candidate ("do not raise *this* reason again"),
while the setting is the global gap between any two utterances and belongs to the
gate. Copying the setting in here would let five different candidates fire in five
minutes, each honouring its own cooldown.

## What ends up in the LLM prompt

`Candidate.reason` is fed to the model verbatim, so **no user text is put in it** -
only words from the lexicons in this file, plus numbers and clock times. A
follow-up therefore cannot be steered by what a forwarded message said, and A and
B additionally only read rows with `origin = 'owner'`. The source message id is in
`payload` for a caller that wants the actual words and can decide to trust them.

**`association_candidates` is the one exception, and it is bounded.** Type E has
nothing to ask about without the memory's own words - a bare "you said something
90 days ago" is the same contentless reason `silence` already produces, and the
judge correctly declines it - so it quotes `RecalledItem.content`, up to
`ASSOCIATION_QUOTE_CHARS` characters. What keeps this from being a hole in the
rule above rather than a stated exception to it: it quotes only items where
`origin == "owner"`. That column is the same unforgeable one CONTRACTS
non-negotiable 10 relies on to keep a forward from steering a tool call; here it
is what decides whether text may reach a prompt at all. An item recalled with any
other origin is dropped before the loop that builds `reason` ever sees its
content.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from typing import Protocol, runtime_checkable

from daemon.clock import now as clock_now
from daemon.clock import parse_iso, to_iso
from daemon.config import Settings
from daemon.memory.base import RecalledItem
from daemon.proactivity.base import Candidate, CandidateKind


@runtime_checkable
class CandidateReader(Protocol):
    """The reads a tick needs. `daemon.memory.store.Store` is the implementation.

    Synchronous, like the rest of `Store`: these are sqlite reads measured in
    milliseconds, and `MemoryRecall` already calls the store this way from an
    async caller. A tick that wants them off the loop can use `asyncio.to_thread`.

    "Conversation" always means `session_kind IN ('interactive', 'voice')`. The
    daemon's own proactive utterances and its reflection passes are not the user
    talking, and counting them would let one proactive message reset the silence
    clock - the loop where speaking is its own excuse not to notice the silence.
    """

    def last_conversation_at(self) -> datetime | None:
        """Timestamp of the most recent conversation message, or `None` if there
        has never been one."""
        ...

    def conversation_between(self, start: datetime, end: datetime) -> list[sqlite3.Row]:
        """Conversation rows with `start <= ts <= end`, oldest first.

        Returns whole rows because the callers need `id`, `ts`, `role`, `content`
        and `origin`; the role and origin filtering is policy and stays in this
        module, next to the comment saying why.
        """
        ...

    def conversation_times(self, since: datetime) -> list[datetime]:
        """Just the timestamps of conversation messages at or after `since`.

        Separate from `conversation_between` because the pattern histogram reads
        two months of history and wants none of the text.
        """
        ...

    def existing_dedup_keys(self, keys: Sequence[str]) -> set[str]:
        """Which of `keys` already appear as `payload.dedup` on any candidate row,
        whatever its state. Bounded by what the caller asks about, so it does not
        grow with the table."""
        ...


@runtime_checkable
class AssociativeRecall(Protocol):
    """The one recall entry point type E may use.

    Declared here rather than imported for the same reason `CandidateReader` is:
    this module depends on one method, not on `MemoryRecall`. `search` is
    deliberately *not* in this protocol - it multiplies by recency decay, which
    buries exactly what type E wants, and it calls `mark_recalled` from a
    background tick that shows nobody anything.
    """

    async def associate(
        self, query: str, *, limit: int = 3, min_age_days: float = 30.0
    ) -> list[RecalledItem]: ...


# --- tuning ------------------------------------------------------------------
# Every number here makes it speak less often or later. That direction is the
# whole design (non-negotiable 7), so a change that loosens one should say why.

FOLLOWUP_HOUR = 20
"""Local hour an open loop with no stated time becomes due. PLAN 6.1's own example
is "내일 발표" -> the next evening: late enough that the thing has happened."""

EVENT_LAG_HOURS = 2
"""How long after a *stated* time to ask how it went. Without this, "내일 밤 10시
발표" gets asked about at 20:00 - two hours before it happens."""

OPEN_LOOP_LOOKBACK_DAYS = 7
"""How far back to scan for things said. Covers the furthest marker (모레, +2d)
plus the follow-up window with room for a daemon that was switched off."""

OPEN_LOOP_TTL_HOURS = 24
"""Follow-up window after due. A day late is still a natural question; three days
late is a machine working through a backlog."""

EMOTION_LOOKBACK_HOURS = 30.0
EMOTION_MIN_QUIET_HOURS = 3.0
"""An emotional follow-up needs the conversation to have *ended*. Following up on
"힘들다" while the two of you are still talking about it is not a follow-up."""

EMOTION_TTL_HOURS = 12

SILENCE_TTL_HOURS = 12
"""Long enough to survive one night of quiet hours, short enough that the number
in the reason ("N시간 전") is still roughly true when it is spoken."""

PATTERN_WINDOW_DAYS = 60
PATTERN_MIN_DAYS = 14
"""Distinct days of history before "the hour they usually talk" means anything.
Below this the histogram is describing a week, not a habit."""

PATTERN_MIN_HITS = 5
PATTERN_MIN_SHARE = 0.3
"""Both must hold: an hour hit 5 times out of 14 days is a habit, and 5 times out
of 200 days is a coincidence that the absolute floor alone would let through."""

PATTERN_TTL_HOURS = 2

MAX_PER_KIND = 3
"""Rows one tick may add per kind. The gate owns the daily *utterance* budget; this
only stops a chatty afternoon from writing forty rows nobody will ever read."""

ASSOCIATION_LOOKBACK = 3
"""How many recent owner messages become the association query. Enough to carry a
topic, few enough that a single stray line does not define it."""

ASSOCIATION_MIN_AGE_DAYS = 30.0
"""Below this it is not an association, it is the conversation."""

ASSOCIATION_QUOTE_CHARS = 200
"""How much of the remembered message reaches the prompt. The judge needs the
words to have anything to ask about - a bare "you said something 90 days ago" is
the contentless reason that makes `silence` produce 빈말 - but the reason is
still a record being shown to a model, so it is bounded."""

ASSOCIATION_TTL_HOURS = 6


# --- Korean surface forms ----------------------------------------------------

_DAY_OFFSETS: tuple[tuple[str, int], ...] = (
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

_TENSE_NEUTRAL = frozenset({"오늘", "이따"})
"""These say *when* without saying *not yet*, so "오늘 발표 했어" would otherwise
become a candidate to ask about a presentation already reported on. They alone get
the past-tense guard; 내일/모레 do not need it and applying it there would throw
away the common "내일 발표 있어, 오늘 준비 많이 했어"."""

_PAST_MARKERS = (
    # Korean past tense contracts into a syllable rather than adding one, so this
    # is a list of contracted forms and not a suffix rule: 하였어->했어, 되었어->
    # 됐어, 보았어->봤어. Not exhaustive, and it does not need to be - a missed
    # past tense costs one unnecessary candidate, and the gate still has to pass it.
    "했", "었", "았", "였", "됐", "갔", "왔", "봤",
)

_EVENTS = (
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

_EVENT_CANCELLED = (
    # Said in the same message, these mean the thing is not happening. A missed
    # cancellation is the expensive false positive here: asking how a cancelled
    # exam went says plainly that nobody was listening.
    "취소", "연기", "미뤄", "없어졌", "안 하", "안 가", "없어", "없다", "없음",
)

_HOUR_RE = re.compile(r"(오전|오후|아침|저녁|밤)?\s*(\d{1,2})\s*시(?!간)")
"""`(?!간)` because "2시간 뒤에" is a duration, and matching `2시` inside it would
schedule the follow-up at 02:00."""

_EMOTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # (canonical word for the reason string, surface forms to look for). The
    # variants exist because the stem changes: 슬프 is not a prefix of 슬퍼.
    ("힘들다", ("힘들", "힘드네", "힘듦")),
    ("지치다", ("지쳤", "지쳐", "지친", "번아웃")),
    ("우울하다", ("우울",)),
    ("불안하다", ("불안",)),
    ("걱정되다", ("걱정",)),
    ("짜증나다", ("짜증", "열받")),
    ("스트레스", ("스트레스",)),
    ("외롭다", ("외롭", "외로워", "외로움", "혼자인")),
    ("슬프다", ("슬프", "슬퍼", "슬픔")),
    ("무섭다", ("무섭", "무서워", "무서움", "두렵", "두려워")),
    ("속상하다", ("속상",)),
    ("답답하다", ("답답",)),
    ("억울하다", ("억울",)),
    ("서럽다", ("서럽", "서러워")),
    ("울었다", ("눈물", "울었", "울고")),
    ("막막하다", ("막막", "막막하")),
    ("자신이 없다", ("자신 없", "자신이 없", "포기하고 싶")),
)
"""Only words a person uses about their own state. This misses most real emotion,
which arrives as "그냥 좀 그래" or a pause, and missing is the right failure: the
budget spent on a wrong guess is a budget the genuine moment does not get."""


# --- generators --------------------------------------------------------------


def open_loop_candidates(reader: CandidateReader, now: datetime) -> list[Candidate]:
    """Type A: something the user said would happen, whose time has now arrived.

    The follow-up window is `[due, due + OPEN_LOOP_TTL_HOURS]` and the candidate is
    generated only inside it, so a daemon that was off for a week does not wake up
    and ask about five stale things at once.
    """
    start = now - timedelta(days=OPEN_LOOP_LOOKBACK_DAYS)
    found: list[Candidate] = []
    for row in reader.conversation_between(start, now):
        if not _is_owner_utterance(row):
            continue
        text = str(row["content"])
        if _contains_any(text, _EVENT_CANCELLED):
            continue
        marker = _day_marker(text)
        event = _first_of(text, _EVENTS)
        if marker is None or event is None:
            continue
        surface, offset = marker
        if surface in _TENSE_NEUTRAL and _contains_any(text, _PAST_MARKERS):
            continue
        said_at = parse_iso(str(row["ts"]))
        due = _due_at(said_at, offset, _stated_hour(text))
        expires = due + timedelta(hours=OPEN_LOOP_TTL_HOURS)
        if not due <= now <= expires:
            continue
        message_id = int(row["id"])
        found.append(
            Candidate(
                kind="open_loop",
                # Only lexicon words and clock times: `surface` and `event` both
                # came from the tuples above, never from the message.
                reason=(
                    f"{_local(said_at):%m월 %d일}에 '{surface} {event}' 이야기를 했고, "
                    f"그 시각({_local(due):%m월 %d일 %H시})이 지났다. "
                    f"어떻게 됐는지 아직 듣지 못했다."
                ),
                payload={
                    "dedup": f"open_loop:{message_id}",
                    "message_id": message_id,
                    "event": event,
                    "when": surface,
                    "said_at": to_iso(said_at),
                },
                due_at=due,
                expires_at=expires,
            )
        )
    # Earliest due first, so if MAX_PER_KIND truncates, the thing that has been
    # waiting longest survives. `dedup_key` breaks ties for a stable order.
    found.sort(key=lambda candidate: (candidate.due_at or now, dedup_key(candidate)))
    return found


def emotional_candidates(reader: CandidateReader, now: datetime) -> list[Candidate]:
    """Type B: an emotional signal, plus enough elapsed time that it is a follow-up.

    Returns at most one. Three messages about the same bad evening are one feeling,
    and generating three candidates for it would spend the whole day's budget
    saying the same thing three ways.
    """
    last = reader.last_conversation_at()
    quiet_since = now - timedelta(hours=EMOTION_MIN_QUIET_HOURS)
    if last is None or last > quiet_since:
        # Still talking, or never talked. Either way this is not a follow-up.
        return []
    start = now - timedelta(hours=EMOTION_LOOKBACK_HOURS)
    for row in reversed(reader.conversation_between(start, quiet_since)):
        if not _is_owner_utterance(row):
            continue
        text = str(row["content"])
        signal = _emotion(text)
        if signal is None:
            continue
        said_at = parse_iso(str(row["ts"]))
        elapsed = _hours_between(said_at, now)
        message_id = int(row["id"])
        return [
            Candidate(
                kind="emotional",
                reason=(
                    f"{elapsed:.0f}시간 전에 '{signal}'는 얘기를 했고, "
                    f"그 뒤로 대화가 없었다. 그 뒤에 어떻게 됐는지 모른다."
                ),
                payload={
                    "dedup": f"emotional:{message_id}",
                    "message_id": message_id,
                    "signal": signal,
                    "said_at": to_iso(said_at),
                    "elapsed_hours": round(elapsed, 1),
                },
                due_at=now,
                expires_at=now + timedelta(hours=EMOTION_TTL_HOURS),
            )
        ]
    return []


def silence_candidates(
    reader: CandidateReader, settings: Settings, now: datetime
) -> list[Candidate]:
    """Type C: `proactive_silence_hours` with no conversation at all.

    Nothing is generated when there has never been a conversation. A daemon whose
    first act on a fresh install is to break a silence it was not part of is not
    reading a gap in a relationship, it is talking to a stranger.
    """
    last = reader.last_conversation_at()
    if last is None:
        return []
    elapsed = _hours_between(last, now)
    if elapsed < settings.proactive_silence_hours:
        return []
    return [
        Candidate(
            kind="silence",
            reason=(
                f"마지막 대화가 {elapsed:.0f}시간 전이고 그 뒤로 아무 말도 오가지 않았다. "
                f"평소 간격보다 길다."
            ),
            payload={
                # The last message's timestamp *is* the identity of this silence:
                # a reply moves it, which is what makes the next quiet stretch a
                # new episode rather than this one repeating.
                "dedup": f"silence:{to_iso(last)}",
                "last_at": to_iso(last),
                "silent_hours": round(elapsed, 1),
            },
            due_at=now,
            expires_at=now + timedelta(hours=SILENCE_TTL_HOURS),
        )
    ]


def pattern_time_candidates(reader: CandidateReader, now: datetime) -> list[Candidate]:
    """Type D: it is an hour this person usually talks, and today there is nothing.

    Local hours throughout, because "the time they usually talk" is a fact about
    their day and not about UTC.
    """
    days = _hours_by_local_day(reader.conversation_times(now - timedelta(days=PATTERN_WINDOW_DAYS)))
    today = _local(now).date()
    # Only the *absence* matters, so any conversation today at all disqualifies -
    # including one in a different hour. They have been in touch.
    if today in days or len(days) < PATTERN_MIN_DAYS:
        return []
    hour = _local(now).hour
    hits = sum(1 for hours in days.values() if hour in hours)
    if hits < max(PATTERN_MIN_HITS, math.ceil(PATTERN_MIN_SHARE * len(days))):
        return []
    return [
        Candidate(
            kind="pattern_time",
            reason=(
                f"최근 {len(days)}일 중 {hits}일은 이 시간(현지 {hour}시)에 대화를 했는데, "
                f"오늘은 아직 한 마디도 없다."
            ),
            payload={
                "dedup": f"pattern_time:{today.isoformat()}",
                "local_hour": hour,
                "observed_days": len(days),
                "hit_days": hits,
            },
            due_at=now,
            expires_at=now + timedelta(hours=PATTERN_TTL_HOURS),
        )
    ]


async def association_candidates(
    recall: AssociativeRecall,
    reader: CandidateReader,
    *,
    now: datetime | None = None,
) -> list[Candidate]:
    """Type E: an old memory the current conversation just brushed against.

    `async` and outside `generate_candidates` because `associate()` awaits the
    embedder and `generate_candidates` is synchronous - see the module docstring.
    `tick.run()` is async and merges the two.

    **This generator quotes the user's own words, which the rest of this module
    does not.** The rule it bends is stated at the top of the file and so is the
    exception: the source id is in `payload` "for a caller that wants the actual
    words and can decide to trust them", and `origin = 'owner'` is what deciding
    looks like. Text that arrived from anywhere else is dropped before it can
    reach a prompt - the same column CONTRACTS non-negotiable 10 relies on. Type
    E cannot work without this: with only elapsed days in the reason it produces
    exactly the 빈말 that `silence` produces.
    """
    moment = now or clock_now()
    recent = [
        str(row["content"])
        for row in reader.conversation_between(moment - timedelta(days=1), moment)
        if _is_owner_utterance(row)
    ][-ASSOCIATION_LOOKBACK:]
    if not recent:
        # Nothing said recently means nothing to associate *from* - a query built
        # out of an empty string would return whatever ranks highest overall,
        # which is not an association with anything.
        return []

    items = await recall.associate(
        " ".join(recent), limit=MAX_PER_KIND, min_age_days=ASSOCIATION_MIN_AGE_DAYS
    )
    found: list[Candidate] = []
    for item in items:
        if item.origin != "owner":
            continue
        if item.message_id is None:
            # The curated tier has no `messages.id`, so there is no stable dedup
            # key for it. Skipping costs a candidate; inventing one would let two
            # unrelated memories collide on the same key and silence the second.
            continue
        quote = " ".join(item.content.split())[:ASSOCIATION_QUOTE_CHARS]
        if not quote:
            continue
        key = f"association:{item.message_id}"
        found.append(
            Candidate(
                kind="association",
                reason=(
                    f"{_local(item.ts):%Y년 %m월 %d일}에 유저가 이런 얘기를 했다: "
                    f"'{quote}'. 지금 대화가 그 기억과 닿아 있다."
                ),
                payload={
                    "dedup": key,
                    "message_id": item.message_id,
                    "recalled_at": to_iso(item.ts),
                    "score": round(item.score, 3),
                },
                due_at=moment,
                expires_at=moment + timedelta(hours=ASSOCIATION_TTL_HOURS),
            )
        )
    spent = reader.existing_dedup_keys([dedup_key(c) for c in found])
    return [c for c in found if dedup_key(c) not in spent][:MAX_PER_KIND]


_KIND_ORDER: tuple[CandidateKind, ...] = ("open_loop", "emotional", "silence", "pattern_time")
"""Stable output order, and **not** a priority. Which kind gets the day's budget is
the gate's decision (`proactive_open_loop_budget` exists for exactly that), and
deciding it here by sort order would put the decision in two places."""


def generate_candidates(
    reader: CandidateReader,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> list[Candidate]:
    """One tick's worth of candidates, already deduplicated against the table.

    Returns rows to insert, in a stable order; it does not write them and it does
    not decide anything. Exceptions are allowed to propagate - a generator that
    silently stops producing looks exactly like a quiet day, and that is the
    failure mode this project keeps shipping.
    """
    if not settings.proactive_enabled:
        # Checked here so a disabled daemon accumulates no backlog to dump the
        # moment it is switched on. The user's own switch, honoured at the source.
        return []
    moment = now or clock_now()
    produced = (
        open_loop_candidates(reader, moment)
        + emotional_candidates(reader, moment)
        + silence_candidates(reader, settings, moment)
        + pattern_time_candidates(reader, moment)
    )
    if not produced:
        return []
    spent = reader.existing_dedup_keys([dedup_key(candidate) for candidate in produced])
    fresh: list[Candidate] = []
    per_kind: dict[str, int] = {}
    for kind in _KIND_ORDER:
        for candidate in produced:
            if candidate.kind != kind or dedup_key(candidate) in spent:
                continue
            if per_kind.get(kind, 0) >= MAX_PER_KIND:
                continue
            per_kind[kind] = per_kind.get(kind, 0) + 1
            fresh.append(candidate)
    return fresh


def dedup_key(candidate: Candidate) -> str:
    """The candidate's identity for deduplication. See the module docstring."""
    return str(candidate.payload["dedup"])


# --- helpers -----------------------------------------------------------------


def _is_owner_utterance(row: sqlite3.Row) -> bool:
    """Only what the owner themselves said, in their own words.

    `origin` is a column precisely so this is decidable (non-negotiable 3):
    relayed text - a forward, an inline-bot result - is not the user telling us
    they have a presentation tomorrow, and treating it as such would let a third
    party schedule the daemon's attention.
    """
    return row["role"] == "user" and row["origin"] == "owner"


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def _first_of(text: str, needles: Sequence[str]) -> str | None:
    """The first of `needles` present, in `needles` order rather than text order,
    so the caller controls precedence."""
    for needle in needles:
        if needle in text:
            return needle
    return None


def _day_marker(text: str) -> tuple[str, int] | None:
    for surface, offset in _DAY_OFFSETS:
        if surface in text:
            return surface, offset
    return None


def _emotion(text: str) -> str | None:
    for canonical, variants in _EMOTIONS:
        for variant in variants:
            if variant not in text:
                continue
            if _negated(text, variant):
                continue
            return canonical
    return None


def _negated(text: str, variant: str) -> bool:
    """Whether this occurrence is denied rather than asserted.

    Korean negates two ways and both are local to the word: `안 힘들어` before it
    and `힘들지 않아` after it. The trailing window is five characters rather than
    three because an adjective takes 하 first - `우울하지 않아` - and it is kept
    narrow so that "우울해서 아무것도 하지 않았어" stays a signal. Only the first
    occurrence is checked; a message that both asserts and denies the same feeling
    is not one this module can read.
    """
    at = text.find(variant)
    before = text[max(0, at - 2) : at]
    after = text[at + len(variant) : at + len(variant) + 5]
    return "안 " in before or before.endswith("안") or "지 않" in after


def _stated_hour(text: str) -> int | None:
    """The hour the user named, in 24h local, or `None` if they named none."""
    match = _HOUR_RE.search(text)
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


def _due_at(said_at: datetime, day_offset: int, stated_hour: int | None) -> datetime:
    """When the follow-up becomes due, as UTC.

    Built in local time and converted, because the offset is a number of *days* on
    the user's calendar: "내일" said at 23:30 KST is a different UTC date already.
    A naive datetime's `.astimezone()` resolves it with the offset in force on that
    date, which is the only form of this that stays correct across a DST boundary.
    """
    target = _local(said_at).date() + timedelta(days=day_offset)
    hour = FOLLOWUP_HOUR
    if stated_hour is not None:
        hour = max(FOLLOWUP_HOUR, min(23, stated_hour + EVENT_LAG_HOURS))
    return datetime.combine(target, time(hour=hour)).astimezone().astimezone(UTC)


def _hours_by_local_day(times: Sequence[datetime]) -> dict[date, set[int]]:
    """Which local hours saw a message, grouped by local calendar day."""
    days: dict[date, set[int]] = {}
    for moment in times:
        local = _local(moment)
        days.setdefault(local.date(), set()).add(local.hour)
    return days


def _hours_between(earlier: datetime, later: datetime) -> float:
    return (later - earlier).total_seconds() / 3600.0


def _local(moment: datetime) -> datetime:
    """The same instant in the machine's local zone, matching `log.local_date`.

    Local, not UTC, wherever a *human* boundary is involved - which day it is, what
    hour they usually talk. docs/PLAN.md 4.2: days are split locally, timestamps
    stay UTC.
    """
    return moment.astimezone()
