"""The M2 tables: curated entries, entity notes, observations.

Kept out of test_store.py because these three tiers have their own rules -
supersession must be atomic, entity `kind` must never be erased, and the
observation table must have no way to rewrite history.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from daemon.memory.base import LoggedMessage
from daemon.memory.store import Store

NOW = datetime(2026, 8, 3, 7, 14, tzinfo=UTC)


@pytest.fixture
def store(db: Any) -> Store:
    return Store(db)


def message(content: str, *, kind: str = "interactive", role: str = "user") -> LoggedMessage:
    return LoggedMessage(
        ts=NOW,
        role=role,  # type: ignore[arg-type]
        content=content,
        origin="owner" if role == "user" else "agent",
        session_kind=kind,  # type: ignore[arg-type]
        modality="text",
        channel="telegram",
        sender_id="42",
    )


def entry(store: Store, body: str, **kwargs: Any) -> int:
    defaults = dict(
        importance=5,
        trigger_phrases=(),
        origin="agent",
        session_kind="reflection",
        modality="text",
        now=NOW,
    )
    return store.insert_entry(body=body, **{**defaults, **kwargs})


# --- the curated tier -------------------------------------------------------


def test_an_entry_is_active_and_readable_back(store: Store) -> None:
    entry(store, "김치찌개를 좋아한다", importance=7, trigger_phrases=("김치찌개", "매운"))

    rows = store.active_entries()
    assert [row["body"] for row in rows] == ["김치찌개를 좋아한다"]
    assert json.loads(rows[0]["trigger_phrases"]) == ["김치찌개", "매운"]
    assert store.count_entries() == 1


def test_a_new_fact_retires_the_old_one_with_the_same_key(store: Store) -> None:
    """docs/PLAN.md 4.4: facts replace, they do not pile up a contradiction."""
    old = entry(store, "여자친구가 있다", supersession_key="relationship")
    new = entry(store, "여자친구가 없다", supersession_key="relationship")

    assert [row["body"] for row in store.active_entries()] == ["여자친구가 없다"]
    assert store.count_entries() == 1

    retired = store.conn.execute("SELECT * FROM memory_entries WHERE id = ?", (old,)).fetchone()
    assert retired["status"] == "retired"
    assert retired["superseded_by"] == new


def test_entries_without_a_key_coexist(store: Store) -> None:
    """No key means no claim of exclusivity - two unrelated facts must both live."""
    entry(store, "고양이를 키운다")
    entry(store, "연희동에 산다")
    assert store.count_entries() == 2


def test_a_failed_supersession_leaves_the_old_fact_active(store: Store) -> None:
    """The retire and the insert are one transaction.

    Two commits would leave a window with no active row for the key, and a failure
    after the retire would lose the fact from the mirror entirely - so the old row
    has to survive a failed replacement.
    """
    entry(store, "여자친구가 있다", supersession_key="relationship")

    with pytest.raises(sqlite3.IntegrityError):
        entry(store, "여자친구가 없다", supersession_key="relationship", importance=99)

    assert [row["body"] for row in store.active_entries()] == ["여자친구가 있다"]
    assert store.count_entries() == 1


def test_the_retire_decision_is_read_inside_the_write_transaction(store: Store) -> None:
    """Which rows to retire is decided by a SELECT, so that SELECT has to be under
    the write lock. It was not: sqlite's legacy transaction handling opens the
    implicit BEGIN on the first *write*, so the read ran in autocommit.

    That is a check-then-act across connections. Replaying the statements with a
    second writer committing in the window: A reads row 1 as active, B supersedes it
    and commits, A retires row 1 again and repoints it - `superseded_by` moves off
    the successor that earned it, and both successors stay active claiming the same
    fact. `daemon reflect` run by hand during the 04:00 pass is two writers, and
    `daemon/app.py` opens a second `Store` for it.

    The window between the two statements is too small to hold open from outside, so
    this asserts the ordering that closes it rather than racing it: the transaction
    is open before the read happens.
    """
    entry(store, "원래 사실")
    statements: list[str] = []
    store.conn.set_trace_callback(lambda sql: statements.append(" ".join(sql.split())))
    try:
        entry(store, "새로운 사실", supersedes=1)
    finally:
        store.conn.set_trace_callback(None)

    read_at = next(
        index
        for index, sql in enumerate(statements)
        if sql.startswith("SELECT id FROM memory_entries")
    )
    began = [index for index, sql in enumerate(statements) if sql.startswith("BEGIN IMMEDIATE")]
    assert began, "the write transaction must be opened explicitly, not by the first write"
    assert began[0] < read_at, "the retire set was read before the write lock was taken"


def test_the_budget_drops_the_least_important_not_the_oldest(store: Store) -> None:
    """This tier is always injected under a budget, so the order it comes back in
    decides what survives truncation."""
    entry(store, "사소한 것", importance=2)
    entry(store, "중요한 것", importance=9)

    assert [row["body"] for row in store.active_entries(limit=1)] == ["중요한 것"]


# --- entity notes -----------------------------------------------------------


def test_an_entity_counts_its_mentions(store: Store) -> None:
    first = store.upsert_entity(
        name="지현", kind="person", file="memory/entities/지현.md", now=NOW
    )
    again = store.upsert_entity(
        name="지현", kind=None, file="memory/entities/지현.md", now=NOW
    )

    assert first == again
    row = store.entity_by_name("지현")
    assert row is not None
    assert row["mention_count"] == 2


def test_a_later_mention_never_erases_a_known_kind(store: Store) -> None:
    """A pass that mentions someone without classifying them must not undo what an
    earlier pass worked out."""
    store.upsert_entity(name="지현", kind="person", file="memory/entities/지현.md", now=NOW)
    store.upsert_entity(name="지현", kind=None, file="memory/entities/지현.md", now=NOW)

    row = store.entity_by_name("지현")
    assert row is not None
    assert row["kind"] == "person"


def test_a_link_reads_from_either_end(store: Store) -> None:
    a = store.upsert_entity(name="지현", kind="person", file="memory/entities/지현.md", now=NOW)
    b = store.upsert_entity(name="연희동", kind="place", file="memory/entities/연희동.md", now=NOW)

    store.link_entities(a, b)

    assert [row["name"] for row in store.links_for(a)] == ["연희동"]
    assert [row["name"] for row in store.links_for(b)] == ["지현"]


def test_linking_an_entity_to_itself_is_dropped_not_raised(store: Store) -> None:
    a = store.upsert_entity(name="지현", kind="person", file="memory/entities/지현.md", now=NOW)
    store.link_entities(a, a)
    assert store.links_for(a) == []


def test_the_same_link_twice_is_not_a_duplicate(store: Store) -> None:
    a = store.upsert_entity(name="지현", kind="person", file="memory/entities/지현.md", now=NOW)
    b = store.upsert_entity(name="연희동", kind="place", file="memory/entities/연희동.md", now=NOW)

    store.link_entities(a, b)
    store.link_entities(b, a)

    assert len(store.links_for(a)) == 1


def test_entities_come_back_most_mentioned_first(store: Store) -> None:
    store.upsert_entity(name="한번", kind=None, file="memory/entities/한번.md", now=NOW)
    for _ in range(3):
        store.upsert_entity(name="자주", kind=None, file="memory/entities/자주.md", now=NOW)

    assert [row["name"] for row in store.entities()] == ["자주", "한번"]


# --- observations -----------------------------------------------------------


def test_an_observation_is_appended_and_unconsumed(store: Store) -> None:
    store.insert_observation(
        body="아침에는 짧은 메시지가 낫다",
        observed_from="2026-08-01/2026-08-03",
        now=NOW,
        confidence=0.7,
    )

    rows = store.unconsumed_observations()
    assert [row["body"] for row in rows] == ["아침에는 짧은 메시지가 낫다"]
    assert rows[0]["confidence"] == 0.7
    assert rows[0]["origin"] == "agent"
    assert store.count_observations() == 1


def test_the_store_offers_no_way_to_rewrite_an_observation(store: Store) -> None:
    """The log clock only counts up. An observation that can be edited later is
    not evidence of anything (docs/PLAN.md 8.1), so there is deliberately no
    update or delete on this table - asserted here so adding one is a decision
    somebody makes on purpose rather than a convenience that slips in.
    """
    forbidden = [
        name
        for name in dir(Store)
        if "observation" in name and any(verb in name for verb in ("update", "delete", "retire"))
    ]
    assert forbidden == []


def test_a_consumed_observation_drops_out_of_the_queue(store: Store) -> None:
    store.insert_observation(body="관찰", observed_from="2026-08-03/2026-08-03", now=NOW)
    store.conn.execute(
        "INSERT INTO persona_rules (body, status, created_at) VALUES ('규칙', 'active', ?)",
        ("2026-08-03T07:14:00Z",),
    )
    store.conn.execute("UPDATE observations SET consumed_by = 1")
    store.conn.commit()

    assert store.unconsumed_observations() == []
    assert store.count_observations() == 1  # still there; it was consumed, not removed


# --- what reflection is allowed to read -------------------------------------


def _record(store: Store, msg: LoggedMessage, *, date: str = "2026-08-03") -> int:
    return store.insert_message(msg, log_file=f"memory/log/{date}.md")


def test_reflection_reads_one_local_day_in_reading_order(store: Store) -> None:
    _record(store, message("첫 번째"))
    _record(store, message("두 번째", role="assistant"))
    _record(store, message("다른 날"), date="2026-08-02")

    rows = store.messages_for_day("2026-08-03")
    assert [row["content"] for row in rows] == ["첫 번째", "두 번째"]


def test_reflection_never_reads_the_daemons_own_speech(store: Store) -> None:
    """Hygiene rule 1: if proactive and reflection output became evidence, the
    loop would amplify itself."""
    _record(store, message("유저가 한 말"))
    _record(store, message("내가 먼저 건 말", kind="proactive", role="assistant"))
    _record(store, message("성찰이 만든 말", kind="reflection", role="assistant"))

    assert [row["content"] for row in store.messages_for_day("2026-08-03")] == ["유저가 한 말"]


def test_reflection_reads_a_row_recall_has_surfaced(store: Store) -> None:
    """Hygiene rule 2 is retired, and this is the assertion that used to say the
    opposite.

    It excluded `recalled = 1` permanently, which on one real day removed 29 of 38
    messages - the persona-relevant ones, leaving wake-word noise - and it blocked
    nothing: recall's hits go into the prompt as a system block, and the only rows
    written are the user's turn and the reply, so injected text is never a row to
    re-extract. See `Store.messages_for_day`.
    """
    fresh = _record(store, message("새 증거"))
    surfaced = _record(store, message("회상이 한 번 보여준 것"))
    store.mark_recalled([surfaced])

    rows = store.messages_for_day("2026-08-03")
    assert [row["id"] for row in rows] == [fresh, surfaced]


def test_the_day_filter_is_the_log_file_not_a_timestamp_range(store: Store) -> None:
    """Days split locally, timestamps are UTC. A KST log file legitimately holds
    records whose UTC date is the day before, so a BETWEEN on `ts` would reflect a
    shifted window - and silently.
    """
    # 07:14Z minus nine hours is the previous UTC day, but the same KST day.
    shifted = replace(message("자기 전에"), ts=NOW - timedelta(hours=9))
    _record(store, shifted, date="2026-08-03")

    assert [row["content"] for row in store.messages_for_day("2026-08-03")] == ["자기 전에"]


def test_a_day_with_nothing_in_it_is_empty_not_an_error(store: Store) -> None:
    assert store.messages_for_day("2020-01-01") == []
