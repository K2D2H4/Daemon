"""The M3 tables: proactive candidates and what was actually said.

The state machine is the interesting part. A candidate that stays `pending`
forever is indistinguishable from one still waiting, and PLAN 6.1's whole design
rests on being able to read back *why* nothing spoke.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from daemon.memory.store import Store

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(db: Any) -> Store:
    return Store(db)


def candidate(store: Store, kind: str = "silence", **kwargs: Any) -> int:
    defaults: dict[str, Any] = {
        "reason": "20시간 동안 대화가 없었다",
        "payload": json.dumps({}),
        "now": NOW,
    }
    return store.insert_candidate(kind=kind, **{**defaults, **kwargs})


def utterance(
    store: Store,
    uid: str,
    *,
    kind: str = "silence",
    at: datetime = NOW,
    text: str = "자기 전에 한마디",
) -> None:
    store.insert_utterance(
        utterance_id=uid,
        candidate_id=None,
        kind=kind,
        text=text,
        route="telegram",
        gate_snapshot=json.dumps({"allowed": True, "why": "ok"}),
        now=at,
    )


# --- candidates -------------------------------------------------------------


def test_a_candidate_starts_pending_and_is_live(store: Store) -> None:
    candidate(store)

    rows = store.live_candidates(now=NOW)
    assert [row["state"] for row in rows] == ["pending"]
    assert rows[0]["kind"] == "silence"


def test_a_candidate_is_not_due_until_its_time(store: Store) -> None:
    candidate(store, kind="open_loop", due_at=NOW + timedelta(hours=6))

    assert store.due_candidates(now=NOW) == []
    assert len(store.due_candidates(now=NOW + timedelta(hours=7))) == 1


def test_a_candidate_with_no_due_time_is_due_now(store: Store) -> None:
    """`silence` has no appointment - the condition either holds or it does not."""
    candidate(store)
    assert len(store.due_candidates(now=NOW)) == 1


def test_an_expired_candidate_is_neither_live_nor_due(store: Store) -> None:
    candidate(store, expires_at=NOW + timedelta(hours=1))
    later = NOW + timedelta(hours=2)

    assert store.live_candidates(now=later) == []
    assert store.due_candidates(now=later) == []


def test_expiring_moves_the_state_rather_than_just_filtering(store: Store) -> None:
    """A row left `pending` forever reads the same as one still waiting, and the
    state machine is what a human consults to understand the silence."""
    candidate(store, expires_at=NOW + timedelta(hours=1))

    assert store.expire_candidates(now=NOW + timedelta(hours=2)) == 1

    row = store.conn.execute("SELECT state FROM proactive_candidates").fetchone()
    assert row["state"] == "expired"


def test_expiring_leaves_a_candidate_with_no_expiry_alone(store: Store) -> None:
    candidate(store)
    assert store.expire_candidates(now=NOW + timedelta(days=365)) == 0


def test_firing_once_with_a_budget_of_one_finishes_the_candidate(store: Store) -> None:
    cid = candidate(store)

    store.mark_candidate_fired(cid, now=NOW)

    row = store.conn.execute("SELECT * FROM proactive_candidates").fetchone()
    assert (row["state"], row["fire_count"]) == ("done", 1)
    assert store.due_candidates(now=NOW) == []


def test_a_candidate_allowed_two_firings_is_not_retired_after_the_first(store: Store) -> None:
    cid = candidate(store, fire_budget=2)

    store.mark_candidate_fired(cid, now=NOW)

    row = store.conn.execute("SELECT * FROM proactive_candidates").fetchone()
    assert (row["state"], row["fire_count"]) == ("fired", 1)
    # Still live, because its own budget is not spent...
    assert len(store.live_candidates(now=NOW)) == 1
    # ...but not offered again until its own cooldown has passed.
    assert store.due_candidates(now=NOW) == []
    assert len(store.due_candidates(now=NOW + timedelta(days=1, seconds=1))) == 1

    store.mark_candidate_fired(cid, now=NOW)
    spent = store.conn.execute("SELECT state FROM proactive_candidates").fetchone()
    assert spent["state"] == "done"


def test_a_candidates_own_cooldown_is_separate_from_the_global_gap(store: Store) -> None:
    """Two cooldowns, deliberately. This one says "do not raise *this* again for a
    day"; the gate owns the gap between any two utterances. Conflated, five
    different candidates could fire in five minutes, each inside its own cooldown.
    """
    cid = candidate(store, fire_budget=3, cooldown_secs=3600)
    store.mark_candidate_fired(cid, now=NOW)

    assert store.due_candidates(now=NOW + timedelta(minutes=59)) == []
    assert len(store.due_candidates(now=NOW + timedelta(minutes=61))) == 1


def test_a_cancelled_candidate_stops_being_offered(store: Store) -> None:
    cid = candidate(store)
    store.set_candidate_state(cid, "cancelled")
    assert store.due_candidates(now=NOW) == []


def test_due_candidates_come_back_soonest_first(store: Store) -> None:
    candidate(store, kind="open_loop", reason="늦은 것", due_at=NOW - timedelta(hours=1))
    candidate(store, kind="open_loop", reason="더 늦은 것", due_at=NOW - timedelta(hours=5))

    assert [row["reason"] for row in store.due_candidates(now=NOW)] == ["더 늦은 것", "늦은 것"]


def test_the_payload_survives_a_round_trip(store: Store) -> None:
    candidate(store, kind="open_loop", payload=json.dumps({"about": "발표", "when": "내일"}))

    row = store.live_candidates(now=NOW)[0]
    assert json.loads(row["payload"]) == {"about": "발표", "when": "내일"}


# --- utterances -------------------------------------------------------------


def test_an_utterance_is_recorded_with_its_gate_snapshot(store: Store) -> None:
    """The snapshot is why the column exists: a bad call has to be diagnosable
    later instead of guessed at."""
    utterance(store, "u1")

    row = store.utterances_since(since=NOW - timedelta(minutes=1))[0]
    assert row["id"] == "u1"
    assert json.loads(row["gate_snapshot"])["why"] == "ok"
    assert row["label"] is None


def test_the_cooldown_reads_the_most_recent_utterance(store: Store) -> None:
    utterance(store, "old", at=NOW - timedelta(hours=5))
    utterance(store, "new", at=NOW - timedelta(minutes=10))

    assert store.last_utterance_at() == NOW - timedelta(minutes=10)


def test_nothing_spoken_yet_is_none_not_an_error(store: Store) -> None:
    assert store.last_utterance_at() is None


def test_the_budget_window_is_a_boundary_the_caller_chooses(store: Store) -> None:
    """`spoken_at` is UTC and the budget is per *local* day, so the caller converts
    its day boundary rather than this method guessing which day it is."""
    utterance(store, "inside", at=NOW - timedelta(hours=2))
    utterance(store, "outside", at=NOW - timedelta(hours=30))

    rows = store.utterances_since(since=NOW - timedelta(hours=12))
    assert [row["id"] for row in rows] == ["inside"]


def test_recent_utterance_texts_is_oldest_first(store: Store) -> None:
    """Task 6: `persona.tics.verbal_tics`'s `said` input, read from this table
    rather than `messages` - see the method's docstring for why. Oldest first,
    matching `Store.recent`'s convention for anything meant to become a window."""
    utterance(store, "old", at=NOW - timedelta(hours=2), text="오래된 말")
    utterance(store, "new", at=NOW - timedelta(minutes=1), text="최근 말")

    assert store.recent_utterance_texts() == ["오래된 말", "최근 말"]


def test_recent_utterance_texts_limit_keeps_the_newest(store: Store) -> None:
    for i in range(5):
        utterance(store, f"u{i}", at=NOW - timedelta(hours=5 - i), text=f"{i}번째")

    assert store.recent_utterance_texts(limit=2) == ["3번째", "4번째"]


def test_labelling_records_the_verdict(store: Store) -> None:
    utterance(store, "u1")

    assert store.label_utterance("u1", "bad", now=NOW) is True

    row = store.utterances_since(since=NOW - timedelta(minutes=1))[0]
    assert row["label"] == "bad"
    assert row["labeled_at"] is not None


def test_labelling_something_that_does_not_exist_says_so(store: Store) -> None:
    """The label arrives from a button press carrying an id, so a stale or forged
    id has to be distinguishable from a successful label."""
    assert store.label_utterance("nope", "good", now=NOW) is False


def test_only_good_or_bad_can_be_stored(store: Store) -> None:
    import sqlite3

    utterance(store, "u1")
    with pytest.raises(sqlite3.IntegrityError):
        store.label_utterance("u1", "maybe", now=NOW)


def test_a_reply_is_a_label_nobody_pressed(store: Store) -> None:
    utterance(store, "u1")
    store.mark_responded("u1")

    assert store.label_counts()["responded"] == 1


def test_the_counts_are_what_doctor_prints(store: Store) -> None:
    """"Is it a stalker or a dead bot" is the M3 gate, and it is not answerable
    from a log line."""
    utterance(store, "a")
    utterance(store, "b")
    utterance(store, "c")
    store.label_utterance("a", "good", now=NOW)
    store.label_utterance("b", "bad", now=NOW)

    counts = store.label_counts()
    assert counts["good"] == 1
    assert counts["bad"] == 1
    assert counts["unlabeled"] == 1


def test_an_utterance_can_point_back_at_its_candidate(store: Store) -> None:
    cid = candidate(store)
    store.insert_utterance(
        utterance_id="u1",
        candidate_id=cid,
        kind="silence",
        text="자기 전에 한마디",
        route="both",
        gate_snapshot="{}",
        now=NOW,
    )

    row = store.utterances_since(since=NOW - timedelta(minutes=1))[0]
    assert row["candidate_id"] == cid
    assert row["route"] == "both"


# --- the 👎 brake's reads -----------------------------------------------------


def test_recent_bad_labels_reads_kind_and_when(store: Store) -> None:
    utterance(store, "u1", kind="association")
    store.label_utterance("u1", "bad", now=NOW)

    assert store.recent_bad_labels(since=NOW - timedelta(minutes=1)) == [("association", NOW)]


def test_recent_bad_labels_excludes_good_and_unlabeled(store: Store) -> None:
    utterance(store, "u1", kind="association")
    utterance(store, "u2", kind="emotional")
    store.label_utterance("u1", "good", now=NOW)

    assert store.recent_bad_labels(since=NOW - timedelta(minutes=1)) == []


def test_recent_bad_labels_are_newest_first(store: Store) -> None:
    utterance(store, "u1", kind="association", at=NOW - timedelta(hours=2))
    utterance(store, "u2", kind="association", at=NOW - timedelta(hours=1))
    store.label_utterance("u1", "bad", now=NOW - timedelta(hours=2))
    store.label_utterance("u2", "bad", now=NOW - timedelta(hours=1))

    rows = store.recent_bad_labels(since=NOW - timedelta(hours=3))
    assert [at for _, at in rows] == [NOW - timedelta(hours=1), NOW - timedelta(hours=2)]


def test_recent_bad_labels_respects_the_since_boundary(store: Store) -> None:
    """The gate passes its own lookback rather than asking the store to guess
    one - see `recent_bad_labels`'s docstring. A row just outside it must not
    come back."""
    utterance(store, "old", at=NOW - timedelta(hours=30))
    store.label_utterance("old", "bad", now=NOW - timedelta(hours=30))

    assert store.recent_bad_labels(since=NOW - timedelta(hours=24)) == []
