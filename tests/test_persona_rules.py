"""`persona/learned.md`, its mirror, and the write order."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from daemon.memory.store import Store
from daemon.persona import rules
from daemon.persona.loader import learned_path
from daemon.persona.rules import LearnedFileDiverged, LearnedRules, Proposal

NOW = datetime(2026, 8, 3, 7, 14, tzinfo=UTC)


@pytest.fixture
def store(db: Any) -> Store:
    return Store(db)


@pytest.fixture
def learned(data_dir: Path, store: Store) -> LearnedRules:
    return LearnedRules(data_dir, store)


# --- the file -----------------------------------------------------------


async def test_a_rule_lands_in_the_markdown_and_the_mirror(
    learned: LearnedRules, data_dir: Path, store: Store
) -> None:
    ids = await learned.add([Proposal(body="아침엔 짧게 말한다", evidence=(1,))], now=NOW)

    assert len(ids) == 1
    text = learned_path(data_dir).read_text(encoding="utf-8")
    assert "아침엔 짧게 말한다" in text
    assert [row["body"] for row in learned.active()] == ["아침엔 짧게 말한다"]


async def test_the_file_is_owner_only(learned: LearnedRules, data_dir: Path) -> None:
    await learned.add([Proposal(body="아침엔 짧게 말한다", evidence=())], now=NOW)
    assert learned_path(data_dir).stat().st_mode & 0o777 == 0o600


async def test_a_multi_line_body_is_folded_to_one_line(
    learned: LearnedRules, data_dir: Path
) -> None:
    await learned.add([Proposal(body="첫 줄\n두 번째 줄", evidence=())], now=NOW)
    assert learned_path(data_dir).read_text(encoding="utf-8").count("- 첫 줄 두 번째 줄") == 1


async def test_a_long_body_is_clamped(learned: LearnedRules) -> None:
    body = "가" * 500
    await learned.add([Proposal(body=body, evidence=())], now=NOW)
    assert len(learned.active()[0]["body"]) == rules.MAX_BODY_CHARS


async def test_an_empty_batch_is_a_no_op(learned: LearnedRules, data_dir: Path) -> None:
    assert await learned.add([], now=NOW) == []
    assert not learned_path(data_dir).exists()


# --- supersession ---------------------------------------------------------


async def test_a_new_rule_retires_the_old_one_sharing_its_key(
    learned: LearnedRules, data_dir: Path
) -> None:
    await learned.add(
        [Proposal(body="아침엔 인사만 한다", evidence=(), supersession_key="morning")], now=NOW
    )
    await learned.add(
        [Proposal(body="아침엔 안부도 묻는다", evidence=(), supersession_key="morning")], now=NOW
    )

    assert [row["body"] for row in learned.active()] == ["아침엔 안부도 묻는다"]
    assert learned_path(data_dir).read_text(encoding="utf-8").count("아침엔") == 1


async def test_two_proposals_with_one_key_in_one_batch_are_resolved_deterministically(
    learned: LearnedRules,
) -> None:
    """No unique index protects `persona_rules` the way `idx_memory_supersession`
    protects the curated tier - applying these one at a time would let the
    second retire the first mid-batch, exactly docs/PLAN.md 8.2.1's defect."""
    ids = await learned.add(
        [
            Proposal(body="첫 번째 제안", evidence=(), supersession_key="k"),
            Proposal(body="두 번째 제안", evidence=(), supersession_key="k"),
        ],
        now=NOW,
    )

    assert len(ids) == 1
    assert [row["body"] for row in learned.active()] == ["첫 번째 제안"]


async def test_resolve_supersessions_keeps_the_first_and_reports_the_rest() -> None:
    a = Proposal(body="a", evidence=(), supersession_key="k")
    b = Proposal(body="b", evidence=(), supersession_key="k")
    c = Proposal(body="c", evidence=(), supersession_key=None)

    kept, discarded = rules.resolve_supersessions([a, b, c])

    assert kept == [a, c]
    assert discarded == [b]


# --- observation consumption ----------------------------------------------


async def test_evidence_observations_are_marked_consumed(
    learned: LearnedRules, store: Store
) -> None:
    obs_id = store.insert_observation(
        body="아침엔 짧게", observed_from="2026-08-01/2026-08-01", now=NOW
    )

    [rule_id] = await learned.add(
        [Proposal(body="아침엔 짧게 말한다", evidence=(obs_id,))], now=NOW
    )

    assert store.unconsumed_observations() == []
    row = store.conn.execute(
        "SELECT consumed_by FROM observations WHERE id = ?", (obs_id,)
    ).fetchone()
    assert row["consumed_by"] == rule_id


# --- the write order -------------------------------------------------------


async def test_the_markdown_is_written_before_the_mirror_commits(
    learned: LearnedRules, data_dir: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asserts the *order*, not just the outcome - a second connection must not
    see the row while the file is being written."""
    seen_by_other_connection: list[int] = []
    real = rules.write_private_replace

    def observe(path: Path, text: str) -> None:
        other = sqlite3.connect(data_dir / "daemon.sqlite3")
        try:
            row = other.execute("SELECT COUNT(*) FROM persona_rules").fetchone()
            seen_by_other_connection.append(int(row[0]))
        finally:
            other.close()
        real(path, text)

    monkeypatch.setattr(rules, "write_private_replace", observe)
    await learned.add([Proposal(body="아직 커밋 안 된 규칙", evidence=())], now=NOW)

    assert seen_by_other_connection == [0]
    assert [row["body"] for row in learned.active()] == ["아직 커밋 안 된 규칙"]


async def test_a_failed_markdown_write_leaves_the_mirror_untouched(
    learned: LearnedRules, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-negotiable 1: if the file cannot be written, the mirror must not
    have moved either."""
    await learned.add([Proposal(body="먼저 있던 규칙", evidence=())], now=NOW)

    def boom(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(rules, "write_private_replace", boom)

    with pytest.raises(OSError):
        await learned.add([Proposal(body="추가하려던 규칙", evidence=())], now=NOW)

    assert [row["body"] for row in learned.active()] == ["먼저 있던 규칙"]


# --- divergence: the file knows about a rule the mirror does not -----------
#
# docs/CONTRACTS.md non-negotiable 1: deleting daemon.sqlite3 must never lose
# user data. `add()` used to compute the whole new file from the mirror alone,
# so the very next ordinary write would silently render a file missing every
# bullet the mirror did not know about. These tests are the reproduction from
# the bug report, not a synthetic stand-in for it.


async def test_add_refuses_when_the_mirror_is_behind_the_file(
    learned: LearnedRules, data_dir: Path, store: Store
) -> None:
    """The exact reproduction: `rm daemon.sqlite3` (a legitimate state per
    non-negotiable 1) leaves the mirror empty while learned.md still has
    every rule a human might be reading right now. The next ordinary `add()`
    must refuse rather than overwrite the file with a fresh one computed from
    the (empty) mirror."""
    bodies = [f"규칙 {i}" for i in range(5)]
    learned_path(data_dir).write_text(rules.render(bodies), encoding="utf-8")
    obs_id = store.insert_observation(
        body="관찰", observed_from="2026-08-01/2026-08-01", now=NOW
    )

    with pytest.raises(LearnedFileDiverged) as exc_info:
        await learned.add([Proposal(body="새 규칙", evidence=(obs_id,))], now=NOW)

    assert sorted(exc_info.value.orphaned_bodies) == sorted(bodies)

    text = learned_path(data_dir).read_text(encoding="utf-8")
    for body in bodies:
        assert body in text
    assert "새 규칙" not in text
    assert learned.active() == [], "nothing should have been written to the mirror either"
    assert [row["id"] for row in store.unconsumed_observations()] == [obs_id], (
        "the observation must not be consumed by a rule that was never added"
    )


async def test_forget_refuses_when_the_mirror_is_behind_the_file(
    learned: LearnedRules, data_dir: Path, store: Store
) -> None:
    """`retire` rewrites the file from the mirror exactly as `add` does, so it had
    the same hole: forgetting one rule on a diverged file would take every
    orphaned bullet with it. A deletion request must never cost the user rules
    they did not name."""
    kept = await learned.add([Proposal(body="미러가 아는 규칙", evidence=())], now=NOW)
    orphan = "미러가 모르는 규칙"
    learned_path(data_dir).write_text(
        rules.render(["미러가 아는 규칙", orphan]), encoding="utf-8"
    )

    with pytest.raises(LearnedFileDiverged) as exc_info:
        await learned.retire(kept[0], why="사람이 지워달라고 했다", now=NOW)

    assert exc_info.value.orphaned_bodies == [orphan]
    text = learned_path(data_dir).read_text(encoding="utf-8")
    assert orphan in text
    assert "미러가 아는 규칙" in text, "the rule named for deletion is still there too"
    assert [int(row["id"]) for row in learned.active()] == kept


async def test_a_crash_after_the_markdown_write_leaves_an_orphan_that_survives(
    learned: LearnedRules, data_dir: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other reproduction: a crash between `add()`'s markdown write and
    its mirror commit. The instant after is the allowed direction (non-
    negotiable 1) - what must not happen is the *next* ordinary `add()`
    treating that orphaned bullet as gone because the mirror never heard
    about it."""
    await learned.add([Proposal(body="R1", evidence=())], now=NOW)

    real_insert = store.insert_persona_rule

    def boom(*_: object, **__: object) -> int:
        raise sqlite3.OperationalError("simulated crash before the mirror commits")

    monkeypatch.setattr(store, "insert_persona_rule", boom)
    with pytest.raises(sqlite3.OperationalError):
        await learned.add([Proposal(body="R2 orphaned", evidence=())], now=NOW)
    monkeypatch.setattr(store, "insert_persona_rule", real_insert)

    # The markdown write happened before the simulated crash, so the file
    # already has the orphan; the mirror does not.
    assert "R2 orphaned" in learned_path(data_dir).read_text(encoding="utf-8")
    assert [row["body"] for row in learned.active()] == ["R1"]

    with pytest.raises(LearnedFileDiverged) as exc_info:
        await learned.add([Proposal(body="R3", evidence=())], now=NOW)
    assert exc_info.value.orphaned_bodies == ["R2 orphaned"]

    text = learned_path(data_dir).read_text(encoding="utf-8")
    assert "R1" in text
    assert "R2 orphaned" in text, "the orphan must survive the next ordinary write"
    assert "R3" not in text


def test_diverged_bodies_is_the_set_difference_by_body() -> None:
    assert rules.diverged_bodies(["a", "b"], ["a"]) == ["b"]
    assert rules.diverged_bodies(["a"], ["a", "b"]) == []
    assert rules.diverged_bodies([], []) == []


# --- rebuild (daemon reindex) ------------------------------------------------


def test_rebuild_restores_an_empty_mirror_from_learned_md(data_dir: Path, store: Store) -> None:
    bodies = [f"규칙 {i}" for i in range(5)]
    learned_path(data_dir).write_text(rules.render(bodies), encoding="utf-8")

    restored = rules.rebuild(data_dir, store)

    assert restored == 5
    assert sorted(row["body"] for row in store.active_persona_rules()) == sorted(bodies)


def test_rebuild_twice_adds_nothing_the_second_time(data_dir: Path, store: Store) -> None:
    bodies = [f"규칙 {i}" for i in range(5)]
    learned_path(data_dir).write_text(rules.render(bodies), encoding="utf-8")

    rules.rebuild(data_dir, store)
    assert rules.rebuild(data_dir, store) == 0
    assert len(store.active_persona_rules()) == 5


async def test_rebuild_never_touches_an_existing_rows_id_or_evidence(
    learned: LearnedRules, data_dir: Path, store: Store
) -> None:
    """`observations.consumed_by` is a foreign key onto `persona_rules(id)` -
    a rebuild that updated or deleted an existing row could fail outright or
    orphan evidence a previous week's pass actually recorded."""
    obs_id = store.insert_observation(
        body="관찰", observed_from="2026-08-01/2026-08-01", now=NOW
    )
    [rule_id] = await learned.add(
        [Proposal(body="기존 규칙", evidence=(obs_id,))], now=NOW
    )

    # A hand-added bullet the mirror has never seen, appended directly to the
    # file the way a crash (or a person) might leave one.
    text = learned_path(data_dir).read_text(encoding="utf-8") + "- 손으로 추가한 규칙\n"
    learned_path(data_dir).write_text(text, encoding="utf-8")

    restored = rules.rebuild(data_dir, store)

    assert restored == 1
    rows = {int(row["id"]): row for row in store.active_persona_rules()}
    assert rows[rule_id]["body"] == "기존 규칙"
    assert json.loads(rows[rule_id]["evidence"]) == [obs_id]
    assert (
        store.conn.execute(
            "SELECT consumed_by FROM observations WHERE id = ?", (obs_id,)
        ).fetchone()["consumed_by"]
        == rule_id
    )


def test_rebuild_with_no_file_does_nothing(data_dir: Path, store: Store) -> None:
    assert rules.rebuild(data_dir, store) == 0


# --- retire (human deletion) ------------------------------------------------


async def test_retire_removes_the_rule_from_the_file_and_the_mirror(
    learned: LearnedRules, data_dir: Path
) -> None:
    [rule_id] = await learned.add([Proposal(body="지울 규칙", evidence=())], now=NOW)

    assert await learned.retire(rule_id, why="틀렸다", now=NOW) is True
    assert learned.active() == []
    assert "지울 규칙" not in learned_path(data_dir).read_text(encoding="utf-8")


async def test_retiring_an_unknown_id_is_false_not_an_exception(learned: LearnedRules) -> None:
    assert await learned.retire(999, why="없음", now=NOW) is False


async def test_retiring_twice_is_false_the_second_time(learned: LearnedRules) -> None:
    [rule_id] = await learned.add([Proposal(body="지울 규칙", evidence=())], now=NOW)
    assert await learned.retire(rule_id, why="틀렸다", now=NOW) is True
    assert await learned.retire(rule_id, why="다시", now=NOW) is False


async def test_retiring_does_not_revive_on_the_same_evidence(
    learned: LearnedRules, store: Store
) -> None:
    """`consumed_by` only ever moves forward (non-negotiable 6) - reverting it
    on a human's delete request would let next week's pass recreate the exact
    rule that was just asked to be forgotten."""
    obs_id = store.insert_observation(
        body="아침엔 짧게", observed_from="2026-08-01/2026-08-01", now=NOW
    )
    [rule_id] = await learned.add(
        [Proposal(body="아침엔 짧게 말한다", evidence=(obs_id,))], now=NOW
    )

    await learned.retire(rule_id, why="싫다", now=NOW)

    assert store.unconsumed_observations() == []


# --- header ------------------------------------------------------------------


def test_the_header_explains_ownership(data_dir: Path) -> None:
    text = rules.render([])
    assert "never touch" in text
    assert "daemon persona forget" in text


# --- annotations (date and evidence count for the prompt) -------------------


async def test_annotations_map_each_body_to_its_date_and_evidence_count(
    learned: LearnedRules,
) -> None:
    """What the prompt needs, read from the columns rather than from the file -
    docs/CONTRACTS.md non-negotiable 3, and `rules.py`'s own docstring names the
    threat: a model that could write its own `created_at` could backdate a rule to
    look established. This design weights rules by date, so it would have both the
    means and the motive."""
    await learned.add(
        [
            Proposal(body="변명을 싫어한다", evidence=(1, 2, 3)),
            Proposal(body="짧은 답을 선호했다", evidence=(4,)),
        ],
        now=datetime(2026, 8, 9, 15, 0, tzinfo=UTC),
    )

    assert learned.annotations() == {
        "변명을 싫어한다": ("2026-08-09", 3),
        "짧은 답을 선호했다": ("2026-08-09", 1),
    }


async def test_annotations_survive_a_row_with_unreadable_evidence(
    learned: LearnedRules, store: Store
) -> None:
    """`evidence` is a JSON column with a CHECK, but a hand-repaired database or a
    future migration can still put something unexpected there. A rule that cannot
    report its count must still reach the prompt with its date - losing the whole
    persona over one malformed column is the failure this repo calls silent
    degradation, and it is worse than a rule that reads as seen once."""
    await learned.add(
        [Proposal(body="변명을 싫어한다", evidence=(1, 2))],
        now=datetime(2026, 8, 9, 15, 0, tzinfo=UTC),
    )
    store.conn.execute("""UPDATE persona_rules SET evidence = '"not-a-list"'""")
    store.conn.commit()

    assert learned.annotations() == {"변명을 싫어한다": ("2026-08-09", 1)}
