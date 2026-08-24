"""The weekly persona-evolution pass.

The model is a fake here, same stance as `test_reflection.py`: what matters is
everything around it - which gates cost a model call and which do not, what a
malformed reply costs, and whether the same evidence can revive a retired
rule.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeProvider

from daemon.config import Route
from daemon.llm.gateway import LLMGateway
from daemon.memory.store import Store
from daemon.persona.evolve import PersonaEvolution, _week_start
from daemon.persona.loader import learned_path, seed_path
from daemon.persona.rules import LearnedFileDiverged, LearnedRules
from daemon.persona.rules import rebuild as rebuild_persona_rules
from daemon.persona.rules import render as render_learned
from daemon.tasks import Task

NOW = datetime(2026, 8, 3, 7, 14, tzinfo=UTC)  # a Monday
WEEK = "2026-08-03"

FULL_REPLY = json.dumps(
    {
        "rules": [
            {"body": "아침엔 인사만 짧게 한다", "evidence": [1, 2], "key": "morning"},
        ]
    },
    ensure_ascii=False,
)


def gateway_for(provider: Any) -> LLMGateway:
    return LLMGateway({provider.name: provider}, {Task.PERSONA_RULE: Route(provider.name, "m")})


@pytest.fixture
def store(db: Any) -> Store:
    return Store(db)


def add_observations(store: Store, n: int, *, start: int = 1) -> list[int]:
    return [
        store.insert_observation(
            body=f"관찰 {i}", observed_from="2026-08-01/2026-08-02", now=NOW
        )
        for i in range(start, start + n)
    ]


def pass_for(
    data_dir: Path, store: Store, reply: str = FULL_REPLY, **kwargs: Any
) -> PersonaEvolution:
    return PersonaEvolution(data_dir, store, gateway_for(FakeProvider(reply)), **kwargs)


# --- the three zero-model-call gates ----------------------------------------


async def test_not_enough_observations_makes_no_model_call(
    data_dir: Path, store: Store
) -> None:
    add_observations(store, 2)
    provider = FakeProvider(FULL_REPLY)
    evolution = PersonaEvolution(data_dir, store, gateway_for(provider), min_observations=5)

    result = await evolution.run(now=NOW)

    assert result.skipped.startswith("not enough observations")
    assert provider.calls == []
    assert not evolution.diary_path(WEEK).exists()


async def test_an_existing_diary_this_week_makes_no_model_call(
    data_dir: Path, store: Store
) -> None:
    add_observations(store, 10)
    evolution = pass_for(data_dir, store, min_observations=5)
    evolution.diary_path(WEEK).parent.mkdir(parents=True, exist_ok=True)
    evolution.diary_path(WEEK).write_text("already ran", encoding="utf-8")

    provider = FakeProvider(FULL_REPLY)
    evolution2 = PersonaEvolution(data_dir, store, gateway_for(provider), min_observations=5)
    result = await evolution2.run(now=NOW)

    assert result.skipped == "already run this week"
    assert provider.calls == []


async def test_force_ignores_the_diary_and_runs_again(data_dir: Path, store: Store) -> None:
    add_observations(store, 10)
    evolution = pass_for(data_dir, store, min_observations=5)
    evolution.diary_path(WEEK).parent.mkdir(parents=True, exist_ok=True)
    evolution.diary_path(WEEK).write_text("already ran", encoding="utf-8")

    result = await evolution.run(now=NOW, force=True)

    assert result.skipped == ""


async def test_a_full_rule_budget_makes_no_model_call(data_dir: Path, store: Store) -> None:
    add_observations(store, 10)
    for n in range(2):
        store.insert_persona_rule(body=f"규칙 {n}", created_at=NOW, evidence=[])

    provider = FakeProvider(FULL_REPLY)
    evolution = PersonaEvolution(
        data_dir, store, gateway_for(provider), min_observations=5, max_active=2
    )
    result = await evolution.run(now=NOW)

    assert result.skipped.startswith("rule budget full")
    assert provider.calls == []


# --- the happy path -----------------------------------------------------


async def test_a_full_pass_adds_a_rule_and_writes_a_diary(data_dir: Path, store: Store) -> None:
    ids = add_observations(store, 10)
    evolution = pass_for(data_dir, store, min_observations=5)

    result = await evolution.run(now=NOW)

    assert result.skipped == ""
    assert result.added == 1
    assert [row["body"] for row in store.active_persona_rules()] == ["아침엔 인사만 짧게 한다"]
    assert "아침엔 인사만 짧게 한다" in learned_path(data_dir).read_text(encoding="utf-8")

    diary = evolution.diary_path(WEEK).read_text(encoding="utf-8")
    assert "아침엔 인사만 짧게 한다" in diary
    assert store.conn.execute(
        "SELECT consumed_by FROM observations WHERE id = ?", (ids[0],)
    ).fetchone()["consumed_by"] is not None


async def test_the_prompt_carries_seed_active_rules_and_observations(
    data_dir: Path, store: Store
) -> None:
    seed_path(data_dir).write_text("나는 다정하다.", encoding="utf-8")
    store.insert_persona_rule(body="기존 활성 규칙", created_at=NOW, evidence=[])
    add_observations(store, 10)

    provider = FakeProvider(FULL_REPLY)
    evolution = PersonaEvolution(data_dir, store, gateway_for(provider), min_observations=5)
    await evolution.run(now=NOW)

    prompt = provider.calls[0][-1].content
    assert "나는 다정하다." in prompt
    assert "기존 활성 규칙" in prompt
    assert "관찰 1" in prompt


async def test_it_routes_as_persona_rule_not_reflection(data_dir: Path, store: Store) -> None:
    add_observations(store, 10)
    provider = FakeProvider(FULL_REPLY)
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PERSONA_RULE: Route(provider.name, "persona-model")}
    )
    await PersonaEvolution(data_dir, store, gateway, min_observations=5).run(now=NOW)

    assert provider.models == ["persona-model"]


# --- seed.md is never touched -----------------------------------------------


async def test_seed_md_is_never_written_to(data_dir: Path, store: Store) -> None:
    path = seed_path(data_dir)
    path.write_text("나는 다정하다.", encoding="utf-8")
    before_mtime = path.stat().st_mtime_ns
    before_text = path.read_text(encoding="utf-8")

    add_observations(store, 10)
    await pass_for(data_dir, store, min_observations=5).run(now=NOW)

    assert path.stat().st_mtime_ns == before_mtime
    assert path.read_text(encoding="utf-8") == before_text


# --- rate limiting and reporting --------------------------------------------


async def test_more_new_rules_than_the_weekly_cap_are_truncated_and_reported(
    data_dir: Path, store: Store
) -> None:
    add_observations(store, 10)
    reply = json.dumps(
        {
            "rules": [
                {"body": f"규칙 {n}", "evidence": [], "key": None}
                for n in range(5)
            ]
        },
        ensure_ascii=False,
    )
    evolution = pass_for(data_dir, store, reply, min_observations=5, max_new=2)

    result = await evolution.run(now=NOW)

    assert result.proposed == 5
    assert result.added == 2
    assert any("dropped" in problem for problem in result.problems)


async def test_a_proposal_duplicating_an_existing_active_rule_is_dropped(
    data_dir: Path, store: Store
) -> None:
    store.insert_persona_rule(body="아침엔 인사만 짧게 한다", created_at=NOW, evidence=[])
    add_observations(store, 10)

    result = await pass_for(data_dir, store, min_observations=5).run(now=NOW)

    assert result.added == 0
    assert any("duplicates an existing active rule" in problem for problem in result.problems)


async def test_two_proposals_with_one_key_in_one_batch_are_resolved_and_reported(
    data_dir: Path, store: Store
) -> None:
    add_observations(store, 10)
    reply = json.dumps(
        {
            "rules": [
                {"body": "첫 번째", "evidence": [], "key": "same"},
                {"body": "두 번째", "evidence": [], "key": "same"},
            ]
        },
        ensure_ascii=False,
    )

    result = await pass_for(data_dir, store, reply, min_observations=5).run(now=NOW)

    assert result.added == 1
    assert [row["body"] for row in store.active_persona_rules()] == ["첫 번째"]
    assert any("kept the earlier one" in problem for problem in result.problems)


# --- evidence is narrowed to what was actually offered ----------------------


async def test_evidence_ids_not_in_the_prompt_are_dropped(data_dir: Path, store: Store) -> None:
    ids = add_observations(store, 10)
    fabricated_id = max(ids) + 1000
    reply = json.dumps(
        {"rules": [{"body": "규칙", "evidence": [ids[0], fabricated_id], "key": None}]}
    )

    result = await pass_for(data_dir, store, reply, min_observations=5).run(now=NOW)

    row = store.conn.execute(
        "SELECT evidence FROM persona_rules WHERE body = '규칙'"
    ).fetchone()
    assert json.loads(row["evidence"]) == [ids[0]]
    # A model inventing evidence must be visible in the diary, not just
    # silently narrowed away.
    assert any(str(fabricated_id) in problem for problem in result.problems)


async def test_two_proposals_citing_the_same_observation_only_the_first_keeps_it(
    data_dir: Path, store: Store
) -> None:
    """`consume_observations`'s `consumed_by IS NULL` guard already lets only
    the first proposal actually claim a shared id - but until this is fixed,
    the second proposal's own `evidence` column still lists it too, so
    `daemon persona`'s "N observation(s)" overstates what that specific rule
    was built from."""
    ids = add_observations(store, 10)
    shared = ids[0]
    reply = json.dumps(
        {
            "rules": [
                {"body": "첫 번째 규칙", "evidence": [shared], "key": None},
                {"body": "두 번째 규칙", "evidence": [shared], "key": None},
            ]
        }
    )

    result = await pass_for(data_dir, store, reply, min_observations=5).run(now=NOW)

    assert result.added == 2
    rows = {
        row["body"]: json.loads(row["evidence"])
        for row in store.conn.execute("SELECT body, evidence FROM persona_rules").fetchall()
    }
    assert rows["첫 번째 규칙"] == [shared]
    assert rows["두 번째 규칙"] == [], "the second proposal must not overstate its own evidence"
    assert any(str(shared) in problem for problem in result.problems)

    first_id = next(
        int(row["id"]) for row in store.active_persona_rules() if row["body"] == "첫 번째 규칙"
    )
    consumed_by = store.conn.execute(
        "SELECT consumed_by FROM observations WHERE id = ?", (shared,)
    ).fetchone()["consumed_by"]
    assert consumed_by == first_id, "the mirror and the reported evidence must agree"


# --- a model that misbehaves -------------------------------------------------


async def test_no_json_at_all_writes_no_diary(data_dir: Path, store: Store) -> None:
    add_observations(store, 10)
    evolution = pass_for(data_dir, store, "죄송해요, 잘 모르겠어요", min_observations=5)

    result = await evolution.run(now=NOW)

    assert result.skipped == ""
    assert result.added == 0
    assert any("did not return a JSON object" in problem for problem in result.problems)
    assert not evolution.diary_path(WEEK).exists()
    assert store.active_persona_rules() == []


async def test_an_unreachable_model_writes_no_diary(data_dir: Path, store: Store) -> None:
    add_observations(store, 10)
    provider = FakeProvider(fail=True)
    evolution = PersonaEvolution(data_dir, store, gateway_for(provider), min_observations=5)

    result = await evolution.run(now=NOW)

    assert result.skipped == ""
    assert any("model unavailable" in problem for problem in result.problems)
    assert not evolution.diary_path(WEEK).exists()


async def test_an_empty_rules_list_is_a_completed_pass_not_a_skip(
    data_dir: Path, store: Store
) -> None:
    add_observations(store, 10)
    evolution = pass_for(data_dir, store, '{"rules": []}', min_observations=5)

    result = await evolution.run(now=NOW)

    assert result.skipped == ""
    assert result.added == 0
    assert evolution.diary_path(WEEK).exists()


# --- a diverged learned.md -----------------------------------------------
#
# The same data-loss defect as test_persona_rules.py's divergence tests, seen
# from the weekly pass: a pass that cannot write must not look like a pass
# that decided there was nothing to add.


async def test_run_refuses_on_a_diverged_file_and_writes_no_diary(
    data_dir: Path, store: Store
) -> None:
    add_observations(store, 10)
    # The mirror is still empty (no persona rules inserted at all), but
    # learned.md already has a rule - the state `rm daemon.sqlite3` leaves.
    learned_path(data_dir).write_text(render_learned(["손으로 있던 규칙"]), encoding="utf-8")

    provider = FakeProvider(FULL_REPLY)
    evolution = PersonaEvolution(data_dir, store, gateway_for(provider), min_observations=5)
    result = await evolution.run(now=NOW)

    # Named in `skipped`, because it is a gate like the other three: with it
    # empty the CLI headline read "ran (10 read -> 0 proposed)", which is what a
    # pass that reached the model and found nothing looks like.
    assert "mirror does not know about" in result.skipped
    assert result.added == 0
    assert any("mirror does not know" in problem for problem in result.problems)
    assert not evolution.diary_path(WEEK).exists()
    # Caught before the model call - it is a zero-cost deterministic gate,
    # same as the other three.
    assert provider.calls == []
    # Nothing was concluded: every observation is still unconsumed.
    assert len(store.unconsumed_observations()) == 10


async def test_rerunning_after_a_reindex_succeeds(data_dir: Path, store: Store) -> None:
    add_observations(store, 10)
    learned_path(data_dir).write_text(render_learned(["손으로 있던 규칙"]), encoding="utf-8")

    evolution = pass_for(data_dir, store, FULL_REPLY, min_observations=5)
    first = await evolution.run(now=NOW)
    assert "mirror does not know about" in first.skipped
    assert any("mirror does not know" in problem for problem in first.problems)
    assert not evolution.diary_path(WEEK).exists()

    restored = rebuild_persona_rules(data_dir, store)
    assert restored == 1

    # No `force` needed: the first attempt never wrote a diary, so the week
    # is still open.
    second = await pass_for(data_dir, store, FULL_REPLY, min_observations=5).run(now=NOW)

    assert second.skipped == ""
    assert second.added == 1
    # `rebuild_persona_rules` stamps its row with the real clock, not `NOW`
    # (same as `curated.rebuild`), so order between the two is not asserted.
    assert sorted(row["body"] for row in store.active_persona_rules()) == sorted(
        ["손으로 있던 규칙", "아침엔 인사만 짧게 한다"]
    )
    assert evolution.diary_path(WEEK).exists()


async def test_run_treats_add_raising_diverged_as_a_backstop_not_a_crash(
    data_dir: Path, store: Store
) -> None:
    """Gate 4 closes the normal case deterministically before the model is
    ever called, so this simulates the race it cannot close: `learned.md`
    diverges in the moment between gate 4's read and `add()`'s own write
    attempt. `add()` is the one method that can actually raise
    `LearnedFileDiverged`, and `run()` must turn that into a reported problem
    - `daemon persona evolve` is an operator command, and an uncaught
    exception there is a traceback where a diagnosis should be.
    """
    add_observations(store, 10)

    class RacingRules(LearnedRules):
        async def add(self, proposals: list[Any], *, now: Any = None) -> list[int]:
            raise LearnedFileDiverged(["어디선가 몰래 추가된 규칙"])

    evolution = PersonaEvolution(
        data_dir,
        store,
        gateway_for(FakeProvider(FULL_REPLY)),
        min_observations=5,
        rules=RacingRules(data_dir, store),
    )

    result = await evolution.run(now=NOW)

    assert result.skipped == ""
    assert result.added == 0
    assert any("어디선가 몰래 추가된 규칙" in problem for problem in result.problems)
    assert not evolution.diary_path(WEEK).exists()


# --- the week -----------------------------------------------------------


def test_week_start_is_the_monday_regardless_of_which_day_it_is() -> None:
    wednesday = datetime(2026, 8, 5, 3, 0, tzinfo=UTC)
    assert _week_start(wednesday) == _week_start(NOW)


# --- retire, at the evolve level --------------------------------------------


async def test_retiring_a_rule_does_not_let_next_weeks_pass_revive_it(
    data_dir: Path, store: Store
) -> None:
    """docs/design/2026-08-05-m4-persona-design.md: `consumed_by` only ever
    moves forward, so a delete request stays honoured because the evidence
    that produced the rule cannot be handed to the model again."""
    ids = add_observations(store, 10)
    # Cite every observation as evidence, so retiring leaves none unconsumed -
    # otherwise the leftover 8 would legitimately support a *new* proposal,
    # which would not be testing revival from the same evidence at all.
    reply = json.dumps(
        {"rules": [{"body": "아침엔 인사만 짧게 한다", "evidence": ids, "key": "morning"}]}
    )
    evolution = pass_for(data_dir, store, reply, min_observations=5)
    await evolution.run(now=NOW)
    [rule_id] = [int(row["id"]) for row in store.active_persona_rules()]
    assert store.unconsumed_observations() == []

    learned = LearnedRules(data_dir, store)
    await learned.retire(rule_id, why="싫다", now=NOW)

    # A second week, same fake reply: with every observation already consumed,
    # there is nothing left to justify the same rule reappearing.
    next_week = datetime(2026, 8, 10, 7, 14, tzinfo=UTC)
    second = await pass_for(data_dir, store, reply, min_observations=5).run(now=next_week)

    assert second.skipped.startswith("not enough observations")
    assert store.active_persona_rules() == []


def test_the_rule_prompt_asks_for_an_observation_not_a_standing_demand() -> None:
    """The prompt used to say `body 는 이 사람에 대한 사실처럼 짧게 한 문장으로 적는다`,
    and a fact carries no strength, so it is read at full weight forever. What that
    produced after five days of terse terminal QA was

        용건 위주의 빠른 응답과 즉각적이고 담백한 피드백을 요구한다

    - a standing demand governing a midnight voice chat as much as a debugging
    session. Naming the unwanted form is the point: `CALLED_BY_NAME` already
    measured that omitting a move is not enough, the move has to be forbidden.
    """
    from daemon.persona.evolve import SYSTEM

    assert "사실처럼" not in SYSTEM
    assert "요구한다" in SYSTEM, "the standing-demand form has to be named to be banned"
    assert "관찰" in SYSTEM
