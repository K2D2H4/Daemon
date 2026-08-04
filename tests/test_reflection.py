"""The daily reflection pass.

The model is a fake here, so what is actually under test is everything around it:
what gets read, what a malformed reply costs, what a half-failure leaves behind,
and whether a day can be reflected on twice.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeProvider

from daemon.config import Route
from daemon.llm.base import Message
from daemon.llm.gateway import LLMGateway
from daemon.memory import curated, entities
from daemon.memory.base import LoggedMessage
from daemon.memory.store import Store
from daemon.reflection import Reflection, artifact_path, extract_json
from daemon.tasks import Task

NOW = datetime(2026, 8, 3, 7, 14, tzinfo=UTC)
DAY = "2026-08-03"

FULL_REPLY = json.dumps(
    {
        "facts": [
            {"body": "연희동에 산다", "importance": 8, "key": "home"},
            {"body": "김치찌개를 좋아한다", "importance": 6},
        ],
        "entities": [
            {
                "name": "지현",
                "kind": "person",
                "note": "연희동 카페에서 만났다고 했다",
                "links": ["연희동"],
            }
        ],
        "observations": [{"body": "아침에는 짧은 메시지가 낫다", "confidence": 0.7}],
    },
    ensure_ascii=False,
)


def gateway_for(provider: Any) -> LLMGateway:
    return LLMGateway({provider.name: provider}, {Task.REFLECTION: Route(provider.name, "m")})


def record(store: Store, content: str, *, role: str = "user", kind: str = "interactive") -> int:
    return store.insert_message(
        LoggedMessage(
            ts=NOW,
            role=role,  # type: ignore[arg-type]
            content=content,
            origin="owner" if role == "user" else "agent",
            session_kind=kind,  # type: ignore[arg-type]
            modality="text",
            channel="telegram",
            sender_id="42",
        ),
        log_file=f"memory/log/{DAY}.md",
    )


@pytest.fixture
def store(db: Any) -> Store:
    return Store(db)


def pass_for(data_dir: Path, store: Store, reply: str = FULL_REPLY) -> Reflection:
    return Reflection(data_dir, store, gateway_for(FakeProvider(reply)))


# --- the happy path ---------------------------------------------------------


async def test_a_day_becomes_facts_entities_and_observations(
    data_dir: Path, store: Store
) -> None:
    record(store, "연희동으로 이사했어")
    record(store, "축하해", role="assistant")

    result = await pass_for(data_dir, store).run(DAY)

    assert result.status == "written"
    assert (result.facts, result.entities, result.observations) == (2, 1, 1)
    assert result.problems == []

    assert curated.read(data_dir) == ["연희동에 산다", "김치찌개를 좋아한다"]
    note = entities.note_path(data_dir, "지현").read_text(encoding="utf-8")
    assert "연희동 카페에서 만났다고 했다" in note
    assert "[[연희동]]" in note
    assert [row["body"] for row in store.unconsumed_observations()] == [
        "아침에는 짧은 메시지가 낫다"
    ]


async def test_the_artifact_is_written_for_a_human_to_check(
    data_dir: Path, store: Store
) -> None:
    """The M2 gate is "an entity graph I did not fix by hand is worth reading", so
    there has to be something to read."""
    record(store, "연희동으로 이사했어")

    await pass_for(data_dir, store).run(DAY)

    text = artifact_path(data_dir, DAY).read_text(encoding="utf-8")
    assert f"# {DAY} 성찰" in text
    assert "연희동에 산다" in text
    assert "지현" in text
    assert "아침에는 짧은 메시지가 낫다" in text


async def test_the_artifact_is_owner_only(data_dir: Path, store: Store) -> None:
    record(store, "연희동으로 이사했어")
    await pass_for(data_dir, store).run(DAY)
    assert artifact_path(data_dir, DAY).stat().st_mode & 0o777 == 0o600


async def test_the_transcript_labels_who_spoke(data_dir: Path, store: Store) -> None:
    """An observation is about how to treat someone, so which turns were theirs is
    the whole basis for one."""
    record(store, "내가 한 말")
    record(store, "내가 답한 말", role="assistant")

    provider = FakeProvider(FULL_REPLY)
    await Reflection(data_dir, store, gateway_for(provider)).run(DAY)

    transcript = [m for m in provider.calls[0] if m.role == "user"][-1].content
    assert "나: 내가 한 말" in transcript
    assert "너: 내가 답한 말" in transcript


# --- idempotence ------------------------------------------------------------


async def test_a_day_is_not_reflected_on_twice(data_dir: Path, store: Store) -> None:
    record(store, "연희동으로 이사했어")
    reflection = pass_for(data_dir, store)

    first = await reflection.run(DAY)
    second = await reflection.run(DAY)

    assert first.status == "written"
    assert second.status == "skipped"
    # Not applied twice: the fact would otherwise retire its own duplicate.
    assert curated.read(data_dir).count("연희동에 산다") == 1


async def test_force_redoes_a_day(data_dir: Path, store: Store) -> None:
    record(store, "연희동으로 이사했어")
    reflection = pass_for(data_dir, store)
    await reflection.run(DAY)

    assert (await reflection.run(DAY, force=True)).status == "written"


async def test_a_day_with_nothing_to_read_is_empty_not_an_error(
    data_dir: Path, store: Store
) -> None:
    result = await pass_for(data_dir, store).run(DAY)
    assert result.status == "empty"
    assert result.ok
    # And no artifact, so tomorrow can still reflect on it if messages arrive.
    assert not artifact_path(data_dir, DAY).exists()


# --- the hygiene rules ------------------------------------------------------


async def test_the_daemons_own_speech_is_not_evidence(data_dir: Path, store: Store) -> None:
    """Hygiene rule 1 (docs/PLAN.md 4.2). If proactive output fed reflection, the
    loop would amplify itself."""
    record(store, "먼저 건 말", role="assistant", kind="proactive")

    result = await pass_for(data_dir, store).run(DAY)
    # `nothing`, not `empty`: the mirror has rows for this day, they were just all
    # ineligible. The day is marked done so it cannot sit in the backlog forever.
    assert result.status == "nothing"


async def test_recalled_messages_are_not_re_extracted(data_dir: Path, store: Store) -> None:
    """Hygiene rule 2. Counting injected context again turns it into new evidence."""
    reused = record(store, "이미 주입된 것")
    store.mark_recalled([reused])

    assert (await pass_for(data_dir, store).run(DAY)).status == "nothing"


# --- a model that misbehaves -------------------------------------------------


async def test_json_in_a_fence_with_a_preamble_still_parses() -> None:
    """A local 4B model does this constantly, and being strict would cost a day."""
    assert extract_json('물론이죠!\n```json\n{"facts": []}\n```') == {"facts": []}


async def test_no_json_at_all_writes_nothing(data_dir: Path, store: Store) -> None:
    """A half-applied pass is worse than a skipped one, because the day gets
    marked done either way."""
    record(store, "연희동으로 이사했어")

    result = await pass_for(data_dir, store, "죄송해요, 잘 모르겠어요").run(DAY)

    assert result.status == "unparseable"
    assert not result.ok
    assert not artifact_path(data_dir, DAY).exists()
    assert curated.read(data_dir) == []
    assert store.count_observations() == 0


async def test_an_unreachable_model_leaves_the_day_for_tomorrow(
    data_dir: Path, store: Store
) -> None:
    record(store, "연희동으로 이사했어")

    result = await Reflection(data_dir, store, gateway_for(FakeProvider(fail=True))).run(DAY)

    assert result.status == "unavailable"
    assert not artifact_path(data_dir, DAY).exists()  # so it is retried


async def test_an_out_of_range_importance_is_clamped_not_rejected(
    data_dir: Path, store: Store
) -> None:
    """`importance` multiplies the recall score, so 999 would let one night's
    conclusion outrank everything the user ever said."""
    record(store, "무슨 말")
    reply = json.dumps({"facts": [{"body": "과장된 사실", "importance": 999}]})

    await pass_for(data_dir, store, reply).run(DAY)

    assert store.active_entries()[0]["importance"] == 10


async def test_a_path_shaped_entity_name_is_reported_and_dropped(
    data_dir: Path, store: Store
) -> None:
    record(store, "무슨 말")
    reply = json.dumps(
        {
            "entities": [
                {"name": "../../persona/seed", "note": "덮어쓰기", "links": []},
                {"name": "지현", "note": "진짜 노트", "links": []},
            ]
        },
        ensure_ascii=False,
    )

    result = await pass_for(data_dir, store, reply).run(DAY)

    assert result.entities == 1
    assert any("unusable entity name" in problem for problem in result.problems)
    assert entities.note_path(data_dir, "지현").exists()


async def test_a_supersession_key_is_narrowed(data_dir: Path, store: Store) -> None:
    """Two spellings of one key would read as two different facts, and the key is
    what retires the old one."""
    record(store, "무슨 말")
    reply = json.dumps(
        {"facts": [{"body": "사는 곳", "key": "  Home Address! "}]}, ensure_ascii=False
    )

    await pass_for(data_dir, store, reply).run(DAY)

    assert store.active_entries()[0]["supersession_key"] == "home_address"


async def test_more_facts_than_the_cap_are_truncated(data_dir: Path, store: Store) -> None:
    """One night must not drown the always-injected tier."""
    record(store, "무슨 말")
    reply = json.dumps({"facts": [{"body": f"사실 {n}"} for n in range(50)]}, ensure_ascii=False)

    result = await pass_for(data_dir, store, reply).run(DAY)

    assert result.facts == 8


async def test_a_list_that_is_not_a_list_is_a_reported_problem(
    data_dir: Path, store: Store
) -> None:
    record(store, "무슨 말")
    reply = json.dumps({"facts": "이건 배열이 아님", "observations": [{"body": "관찰"}]})

    result = await pass_for(data_dir, store, reply).run(DAY)

    assert result.status == "written"
    assert result.facts == 0
    assert result.observations == 1
    assert any("was not a list" in problem for problem in result.problems)


async def test_an_empty_conclusion_is_still_a_written_day(
    data_dir: Path, store: Store
) -> None:
    """Nothing worth remembering is a legitimate answer, and the day must not be
    reread forever because of it."""
    record(store, "ㅇㅇ")

    result = await pass_for(data_dir, store, '{"facts": [], "entities": []}').run(DAY)

    assert result.status == "written"
    assert "정리할 만한 것이 없었다" in artifact_path(data_dir, DAY).read_text(encoding="utf-8")


# --- catching up ------------------------------------------------------------


async def test_catch_up_reflects_every_past_day_but_not_today(
    data_dir: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Today is still being written to; reflecting on a partial day would mark it
    done and lose the evening."""
    import daemon.reflection as reflection_module

    log_dir = data_dir / "memory" / "log"
    for day in ("2026-08-01", "2026-08-02", DAY):
        (log_dir / f"{day}.md").write_text(f"# {day}\n", encoding="utf-8")
        store.insert_message(
            LoggedMessage(
                ts=NOW,
                role="user",
                content=f"{day} 에 한 말",
                origin="owner",
                session_kind="interactive",
                modality="text",
                channel="telegram",
                sender_id="42",
            ),
            log_file=f"memory/log/{day}.md",
        )

    monkeypatch.setattr(reflection_module, "clock_now", lambda: NOW)
    results = await pass_for(data_dir, store).catch_up()

    assert [result.date for result in results] == ["2026-08-01", "2026-08-02"]
    assert not artifact_path(data_dir, DAY).exists()


async def test_catch_up_is_bounded(data_dir: Path, store: Store) -> None:
    """A first run over months of history must not become one unbounded batch of
    model calls."""
    log_dir = data_dir / "memory" / "log"
    for day in range(1, 20):
        (log_dir / f"2026-07-{day:02d}.md").write_text("x", encoding="utf-8")

    results = await pass_for(data_dir, store).catch_up(limit=3)
    assert len(results) == 3


async def test_pending_days_excludes_what_was_already_done(
    data_dir: Path, store: Store
) -> None:
    log_dir = data_dir / "memory" / "log"
    for day in ("2026-08-01", "2026-08-02"):
        (log_dir / f"{day}.md").write_text("x", encoding="utf-8")
    artifact_path(data_dir, "2026-08-01").parent.mkdir(parents=True, exist_ok=True)
    artifact_path(data_dir, "2026-08-01").write_text("done", encoding="utf-8")

    assert pass_for(data_dir, store).pending_days() == ["2026-08-02"]


# --- the prompt reaches the right task --------------------------------------


async def test_it_routes_as_reflection_not_chat(data_dir: Path, store: Store) -> None:
    """Reflection quality propagates to the whole graph, so it is routed
    separately from chat on purpose (docs/PLAN.md 3.2)."""
    record(store, "무슨 말")
    provider = FakeProvider(FULL_REPLY)
    gateway = LLMGateway(
        {provider.name: provider},
        {Task.REFLECTION: Route(provider.name, "reflection-model")},
    )

    await Reflection(data_dir, store, gateway).run(DAY)

    assert provider.models == ["reflection-model"]
    assert isinstance(provider.calls[0][0], Message)
    assert provider.calls[0][0].role == "system"


async def test_a_note_is_dated_by_the_day_it_is_about(data_dir: Path, store: Store) -> None:
    """Found by running the pass for real: the note said today, not the day it had
    just read. A first run over months of history would stamp every section with
    the date of the run, and the note is supposed to read as a history.
    """
    record(store, "연희동으로 이사했어")

    await pass_for(data_dir, store).run(DAY)

    note = entities.note_path(data_dir, "지현").read_text(encoding="utf-8")
    assert f"## {DAY}" in note


async def test_two_facts_with_one_key_keep_the_more_important(
    data_dir: Path, store: Store
) -> None:
    """Found by running the pass for real. A local model keyed both halves of a
    move `location`, so the second retired the first and `core.md` kept the *less*
    important one while the artifact claimed both - a silent inversion.
    """
    record(store, "연희동으로 이사했어")
    reply = json.dumps(
        {
            "facts": [
                {"body": "이주 위치: 연희동", "importance": 8, "key": "location"},
                {"body": "이전 주소: 망원동", "importance": 3, "key": "location"},
            ]
        },
        ensure_ascii=False,
    )

    result = await pass_for(data_dir, store, reply).run(DAY)

    assert result.facts == 1
    assert curated.read(data_dir) == ["이주 위치: 연희동"]
    assert any("two facts keyed 'location'" in problem for problem in result.problems)
    # The artifact is rendered from the same filtered conclusion, so it must not
    # claim a fact that was dropped.
    artifact = artifact_path(data_dir, DAY).read_text(encoding="utf-8")
    assert "망원동" not in artifact


async def test_facts_without_keys_are_never_deduplicated(
    data_dir: Path, store: Store
) -> None:
    """No key means no claim of exclusivity, so two unrelated facts must both land
    even if one repeats a word."""
    record(store, "무슨 말")
    reply = json.dumps(
        {"facts": [{"body": "고양이를 키운다"}, {"body": "고양이 이름은 나비"}]},
        ensure_ascii=False,
    )

    result = await pass_for(data_dir, store, reply).run(DAY)
    assert result.facts == 2


async def test_a_day_the_mirror_has_not_caught_up_on_stays_pending(
    data_dir: Path, store: Store
) -> None:
    """Reported by running it: a day with no eligible messages never left the
    backlog, so doctor nagged about a day `reflect` could not clear. But marking
    every such day done is just as wrong - an unmirrored day would be skipped
    permanently, and a deleted mirror is a state the contract calls legitimate.
    So the mirror decides which case this is.
    """
    log_dir = data_dir / "memory" / "log"
    (log_dir / f"{DAY}.md").write_text(f"# {DAY}\n", encoding="utf-8")

    result = await pass_for(data_dir, store).run(DAY)

    assert result.status == "empty"
    assert "reindex" in result.detail
    assert pass_for(data_dir, store).pending_days() == [DAY]


async def test_a_mirrored_day_with_nothing_eligible_leaves_the_backlog(
    data_dir: Path, store: Store
) -> None:
    log_dir = data_dir / "memory" / "log"
    (log_dir / "2026-08-01.md").write_text("# 2026-08-01\n", encoding="utf-8")
    store.insert_message(
        LoggedMessage(
            ts=NOW,
            role="assistant",
            content="내가 먼저 건 말",
            origin="agent",
            session_kind="proactive",
            modality="text",
            channel="telegram",
            sender_id="42",
        ),
        log_file="memory/log/2026-08-01.md",
    )

    result = await pass_for(data_dir, store).run("2026-08-01")

    assert result.status == "nothing"
    assert pass_for(data_dir, store).pending_days() == []


async def test_catch_up_drops_today_before_applying_the_cap(
    data_dir: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slicing first meant a run where today was pending processed limit-1 days,
    so it fell one day further behind on every run."""
    import daemon.reflection as reflection_module

    log_dir = data_dir / "memory" / "log"
    for day in ("2026-08-01", "2026-08-02", DAY):
        (log_dir / f"{day}.md").write_text("x", encoding="utf-8")

    monkeypatch.setattr(reflection_module, "clock_now", lambda: NOW)
    results = await pass_for(data_dir, store).catch_up(limit=2, now=NOW)

    assert [result.date for result in results] == ["2026-08-01", "2026-08-02"]


# --- trigger phrases --------------------------------------------------------


async def test_a_fact_carries_the_trigger_phrases_the_model_proposed(
    data_dir: Path, store: Store
) -> None:
    """Reported by an agent reviewing recall: the matching was implemented, the
    column accepted a value, and nothing produced one. Complete, tested, and
    reachable by nothing."""
    record(store, "연희동으로 이사했어")
    reply = json.dumps(
        {"facts": [{"body": "연희동에 산다", "triggers": ["이사", "연희동"]}]},
        ensure_ascii=False,
    )

    await pass_for(data_dir, store, reply).run(DAY)

    assert json.loads(store.active_entries()[0]["trigger_phrases"]) == ["이사", "연희동"]


async def test_triggers_are_bounded_and_deduplicated(data_dir: Path, store: Store) -> None:
    """Recall matches every phrase against every query on the voice latency path,
    and a phrase long enough to be a sentence matches nothing."""
    record(store, "무슨 말")
    reply = json.dumps(
        {
            "facts": [
                {
                    "body": "사실",
                    "triggers": ["이사", "이사", " 이사 ", "가" * 40, 42, "연희동"],
                }
            ]
        },
        ensure_ascii=False,
    )

    await pass_for(data_dir, store, reply).run(DAY)

    assert json.loads(store.active_entries()[0]["trigger_phrases"]) == ["이사", "연희동"]


async def test_a_fact_with_no_triggers_still_lands(data_dir: Path, store: Store) -> None:
    record(store, "무슨 말")
    reply = json.dumps({"facts": [{"body": "사실"}]}, ensure_ascii=False)

    result = await pass_for(data_dir, store, reply).run(DAY)

    assert result.facts == 1
    assert json.loads(store.active_entries()[0]["trigger_phrases"]) == []


async def test_triggers_that_are_not_a_list_are_reported(data_dir: Path, store: Store) -> None:
    record(store, "무슨 말")
    reply = json.dumps({"facts": [{"body": "사실", "triggers": "이사"}]}, ensure_ascii=False)

    result = await pass_for(data_dir, store, reply).run(DAY)

    assert result.facts == 1
    assert any("triggers was not a list" in problem for problem in result.problems)


async def test_the_artifact_shows_the_triggers(data_dir: Path, store: Store) -> None:
    """They decide when a fact resurfaces, so a human checking the pass has to be
    able to see them."""
    record(store, "무슨 말")
    reply = json.dumps(
        {"facts": [{"body": "연희동에 산다", "triggers": ["이사"]}]}, ensure_ascii=False
    )

    await pass_for(data_dir, store, reply).run(DAY)

    assert "triggers: 이사" in artifact_path(data_dir, DAY).read_text(encoding="utf-8")
