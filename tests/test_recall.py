"""Lane 1 recall: the two lanes, the score, and every way it is allowed to fail.

Written in Korean on purpose. Recall in English is a solved problem - FTS5 alone
carries it - and every interesting failure in this module comes from Korean
morphology meeting a tokenizer that does not know about it. An English test suite
here would pass while the product did not work.

The load-bearing test is `test_fts5_misses_a_korean_substring_...`: it is the
evidence for pulling the vector index forward from M2 into M1b (docs/PLAN.md 4.3).
If it ever starts passing without the vector lane, that decision should be
revisited.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Literal

import numpy as np
import pytest

from daemon.llm.base import Embedder, ProviderError
from daemon.memory.base import LoggedMessage, Recall
from daemon.memory.curated import MAX_INJECTED
from daemon.memory.recall import HALF_LIFE_DAYS, MemoryRecall, fts_query
from daemon.memory.store import Store

NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC)


# --- deterministic fake embedders --------------------------------------------


class GramEmbedder:
    """Hashed character bigrams. Deterministic, offline, no model.

    Not semantic, and not pretending to be: what it does capture is the one thing
    FTS5 structurally cannot, a substring inside a longer Korean token (`찌개`
    inside `김치찌개`). blake2b rather than `hash()` because Python's string hash
    is salted per process, and a test that passes on Tuesday is not a test.
    """

    name = "fake"
    model = "fake-gram"
    dimensions = 64

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail:
            raise ProviderError("fake embedder was told to fail")
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        compact = "".join(c for c in text if not c.isspace())
        for start in range(max(len(compact) - 1, 0)):
            digest = hashlib.blake2b(compact[start : start + 2].encode(), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return [float(value) for value in vector]


class FlatEmbedder:
    """Every text gets the same unit vector, so every cosine is 1.0 and only
    recency and importance can reorder anything."""

    name = "flat"
    model = "flat"
    dimensions = 4

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class WideEmbedder(FlatEmbedder):
    """Same name, different width: a model re-tagged under an existing name."""

    dimensions = 8

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]


# --- fixtures ----------------------------------------------------------------


def message(
    content: str,
    *,
    ts: datetime | None = None,
    role: Literal["user", "assistant"] = "user",
) -> LoggedMessage:
    return LoggedMessage(
        ts=ts or NOW - timedelta(days=1),
        role=role,
        content=content,
        origin="owner" if role == "user" else "agent",
        session_kind="interactive",
        modality="text",
        channel="telegram",
        sender_id="42",
    )


@pytest.fixture
def store(db: sqlite3.Connection) -> Store:
    return Store(db)


def add(store: Store, content: str, *, days_ago: float = 1.0) -> int:
    return store.insert_message(
        message(content, ts=NOW - timedelta(days=days_ago)),
        log_file="memory/log/2026-08-02.md",
    )


async def seed(store: Store, embedder: Embedder | None, *texts: str) -> MemoryRecall:
    for text in texts:
        add(store, text)
    recall = MemoryRecall(store, embedder, now=NOW)
    await recall.backfill()
    return recall


def entry(
    store: Store,
    body: str,
    *,
    importance: int = 5,
    triggers: tuple[str, ...] = (),
    origin: str = "agent",
    key: str | None = None,
) -> int:
    """One curated fact in the mirror. The markdown side is `tests/test_curated.py`;
    recall only ever reads `memory_entries`."""
    return store.insert_entry(
        body=body,
        importance=importance,
        trigger_phrases=triggers,
        origin=origin,
        session_kind="reflection",
        modality="text",
        now=NOW,
        supersession_key=key,
    )


def contents(items: list) -> list[str]:
    return [item.content for item in items]


def injected(items: list) -> list[str]:
    return [item.content for item in items if item.reason.startswith("curated")]


def searched(items: list) -> list[str]:
    return [item.content for item in items if not item.reason.startswith("curated")]


def spy_on_loads(store: Store) -> list[str]:
    """Record every full load of the vector index, which is the thing that must stop
    happening once per turn. Wrapping the store rather than reading a private: what
    matters is the sqlite work, not how recall remembers it."""
    calls: list[str] = []
    original = store.load_embeddings

    def counted(model: str) -> tuple[list[int], np.ndarray]:
        calls.append(model)
        return original(model)

    store.load_embeddings = counted  # type: ignore[method-assign]
    return calls


# --- the protocol ------------------------------------------------------------


def test_satisfies_the_recall_protocol(store: Store) -> None:
    assert isinstance(MemoryRecall(store), Recall)


# --- fts_query: user text is not query syntax --------------------------------


def test_fts_query_quotes_every_token() -> None:
    assert fts_query("어제 김치찌개 먹었어") == '"어제" OR "김치찌개" OR "먹었어"'


def test_fts_query_strips_everything_fts5_would_read_as_syntax() -> None:
    """Each of these is a real thing a user types, and each one is FTS5 syntax."""
    assert fts_query('어제 "김치찌개" 먹었나?') == '"어제" OR "김치찌개" OR "먹었나"'
    assert fts_query("C:\\Users 경로 알려줘") == '"C" OR "Users" OR "경로" OR "알려줘"'
    assert fts_query("NEAR 라는 단어") == '"NEAR" OR "라는" OR "단어"'
    assert fts_query("3*4 계산해줘") == '"3" OR "4" OR "계산해줘"'
    assert fts_query("^시작 -빼기 (괄호)") == '"시작" OR "빼기" OR "괄호"'


def test_fts_query_is_empty_when_there_is_nothing_to_search() -> None:
    assert fts_query("") == ""
    assert fts_query('?! "" *** :::') == ""


def test_fts_query_caps_a_pasted_wall_of_text() -> None:
    query = fts_query(" ".join(f"단어{i}" for i in range(200)))
    assert query.count(" OR ") == 31  # MAX_QUERY_TOKENS - 1


@pytest.mark.parametrize(
    "query",
    [
        '어제 "김치찌개" 먹었나?',
        "C:\\Users 경로",
        "NEAR AND OR NOT",
        "김치* 찌개**",
        "^^^ ::: (((",
        '"',
        "*",
        ":",
        "",
        "   ",
        "김치찌개",
    ],
)
async def test_search_never_raises_on_fts5_syntax_in_the_query(store: Store, query: str) -> None:
    """FTS5 parses a bound value as query syntax, so an unescaped `:` is an
    OperationalError in the middle of a conversation. Nothing here may raise."""
    recall = await seed(store, GramEmbedder(), "어제 저녁에 김치찌개 먹었어", "모카 밥 줬어")
    await recall.search(query, limit=3)


@pytest.mark.parametrize(
    "query",
    [
        '어제 "김치찌개" 먹었나?',
        "C:\\Users 경로",
        "NEAR AND OR NOT",
        "김치* 찌개**",
        "^^^ ::: (((",
        '"',
        "*",
        ":",
    ],
)
async def test_no_user_input_produces_a_malformed_fts_query(
    store: Store, query: str, caplog: pytest.LogCaptureFixture
) -> None:
    """`store.search_fts` swallows an OperationalError so a bad query cannot kill
    a turn, which would also hide a broken query builder. So assert the stronger
    thing: sqlite never rejects what `fts_query` produces in the first place."""
    recall = await seed(store, None, "어제 저녁에 김치찌개 먹었어")

    with caplog.at_level("WARNING"):
        await recall.search(query, limit=3)

    assert "sqlite rejected" not in caplog.text


# --- the reason the vector lane exists ---------------------------------------


async def test_keyword_lane_finds_a_whole_korean_token(store: Store) -> None:
    recall = await seed(store, None, "어제 저녁에 김치찌개 먹었어", "오늘은 클라이밍 갔어")
    items = await recall.search("김치찌개 맛있었어?", limit=3)

    assert contents(items) == ["어제 저녁에 김치찌개 먹었어"]
    assert items[0].reason == "keyword"


@pytest.mark.parametrize("query", ["김치", "찌개", "어제는 뭐 먹었지"])
async def test_fts5_misses_a_korean_substring_that_the_vector_lane_catches(
    store: Store, query: str
) -> None:
    """The measurement docs/PLAN.md 4.3 rests on, executable.

    `unicode61` splits on non-alphanumerics and knows nothing about Korean
    morphology, so a token is only ever matched whole: `김치찌개` is one token, and
    `어제는` and `어제` are two unrelated ones. In English FTS5 alone would carry
    recall. Here it returns nothing at all, and only the vector lane answers -
    which is why the vector index moved from M2 into M1b.
    """
    stored = "어제 저녁에 김치찌개 먹었어"
    keyword_only = await seed(store, None, stored, "오늘은 클라이밍 갔어", "모카 밥 줬어")
    assert await keyword_only.search(query, limit=3) == []

    hybrid = MemoryRecall(store, GramEmbedder(), now=NOW)
    await hybrid.backfill()
    items = await hybrid.search(query, limit=3)

    assert stored in contents(items)
    assert next(item for item in items if item.content == stored).reason == "vector"


async def test_reason_is_both_when_the_lanes_agree(store: Store) -> None:
    recall = await seed(store, GramEmbedder(), "어제 저녁에 김치찌개 먹었어", "모카 밥 줬어")
    items = await recall.search("김치찌개", limit=3)

    top = items[0]
    assert top.content == "어제 저녁에 김치찌개 먹었어"
    assert top.reason == "both"


async def test_an_agreed_hit_outranks_a_single_lane_hit(store: Store) -> None:
    """Summing the lanes is what makes agreement mean something."""
    recall = await seed(store, GramEmbedder(), "어제 저녁에 김치찌개 먹었어", "김치찌개")

    items = await recall.search("어제 저녁에 김치찌개 먹었어", limit=5)

    assert items[0].content == "어제 저녁에 김치찌개 먹었어"
    assert items[0].reason == "both"


# --- scoring -----------------------------------------------------------------


async def test_recency_decay_reorders_identical_similarity(store: Store) -> None:
    """Same cosine for every message, so only the 30-day decay can order them.

    Inserted out of order on purpose: with equal scores the tie-break is id
    descending, so insertion order matching recency order would let this pass
    with no decay at all.
    """
    add(store, "최근 얘기", days_ago=0)
    add(store, "오래된 얘기", days_ago=60)
    add(store, "중간쯤 얘기", days_ago=30)
    recall = MemoryRecall(store, FlatEmbedder(), now=NOW)
    await recall.backfill()

    items = await recall.search("아무거나", limit=3)

    assert contents(items) == ["최근 얘기", "중간쯤 얘기", "오래된 얘기"]


async def test_a_half_life_old_memory_scores_half(store: Store) -> None:
    add(store, "지금 얘기", days_ago=0)
    add(store, "한 반감기 전 얘기", days_ago=HALF_LIFE_DAYS)
    recall = MemoryRecall(store, FlatEmbedder(), now=NOW)
    await recall.backfill()

    fresh, old = await recall.search("아무거나", limit=2)

    assert old.score == pytest.approx(fresh.score / 2, rel=1e-3)


async def test_limit_is_respected(store: Store) -> None:
    recall = await seed(store, FlatEmbedder(), *[f"메시지 {i}" for i in range(30)])
    assert len(await recall.search("메시지", limit=4)) == 4


async def test_nothing_stored_means_nothing_recalled(store: Store) -> None:
    recall = MemoryRecall(store, GramEmbedder(), now=NOW)
    assert await recall.search("어제 뭐 먹었지", limit=5) == []


# --- degrading rather than failing -------------------------------------------


async def test_without_an_embedder_the_keyword_lane_still_answers(store: Store) -> None:
    recall = await seed(store, None, "어제 저녁에 김치찌개 먹었어")
    items = await recall.search("김치찌개", limit=3)

    assert contents(items) == ["어제 저녁에 김치찌개 먹었어"]
    assert items[0].reason == "keyword"


async def test_an_embedder_that_raises_degrades_to_keyword_only(store: Store) -> None:
    """The worst place for an exception is the middle of a conversation."""
    add(store, "어제 저녁에 김치찌개 먹었어")
    recall = MemoryRecall(store, GramEmbedder(fail=True), now=NOW)

    items = await recall.search("김치찌개", limit=3)

    assert contents(items) == ["어제 저녁에 김치찌개 먹었어"]
    assert items[0].reason == "keyword"


async def test_a_vector_index_of_a_different_width_is_ignored(store: Store) -> None:
    """A model re-tagged under the same name, or a half-finished backfill.
    Scoring against the wrong vector space is worse than not scoring."""
    await seed(store, FlatEmbedder(), "어제 저녁에 김치찌개 먹었어")
    swapped = MemoryRecall(store, WideEmbedder(), now=NOW)

    items = await swapped.search("김치찌개", limit=3)

    assert [item.reason for item in items] == ["keyword"]


async def test_index_failure_never_reaches_the_caller(store: Store) -> None:
    """`index` runs right after the markdown was written, so the user's words are
    already safe. Raising here would cost them the reply instead."""
    message_id = add(store, "어제 저녁에 김치찌개 먹었어")
    recall = MemoryRecall(store, GramEmbedder(fail=True), now=NOW)

    await recall.index(message_id, "어제 저녁에 김치찌개 먹었어")

    assert store.load_embeddings("fake-gram")[0] == []


async def test_backfill_reports_partial_progress_instead_of_raising(store: Store) -> None:
    for i in range(3):
        add(store, f"메시지 {i}")
    recall = MemoryRecall(store, GramEmbedder(fail=True), now=NOW)

    assert await recall.backfill() == 0


async def test_backfill_without_an_embedder_is_a_no_op(store: Store) -> None:
    add(store, "어제 저녁에 김치찌개 먹었어")
    assert await MemoryRecall(store, None, now=NOW).backfill() == 0


# --- indexing ----------------------------------------------------------------


async def test_index_makes_a_message_findable_without_a_restart(store: Store) -> None:
    """The matrix is cached, so a message indexed after the first search has to
    invalidate it or it stays invisible for the life of the process."""
    embedder = GramEmbedder()
    recall = MemoryRecall(store, embedder, now=NOW)
    assert await recall.search("찌개", limit=3) == []

    message_id = add(store, "어제 저녁에 김치찌개 먹었어")
    await recall.index(message_id, "어제 저녁에 김치찌개 먹었어")

    assert contents(await recall.search("찌개", limit=3)) == ["어제 저녁에 김치찌개 먹었어"]


async def test_index_skips_blank_text(store: Store) -> None:
    embedder = GramEmbedder()
    await MemoryRecall(store, embedder, now=NOW).index(add(store, "..."), "   ")
    assert embedder.calls == []


async def test_backfill_only_touches_what_is_missing(store: Store) -> None:
    recall = await seed(store, GramEmbedder(), "첫 번째", "두 번째")
    add(store, "세 번째")

    assert await recall.backfill() == 1
    assert len(store.load_embeddings("fake-gram")[0]) == 3


# --- the cached matrix is updated, not thrown away ----------------------------
# The audit item due before M2 (docs/PLAN.md): `index` used to invalidate the cache,
# so the next search reloaded every vector from sqlite - synchronously, on the event
# loop, on the voice latency path. Measured on an M4 Max at dim 1024: 23 ms at 10k
# vectors, 121 ms at 50k, per turn.


async def test_a_turn_does_not_reload_the_whole_vector_matrix(store: Store) -> None:
    recall = await seed(store, GramEmbedder(), "어제 저녁에 김치찌개 먹었어")
    loads = spy_on_loads(store)

    await recall.search("찌개", limit=3)
    assert len(loads) == 1  # the cold cache, once

    for i in range(3):  # three turns' worth of new messages
        text = f"새로운 이야기 {i} 클라이밍"
        await recall.index(add(store, text), text)
        assert text in contents(await recall.search(text, limit=5))

    assert len(loads) == 1


async def test_indexing_before_the_first_search_loads_nothing(store: Store) -> None:
    """The load stays lazy. `index` runs on the turn path just after the reply, and a
    process that has never searched has no matrix worth building yet."""
    recall = MemoryRecall(store, GramEmbedder(), now=NOW)
    loads = spy_on_loads(store)

    await recall.index(add(store, "김치찌개"), "김치찌개")

    assert loads == []
    assert contents(await recall.search("찌개", limit=3)) == ["김치찌개"]


async def test_re_indexing_a_message_replaces_its_vector_rather_than_adding_one(
    store: Store,
) -> None:
    """`upsert_embedding` is a REPLACE, so an id already in the cache is an overwrite.
    A cache that appended would keep scoring the superseded vector for ever, under an
    id that now means something else."""
    recall = MemoryRecall(store, GramEmbedder(), now=NOW)
    # Message text the keyword lane cannot match, so only the vector answers.
    message_id = add(store, "옛날 얘기")
    await recall.index(message_id, "김치찌개")
    assert contents(await recall.search("김치찌개", limit=3)) == ["옛날 얘기"]

    await recall.index(message_id, "클라이밍")

    assert await recall.search("김치찌개", limit=3) == []
    assert contents(await recall.search("클라이밍", limit=3)) == ["옛날 얘기"]


async def test_a_new_width_after_a_model_swap_reloads_instead_of_corrupting(
    store: Store,
) -> None:
    """`WideEmbedder` is `FlatEmbedder`'s model name at a different width - a model
    retagged in place. The cache is holding the old width, so appending the first
    new-width vector into it would either raise on the turn path or, worse, score two
    vector spaces against each other. Reloading is what lets the lane recover."""
    await seed(store, FlatEmbedder(), "옛날 벡터")
    swapped = MemoryRecall(store, WideEmbedder(), now=NOW)
    assert await swapped.search("아무거나", limit=3) == []
    assert "dimension mismatch" in swapped.vector_lane_status()

    await swapped.index(add(store, "새 벡터"), "새 벡터")

    assert contents(await swapped.search("아무거나", limit=3)) == ["새 벡터"]
    assert swapped.vector_lane_status() == "ok"


async def test_backfill_lands_in_a_warm_cache(store: Store) -> None:
    """Backfill inserts a batch at a time, and it can run while the daemon is serving
    turns (daemon/app.py runs it at startup in chunks). Whatever it embeds has to be
    findable without a reload."""
    recall = await seed(store, GramEmbedder(), "어제 저녁에 김치찌개 먹었어")
    await recall.search("찌개", limit=3)
    loads = spy_on_loads(store)
    add(store, "클라이밍 갔어")

    assert await recall.backfill() == 1

    assert "클라이밍 갔어" in contents(await recall.search("클라이밍", limit=3))
    assert loads == []


async def test_a_zero_vector_never_becomes_nan_in_the_cache(store: Store) -> None:
    """A single character has no bigrams, so `GramEmbedder` hands back a zero vector -
    as a real embedder can for punctuation or an empty transcript. Dividing by that
    norm would put NaN in the matrix every later search multiplies against, and
    `Store.load_embeddings` guards it (`where=norms > 0`) precisely because unit rows
    are what make a dot product a cosine. The incremental path has to guard it too."""
    recall = await seed(store, GramEmbedder(), "어제 저녁에 김치찌개 먹었어")
    await recall.search("찌개", limit=3)

    await recall.index(add(store, "한"), "한")

    _, matrix = recall._embeddings()  # noqa: SLF001 - the invariant, not a detail
    assert np.isfinite(matrix).all()
    assert contents(await recall.search("찌개", limit=3)) == ["어제 저녁에 김치찌개 먹었어"]


# --- the curated tier: always injected, never searched -----------------------
# docs/PLAN.md 4.1 layer 2. The whole point of the layer is the boundary: the
# episodic log is large and searched, this is small and unconditional.


async def test_the_curated_tier_is_injected_when_the_search_finds_nothing(store: Store) -> None:
    """Unconditional means unconditional: a question that matches no message at all
    still gets what we know about the user."""
    entry(store, "김치찌개를 좋아한다")
    recall = MemoryRecall(store, None, now=NOW)

    items = await recall.search("전혀 관계없는 질문", limit=3)

    assert contents(items) == ["김치찌개를 좋아한다"]
    assert items[0].reason == "curated"


async def test_injecting_the_curated_tier_costs_no_embedder_call(store: Store) -> None:
    """docs/CONTRACTS.md non-negotiable 2. The tier is read on every turn including
    voice turns, so one embedder round trip here would be 117 ms of unconditional
    latency (docs/PLAN.md 4.3.1). The single call this makes is the query's."""
    entry(store, "김치찌개를 좋아한다")
    embedder = GramEmbedder()

    items = await MemoryRecall(store, embedder, now=NOW).search("뭐 좋아해?", limit=3)

    assert injected(items) == ["김치찌개를 좋아한다"]
    assert embedder.calls == [["뭐 좋아해?"]]


async def test_the_curated_tier_does_not_spend_the_search_limit(store: Store) -> None:
    """Injected, not ranked: the tier is appended past the cut, so it neither
    crowds out search hits nor moves their positions."""
    recall = await seed(store, FlatEmbedder(), *[f"메시지 {i}" for i in range(10)])
    for i in range(3):
        entry(store, f"큐레이션된 사실 {i}")

    items = await recall.search("메시지", limit=4)

    assert len(searched(items)) == 4
    assert len(injected(items)) == 3
    assert [item.reason for item in items][-3:] == ["curated"] * 3


async def test_the_injection_budget_drops_the_least_important_fact(store: Store) -> None:
    """`Store.active_entries` orders by importance for exactly this moment: when the
    budget truncates, the fact that goes must be the least important, not the
    oldest."""
    for i in range(MAX_INJECTED):
        entry(store, f"중요한 사실 {i}", importance=9)
    entry(store, "사소한 사실", importance=1)

    items = await MemoryRecall(store, None, now=NOW).search("아무거나", limit=3)

    assert len(items) == MAX_INJECTED
    assert "사소한 사실" not in contents(items)


async def test_a_trigger_phrase_rescues_a_fact_the_budget_would_have_dropped(
    store: Store,
) -> None:
    """docs/PLAN.md 4.3, "+ 트리거 구절 매칭". Without this the phrases are decorative:
    a fact the user just named by name stays invisible because fifty more important
    ones exist."""
    for i in range(MAX_INJECTED):
        entry(store, f"중요한 사실 {i}", importance=9)
    entry(store, "모카는 고양이 이름이다", importance=1, triggers=("모카",))
    recall = MemoryRecall(store, None, now=NOW)

    assert "모카는 고양이 이름이다" not in contents(await recall.search("오늘 뭐 하지", limit=3))

    items = await recall.search("모카 밥 줬어?", limit=3)

    assert items[0].content == "모카는 고양이 이름이다"
    assert items[0].reason == "curated-trigger"


async def test_a_trigger_phrase_matches_inside_an_inflected_korean_word(store: Store) -> None:
    """Substring, not token. The premise of this whole module is that Korean
    morphology defeats whole-token matching, so a phrase trigger that required token
    boundaries would miss "어제" in "어제는" - the same failure as the FTS5 lane."""
    entry(store, "어제 회의가 있었다", triggers=("어제",))

    items = await MemoryRecall(store, None, now=NOW).search("어제는 뭐 했지", limit=3)

    assert items[0].reason == "curated-trigger"


async def test_a_trigger_phrase_is_a_phrase_not_a_bag_of_words(store: Store) -> None:
    entry(store, "생일 선물로 등산화를 원한다", triggers=("생일 선물",))

    items = await MemoryRecall(store, None, now=NOW).search("선물 뭐 살까", limit=3)

    assert items[0].reason == "curated"  # injected anyway, but not as a trigger hit


@pytest.mark.parametrize("stored", ["{}", "[1, 2]", '[""]', "null", '{"a": ["모카"]}'])
async def test_unreadable_trigger_phrases_do_not_break_the_turn(store: Store, stored: str) -> None:
    """The column's CHECK proves the JSON parses, not that it is a list of strings, so
    a hand edit or a future writer can put any of these there. The tier is injected on
    every turn, so raising here would break every turn rather than one."""
    entry_id = entry(store, "김치찌개를 좋아한다", triggers=("모카",))
    store.conn.execute(
        "UPDATE memory_entries SET trigger_phrases = ? WHERE id = ?", (stored, entry_id)
    )
    store.conn.commit()

    items = await MemoryRecall(store, None, now=NOW).search("모카 밥 줬어?", limit=3)

    assert contents(items) == ["김치찌개를 좋아한다"]
    assert items[0].reason == "curated"


async def test_a_superseded_fact_is_not_injected(store: Store) -> None:
    """Injecting a retired fact means telling the model something known to be false."""
    entry(store, "여자친구가 있다", key="relationship")
    entry(store, "여자친구가 없다", key="relationship")

    items = await MemoryRecall(store, None, now=NOW).search("아무거나", limit=3)

    assert contents(items) == ["여자친구가 없다"]


async def test_an_injected_fact_from_relayed_text_keeps_its_origin(store: Store) -> None:
    """Reflection can conclude something from a forwarded message, and `origin` is
    the column that keeps that distinguishable. Recall replaying it as the owner's
    own words is the failure the column exists to prevent."""
    entry(store, "누군가 전달한 주장", origin="untrusted")

    item = (await MemoryRecall(store, None, now=NOW).search("아무거나", limit=3))[0]

    assert item.origin == "untrusted"


async def test_an_injected_fact_carries_its_importance_and_is_nobody_s_words(
    store: Store,
) -> None:
    entry(store, "김치찌개를 좋아한다", importance=9)

    item = (await MemoryRecall(store, None, now=NOW).search("아무거나", limit=3))[0]

    assert item.score == 9.0
    # Not "user" and not "assistant": a curated fact is a conclusion about the user,
    # and labelling it as either side's speech would make the daemon's own inference
    # come back as something that was said.
    assert item.role == "memory"
    assert item.ts == NOW


# --- provenance / hygiene ----------------------------------------------------


async def test_recalled_rows_are_marked_so_reflection_cannot_re_extract_them(
    store: Store,
) -> None:
    """docs/PLAN.md 4.2 hygiene rule 2."""
    add(store, "어제 저녁에 김치찌개 먹었어")
    add(store, "관계없는 얘기")
    recall = MemoryRecall(store, None, now=NOW)

    await recall.search("김치찌개", limit=3)

    marked = {
        row["content"]
        for row in store.conn.execute("SELECT content FROM messages WHERE recalled = 1")
    }
    assert marked == {"어제 저녁에 김치찌개 먹었어"}


async def test_marking_recalled_does_not_corrupt_the_fts_index(store: Store) -> None:
    """Updating a row fires the FTS trigger, which deletes and reinserts its
    entry. A second identical search has to return the same thing."""
    recall = await seed(store, None, "어제 저녁에 김치찌개 먹었어")

    first = await recall.search("김치찌개", limit=3)
    second = await recall.search("김치찌개", limit=3)

    assert contents(first) == contents(second) == ["어제 저녁에 김치찌개 먹었어"]


async def test_recalled_items_carry_the_timestamp_and_role(store: Store) -> None:
    store.insert_message(
        message("맛있었어?", ts=NOW - timedelta(days=2), role="assistant"),
        log_file="memory/log/2026-08-01.md",
    )
    recall = MemoryRecall(store, None, now=NOW)

    item = (await recall.search("맛있었어", limit=1))[0]

    assert item.role == "assistant"
    assert item.ts == NOW - timedelta(days=2)


# --- vector storage ----------------------------------------------------------


def test_vector_survives_the_blob_round_trip(store: Store) -> None:
    message_id = add(store, "어제 저녁에 김치찌개 먹었어")
    original = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)

    store.upsert_embedding(message_id, "test-model", original)
    ids, matrix = store.load_embeddings("test-model")

    assert ids == [message_id]
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix[0], original, rtol=1e-6)


def test_load_embeddings_normalises_so_a_dot_product_is_a_cosine(store: Store) -> None:
    store.upsert_embedding(add(store, "하나"), "test-model", [3.0, 4.0, 0.0, 0.0])

    _, matrix = store.load_embeddings("test-model")

    assert float(np.linalg.norm(matrix[0])) == pytest.approx(1.0)


def test_upsert_replaces_rather_than_duplicates(store: Store) -> None:
    message_id = add(store, "하나")
    store.upsert_embedding(message_id, "test-model", [1.0, 0.0])
    store.upsert_embedding(message_id, "test-model", [0.0, 1.0])

    ids, matrix = store.load_embeddings("test-model")

    assert ids == [message_id]
    np.testing.assert_allclose(matrix[0], [0.0, 1.0])


def test_load_embeddings_drops_rows_of_a_stale_width(store: Store) -> None:
    """A model re-tagged without renaming: two vector spaces under one key.
    Reshaping them into one matrix would raise, so the old width is dropped."""
    narrow = add(store, "옛날 벡터")
    wide = add(store, "새 벡터")
    store.upsert_embedding(narrow, "test-model", [1.0, 0.0])
    store.upsert_embedding(wide, "test-model", [1.0, 0.0, 0.0, 0.0])

    ids, matrix = store.load_embeddings("test-model")

    assert ids == [wide]
    assert matrix.shape == (1, 4)


def test_load_embeddings_of_an_unknown_model_is_empty(store: Store) -> None:
    ids, matrix = store.load_embeddings("never-used")
    assert ids == []
    assert matrix.shape[0] == 0


def test_messages_without_embedding_is_per_model(store: Store) -> None:
    first = add(store, "하나")
    second = add(store, "둘")
    store.upsert_embedding(first, "model-a", [1.0, 0.0])

    pending_a = store.messages_without_embedding("model-a", 10)
    pending_b = store.messages_without_embedding("model-b", 10)

    assert [row["id"] for row in pending_a] == [second]
    # Newest first: recency decay scores recent messages highest, so a backfill
    # that runs out of budget must have covered those rather than the oldest.
    assert [row["id"] for row in pending_b] == [second, first]


def test_delete_embeddings_only_drops_its_own_model(store: Store) -> None:
    first = add(store, "하나")
    second = add(store, "둘")
    store.upsert_embedding(first, "model-a", [1.0, 0.0])
    store.upsert_embedding(second, "model-b", [1.0, 0.0])

    assert store.delete_embeddings("model-a") == 1
    assert store.load_embeddings("model-a")[0] == []
    assert store.load_embeddings("model-b")[0] == [second]


def test_deleting_a_message_takes_its_vector_with_it(store: Store) -> None:
    """ON DELETE CASCADE, which needs `PRAGMA foreign_keys` actually on."""
    message_id = add(store, "하나")
    store.upsert_embedding(message_id, "test-model", [1.0, 0.0])

    store.conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
    store.conn.commit()

    assert store.load_embeddings("test-model")[0] == []


def test_upsert_rejects_an_empty_vector(store: Store) -> None:
    with pytest.raises(ValueError):
        store.upsert_embedding(add(store, "하나"), "test-model", [])


# --- associate: PLAN 6.1 type E ----------------------------------------------
# A separate entry point because `search` is wrong for this in two ways, and both
# of them would fail silently rather than loudly.


async def test_associate_finds_an_old_memory_search_would_have_buried(
    store: Store,
) -> None:
    """Recency decay exists to bury old messages. Type E wants exactly those, so a
    90-day-old memory must outrank nothing and still be returned."""
    add(store, "제주도에서 본 바다가 아직 기억나", days_ago=90)
    recall = MemoryRecall(store, GramEmbedder(), now=NOW)
    await recall.backfill()

    hits = await recall.associate("바다", min_age_days=30)

    assert [item.content for item in hits] == ["제주도에서 본 바다가 아직 기억나"]


async def test_associate_never_marks_a_message_as_recalled(store: Store) -> None:
    """The hazard that kept type E unbuilt.

    `search` marks its hits so reflection does not re-extract what the model was
    just shown (docs/PLAN.md 4.2 rule 2). From a five-minute background tick that
    shows nobody anything, and `messages_for_day` would then drop those rows from
    reflection **permanently** - a generator quietly starving the reflection pass.
    """
    message_id = add(store, "제주도에서 본 바다", days_ago=90)
    recall = MemoryRecall(store, GramEmbedder(), now=NOW)
    await recall.backfill()

    assert await recall.associate("바다", min_age_days=30)

    row = store.messages_by_ids([message_id])[message_id]
    assert row["recalled"] == 0


async def test_search_still_marks_what_it_showed(store: Store) -> None:
    """The guard above must not have turned the hygiene rule off for turns."""
    message_id = add(store, "제주도에서 본 바다", days_ago=1)
    recall = MemoryRecall(store, GramEmbedder(), now=NOW)
    await recall.backfill()

    assert await recall.search("바다")

    row = store.messages_by_ids([message_id])[message_id]
    assert row["recalled"] == 1


async def test_associate_excludes_anything_recent(store: Store) -> None:
    """A memory from this morning is not an association, it is the conversation."""
    add(store, "오늘 아침에 본 바다", days_ago=1)
    add(store, "작년에 본 바다", days_ago=200)
    recall = MemoryRecall(store, GramEmbedder(), now=NOW)
    await recall.backfill()

    hits = await recall.associate("바다", min_age_days=30)

    assert [item.content for item in hits] == ["작년에 본 바다"]


async def test_associate_does_not_penalise_a_memory_for_being_old(store: Store) -> None:
    """The property that makes this a separate entry point.

    Two identical memories 260 days apart must score the *same*. Under `search` the
    older one arrives at a fraction of the newer - at a 30-day half-life, 260 days
    is about 1/400 - so type E would degenerate into a slow version of `search`.

    Asserted on equal text rather than on which of two different texts wins:
    repetition is not a stronger match in either lane. bm25 normalises for document
    length and cosine similarity is unchanged by repeating a token, so an
    "obviously stronger" longer match actually scores *lower*. Learned by writing
    the wrong test first.
    """
    add(store, "제주도 바다", days_ago=300)
    add(store, "제주도 바다", days_ago=40)
    recall = MemoryRecall(store, GramEmbedder(), now=NOW)
    await recall.backfill()

    hits = await recall.associate("바다", min_age_days=30, limit=2)

    assert len(hits) == 2
    assert hits[0].score == hits[1].score


async def test_associate_with_nothing_old_enough_is_empty(store: Store) -> None:
    add(store, "어제 본 바다", days_ago=1)
    recall = MemoryRecall(store, GramEmbedder(), now=NOW)
    await recall.backfill()

    assert await recall.associate("바다", min_age_days=30) == []


async def test_associate_with_an_empty_query_makes_no_call(store: Store) -> None:
    """The tick runs every five minutes; an empty query must not cost an embedder
    round trip to learn it found nothing."""
    embedder = GramEmbedder()
    recall = MemoryRecall(store, embedder, now=NOW)
    before = len(getattr(embedder, "calls", []))

    assert await recall.associate("   ") == []
    assert len(getattr(embedder, "calls", [])) == before


async def test_a_reindexed_row_is_excluded_from_associate(store: Store) -> None:
    """Finding 2 (whole-branch review): `daemon reindex` infers `origin='owner'`
    for every rebuilt `role='user'` row, because the markdown it rebuilds from
    carries no provenance (non-negotiable 3) - so a forward that was originally
    `origin='untrusted'` comes back `'owner'` after a rebuild. `associate` is
    type E's only entry point into recall and quotes `origin == 'owner'` items
    verbatim into the judge's prompt, so a `reindexed` row must be excluded
    regardless of what its `origin` column says."""
    store.insert_message(
        message("제주도에서 본 바다가 아직 기억나", ts=NOW - timedelta(days=90)),
        log_file="memory/log/2026-08-02.md",
        reindexed=True,
    )
    recall = MemoryRecall(store, GramEmbedder(), now=NOW)
    await recall.backfill()

    assert await recall.associate("바다", min_age_days=30) == []


async def test_an_un_reindexed_row_is_the_control_for_the_test_above(store: Store) -> None:
    """Same row, same age, same query, `reindexed` left `False` - proves the
    exclusion above is about the flag and not about the row being unfindable for
    some other reason."""
    store.insert_message(
        message("제주도에서 본 바다가 아직 기억나", ts=NOW - timedelta(days=90)),
        log_file="memory/log/2026-08-02.md",
        reindexed=False,
    )
    recall = MemoryRecall(store, GramEmbedder(), now=NOW)
    await recall.backfill()

    assert await recall.associate("바다", min_age_days=30)


async def test_search_still_returns_a_reindexed_row(store: Store) -> None:
    """The exclusion above is `associate`-only. `search` - the ordinary recall
    lane read every turn - must not lose reindexed history; it only stops
    letting type E treat a rebuilt `'owner'` as observed rather than inferred.
    Rendering the distinction at read time is `daemon/companion.py`'s job, not
    this lane's."""
    store.insert_message(
        message("제주도에서 본 바다", ts=NOW - timedelta(days=1)),
        log_file="memory/log/2026-08-02.md",
        reindexed=True,
    )
    recall = MemoryRecall(store, GramEmbedder(), now=NOW)
    await recall.backfill()

    assert await recall.search("바다")


async def test_associate_degrades_to_the_keyword_lane_like_search_does(
    store: Store,
) -> None:
    """No embedder is a degradation, not a shutdown - the same choice `search`
    makes.

    Keyword-only association is narrow, because `unicode61` matches a Korean token
    only whole (docs/PLAN.md 4.3), so an exact word still hits and an inflected one
    does not. Whether that is worth a proactive utterance is the *generator's*
    call; recall's job is to return what it can find and score it honestly.
    """
    add(store, "제주도 바다", days_ago=90)
    recall = MemoryRecall(store, None, now=NOW)

    hits = await recall.associate("바다", min_age_days=30)

    assert [item.reason for item in hits] == ["keyword"]
