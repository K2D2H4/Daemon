"""Lane 1 recall (docs/PLAN.md 4.3). Zero LLM calls, two lanes, one score.

The hard constraint first: **no model call happens here.** Recall runs on every
turn including voice turns, where the whole round trip has a sub-second budget.
Embedding the query is allowed - that is a local model, single-digit
milliseconds - and nothing else that thinks.

Why two lanes rather than FTS5 alone. SQLite's `unicode61` tokenizer does not
know Korean morphology, and the failure is not subtle:

    stored: "어제 저녁에 김치찌개 먹었어"
    query "김치찌개"  -> hit           (whole token)
    query "김치"      -> MISS          (not a token boundary)
    query "찌개"      -> MISS          (mid-token)
    query "어제는"    -> MISS          ("어제는" and "어제" are different tokens)

In English, FTS5 alone carries recall. In Korean it does not, which is why the
vector index moved from M2 into M1b - the M1b gate ("quotes yesterday
accurately") is unreachable in Korean without it. A morphological analyser would
be the other answer and was declined: too heavy a dependency for self-hosters.
`tests/test_recall.py` pins the above behaviour so a tokenizer change cannot
quietly invalidate the reasoning.

Scoring, per docs/PLAN.md 4.3:

    score = similarity x exp recency decay (30d half-life) x importance

`similarity` is the sum of the two lanes' contributions, each in [0, 1], so a
message both lanes agree on outranks one only a single lane found. The result is
an ordering key, not a calibrated probability - nothing downstream compares
scores across queries.

`importance` is 1.0 for every row in M1b: `messages` has no importance column and
should not get one, because a per-message 1-10 rating is a model judgement and
messages are raw observations. It lands in M2 with `memory_entries` (already in
schema.sql, with the column), and `_importance` below is the single place that
changes when the curated tier arrives.

Degrading is a feature. No embedder, an unreachable Ollama, a malformed FTS
query: each one narrows recall and logs, and none of them raises. A worse recall
result costs the model some context; an exception costs the user their turn.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import sqlite3
from datetime import datetime

import numpy as np

from daemon.clock import now as clock_now
from daemon.llm.base import Embedder
from daemon.memory.base import RecalledItem
from daemon.memory.log import from_iso
from daemon.memory.store import Store

logger = logging.getLogger(__name__)

HALF_LIFE_DAYS = 30.0
"""docs/PLAN.md 4.3. A month-old memory is worth half a fresh one."""

CANDIDATE_FACTOR = 4
"""Each lane over-fetches this many times `limit` before scoring, because recency
and importance reorder the pool - taking exactly `limit` per lane would let a
stale top hit crowd out a fresher one that recency would have promoted."""

MIN_CANDIDATES = 20

MAX_QUERY_TOKENS = 32
"""A pasted wall of text would otherwise become a 400-term FTS query."""

BACKFILL_BATCH = 32

VECTOR_LANE_BUDGET_SECONDS = 3.0
"""Wall-clock ceiling for the query embedding. Generous enough for a cold model
load to sometimes make it, short enough that a stalled Ollama costs one turn of
keyword-only recall instead of half a minute of silence."""

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
"""Word characters only, matching what `unicode61` treats as token content.
Everything it drops - quotes, `*`, `:`, `^`, `-`, parens - is FTS5 *syntax*, and
that is exactly the point: see `fts_query`."""


def fts_query(text: str) -> str:
    """Turn arbitrary user text into a safe FTS5 MATCH expression.

    FTS5 parses a *bound parameter* as query syntax, so `MATCH ?` with raw user
    text is not safe the way ordinary SQL binding is safe. Real user utterances
    that break it:

        어제 "김치찌개" 먹었나?   ->  unbalanced quote handling, phrase semantics
        C:\\Users 경로 알려줘      ->  `C:` reads as a column filter, "no such
                                      column: C"  ->  OperationalError
        NEAR 라는 단어            ->  `NEAR` is an operator, not a term
        3*4 계산해줘              ->  `*` is the prefix operator

    So: extract word tokens, wrap each in double quotes (a quoted token is a
    literal, and having stripped every non-word character there is no quote left
    inside to escape), and OR them together. OR rather than FTS5's implicit AND
    because this is recall, not retrieval - one matching term is a lead worth
    scoring, and requiring all of them makes any question longer than four words
    return nothing.

    Returns "" when there is nothing to search for; callers skip the lane.
    """
    tokens = _TOKEN_RE.findall(text)[:MAX_QUERY_TOKENS]
    return " OR ".join(f'"{token}"' for token in tokens)


class MemoryRecall:
    """`Recall` over the sqlite mirror: FTS5 keyword lane + numpy vector lane."""

    def __init__(
        self,
        store: Store,
        embedder: Embedder | None = None,
        *,
        now: datetime | None = None,
        half_life_days: float = HALF_LIFE_DAYS,
    ) -> None:
        self._store = store
        self._embedder = embedder
        # Configurable because DAEMON_RECALL_HALF_LIFE_DAYS exists: a setting the
        # code ignores is worse than no setting, since it reads as a tuning knob
        # that has been tried.
        self._half_life_days = half_life_days
        # Pinned time makes recency decay - and therefore the golden set's pass
        # rate - reproducible. None means read the clock per call.
        self._now = now
        self._matrix: np.ndarray | None = None
        self._ids: list[int] = []
        # The vector lane can die three ways - no embedder reachable, a dimension
        # mismatch after a model swap, an unfinished backfill - and all three look
        # identical from outside: no exception, no failure, just Korean recall
        # quietly capped at the keyword-only ceiling (measured: 50% against 93%).
        # So the state is recorded rather than only logged, and /health reports it.
        self._vector_lane_error: str | None = None

    # --- search -------------------------------------------------------------

    async def search(self, query: str, *, limit: int = 8) -> list[RecalledItem]:
        pool = max(limit * CANDIDATE_FACTOR, MIN_CANDIDATES)
        keyword = self._keyword_lane(query, pool)
        vector = await self._vector_lane(query, pool)
        if not keyword and not vector:
            return []

        rows = self._store.messages_by_ids(list(keyword.keys() | vector.keys()))
        now = self._now or clock_now()
        scored: list[tuple[float, int, RecalledItem]] = []
        for message_id, row in rows.items():
            kw = keyword.get(message_id, 0.0)
            vec = vector.get(message_id, 0.0)
            ts = from_iso(row["ts"])
            score = (kw + vec) * _decay(ts, now, self._half_life_days) * _importance(row)
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    message_id,
                    RecalledItem(
                        content=row["content"],
                        ts=ts,
                        role=row["role"],
                        score=score,
                        reason=_reason(kw > 0, vec > 0),
                        # Carried, not dropped: the column is unforgeable so the
                        # renderer can tell relayed text from the owner's own.
                        origin=row["origin"],
                    ),
                )
            )

        # id descending breaks score ties towards the newer message, which is the
        # same preference the decay expresses at coarser resolution.
        scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        top = scored[:limit]
        self._store.mark_recalled([message_id for _, message_id, _ in top])
        return [item for _, _, item in top]

    def _keyword_lane(self, query: str, pool: int) -> dict[int, float]:
        """message id -> similarity in (0, 1], best hit 1.0.

        bm25 is negative and unbounded, so it cannot be summed with a cosine as
        it stands. Normalising against the best hit *within this query* makes the
        two lanes commensurable, which is all the merge needs; the price is that
        a query whose only keyword hit is weak still scores that hit 1.0. Given
        the tokens come from the query itself, a hit means the message literally
        contains one of the user's words, which in Korean is a strong signal
        precisely because whole-token matching is so strict.
        """
        match = fts_query(query)
        if not match:
            return {}
        hits = self._store.search_fts(match, pool)
        if not hits:
            return {}
        best = min(rank for _, rank in hits)  # most negative bm25 = best match
        if best >= 0:
            return {int(row["id"]): 1.0 for row, _ in hits}
        return {int(row["id"]): min(1.0, rank / best) for row, rank in hits}

    async def _vector_lane(self, query: str, pool: int) -> dict[int, float]:
        """message id -> cosine in [0, 1]. Empty when the lane is unavailable."""
        if self._embedder is None:
            self._vector_lane_error = "no embedder configured"
            return {}
        try:
            # Bounded on purpose. The only limit otherwise is httpx's 30 s, and a
            # cold Ollama reloading an unloaded model is slow rather than broken -
            # so a lane declared to have a sub-second budget would sit there for
            # half a minute, in voice mode as pure silence. One keyword-only turn
            # is the better trade; the next turn finds the model warm.
            async with asyncio.timeout(VECTOR_LANE_BUDGET_SECONDS):
                vectors = await self._embedder.embed([query])
        except (Exception, TimeoutError) as exc:  # noqa: BLE001 - module docstring
            self._vector_lane_error = f"embedder failed: {type(exc).__name__}"
            logger.warning("recall: vector lane unavailable, keyword only (%s)", exc)
            return {}
        if not vectors:
            self._vector_lane_error = "embedder returned nothing"
            return {}

        ids, matrix = self._embeddings()
        if matrix.shape[0] == 0:
            self._vector_lane_error = "no vectors indexed yet - backfill has not run"
            return {}
        probe = np.asarray(vectors[0], dtype=np.float32)
        if probe.shape[0] != matrix.shape[1]:
            # A model change that kept the same name, or a half-finished
            # backfill. Silence beats scoring against the wrong vector space.
            self._vector_lane_error = (
                f"dimension mismatch: query {probe.shape[0]}, index {matrix.shape[1]}"
            )
            logger.warning(
                "recall: query is %d-dim but the index is %d-dim; run backfill",
                probe.shape[0],
                matrix.shape[1],
            )
            return {}

        # errstate because Apple's Accelerate BLAS leaves the FP exception flags
        # set after a perfectly ordinary float32 matmul, so numpy reports
        # "divide by zero encountered in matmul" on every single recall. Verified
        # spurious: the result agrees with a float64 einsum to 8e-8. Suppressed
        # here rather than globally so a real NaN elsewhere still surfaces.
        with np.errstate(all="ignore"):
            similarities = matrix @ probe
        k = min(pool, similarities.shape[0])
        # argpartition is O(N) against argsort's O(N log N); at 50k messages the
        # whole lane measures ~1 ms, which is what keeps voice inside budget.
        top = np.argpartition(-similarities, k - 1)[:k]
        found = {}
        for index in top:
            similarity = float(similarities[index])
            if similarity > 0:  # a negative cosine is evidence against, not for
                found[ids[index]] = min(1.0, similarity)
        # Cleared only here, on a lane that actually answered, so a stale error
        # cannot make /health look worse than it is - or better.
        self._vector_lane_error = None
        return found

    def _embeddings(self) -> tuple[list[int], np.ndarray]:
        """The whole vector index, cached across searches.

        Invalidated by our own writes only. Nothing else writes this table while
        the daemon runs - docs/CONTRACTS.md non-negotiable 9 makes it a single
        process - so a validity token queried per search would cost latency on
        every turn to catch a case that cannot happen without a restart.
        """
        if self._matrix is None:
            self._ids, self._matrix = self._store.load_embeddings(self._model())
            logger.debug("recall: loaded %d vectors for %s", len(self._ids), self._model())
        return self._ids, self._matrix

    def _model(self) -> str:
        return "" if self._embedder is None else self._embedder.model

    # --- indexing -----------------------------------------------------------

    async def index(self, message_id: int, text: str) -> None:
        """Embed and store one message. Never raises: this is called just after
        the message was recorded, and the markdown - the source of truth - already
        has it. Losing a vector costs recall quality until the next backfill;
        raising here would cost the user the reply to the message itself."""
        if self._embedder is None or not text.strip():
            return
        try:
            vectors = await self._embedder.embed([text])
            self._store.upsert_embedding(message_id, self._embedder.model, vectors[0])
        except Exception as exc:  # noqa: BLE001 - a failed index must not fail a turn
            logger.warning("recall: could not index message %d (%s)", message_id, exc)
            return
        self._matrix = None

    def vector_lane_status(self) -> str:
        """`ok`, or why the vector lane is not answering."""
        return self._vector_lane_error or "ok"

    def vector_count(self) -> int:
        """How many vectors are loaded. Distinguishes "no embedder" from
        "embedder fine, backfill unfinished" without reading the log."""
        return len(self._ids)

    async def backfill(self, limit: int = 500) -> int:
        """Embed messages this model has no vector for, returning how many landed.

        Run after a re-index from the markdown (which recreates rows with new ids),
        after switching embedding models, and at startup so a turn dropped by
        `index` heals itself. Reports what it managed rather than raising: a cold
        Ollama at startup should not stop the daemon from serving text.
        """
        if self._embedder is None:
            return 0
        rows = self._store.messages_without_embedding(self._embedder.model, limit)
        done = 0
        for start in range(0, len(rows), BACKFILL_BATCH):
            batch = rows[start : start + BACKFILL_BATCH]
            try:
                vectors = await self._embedder.embed([row["content"] for row in batch])
            except Exception as exc:  # noqa: BLE001 - partial progress is progress
                logger.warning("recall: backfill stopped after %d message(s) (%s)", done, exc)
                break
            for row, vector in zip(batch, vectors, strict=False):
                self._store.upsert_embedding(int(row["id"]), self._embedder.model, vector)
                done += 1
        if done:
            self._matrix = None
        return done


# --- scoring ----------------------------------------------------------------


def _decay(ts: datetime, now: datetime, half_life_days: float = HALF_LIFE_DAYS) -> float:
    age_days = max((now - ts).total_seconds(), 0.0) / 86400.0
    return math.exp(-math.log(2.0) * age_days / half_life_days)


def _importance(row: sqlite3.Row) -> float:
    """Flat 1.0 in M1b - see the module docstring. The seam for M2's
    `memory_entries.importance` (1-10, already in schema.sql) is right here."""
    return 1.0


def _reason(by_keyword: bool, by_vector: bool) -> str:
    if by_keyword and by_vector:
        return "both"
    return "keyword" if by_keyword else "vector"
