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

**Two tiers come back, and keeping them apart is the point** (docs/PLAN.md 4.1).
The episodic log is large and *searched* - that is the scoring above. The curated
tier (`memory/core.md`, mirrored in `memory_entries`) is small and *always
injected*: no query, no similarity, no decay, and no embedder call, because it is
unconditional and this lane runs on every voice turn. Mixing the two blows the
context window or makes recall meaningless, so they are merged only at the very
end of `search`, with the injected tier appended after the search hits so that it
competes for none of `limit`'s slots.

`importance` (1-10) therefore multiplies nothing here. `messages` has no
importance column and should not get one - rating a raw observation is a
judgement about evidence rather than evidence - so `_importance` below is 1.0 for
every searched row. It is the curated tier that carries importance, as the
ordering key its injection budget truncates by (`Store.active_entries`). That is
a narrower reading of PLAN 4.3's formula than the formula states, and the
deliberate one: nothing this module scores has an importance to multiply by.

Degrading is a feature. No embedder, an unreachable Ollama, a malformed FTS
query: each one narrows recall and logs, and none of them raises. A worse recall
result costs the model some context; an exception costs the user their turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np

from daemon.clock import now as clock_now
from daemon.llm.base import Embedder
from daemon.memory.base import RecalledItem
from daemon.memory.curated import MAX_INJECTED
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

TRIGGER_SCAN = 10 * MAX_INJECTED
"""How far below the injection budget a trigger phrase can reach.

The curated tier is ordered importance DESC and cut off at `MAX_INJECTED`, so
without a wider read a fact the user just named by phrase is unreachable the
moment a budget's worth of more important facts exist - which makes `trigger_phrases`
(docs/PLAN.md 4.3, "+ 트리거 구절 매칭") decorative. Ten budgets deep, because the
scan is what it costs: measured on an M4 Max at 500 active entries, reading them
is 0.75 ms and matching their phrases 0.36 ms, the same order as the FTS5 lane's
1.9 ms and under 1% of the embedder round trip that dominates the turn. It is not
free at absurd sizes - 50k active entries measures 10 ms, because the tier's
ordering has no covering index - but that is a tier that stopped being curated.
"""

CURATED_ROLE = "memory"
"""`role` for an injected curated item. `memory_entries` has no role column and
should not have one: the tier is not something either side said, it is a
conclusion reflection reached, and labelling it "user" would let the daemon's own
inference come back as the user's words."""

SPARE_ROWS = 256
"""Rows kept free at the end of the cached vector matrix so appending one costs no
copy. See `_grow` for why the buffer is over-allocated at all."""

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
        # The vector index, cached and kept up to date in place rather than
        # reloaded - see `_embeddings` and `_remember`. `_buffer is None` means
        # cold; `_buffer[:_rows]` is the matrix; `_row_of` maps a message id to its
        # row so an upsert can overwrite instead of appending a second vector.
        self._buffer: np.ndarray | None = None
        self._rows = 0
        self._ids: list[int] = []
        self._row_of: dict[int, int] = {}
        # The vector lane can die three ways - no embedder reachable, a dimension
        # mismatch after a model swap, an unfinished backfill - and all three look
        # identical from outside: no exception, no failure, just Korean recall
        # quietly capped at the keyword-only ceiling (measured: 50% against 93%).
        # So the state is recorded rather than only logged, and /health reports it.
        self._vector_lane_error: str | None = None

    # --- search -------------------------------------------------------------

    async def search(self, query: str, *, limit: int = 8) -> list[RecalledItem]:
        """`limit` search hits, then the always-injected curated tier after them.

        One list because `Recall.search` in memory/base.py is the seam both callers
        (daemon/loop.py, daemon/voice/conversation.py) use and it is frozen: a
        second return value or an attribute nobody reads would leave the curated
        tier written every night and injected never, which is the M2 gap this
        closes. `limit` still means what it meant - the tier is *appended*, past
        the cut - so a caller measuring the rank of a search hit (evals/golden_set)
        reads the same index it read before.
        """
        # Read before the lanes so an unhelpful query still injects the tier.
        curated = self._curated_tier(query)
        pool = max(limit * CANDIDATE_FACTOR, MIN_CANDIDATES)
        keyword = self._keyword_lane(query, pool)
        vector = await self._vector_lane(query, pool)
        if not keyword and not vector:
            return curated

        top = self._score(keyword, vector, decay=True)[:limit]
        # Messages only. A curated entry is reflection's own output, so marking it
        # would mean nothing to the hygiene rule this serves (docs/PLAN.md 4.2 rule
        # 2, which guards `messages_for_day` against re-extracting what was already
        # injected) - and `memory_entries` has no such column.
        self._store.mark_recalled([message_id for _, message_id, _ in top])
        return [item for _, _, item in top] + curated

    def _score(
        self,
        keyword: dict[int, float],
        vector: dict[int, float],
        *,
        decay: bool,
        older_than: datetime | None = None,
    ) -> list[tuple[float, int, RecalledItem]]:
        """Lane hits turned into scored items, best first.

        `decay=False` and `older_than` exist for `associate`, which wants what
        recency decay is there to bury. Shared with `search` so the two cannot
        drift in how a hit becomes an item - only in how it is ranked.
        """
        rows = self._store.messages_by_ids(list(keyword.keys() | vector.keys()))
        now = self._now or clock_now()
        scored: list[tuple[float, int, RecalledItem]] = []
        for message_id, row in rows.items():
            kw = keyword.get(message_id, 0.0)
            vec = vector.get(message_id, 0.0)
            ts = from_iso(row["ts"])
            if older_than is not None and ts >= older_than:
                continue
            recency = _decay(ts, now, self._half_life_days) if decay else 1.0
            score = (kw + vec) * recency * _importance(row)
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
        return scored

    async def associate(
        self, query: str, *, limit: int = 3, min_age_days: float = 30.0
    ) -> list[RecalledItem]:
        """Old memories strongly connected to `query` - PLAN 6.1's type E.

        A separate entry point rather than a flag on `search`, because `search` is
        wrong for this twice and both are load-bearing:

        1. **It multiplies by recency decay.** At a 30-day half-life a
           three-month-old memory arrives at 0.125x, so the items type E is
           looking for are exactly the ones the scoring exists to bury. Here decay
           is off and `min_age_days` is a *floor*: anything recent is excluded,
           because a memory from this morning is not an association, it is the
           conversation.
        2. **It calls `mark_recalled`.** That is right for a turn - the hygiene rule
           in PLAN 4.2 stops reflection re-extracting what the model was just
           shown. It is wrong from a five-minute background tick, which shows
           nobody anything: those rows would be flagged as already-seen and
           `messages_for_day` would drop them from reflection **permanently**. A
           generator that silently starves the reflection pass is worse than an
           absent one, which is why type E shipped as silence until this existed.

        Makes no model call beyond the embedder, same as Lane 1.
        """
        if not query.strip():
            return []
        pool = max(limit * 8, 40)
        keyword = self._keyword_lane(query, pool)
        vector = await self._vector_lane(query, pool)
        if not keyword and not vector:
            return []
        cutoff = (self._now or clock_now()) - timedelta(days=min_age_days)
        scored = self._score(keyword, vector, decay=False, older_than=cutoff)
        return [item for _, _, item in scored[:limit]]

    # --- the curated tier (docs/PLAN.md 4.1 layer 2) -------------------------

    def _curated_tier(self, query: str) -> list[RecalledItem]:
        """The always-injected facts: triggered first, then most important, cut at
        the budget.

        Injected, not searched: no similarity, no recency decay, and above all **no
        embedder call**. Non-negotiable 2 gives this lane zero model calls, and
        unlike the vector lane - which spends one round trip on a query that may be
        worth it - this tier runs on every single turn including voice, so a round
        trip here would be an unconditional tax on the latency path. sqlite instead,
        and it is cheap: measured on an M4 Max, 0.07 ms at 50 active entries and
        0.75 ms at 500, against the 117 ms embedder call that dominates the turn
        (docs/PLAN.md 4.3.1).

        `Store.active_entries` orders by importance DESC precisely so truncation
        drops the least important fact rather than the oldest one.
        """
        rows = self._store.active_entries(TRIGGER_SCAN)
        if not rows:
            return []
        lowered = query.casefold()
        triggered: list[RecalledItem] = []
        rest: list[RecalledItem] = []
        for row in rows:
            hit = _trigger_hit(row, lowered)
            (triggered if hit else rest).append(_curated_item(row, triggered=hit))
        # Triggered first, so the budget cannot truncate away a fact the query
        # literally named. Both partitions keep the importance order they arrived
        # in, because appending in row order is stable.
        return (triggered + rest)[:MAX_INJECTED]

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
            # Not recorded as an error: "the index is empty" is a fact anyone can
            # count at any time, and remembering it made the status stale the
            # moment the first vector landed - reporting a problem that had
            # already fixed itself. Real failures (no embedder, wrong dimension)
            # stay remembered, because those are not visible from a count.
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
        """The whole vector index, cached across searches and updated in place.

        Loaded once, then maintained by our own writes only. Nothing else writes
        this table while the daemon runs - docs/CONTRACTS.md non-negotiable 9 makes
        it a single process - so a validity token queried per search would cost
        latency on every turn to catch a case that cannot happen without a restart,
        and by the same argument an incrementally updated cache cannot silently miss
        a row: every row that lands goes through `index` or `backfill`.

        What this replaces is the cache being *invalidated* by those writes, which
        moved a full reload onto the next turn - synchronous, on the event loop, on
        the voice latency path, and growing with history. Measured on an M4 Max at
        dim 1024: 23 ms at 10k vectors, 121 ms at 50k, every turn. Now the reload
        happens once per process (or once per embedding-model change) and a turn
        costs two in-place row writes - 0.005 ms at both sizes, measured through
        this code. docs/PLAN.md's audit item quotes 183 ms at 10k; this build does
        not reproduce that number, but the defect is the same one and it scales
        linearly either way.
        """
        buffer = self._buffer
        if buffer is None:
            ids, matrix = self._store.load_embeddings(self._model())
            buffer = self._adopt(ids, matrix)
            logger.debug("recall: loaded %d vectors for %s", len(self._ids), self._model())
        return self._ids, buffer[: self._rows]

    def _adopt(self, ids: list[int], matrix: np.ndarray) -> np.ndarray:
        """Take a freshly loaded index and leave room to grow into."""
        self._ids = list(ids)
        self._row_of = {message_id: row for row, message_id in enumerate(ids)}
        self._rows = len(ids)
        # Width 0 for a model with no vectors yet; the first `_remember` then finds
        # that mismatch, reloads, and picks the real width up from sqlite.
        buffer = np.zeros(
            (self._rows + max(self._rows // 4, SPARE_ROWS), matrix.shape[1]), np.float32
        )
        buffer[: self._rows] = matrix
        self._buffer = buffer
        return buffer

    def _remember(self, message_id: int, vector: Sequence[float]) -> None:
        """Fold one just-stored vector into the cached matrix. Never raises.

        Called after the row is safely in sqlite, so the worst case here is a cache
        that has to reload - never a lost vector.
        """
        buffer = self._buffer
        if buffer is None:
            return  # cold: the next search loads the index with this row in it
        row = np.asarray(vector, dtype=np.float32)
        if row.ndim != 1 or row.shape[0] != buffer.shape[1]:
            # An embedding model swapped under the same name, or the first vector
            # after loading an empty index (width 0). Reload rather than reshape:
            # `Store.load_embeddings` is the single place that decides which width
            # wins and drops the rows of the losing one, and appending a foreign
            # width here would instead score two vector spaces against each other -
            # the one failure `_vector_lane`'s dimension check cannot see, because
            # the query would match the matrix.
            self._buffer = None
            return
        # Unit rows are the invariant that makes a dot product a cosine
        # (`Store.load_embeddings` re-normalises on the way out and so does this,
        # rather than trusting the embedder to have done it). A zero vector has no
        # direction, and `load_embeddings` leaves it alone; leaving it zero here
        # keeps the cache identical to what a reload would produce, and it scores 0
        # either way.
        norm = float(np.linalg.norm(row))
        if norm > 0:
            row = row / norm
        existing = self._row_of.get(message_id)
        if existing is not None:
            # `upsert_embedding` is a REPLACE, so an id already held is an
            # overwrite. Appending would leave the superseded vector in the matrix,
            # still being scored, under an id that now means something else.
            buffer[existing] = row
            return
        if self._rows == buffer.shape[0]:
            buffer = self._grow(buffer)
        buffer[self._rows] = row
        # `load_embeddings` returns ids ordered by message_id and appends here do not
        # preserve that (backfill walks newest first). Nothing needs the order: the
        # only thing asked of `_ids` is which message row i belongs to.
        self._ids.append(message_id)
        self._row_of[message_id] = self._rows
        self._rows += 1

    def _grow(self, buffer: np.ndarray) -> np.ndarray:
        """Make room for more rows, by a fraction rather than a fixed amount.

        A copy per append is what the over-allocation buys off: appending with
        `np.vstack` measures 12 ms of blocked event loop per turn at 50k x 1024,
        against 0.005 ms here. The copy still happens when the buffer fills - 3.2 ms
        at 10k x 1024 - but once every `_rows // 4` appends, so ~2500 turns apart at
        that size.

        A quarter rather than the textbook doubling because this matrix is already
        the daemon's largest resident object - 205 MB at 50k x 1024 - and holding a
        spare copy of all of it is a worse trade than four row-copies per append.
        """
        bigger = np.zeros(
            (self._rows + max(self._rows // 4, SPARE_ROWS), buffer.shape[1]), np.float32
        )
        bigger[: self._rows] = buffer[: self._rows]
        self._buffer = bigger
        return bigger

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
        self._remember(message_id, vectors[0])

    def vector_lane_status(self) -> str:
        """`ok`, or why the vector lane is not answering."""
        return self._vector_lane_error or "ok"

    def vector_count(self) -> int:
        """How many vectors exist for the current model.

        Counted in sqlite rather than read off `_ids`, which is only as fresh as the
        cache: before the first search there is no cache at all, so a freshly
        indexed message looked like no progress whatsoever. (That is now the only
        gap - `index` keeps a warm cache current - and it is still the wrong number
        to report from, because it is exactly the startup window where a backfill's
        progress is what somebody is watching.) Distinguishes "no embedder" from
        "embedder fine, backfill unfinished".
        """
        if self._embedder is None:
            return 0
        return self._store.count_embeddings(self._embedder.model)

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
                # Same path as a single turn's `index`, batch or not: hundreds of
                # in-place row writes are still cheaper than the one full reload
                # that invalidating the cache used to buy, and one path means the
                # startup backfill cannot leave the cache in a state a turn never
                # produces. With a cold cache each of these is a no-op and the
                # first search loads all of it at once.
                self._remember(int(row["id"]), vector)
                done += 1
        return done


# --- scoring ----------------------------------------------------------------


def _decay(ts: datetime, now: datetime, half_life_days: float = HALF_LIFE_DAYS) -> float:
    age_days = max((now - ts).total_seconds(), 0.0) / 86400.0
    return math.exp(-math.log(2.0) * age_days / half_life_days)


def _importance(row: sqlite3.Row) -> float:
    """Flat 1.0, and staying that way - see the module docstring. `messages` has no
    importance column; the 1-10 rating lives on `memory_entries`, which this
    function is never asked about because the curated tier is injected rather than
    scored (`MemoryRecall._curated_tier`)."""
    return 1.0


def _trigger_hit(row: sqlite3.Row, query_casefolded: str) -> bool:
    """Does the query contain one of this entry's trigger phrases?

    docs/PLAN.md 4.3 lists phrase matching as part of Lane 1. Substring rather than
    token matching, deliberately: the reason the vector lane exists at all is that
    `unicode61` matches a Korean token only whole, so a phrase trigger that
    respected token boundaries would miss the phrase "어제" in "어제는" - the exact
    failure documented at the top of this file. A phrase is also a phrase, not a
    bag of words, so "생일 선물" must not fire on "선물" alone.

    A malformed `trigger_phrases` is ignored, never fatal. schema.sql's CHECK proves
    only that the value is valid JSON, so `{}`, `[1, 2]` or `null` can be in there -
    a hand edit, a future writer - and this runs on every turn, so raising would
    break every turn rather than one. The two `isinstance` guards are what actually
    hold; the `except` is for a JSON value SQLite accepted and Python will not.
    """
    try:
        phrases = json.loads(row["trigger_phrases"])
    except (TypeError, ValueError):
        logger.warning("recall: entry %s has unreadable trigger phrases", row["id"])
        return False
    if not isinstance(phrases, list):
        return False
    return any(
        isinstance(phrase, str) and phrase.strip() and phrase.casefold() in query_casefolded
        for phrase in phrases
    )


def _curated_item(row: sqlite3.Row, *, triggered: bool) -> RecalledItem:
    """One always-injected fact as a `RecalledItem`."""
    return RecalledItem(
        content=row["body"],
        # When the fact was last true, not when it was first written: a
        # supersession retires the old row and stamps the new one.
        ts=from_iso(row["updated_at"]),
        role=CURATED_ROLE,
        # Its importance, not a similarity, and not comparable with a search hit's
        # score - this tier is not ranked against the log, it is injected alongside
        # it. Reported rather than flattened to 1.0 so a caller looking at the block
        # can see which facts the budget would drop first.
        score=float(row["importance"]),
        reason="curated-trigger" if triggered else "curated",
        # Carried verbatim, like a message's: a fact reflection drew from relayed
        # text is 'untrusted' and must not be rendered as the owner's own words.
        origin=row["origin"],
    )


def _reason(by_keyword: bool, by_vector: bool) -> str:
    if by_keyword and by_vector:
        return "both"
    return "keyword" if by_keyword else "vector"
