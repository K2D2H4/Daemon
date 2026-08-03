"""Local embedder (Ollama `/api/embed`).

Default model is `bge-m3`: multilingual, strong on Korean, and the reason the
vector lane was pulled forward from M2 into M1b at all (docs/PLAN.md 4.3 - FTS5's
`unicode61` tokenizer does not know Korean morphology, so keyword recall alone
cannot pass the M1b gate in Korean). Configurable, because a self-hoster who
already has `nomic-embed-text` or `snowflake-arctic-embed2` pulled should not be
made to download another gigabyte.

Two things this module owes its caller:

  * **L2-normalised vectors**, per the `Embedder` contract. Recall scores with a
    plain dot product and never computes a norm, so an unnormalised vector here
    does not fail loudly - it quietly skews every score.
  * **A bounded wait.** This runs on every turn including voice turns, so an
    unbounded call would hang the conversation rather than degrade it.

Deliberately no retry, unlike `providers/ollama.py`. A retry doubles the worst
case on the latency path, and the caller that matters (`MemoryRecall.search`)
already degrades to keyword-only on failure - a slow turn is worse than a
keyword-only turn.
"""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np

from daemon.llm.base import ProviderError

DEFAULT_MODEL = "bge-m3"

DEFAULT_TIMEOUT = 30.0
"""Enough for a cold model load, short enough that a dead Ollama degrades recall
within one turn instead of stalling it."""

KNOWN_DIMENSIONS: dict[str, int] = {
    "bge-m3": 1024,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed2": 1024,
}
"""Width for models we can name up front, so `dimensions` is readable before the
first call. An unlisted model reports 0 until it has answered once."""

MAX_BATCH = 64
"""Ollama loads the whole batch into memory at once; a backfill over a year of
history in one request is how you OOM a laptop."""


class OllamaEmbedder:
    name = "ollama"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.dimensions = KNOWN_DIMENSIONS.get(model, 0)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._base_url = base_url.rstrip("/")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), MAX_BATCH):
            vectors.extend(await self._embed_batch(texts[start : start + MAX_BATCH]))
        return vectors

    async def health(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/api/tags")
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # --- internals ----------------------------------------------------------

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{self._base_url}/api/embed"
        payload = {"model": self.model, "input": texts}
        try:
            response = await self._client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama embedder unreachable at {url}: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"ollama embedder returned HTTP {response.status_code} "
                f"{response.text[:200]}"
            )
        try:
            data: dict[str, Any] = response.json()
        except ValueError as exc:
            raise ProviderError(f"ollama embedder returned a non-JSON body: {exc}") from exc

        raw = data.get("embeddings")
        if not isinstance(raw, list) or len(raw) != len(texts):
            raise ProviderError(
                f"ollama embedder returned {type(raw).__name__} for {len(texts)} input(s), "
                f"expected a list of {len(texts)} vectors: {str(data)[:200]}"
            )
        return self._normalise(raw)

    def _normalise(self, raw: list[Any]) -> list[list[float]]:
        try:
            matrix = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"ollama embedder returned non-numeric vectors: {exc}") from exc
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ProviderError(
                f"ollama embedder returned vectors of shape {matrix.shape}, expected (n, dim)"
            )

        width = int(matrix.shape[1])
        if self.dimensions == 0:
            self.dimensions = width
        elif width != self.dimensions:
            # Two vector spaces under one `model` string would be silently mixed
            # in the embeddings table and every cosine after that is nonsense.
            raise ProviderError(
                f"ollama embedder {self.model!r} returned {width}-dim vectors, "
                f"expected {self.dimensions}"
            )

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # A zero vector (empty input, degenerate model) has no direction to
        # preserve; leaving it at zero scores 0 against everything, which is the
        # honest answer and beats a NaN propagating into every recall score.
        np.divide(matrix, norms, out=matrix, where=norms > 0)
        return [[float(value) for value in row] for row in matrix]
