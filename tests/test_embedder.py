"""The Ollama embedder, exercised with httpx MockTransport. No socket is opened.

The contract being pinned: L2-normalised vectors out, every failure as
ProviderError, and a bounded wait. Recall scores with a bare dot product, so an
unnormalised vector would not fail loudly - it would quietly skew every score.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import httpx
import pytest

from daemon.llm.base import Embedder, ProviderError
from daemon.llm.embedders.ollama import KNOWN_DIMENSIONS, MAX_BATCH, OllamaEmbedder

Handler = Callable[[httpx.Request], httpx.Response]


def mock_client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def responder(embeddings: list[list[float]], status: int = 200) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"model": "bge-m3", "embeddings": embeddings})

    return handler


def test_satisfies_the_embedder_protocol() -> None:
    assert isinstance(OllamaEmbedder(), Embedder)


def test_known_model_reports_its_width_before_the_first_call() -> None:
    assert OllamaEmbedder(model="bge-m3").dimensions == KNOWN_DIMENSIONS["bge-m3"]
    assert OllamaEmbedder(model="something-nobody-listed").dimensions == 0


async def test_posts_one_batch_to_api_embed() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0], [0.0, 1.0]]})

    async with mock_client(handler) as client:
        embedder = OllamaEmbedder("http://127.0.0.1:11434/", "tiny", client=client)
        vectors = await embedder.embed(["어제 김치찌개 먹었어", "모카 밥 줬어"])

    assert len(seen) == 1
    assert str(seen[0].url) == "http://127.0.0.1:11434/api/embed"
    assert json.loads(seen[0].read()) == {
        "model": "tiny",
        "input": ["어제 김치찌개 먹었어", "모카 밥 줬어"],
    }
    assert len(vectors) == 2


async def test_a_known_models_declared_width_is_enforced() -> None:
    """`bge-m3` is 1024-wide in the constant table. A server answering with
    something else is serving a different model under that name."""
    async with mock_client(responder([[1.0, 0.0]])) as client:
        with pytest.raises(ProviderError, match="expected 1024"):
            await OllamaEmbedder(model="bge-m3", client=client).embed(["안녕"])


async def test_normalises_what_the_server_returns() -> None:
    """Ollama does not promise unit vectors, and the `Embedder` contract does."""
    async with mock_client(responder([[3.0, 4.0], [0.0, -2.0]])) as client:
        vectors = await OllamaEmbedder(model="tiny", client=client).embed(["a", "b"])

    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert vectors[1] == pytest.approx([0.0, -1.0])
    for vector in vectors:
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-6)


async def test_a_zero_vector_stays_zero_rather_than_becoming_nan() -> None:
    """A NaN here would propagate into every recall score in the index."""
    async with mock_client(responder([[0.0, 0.0, 0.0]])) as client:
        vectors = await OllamaEmbedder(model="tiny", client=client).embed(["   "])

    assert vectors == [[0.0, 0.0, 0.0]]


async def test_learns_its_width_from_the_first_answer() -> None:
    async with mock_client(responder([[1.0, 0.0, 0.0, 0.0]])) as client:
        embedder = OllamaEmbedder(model="unlisted", client=client)
        assert embedder.dimensions == 0
        await embedder.embed(["안녕"])
        assert embedder.dimensions == 4


async def test_rejects_a_width_change_under_the_same_model_name() -> None:
    """Two vector spaces under one `model` string would be mixed in the
    embeddings table and every cosine after that is meaningless."""
    calls = [[[1.0, 0.0]], [[1.0, 0.0, 0.0]]]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": calls.pop(0)})

    async with mock_client(handler) as client:
        embedder = OllamaEmbedder(model="retagged", client=client)
        await embedder.embed(["안녕"])
        with pytest.raises(ProviderError, match="expected 2"):
            await embedder.embed(["또 안녕"])


async def test_empty_input_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call should happen for an empty batch")

    async with mock_client(handler) as client:
        assert await OllamaEmbedder(client=client).embed([]) == []


async def test_splits_a_large_batch() -> None:
    """A backfill over a year of history in one request is how a laptop OOMs."""
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        count = len(json.loads(request.read())["input"])
        sizes.append(count)
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]] * count})

    async with mock_client(handler) as client:
        texts = [f"메시지 {i}" for i in range(MAX_BATCH + 5)]
        vectors = await OllamaEmbedder(model="tiny", client=client).embed(texts)

    assert sizes == [MAX_BATCH, 5]
    assert len(vectors) == len(texts)


# --- failure paths ----------------------------------------------------------


async def test_unreachable_server_becomes_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nothing is listening")

    async with mock_client(handler) as client:
        with pytest.raises(ProviderError, match="unreachable"):
            await OllamaEmbedder(client=client).embed(["안녕"])


@pytest.mark.parametrize("status", [400, 404, 429, 500, 503])
async def test_http_errors_become_provider_error(status: int) -> None:
    async with mock_client(responder([[1.0]], status=status)) as client:
        with pytest.raises(ProviderError, match=f"HTTP {status}"):
            await OllamaEmbedder(client=client).embed(["안녕"])


async def test_non_json_body_becomes_provider_error() -> None:
    async with mock_client(lambda request: httpx.Response(200, text="<html>")) as client:
        with pytest.raises(ProviderError, match="non-JSON"):
            await OllamaEmbedder(client=client).embed(["안녕"])


@pytest.mark.parametrize(
    "body",
    [
        {"model": "bge-m3"},  # no embeddings key at all
        {"embeddings": []},  # fewer vectors than inputs
        {"embeddings": [[1.0], [2.0]]},  # more vectors than inputs
        {"embeddings": "nope"},
    ],
)
async def test_malformed_payload_becomes_provider_error(body: dict) -> None:
    async with mock_client(lambda request: httpx.Response(200, json=body)) as client:
        with pytest.raises(ProviderError):
            await OllamaEmbedder(client=client).embed(["안녕"])


async def test_scalar_vectors_become_provider_error() -> None:
    async with mock_client(lambda request: httpx.Response(200, json={"embeddings": [1.0]})) as c:
        with pytest.raises(ProviderError, match="shape"):
            await OllamaEmbedder(model="tiny", client=c).embed(["안녕"])


async def test_health_is_false_when_nothing_answers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with mock_client(handler) as client:
        assert await OllamaEmbedder(client=client).health() is False
