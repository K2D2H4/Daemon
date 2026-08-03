"""Provider HTTP layer, exercised with httpx MockTransport.

Nothing here opens a socket: MockTransport answers every request in-process.
What is being tested is the request shape, the response mapping, and that every
failure comes out as ProviderError with at most one retry.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from daemon.llm.base import Message, ProviderError
from daemon.llm.providers.anthropic import AnthropicProvider
from daemon.llm.providers.ollama import OllamaProvider

PROMPT = [Message(role="system", content="seed"), Message(role="user", content="yo")]

Handler = Callable[[httpx.Request], httpx.Response]


def mock_client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- ollama -----------------------------------------------------------------

OLLAMA_OK = {
    "model": "qwen3:14b",
    "message": {"role": "assistant", "content": "hi there"},
    "prompt_eval_count": 11,
    "eval_count": 3,
}


async def test_ollama_posts_a_non_streaming_chat_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OLLAMA_OK)

    async with mock_client(handler) as client:
        provider = OllamaProvider("http://127.0.0.1:11434/", client=client)
        completion = await provider.complete(
            PROMPT, model="qwen3:14b", max_output_tokens=64, temperature=0.4
        )

    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/chat"
    assert body["stream"] is False
    assert body["messages"] == [
        {"role": "system", "content": "seed"},
        {"role": "user", "content": "yo"},
    ]
    assert body["options"] == {"num_predict": 64, "temperature": 0.4}
    assert completion.text == "hi there"
    assert completion.model == "qwen3:14b"
    assert (completion.input_tokens, completion.output_tokens) == (11, 3)


async def test_ollama_retries_a_server_error_exactly_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="model loading")

    async with mock_client(handler) as client:
        provider = OllamaProvider(client=client)
        with pytest.raises(ProviderError, match="HTTP 503"):
            await provider.complete(PROMPT, model="qwen3:14b")

    assert calls == 2


async def test_ollama_does_not_retry_a_client_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, text="model 'nope' not found")

    async with mock_client(handler) as client:
        provider = OllamaProvider(client=client)
        with pytest.raises(ProviderError, match="rejected the request"):
            await provider.complete(PROMPT, model="nope")

    assert calls == 1


async def test_ollama_turns_an_unreachable_daemon_into_a_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with mock_client(handler) as client:
        provider = OllamaProvider(client=client)
        with pytest.raises(ProviderError, match="unreachable"):
            await provider.complete(PROMPT, model="qwen3:14b")


async def test_ollama_reports_a_malformed_response_instead_of_returning_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "qwen3:14b"})

    async with mock_client(handler) as client:
        provider = OllamaProvider(client=client)
        with pytest.raises(ProviderError, match="no message content"):
            await provider.complete(PROMPT, model="qwen3:14b")


async def test_ollama_health_never_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with mock_client(handler) as client:
        assert await OllamaProvider(client=client).health() is False


# --- anthropic --------------------------------------------------------------

ANTHROPIC_OK = {
    "model": "claude-sonnet-4-5-20250929",
    "content": [{"type": "text", "text": "hello"}],
    "usage": {"input_tokens": 5, "output_tokens": 2},
    "stop_reason": "end_turn",
}


async def test_anthropic_hoists_system_turns_and_sets_max_tokens() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ANTHROPIC_OK)

    async with mock_client(handler) as client:
        provider = AnthropicProvider("secret-key", client=client)
        completion = await provider.complete(PROMPT, model="claude-sonnet-4-5-20250929")

    request = seen[0]
    body = json.loads(request.content)
    assert request.headers["x-api-key"] == "secret-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "seed"
    assert body["messages"] == [{"role": "user", "content": "yo"}]
    assert body["max_tokens"] == 2048
    assert "temperature" not in body
    assert completion.text == "hello"
    assert (completion.input_tokens, completion.output_tokens) == (5, 2)
    assert completion.meta["stop_reason"] == "end_turn"


async def test_anthropic_retries_a_rate_limit_exactly_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=ANTHROPIC_OK)

    async with mock_client(handler) as client:
        provider = AnthropicProvider("k", client=client)
        completion = await provider.complete(PROMPT, model="claude-x")

    assert calls == 2
    assert completion.text == "hello"


async def test_anthropic_does_not_retry_a_bad_key() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="invalid x-api-key")

    async with mock_client(handler) as client:
        provider = AnthropicProvider("wrong", client=client)
        with pytest.raises(ProviderError, match="HTTP 401"):
            await provider.complete(PROMPT, model="claude-x")

    assert calls == 1


async def test_anthropic_needs_a_key() -> None:
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider("")


async def test_anthropic_needs_at_least_one_non_system_turn() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not reach the network")

    async with mock_client(handler) as client:
        provider = AnthropicProvider("k", client=client)
        with pytest.raises(ProviderError, match="at least one user"):
            await provider.complete(
                [Message(role="system", content="seed")], model="claude-x"
            )
