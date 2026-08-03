"""Provider HTTP layer, exercised with httpx MockTransport.

Nothing here opens a socket: MockTransport answers every request in-process.
What is being tested is the request shape, the response mapping, and that every
failure comes out as ProviderError with at most one retry.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable

import httpx
import pytest

from daemon.llm.base import Message, ProviderError
from daemon.llm.providers.anthropic import AnthropicProvider
from daemon.llm.providers.gemini import GeminiProvider
from daemon.llm.providers.ollama import OllamaProvider
from daemon.llm.providers.openai import OpenAIProvider

PROMPT = [Message(role="system", content="seed"), Message(role="user", content="yo")]

Handler = Callable[[httpx.Request], httpx.Response]

SECRET = "sk-live-do-not-leak-0123456789"
"""Stand-in API key. Every hosted provider is asserted never to reveal it - this
project has leaked a key through an exception chain twice."""

KOREAN = "어제 얘기했던 그거, 어떻게 됐어?"
KOREAN_REPLY = "아직 그대로야. 오늘 저녁에 다시 볼게."


def mock_client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def chain_text(exc: BaseException) -> str:
    """Everything a traceback or a log line could show: each exception still
    attached through `__cause__`/`__context__`, as message and as repr.

    `raise ... from None` does not help here - it sets a suppression flag but
    leaves `__context__` populated - so the providers keep secrets out of the
    chain instead, and this is what proves it.
    """
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts += [str(current), repr(current)]
        current = current.__cause__ or current.__context__
    return "\n".join(parts)


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


# --- openai -----------------------------------------------------------------

OPENAI_OK = {
    "model": "gpt-5.1",
    "status": "completed",
    "output": [
        # A reasoning item sits next to the reply in the same array; only the
        # message item's output_text parts are the answer.
        {"type": "reasoning", "summary": []},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello"}],
        },
    ],
    "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
}


async def test_openai_hoists_system_turns_and_opts_out_of_server_side_storage() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OPENAI_OK)

    async with mock_client(handler) as client:
        provider = OpenAIProvider(SECRET, client=client)
        completion = await provider.complete(PROMPT, model="gpt-5.1", max_output_tokens=64)

    request = seen[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1/responses"
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert body["instructions"] == "seed"
    assert body["input"] == [{"role": "user", "content": "yo"}]
    assert body["max_output_tokens"] == 64
    assert body["store"] is False
    assert "temperature" not in body
    assert completion.text == "hello"
    assert completion.model == "gpt-5.1"
    assert (completion.input_tokens, completion.output_tokens) == (5, 2)
    assert completion.meta["stop_reason"] == "completed"


async def test_openai_reports_why_a_reply_was_truncated() -> None:
    truncated = {
        **OPENAI_OK,
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=truncated)

    async with mock_client(handler) as client:
        completion = await OpenAIProvider(SECRET, client=client).complete(
            PROMPT, model="gpt-5.1", max_output_tokens=1
        )

    # The partial text is still worth handing back; the reason is why it is short.
    assert completion.text == "hello"
    assert completion.meta["stop_reason"] == "max_output_tokens"


async def test_openai_retries_a_rate_limit_exactly_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=OPENAI_OK)

    async with mock_client(handler) as client:
        completion = await OpenAIProvider(SECRET, client=client).complete(
            PROMPT, model="gpt-5.1"
        )

    assert calls == 2
    assert completion.text == "hello"


async def test_openai_treats_a_server_error_as_transient() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500, text="internal error")

    async with mock_client(handler) as client:
        provider = OpenAIProvider(SECRET, client=client)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(PROMPT, model="gpt-5.1")

    assert calls == 2
    # Transient wording, not the permanent "rejected the request".
    assert str(caught.value) == "openai returned HTTP 500"


async def test_openai_does_not_retry_a_bad_key() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="invalid_api_key")

    async with mock_client(handler) as client:
        provider = OpenAIProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="rejected the request: HTTP 401"):
            await provider.complete(PROMPT, model="gpt-5.1")

    assert calls == 1


async def test_openai_does_not_retry_an_unknown_model() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, text="model not found")

    async with mock_client(handler) as client:
        provider = OpenAIProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="rejected the request: HTTP 404"):
            await provider.complete(PROMPT, model="nope")

    assert calls == 1


async def test_openai_turns_a_timeout_into_a_provider_error_after_one_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out")

    async with mock_client(handler) as client:
        provider = OpenAIProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="unreachable"):
            await provider.complete(PROMPT, model="gpt-5.1")

    assert calls == 2


async def test_openai_reports_an_empty_response_instead_of_an_empty_reply() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "completed", "output": []})

    async with mock_client(handler) as client:
        provider = OpenAIProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="no text content"):
            await provider.complete(PROMPT, model="gpt-5.1")


async def test_openai_reports_an_unexpected_shape_without_echoing_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        # Not the documented shape at all, and it quotes the key back at us.
        return httpx.Response(200, json={"output": 42, "sent_key": SECRET})

    async with mock_client(handler) as client:
        provider = OpenAIProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="unreadable response") as caught:
            await provider.complete(PROMPT, model="gpt-5.1")

    assert SECRET not in chain_text(caught.value)


async def test_openai_reports_a_non_json_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    async with mock_client(handler) as client:
        provider = OpenAIProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="non-JSON body"):
            await provider.complete(PROMPT, model="gpt-5.1")


async def test_openai_reports_a_json_body_that_is_not_an_object() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2])

    async with mock_client(handler) as client:
        provider = OpenAIProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="JSON list, not an object"):
            await provider.complete(PROMPT, model="gpt-5.1")


async def test_openai_needs_a_key() -> None:
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAIProvider("")


async def test_openai_needs_at_least_one_non_system_turn() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not reach the network")

    async with mock_client(handler) as client:
        provider = OpenAIProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="at least one user"):
            await provider.complete([Message(role="system", content="seed")], model="gpt-5.1")


async def test_openai_round_trips_korean() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                **OPENAI_OK,
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": KOREAN_REPLY}],
                    }
                ],
            },
        )

    async with mock_client(handler) as client:
        completion = await OpenAIProvider(SECRET, client=client).complete(
            [Message(role="user", content=KOREAN)], model="gpt-5.1"
        )

    assert json.loads(seen[0].content)["input"][0]["content"] == KOREAN
    assert completion.text == KOREAN_REPLY


async def test_openai_keeps_the_key_out_of_an_error_it_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    def handler(request: httpx.Request) -> httpx.Response:
        # What the real 401 body does: it quotes the key that was sent.
        return httpx.Response(401, text=f"Incorrect API key provided: {SECRET}.")

    async with mock_client(handler) as client:
        provider = OpenAIProvider(SECRET, client=client)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(PROMPT, model="gpt-5.1")

    exc = caught.value
    assert SECRET not in str(exc)
    assert SECRET not in repr(exc)
    # Raised outside the except block, so there is no exception chain to inspect.
    assert exc.__context__ is None
    assert SECRET not in chain_text(exc)
    assert SECRET not in caplog.text
    assert "<redacted>" in str(exc)


async def test_openai_keeps_the_key_out_of_a_transport_failure_chain(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed to connect to {request.url}")

    async with mock_client(handler) as client:
        provider = OpenAIProvider(SECRET, client=client)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(PROMPT, model="gpt-5.1")

    exc = caught.value
    # The cause is deliberately kept (never `from None`), so the guarantee has to
    # come from there being no secret in it.
    assert isinstance(exc.__cause__, httpx.ConnectError)
    assert SECRET not in chain_text(exc)
    assert SECRET not in caplog.text


async def test_openai_sends_the_key_as_a_header_and_never_in_the_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OPENAI_OK)

    async with mock_client(handler) as client:
        await OpenAIProvider(SECRET, client=client).complete(PROMPT, model="gpt-5.1")

    assert SECRET not in str(seen[0].url)
    assert seen[0].headers["authorization"] == f"Bearer {SECRET}"
    # httpx logs the request line; a key in the URL would land in the log file.
    assert SECRET not in caplog.text


# --- gemini -----------------------------------------------------------------

GEMINI_OK = {
    "modelVersion": "gemini-2.5-flash",
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "hello"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 5,
        "candidatesTokenCount": 2,
        "totalTokenCount": 7,
    },
}


async def test_gemini_puts_the_system_turn_in_system_instruction() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=GEMINI_OK)

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        completion = await provider.complete(
            PROMPT, model="gemini-2.5-flash", max_output_tokens=64, temperature=0.4
        )

    request = seen[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1beta/models/gemini-2.5-flash:generateContent"
    assert body["systemInstruction"] == {"parts": [{"text": "seed"}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "yo"}]}]
    assert body["generationConfig"] == {"maxOutputTokens": 64, "temperature": 0.4}
    assert completion.text == "hello"
    assert completion.model == "gemini-2.5-flash"
    assert (completion.input_tokens, completion.output_tokens) == (5, 2)
    assert completion.meta["stop_reason"] == "STOP"


async def test_gemini_maps_assistant_turns_to_the_model_role() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=GEMINI_OK)

    async with mock_client(handler) as client:
        await GeminiProvider(SECRET, client=client).complete(
            [
                Message(role="user", content="yo"),
                Message(role="assistant", content="hi"),
                Message(role="user", content="again"),
            ],
            model="gemini-2.5-flash",
        )

    roles = [turn["role"] for turn in json.loads(seen[0].content)["contents"]]
    assert roles == ["user", "model", "user"]


async def test_gemini_accepts_a_model_id_with_the_models_prefix() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=GEMINI_OK)

    async with mock_client(handler) as client:
        await GeminiProvider(SECRET, client=client).complete(
            PROMPT, model="models/gemini-2.5-flash"
        )

    assert seen[0].url.path == "/v1beta/models/gemini-2.5-flash:generateContent"


async def test_gemini_does_not_double_count_thinking_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                **GEMINI_OK,
                # thoughtsTokenCount is a breakdown of candidatesTokenCount on
                # this API, not an extra charge on top of it.
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 500,
                    "thoughtsTokenCount": 480,
                    "totalTokenCount": 505,
                },
            },
        )

    async with mock_client(handler) as client:
        completion = await GeminiProvider(SECRET, client=client).complete(
            PROMPT, model="gemini-2.5-flash"
        )

    assert completion.output_tokens == 500


async def test_gemini_retries_a_rate_limit_exactly_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, text="RESOURCE_EXHAUSTED")
        return httpx.Response(200, json=GEMINI_OK)

    async with mock_client(handler) as client:
        completion = await GeminiProvider(SECRET, client=client).complete(
            PROMPT, model="gemini-2.5-flash"
        )

    assert calls == 2
    assert completion.text == "hello"


async def test_gemini_treats_a_server_error_as_transient() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="UNAVAILABLE")

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(PROMPT, model="gemini-2.5-flash")

    assert calls == 2
    assert str(caught.value) == "gemini returned HTTP 503"


async def test_gemini_does_not_retry_a_refused_key_and_names_the_standard_key_trap() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403, text="PERMISSION_DENIED")

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="rejected the request: HTTP 403") as caught:
            await provider.complete(PROMPT, model="gemini-2.5-flash")

    assert calls == 1
    assert "September 2026" in str(caught.value)


async def test_gemini_does_not_retry_a_401_either() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401, text="UNAUTHENTICATED")

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="rejected the request: HTTP 401"):
            await provider.complete(PROMPT, model="gemini-2.5-flash")

    assert calls == 1


async def test_gemini_does_not_retry_an_unknown_model() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, text="NOT_FOUND")

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="rejected the request: HTTP 404") as caught:
            await provider.complete(PROMPT, model="nope")

    assert calls == 1
    # The Standard-key hint belongs to 401/403 only; a bad model id is not that.
    assert "September 2026" not in str(caught.value)


async def test_gemini_turns_a_timeout_into_a_provider_error_after_one_retry() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out")

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="unreachable"):
            await provider.complete(PROMPT, model="gemini-2.5-flash")

    assert calls == 2


async def test_gemini_reports_a_blocked_prompt_instead_of_an_empty_reply() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []}
        )

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="no text content") as caught:
            await provider.complete(PROMPT, model="gemini-2.5-flash")

    # The reason has to survive into the message or the failure is unexplainable.
    assert "SAFETY" in str(caught.value)


async def test_gemini_reports_an_unexpected_shape_without_echoing_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"candidates": "nope", "sent_key": SECRET})

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="unreadable response") as caught:
            await provider.complete(PROMPT, model="gemini-2.5-flash")

    assert SECRET not in chain_text(caught.value)


async def test_gemini_reports_a_non_json_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="non-JSON body"):
            await provider.complete(PROMPT, model="gemini-2.5-flash")


async def test_gemini_reports_a_json_body_that_is_not_an_object() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2])

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="JSON list, not an object"):
            await provider.complete(PROMPT, model="gemini-2.5-flash")


async def test_gemini_needs_a_key() -> None:
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        GeminiProvider("")


async def test_gemini_needs_at_least_one_non_system_turn() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not reach the network")

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        with pytest.raises(ProviderError, match="at least one user"):
            await provider.complete(
                [Message(role="system", content="seed")], model="gemini-2.5-flash"
            )


async def test_gemini_round_trips_korean() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                **GEMINI_OK,
                "candidates": [
                    {"content": {"parts": [{"text": KOREAN_REPLY}]}, "finishReason": "STOP"}
                ],
            },
        )

    async with mock_client(handler) as client:
        completion = await GeminiProvider(SECRET, client=client).complete(
            [Message(role="user", content=KOREAN)], model="gemini-2.5-flash"
        )

    body = json.loads(seen[0].content)
    assert body["contents"][0]["parts"][0]["text"] == KOREAN
    assert completion.text == KOREAN_REPLY


async def test_gemini_keeps_the_key_out_of_an_error_it_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=f"API_KEY_INVALID: {SECRET}")

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(PROMPT, model="gemini-2.5-flash")

    exc = caught.value
    assert SECRET not in str(exc)
    assert SECRET not in repr(exc)
    # Raised outside the except block, so there is no exception chain to inspect.
    assert exc.__context__ is None
    assert SECRET not in chain_text(exc)
    assert SECRET not in caplog.text
    assert "<redacted>" in str(exc)


async def test_gemini_keeps_the_key_out_of_a_transport_failure_chain(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed to connect to {request.url}")

    async with mock_client(handler) as client:
        provider = GeminiProvider(SECRET, client=client)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(PROMPT, model="gemini-2.5-flash")

    exc = caught.value
    assert isinstance(exc.__cause__, httpx.ConnectError)
    # This is the assertion that fails the moment anyone switches to `?key=`:
    # the URL is inside the transport error, and the transport error is the cause.
    assert SECRET not in chain_text(exc)
    assert SECRET not in caplog.text


async def test_gemini_sends_the_key_as_a_header_and_never_in_the_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=GEMINI_OK)

    async with mock_client(handler) as client:
        await GeminiProvider(SECRET, client=client).complete(PROMPT, model="gemini-2.5-flash")

    request = seen[0]
    assert request.headers["x-goog-api-key"] == SECRET
    assert SECRET not in str(request.url)
    assert "key=" not in str(request.url)
    assert SECRET not in caplog.text


async def test_gemini_health_never_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with mock_client(handler) as client:
        assert await GeminiProvider(SECRET, client=client).health() is False


async def test_openai_health_never_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with mock_client(handler) as client:
        assert await OpenAIProvider(SECRET, client=client).health() is False
