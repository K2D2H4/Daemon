"""Gateway routing and the fallback policy.

No network: every provider here is the FakeProvider from conftest.
"""

from __future__ import annotations

import pytest
from conftest import FakeProvider

from daemon.config import ConfigError, Route
from daemon.llm.base import Message, ProviderError
from daemon.llm.gateway import LLMGateway
from daemon.tasks import Task

PROMPT = [Message(role="user", content="hi")]


def named(name: str, reply: str = "ok", *, fail: bool = False) -> FakeProvider:
    provider = FakeProvider(reply, fail=fail)
    provider.name = name
    return provider


async def test_each_task_reaches_its_own_provider_and_model() -> None:
    local = named("ollama", "from local")
    hosted = named("anthropic", "from hosted")
    gateway = LLMGateway(
        {"ollama": local, "anthropic": hosted},
        {
            Task.PROACTIVE_JUDGE: Route("ollama", "qwen3:14b"),
            Task.REFLECTION: Route("anthropic", "claude-x"),
        },
    )

    judged = await gateway.complete(Task.PROACTIVE_JUDGE, PROMPT)
    reflected = await gateway.complete(Task.REFLECTION, PROMPT)

    assert judged.text == "from local"
    assert reflected.text == "from hosted"
    assert local.models == ["qwen3:14b"]
    assert hosted.models == ["claude-x"]


async def test_unrouted_task_raises_instead_of_guessing() -> None:
    gateway = LLMGateway({"ollama": named("ollama")}, {Task.CHAT_TEXT: Route("ollama", "m")})

    with pytest.raises(ConfigError, match="no provider routed for task 'reflection'"):
        await gateway.complete(Task.REFLECTION, PROMPT)


async def test_routing_to_an_unregistered_provider_raises() -> None:
    gateway = LLMGateway({"ollama": named("ollama")}, {Task.CHAT_VOICE: Route("gemini", "g")})

    with pytest.raises(ConfigError, match="not registered"):
        await gateway.complete(Task.CHAT_VOICE, PROMPT)


# --- fallback: only when configured, and only once --------------------------


async def test_provider_error_propagates_when_no_fallback_is_configured() -> None:
    broken = named("anthropic", fail=True)
    spare = named("ollama", "should not be used")
    gateway = LLMGateway(
        {"anthropic": broken, "ollama": spare},
        {Task.CHAT_TEXT: Route("anthropic", "claude-x")},
    )

    with pytest.raises(ProviderError):
        await gateway.complete(Task.CHAT_TEXT, PROMPT)

    assert spare.calls == [], "a registered provider is not a fallback unless configured"


async def test_configured_fallback_is_tried_exactly_once() -> None:
    broken = named("anthropic", fail=True)
    spare = named("ollama", "from fallback")
    gateway = LLMGateway(
        {"anthropic": broken, "ollama": spare},
        {Task.CHAT_TEXT: Route("anthropic", "claude-x")},
        fallback=Route("ollama", "qwen3:14b"),
    )

    completion = await gateway.complete(Task.CHAT_TEXT, PROMPT)

    assert completion.text == "from fallback"
    assert len(broken.calls) == 1
    assert spare.models == ["qwen3:14b"]


async def test_failing_fallback_is_not_retried_further() -> None:
    broken = named("anthropic", fail=True)
    also_broken = named("ollama", fail=True)
    gateway = LLMGateway(
        {"anthropic": broken, "ollama": also_broken},
        {Task.CHAT_TEXT: Route("anthropic", "claude-x")},
        fallback=Route("ollama", "qwen3:14b"),
    )

    with pytest.raises(ProviderError):
        await gateway.complete(Task.CHAT_TEXT, PROMPT)

    assert len(also_broken.calls) == 1


async def test_fallback_pointing_at_the_failing_provider_is_not_retried() -> None:
    broken = named("ollama", fail=True)
    gateway = LLMGateway(
        {"ollama": broken},
        {Task.CHAT_TEXT: Route("ollama", "qwen3:14b")},
        fallback=Route("ollama", "qwen3:14b"),
    )

    with pytest.raises(ProviderError):
        await gateway.complete(Task.CHAT_TEXT, PROMPT)

    assert len(broken.calls) == 1


async def test_call_is_logged_with_task_provider_and_tokens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # This log line is the evidence base for cost tracking, so it is behaviour.
    gateway = LLMGateway(
        {"ollama": named("ollama")}, {Task.CHAT_TEXT: Route("ollama", "qwen3:14b")}
    )

    with caplog.at_level("INFO", logger="daemon.llm.gateway"):
        await gateway.complete(Task.CHAT_TEXT, PROMPT)

    line = caplog.messages[-1]
    assert "task=chat_text" in line
    assert "provider=ollama" in line
    assert "model=qwen3:14b" in line
    assert "in_tokens=1" in line
    assert "out_tokens=1" in line
