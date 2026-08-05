"""Task -> Provider routing.

The only LLM entry point above this layer. Callers name a Task; the gateway
decides which provider and model serves it (docs/PLAN.md 3.2).

Every call is logged with task, provider, model and token counts. That log is
the evidence base for the hard budget breaker, so it is not decoration.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from time import perf_counter

from daemon.config import ConfigError, Route
from daemon.llm.base import Completion, Message, Provider, ProviderError, ToolSpec
from daemon.tasks import Task

logger = logging.getLogger(__name__)


class LLMGateway:
    def __init__(
        self,
        providers: Mapping[str, Provider],
        routing: Mapping[Task, Route],
        *,
        fallback: Route | None = None,
    ) -> None:
        self._providers = dict(providers)
        self._routing = dict(routing)
        self._fallback = fallback

    async def complete(
        self,
        task: Task,
        messages: list[Message],
        *,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> Completion:
        """Route one call. Falls back at most once, and only if configured."""
        route = self._routing.get(task)
        if route is None:
            raise ConfigError(f"no provider routed for task {task.value!r}")

        try:
            return await self._call(task, route, messages, max_output_tokens, temperature, tools)
        except ProviderError as exc:
            fallback = self._fallback
            if fallback is None or fallback.provider == route.provider:
                logger.warning(
                    "llm.failed task=%s provider=%s model=%s fallback=none error=%s",
                    task.value,
                    route.provider,
                    route.model,
                    exc,
                )
                raise
            logger.warning(
                "llm.fallback task=%s from=%s to=%s error=%s",
                task.value,
                route.provider,
                fallback.provider,
                exc,
            )
            # One attempt only. A chain of fallbacks hides a broken setup and
            # spends money doing it.
            return await self._call(
                task, fallback, messages, max_output_tokens, temperature, tools, fallback=True
            )

    async def _call(
        self,
        task: Task,
        route: Route,
        messages: list[Message],
        max_output_tokens: int | None,
        temperature: float | None,
        tools: Sequence[ToolSpec] | None = None,
        *,
        fallback: bool = False,
    ) -> Completion:
        provider = self._providers.get(route.provider)
        if provider is None:
            # Configuration bug, not a runtime failure: do not treat as fallback-worthy.
            raise ConfigError(
                f"task {task.value!r} routes to provider {route.provider!r}, "
                f"which is not registered (registered: {', '.join(sorted(self._providers))})"
            )

        started = perf_counter()
        completion = await provider.complete(
            messages,
            model=route.model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            tools=tools,
        )
        logger.info(
            "llm.call task=%s provider=%s model=%s in_tokens=%d out_tokens=%d ms=%d "
            "tools_offered=%d tools_asked=%d fallback=%s",
            task.value,
            route.provider,
            completion.model,
            completion.input_tokens,
            completion.output_tokens,
            (perf_counter() - started) * 1000,
            len(tools or ()),
            len(completion.tool_calls),
            fallback,
        )
        return completion
