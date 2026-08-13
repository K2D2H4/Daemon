# OpenAI-compatible provider — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Qwen, Kimi, DeepSeek, OpenRouter or a self-hosted server be Daemon's main model, chosen during `daemon setup` with an API key.

**Architecture:** One new provider module speaking OpenAI's **Chat Completions** API, with the endpoint supplied by the user. It joins `HOSTED_PROVIDERS` as a fourth name, `openai_compatible`. Vendor identity is not a provider name — onboarding prefills a base URL and model from a table, and `.env` stores only the URL.

**Tech Stack:** Python 3.11+, `httpx` (raw HTTP, no vendor SDK), `pydantic-settings`, `pytest` with `httpx.MockTransport`.

**Spec:** [2026-08-11-openai-compatible-provider-design.md](../specs/2026-08-11-openai-compatible-provider-design.md)

## Global Constraints

- **`daemon/llm/providers/openai.py` must not be modified.** Any diff touching it is out of scope. It speaks Responses; this work speaks Chat Completions.
- **Provider name is exactly `openai_compatible`.** `Settings.provider_model` resolves `getattr(self, f"{provider}_model")` and builds error text as `DAEMON_{provider.upper()}_MODEL`, so the name fixes the module path, the settings field and every env key.
- **Only `daemon/app.py` may import a concrete provider** (`docs/CONTRACTS.md` layering rule). Its imports are function-local.
- **No test may touch the network, an API key, a microphone or a speaker** (`tests/CLAUDE.md`). Live checks live in `evals/`, run by hand.
- **At least one Korean case** for any module touching text.
- **Providers retry at most once internally** and raise `ProviderError`, never a vendor exception — the gateway decides about fallback (`daemon/llm/base.py`).
- **The API key must never reach an exception chain.** This project has leaked one twice; `tests/test_providers.py` asserts against it for every hosted provider.
- Admin UI copy stays **English-only** (Silkscreen has no Hangul).
- Run `python3 -m ruff check .` before every commit.

---

### Task 1: The provider module

**Files:**
- Create: `daemon/llm/providers/openai_compatible.py`
- Test: `tests/test_providers.py` (append a new section at the end)

**Interfaces:**
- Consumes: `Completion`, `Message`, `ImageBlock`, `ProviderError`, `ToolCall`, `ToolSpec`, `decode_tool_arguments`, `synthesise_call_id` from `daemon.llm.base`.
- Produces: `OpenAICompatibleProvider(api_key: str, base_url: str, *, timeout: float = 60.0, client: httpx.AsyncClient | None = None)` with `name = "openai_compatible"`, `async complete(...) -> Completion`, `async health() -> bool`, `async aclose() -> None`. Module-level helpers `_messages(messages: list[Message]) -> list[dict[str, Any]]` and `_tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_providers.py`:

```python
# --- openai-compatible -------------------------------------------------------

COMPAT_OK = {
    "model": "qwen-plus",
    "choices": [
        {"message": {"role": "assistant", "content": KOREAN_REPLY}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 5},
}
COMPAT_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


async def test_compatible_posts_a_chat_completions_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=COMPAT_OK)

    async with mock_client(handler) as client:
        provider = OpenAICompatibleProvider(SECRET, COMPAT_URL + "/", client=client)
        completion = await provider.complete(
            [Message(role="system", content="seed"), Message(role="user", content=KOREAN)],
            model="qwen-plus",
            max_output_tokens=64,
            temperature=0.4,
        )

    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/compatible-mode/v1/chat/completions"
    assert seen[0].headers["authorization"] == f"Bearer {SECRET}"
    # The system turn stays a role in the array - unlike Responses, which hoists it.
    assert body["messages"] == [
        {"role": "system", "content": "seed"},
        {"role": "user", "content": KOREAN},
    ]
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.4
    assert completion.text == KOREAN_REPLY
    assert completion.model == "qwen-plus"
    assert (completion.input_tokens, completion.output_tokens) == (12, 5)
    assert completion.meta["stop_reason"] == "stop"


async def test_compatible_declares_tools_nested_under_function() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=COMPAT_OK)

    spec = ToolSpec(
        name="read_file",
        description="Read a file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    async with mock_client(handler) as client:
        provider = OpenAICompatibleProvider(SECRET, COMPAT_URL, client=client)
        await provider.complete(PROMPT, model="qwen-plus", tools=[spec])

    body = json.loads(seen[0].content)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]


async def test_compatible_reads_tool_calls_with_json_string_arguments() -> None:
    answer = {
        "model": "qwen-plus",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "메모.txt"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=answer)

    async with mock_client(handler) as client:
        provider = OpenAICompatibleProvider(SECRET, COMPAT_URL, client=client)
        completion = await provider.complete(PROMPT, model="qwen-plus")

    assert completion.text == ""
    assert completion.tool_calls == (
        ToolCall(id="call_abc", name="read_file", arguments={"path": "메모.txt"}),
    )


async def test_compatible_synthesises_an_id_when_the_endpoint_issues_none() -> None:
    answer = {
        "model": "local",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "list_dir", "arguments": {"path": "."}}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=answer)

    async with mock_client(handler) as client:
        provider = OpenAICompatibleProvider(SECRET, COMPAT_URL, client=client)
        completion = await provider.complete(PROMPT, model="local")

    assert completion.tool_calls[0].id == "list_dir-0"
    assert completion.tool_calls[0].name == "list_dir"


def test_compatible_sends_a_tool_result_as_its_own_role() -> None:
    from daemon.llm.providers.openai_compatible import _messages

    turns = _messages(
        [
            Message(
                role="assistant",
                content="",
                tool_calls=(ToolCall(id="call_abc", name="read_file", arguments={"path": "a"}),),
            ),
            Message(role="tool", content="파일 내용", tool_call_id="call_abc"),
        ]
    )
    assert turns[0]["tool_calls"][0]["id"] == "call_abc"
    assert turns[0]["tool_calls"][0]["function"]["arguments"] == '{"path": "a"}'
    assert turns[1] == {"role": "tool", "tool_call_id": "call_abc", "content": "파일 내용"}


def test_compatible_encodes_an_image_as_a_data_uri() -> None:
    from daemon.llm.providers.openai_compatible import _messages

    turns = _messages(
        [
            Message(
                role="user",
                content="what is this",
                images=(ImageBlock(b"\xff\xd8\xff", "image/jpeg"),),
            )
        ]
    )
    encoded = base64.b64encode(b"\xff\xd8\xff").decode()
    assert turns[0]["content"][1]["image_url"]["url"] == f"data:image/jpeg;base64,{encoded}"


async def test_compatible_retries_a_server_error_exactly_once() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, text="upstream busy")

    async with mock_client(handler) as client:
        provider = OpenAICompatibleProvider(SECRET, COMPAT_URL, client=client)
        with pytest.raises(ProviderError, match="HTTP 503"):
            await provider.complete(PROMPT, model="qwen-plus")

    assert calls == 2


async def test_compatible_does_not_retry_a_client_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(402, text="insufficient credits")

    async with mock_client(handler) as client:
        provider = OpenAICompatibleProvider(SECRET, COMPAT_URL, client=client)
        with pytest.raises(ProviderError, match="rejected the request"):
            await provider.complete(PROMPT, model="qwen-plus")

    assert calls == 1


async def test_compatible_never_reveals_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"Incorrect API key provided: {SECRET}")

    async with mock_client(handler) as client:
        provider = OpenAICompatibleProvider(SECRET, COMPAT_URL, client=client)
        with pytest.raises(ProviderError) as caught:
            await provider.complete(PROMPT, model="qwen-plus")

    assert SECRET not in chain_text(caught.value)


async def test_compatible_rejects_a_bare_array_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"nope": True}])

    async with mock_client(handler) as client:
        provider = OpenAICompatibleProvider(SECRET, COMPAT_URL, client=client)
        with pytest.raises(ProviderError, match="not an object"):
            await provider.complete(PROMPT, model="qwen-plus")


async def test_compatible_reports_an_empty_answer_instead_of_returning_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "qwen-plus", "choices": []})

    async with mock_client(handler) as client:
        provider = OpenAICompatibleProvider(SECRET, COMPAT_URL, client=client)
        with pytest.raises(ProviderError, match="no text content"):
            await provider.complete(PROMPT, model="qwen-plus")


async def test_compatible_needs_both_a_key_and_an_endpoint() -> None:
    with pytest.raises(ProviderError, match="OPENAI_COMPATIBLE_API_KEY"):
        OpenAICompatibleProvider("", COMPAT_URL)
    with pytest.raises(ProviderError, match="DAEMON_OPENAI_COMPATIBLE_BASE_URL"):
        OpenAICompatibleProvider(SECRET, "")


async def test_compatible_health_never_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with mock_client(handler) as client:
        provider = OpenAICompatibleProvider(SECRET, COMPAT_URL, client=client)
        assert await provider.health() is False
```

Add the import beside the other providers at the top of the file:

```python
from daemon.llm.providers.openai_compatible import OpenAICompatibleProvider
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_providers.py -k compatible -v`
Expected: collection error — `ModuleNotFoundError: No module named 'daemon.llm.providers.openai_compatible'`

- [ ] **Step 3: Write the provider**

Create `daemon/llm/providers/openai_compatible.py`:

```python
"""Any endpoint that speaks OpenAI's Chat Completions API.

Qwen (Alibaba Model Studio), Kimi (Moonshot), DeepSeek, OpenRouter and a
self-hosted vLLM or LM Studio all expose the same surface, so one module serves
them all and they differ only by `base_url` and key.

Deliberately not a refactor of `providers/openai.py`, which speaks the newer
Responses API. Merging the two would make a single provider answer to two vendor
identities, so someone running Qwen would read `provider=openai` in every
gateway log line and in `.env` - the choice displayed would not be the choice
stored. The ~40 lines of shared HTTP scaffolding are copied on purpose, the same
trade this package already makes for `MODELS_URL` and `SENSITIVITIES`.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from typing import Any

import httpx

from daemon.llm.base import (
    Completion,
    Message,
    ProviderError,
    ToolCall,
    ToolSpec,
    decode_tool_arguments,
    synthesise_call_id,
)

DEFAULT_TIMEOUT = 60.0


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError("openai_compatible provider needs OPENAI_COMPATIBLE_API_KEY")
        if not base_url:
            raise ProviderError(
                "openai_compatible provider needs DAEMON_OPENAI_COMPATIBLE_BASE_URL"
            )
        self._api_key = api_key
        """Kept only so `_redact` can strip it back out of an upstream body: a 401
        body often echoes the key that was sent, and that body goes into an
        exception message."""
        self._base_url = base_url.rstrip("/")
        self._chat_url = f"{self._base_url}/chat/completions"
        self._models_url = f"{self._base_url}/models"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        tools: Sequence[ToolSpec] | None = None,
    ) -> Completion:
        turns = _messages(messages)
        if not turns:
            raise ProviderError("openai_compatible needs at least one message")

        payload: dict[str, Any] = {"model": model, "messages": turns}
        if max_output_tokens is not None:
            # `max_tokens`, not Responses' `max_output_tokens`. Compatible
            # endpoints implement the classic spelling; the newer
            # `max_completion_tokens` is not universal among them.
            payload["max_tokens"] = max_output_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            # Nested under `function`, where Responses puts these fields flat.
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.parameters,
                    },
                }
                for spec in tools
            ]

        data = await self._post(payload)
        try:
            choices = data.get("choices") or []
            message = choices[0].get("message") or {} if choices else {}
            text = _text(message.get("content"))
            calls = _tool_calls(message)
            usage = data.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens", 0))
            output_tokens = int(usage.get("completion_tokens", 0))
            finish = str(choices[0].get("finish_reason", "")) if choices else ""
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            # Per llm/base.py every failure leaves as ProviderError - including a
            # body that simply is not shaped like the documented one.
            raise ProviderError(
                f"openai_compatible returned an unreadable response: "
                f"{self._redact(repr(data))}"
            ) from exc

        if not text and not calls:
            # A turn that only asked for tools has no text, and that one is fine.
            # Everything else here is a refusal or an empty answer.
            raise ProviderError(
                f"openai_compatible returned no text content: {self._redact(repr(data))}"
            )

        return Completion(
            text=text,
            model=str(data.get("model", model)),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            meta={"stop_reason": finish},
            tool_calls=calls,
        )

    async def health(self) -> bool:
        """Cheapest authenticated endpoint that costs no tokens. Optional in the
        compatible spec, so a False here means "could not confirm", not "down"."""
        try:
            response = await self._client.get(self._models_url, headers=self._headers)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _redact(self, text: str) -> str:
        """Remove the key from anything upstream that is about to be quoted.

        Nothing here is raised `from None`: suppressing a context does not remove
        it from `__context__`, so the only real defence is that no secret is in
        the chain to begin with - the key is a header, never a URL.
        """
        return text.replace(self._api_key, "<redacted>")

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST with exactly one retry (llm/base.py: the gateway decides about
        fallback, providers do not build retry chains)."""
        for attempt in (1, 2):
            may_retry = attempt == 1
            try:
                response = await self._client.post(
                    self._chat_url, json=payload, headers=self._headers
                )
            except httpx.HTTPError as exc:
                if may_retry:
                    continue
                # `exc` names the URL, which carries no secret here.
                raise ProviderError(f"openai_compatible unreachable: {exc}") from exc

            if response.status_code == 429 or response.status_code >= 500:
                if may_retry:
                    continue
                raise ProviderError(f"openai_compatible returned HTTP {response.status_code}")
            if response.status_code >= 400:
                # Permanent - 401 bad key, 402 no credit (OpenRouter's answer for a
                # paid model on an unfunded account), 404 wrong endpoint or model.
                raise ProviderError(
                    f"openai_compatible rejected the request: HTTP {response.status_code} "
                    f"{self._redact(response.text[:200])}"
                )
            try:
                data: dict[str, Any] = response.json()
            except ValueError as exc:
                raise ProviderError(
                    f"openai_compatible returned a non-JSON body: {exc}"
                ) from exc
            if not isinstance(data, dict):
                # A proxy in front of one of these endpoints can answer with a
                # bare JSON array.
                raise ProviderError(
                    f"openai_compatible returned a JSON {type(data).__name__}, not an object"
                )
            return data

        raise ProviderError("openai_compatible call failed")  # unreachable


def _text(content: object) -> str:
    """The reply text, whether the endpoint sent a string or the multipart form.

    A plain string is the common answer. Some gateways mirror the *request*
    shape back and send a list of typed parts, and treating that as empty would
    turn a perfectly good reply into `no text content`.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    """`tool_calls` off an assistant turn, with arguments already decoded.

    The id is synthesised when the endpoint issues none: a self-hosted server may
    omit it, and `daemon/loop.py` pairs a result back to its request by that id.
    `synthesise_call_id`'s inverse, `call_name`, recovers the function name from
    the shape this produces.
    """
    raw = message.get("tool_calls") or []
    if not isinstance(raw, list):
        return ()
    calls: list[ToolCall] = []
    for index, call in enumerate(raw):
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        name = str(function.get("name", ""))
        calls.append(
            ToolCall(
                id=str(call.get("id") or synthesise_call_id(name, index)),
                name=name,
                arguments=decode_tool_arguments(function.get("arguments")),
            )
        )
    return tuple(calls)


def _messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Neutral messages as Chat Completions turns.

    Closer to the neutral shape than Responses is: `role: tool` and
    `tool_call_id` are this API's own spelling, so `daemon/llm/base.py` needs no
    translation for them. The system turn also stays in the array rather than
    being hoisted to a top-level field.
    """
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            turns.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id or "",
                    "content": message.content,
                }
            )
            continue
        if message.role == "assistant" and message.tool_calls:
            turns.append(
                {
                    "role": "assistant",
                    # Explicitly null rather than absent: several endpoints reject
                    # an assistant turn that carries neither content nor the key.
                    "content": message.content or None,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                # Back to a string: this API sends and expects
                                # JSON text here.
                                "arguments": json.dumps(
                                    call.arguments, ensure_ascii=False
                                ),
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
            )
            continue
        if message.role == "user" and message.images:
            content: list[dict[str, Any]] = []
            if message.content:
                content.append({"type": "text", "text": message.content})
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.media_type};base64,"
                        f"{base64.b64encode(img.data).decode()}"
                    },
                }
                for img in message.images
            )
            turns.append({"role": "user", "content": content})
            continue
        turns.append({"role": message.role, "content": message.content})
    return turns
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_providers.py -k compatible -v`
Expected: PASS, all 13.

Then the whole provider file, to confirm nothing else moved:
Run: `python3 -m pytest tests/test_providers.py -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
python3 -m ruff check .
git add daemon/llm/providers/openai_compatible.py tests/test_providers.py
git commit -m "llm: a provider for every endpoint that speaks Chat Completions"
```

---

### Task 2: Config surface and assembly

These land together on purpose. `tests/test_reachable.py` fails the moment a name enters `PROVIDER_KEY_ENV` unless the module exists **and** `app.py` builds it, so splitting them would leave the suite red between two commits.

**Files:**
- Modify: `daemon/config.py` (`PROVIDER_KEY_ENV`, `HOSTED_PROVIDERS`, `Settings` fields, a validator, `_provider_problems`)
- Modify: `daemon/app.py` (`_build_providers`)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `OpenAICompatibleProvider` from Task 1.
- Produces: `Settings.openai_compatible_api_key`, `.openai_compatible_model`, `.openai_compatible_base_url`; `PROVIDER_KEY_ENV["openai_compatible"] == "OPENAI_COMPATIBLE_API_KEY"`; `HOSTED_PROVIDERS` gains `"openai_compatible"`. Task 3 reads all of these.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
COMPAT_ENV = {
    "DAEMON_PRESET": "balanced",
    "DAEMON_HOSTED_PROVIDER": "openai_compatible",
    "OPENAI_COMPATIBLE_API_KEY": "sk-compat-test",
    "DAEMON_OPENAI_COMPATIBLE_MODEL": "qwen-plus",
    "DAEMON_OPENAI_COMPATIBLE_BASE_URL": "https://api.deepseek.com/v1",
}


def test_openai_compatible_is_a_choosable_hosted_provider() -> None:
    settings = Settings(**COMPAT_ENV)
    route = settings.route_for(Task.CHAT_TEXT)
    assert route == Route("openai_compatible", "qwen-plus")


def test_openai_compatible_needs_a_base_url() -> None:
    settings = Settings(**{**COMPAT_ENV, "DAEMON_OPENAI_COMPATIBLE_BASE_URL": ""})
    problems = settings.problems()
    assert any("DAEMON_OPENAI_COMPATIBLE_BASE_URL" in problem for problem in problems)


def test_openai_compatible_base_url_drops_a_trailing_slash() -> None:
    settings = Settings(
        **{**COMPAT_ENV, "DAEMON_OPENAI_COMPATIBLE_BASE_URL": "https://api.deepseek.com/v1/"}
    )
    assert settings.openai_compatible_base_url == "https://api.deepseek.com/v1"


def test_openai_compatible_base_url_rejects_the_full_endpoint_and_says_what_to_use() -> None:
    with pytest.raises(ValidationError) as caught:
        Settings(
            **{
                **COMPAT_ENV,
                "DAEMON_OPENAI_COMPATIBLE_BASE_URL": (
                    "https://api.deepseek.com/v1/chat/completions"
                ),
            }
        )
    assert "https://api.deepseek.com/v1" in str(caught.value)


def test_openai_compatible_base_url_must_be_http() -> None:
    with pytest.raises(ValidationError):
        Settings(**{**COMPAT_ENV, "DAEMON_OPENAI_COMPATIBLE_BASE_URL": "api.deepseek.com"})
```

Check the top of `tests/test_config.py` for the imports it already has; add `from pydantic import ValidationError` and `pytest` only if absent. `problems()` is the existing method used by the other provider-problem tests in that file — match whatever name they call.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_config.py -k compatible -v`
Expected: FAIL — `unknown DAEMON_HOSTED_PROVIDER 'openai_compatible'`.

- [ ] **Step 3: Add the config surface**

In `daemon/config.py`, beside the other provider constants:

```python
OPENAI_COMPATIBLE = "openai_compatible"
```

Add to `PROVIDER_KEY_ENV`:

```python
    OPENAI_COMPATIBLE: "OPENAI_COMPATIBLE_API_KEY",
```

Extend `HOSTED_PROVIDERS`:

```python
HOSTED_PROVIDERS = ("anthropic", "openai", "gemini", "openai_compatible")
"""What DAEMON_HOSTED_PROVIDER accepts. Ollama is not here - "hosted" is the
opposite of local, and the offline preset never resolves HOSTED at all.

`openai_compatible` is one name for many vendors on purpose: Qwen, Kimi,
DeepSeek, OpenRouter and a self-hosted server differ by endpoint, not by
protocol, so the endpoint is configuration and the protocol is the provider."""
```

Add the settings fields beside the other model ids:

```python
    openai_compatible_model: str = Field(
        default="", alias="DAEMON_OPENAI_COMPATIBLE_MODEL"
    )
    openai_compatible_base_url: str = Field(
        default="", alias="DAEMON_OPENAI_COMPATIBLE_BASE_URL"
    )
    """Which OpenAI-compatible endpoint answers, up to and including the version
    segment - `https://api.deepseek.com/v1`, not the `/chat/completions` below it.

    No default, deliberately, for the reason `DAEMON_GEMINI_LIVE_MODEL` has none:
    a guessed endpoint fails at the first conversation instead of at startup."""
```

Add the key field beside the other secrets (match the existing `anthropic_api_key` declaration style):

```python
    openai_compatible_api_key: str = Field(default="", alias="OPENAI_COMPATIBLE_API_KEY")
```

Add the validator beside the others:

```python
    @field_validator("openai_compatible_base_url", mode="before")
    @classmethod
    def _clean_base_url(cls, value: object) -> object:
        """Strip the trailing slash, and refuse the whole endpoint URL.

        Vendor docs print `.../v1/chat/completions`, and pasting that whole line
        is the predictable mistake. Left alone the provider appends the path a
        second time and the resulting 404 explains nothing. Rejected rather than
        quietly repaired, and the message carries the value to use instead - the
        same choice QUIET_HOURS_RE makes: loud beats degraded.
        """
        if not isinstance(value, str):
            return value
        text = value.strip().rstrip("/")
        if not text:
            return ""
        if not text.startswith(("http://", "https://")):
            raise ValueError(
                f"DAEMON_OPENAI_COMPATIBLE_BASE_URL must start with http:// or https:// - "
                f"got {text!r}"
            )
        if text.endswith("/chat/completions"):
            raise ValueError(
                "DAEMON_OPENAI_COMPATIBLE_BASE_URL must not include /chat/completions - "
                f"use {text.removesuffix('/chat/completions')}"
            )
        return text
```

In `_provider_problems`, after the existing model check, add:

```python
        if provider == OPENAI_COMPATIBLE and not self.openai_compatible_base_url:
            found.append(
                f"{context} routes to {provider!r} but no endpoint is set "
                "(DAEMON_OPENAI_COMPATIBLE_BASE_URL)"
            )
```

- [ ] **Step 4: Wire it into the app**

In `daemon/app.py:_build_providers`, add the import beside the others:

```python
    from daemon.llm.providers.openai_compatible import OpenAICompatibleProvider
```

and the branch, before the `else`:

```python
        elif name == OPENAI_COMPATIBLE:
            providers[name] = OpenAICompatibleProvider(
                settings.openai_compatible_api_key,
                settings.openai_compatible_base_url,
            )
```

Import `OPENAI_COMPATIBLE` from `daemon.config` alongside `OLLAMA`, `ANTHROPIC`, `OPENAI`, `GEMINI`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_config.py -k compatible -v`
Expected: PASS.

Run: `python3 -m pytest tests/test_reachable.py -v`
Expected: PASS — including `test_every_nameable_provider_can_be_built[openai_compatible]`, which needs no `PENDING_PROVIDERS` entry because the module exists and `app.py` names it.

Run: `python3 -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Lint and commit**

```bash
python3 -m ruff check .
git add daemon/config.py daemon/app.py tests/test_config.py
git commit -m "config: a fourth hosted provider, whose endpoint is the user's to name"
```

---

### Task 3: Onboarding

**Files:**
- Modify: `daemon/setup.py` (vendor table, `HOSTED_CHOICES`, `_choose_hosted`, `needs_for`, `check_openai_compatible`, `Checks`, `LISTED_BY`, `_list_with_saved_key`, `Wizard._verify`)
- Modify: `daemon/cli.py` (`_doctor`'s config line)
- Test: `tests/test_setup.py`

**Interfaces:**
- Consumes: `OPENAI_COMPATIBLE`, `HOSTED_PROVIDERS`, `Settings.openai_compatible_*` from Task 2.
- Produces: `COMPATIBLE_VENDORS: tuple[Vendor, ...]`, `Vendor(name, label, base_url, model, keys_url)`, `vendor_label(base_url: str) -> str`, `check_openai_compatible(key: str, base_url: str, model: str) -> Verdict`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_setup.py`:

```python
def test_needs_asks_for_endpoint_key_and_model_when_compatible_is_chosen() -> None:
    needs = needs_for(
        {"DAEMON_PRESET": "balanced", "DAEMON_HOSTED_PROVIDER": "openai_compatible"}
    )
    keys = [need.key for need in needs]
    assert "DAEMON_OPENAI_COMPATIBLE_BASE_URL" in keys
    assert "OPENAI_COMPATIBLE_API_KEY" in keys
    assert "DAEMON_OPENAI_COMPATIBLE_MODEL" in keys
    # The endpoint is asked before the key, because the probe needs it.
    assert keys.index("DAEMON_OPENAI_COMPATIBLE_BASE_URL") < keys.index(
        "OPENAI_COMPATIBLE_API_KEY"
    )


def test_offline_never_asks_about_a_compatible_endpoint() -> None:
    needs = needs_for({"DAEMON_PRESET": "offline"})
    keys = [need.key for need in needs]
    assert not any(key.startswith("DAEMON_OPENAI_COMPATIBLE") for key in keys)
    assert "OPENAI_COMPATIBLE_API_KEY" not in keys


def test_vendor_label_names_a_known_endpoint_and_echoes_an_unknown_one() -> None:
    assert "Qwen" in vendor_label("https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    assert vendor_label("https://llm.internal/v1") == "https://llm.internal/v1"


def test_compatible_probe_lists_models_when_the_endpoint_has_them() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(
            200, json={"data": [{"id": "qwen-plus"}, {"id": "qwen-max"}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        verdict = check_openai_compatible(
            "sk-x", "https://x.test/v1", "qwen-plus", client=client
        )

    assert verdict.ok
    assert verdict.models["DAEMON_OPENAI_COMPATIBLE_MODEL"] == ("qwen-plus", "qwen-max")


def test_compatible_probe_falls_back_to_a_one_token_chat_when_models_is_missing() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("/models"):
            return httpx.Response(404, text="not found")
        body = json.loads(request.content)
        assert body["max_tokens"] == 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        verdict = check_openai_compatible(
            "sk-x", "https://x.test/v1", "custom-model", client=client
        )

    assert verdict.ok
    assert any(path.endswith("/chat/completions") for path in seen)
    # No list to offer, so the wizard falls through to NO_LIST_NOTE.
    assert not verdict.models


def test_compatible_probe_does_not_fall_back_on_a_rejected_key() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(401, text="bad key")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        verdict = check_openai_compatible(
            "sk-bad", "https://x.test/v1", "qwen-plus", client=client
        )

    assert not verdict.ok
    # 401 is a definitive answer about the key; a second call would only cost time.
    assert not any(path.endswith("/chat/completions") for path in seen)


def test_compatible_probe_never_reveals_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request for key sk-secret-abc")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        verdict = check_openai_compatible(
            "sk-secret-abc", "https://x.test/v1", "m", client=client
        )

    assert "sk-secret-abc" not in verdict.detail
```

Add whatever of `httpx`, `json`, `needs_for`, `check_openai_compatible`, `vendor_label` the file does not already import.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_setup.py -k "compatible or vendor" -v`
Expected: FAIL — `ImportError: cannot import name 'check_openai_compatible'`.

- [ ] **Step 3: Add the vendor table and the fourth choice**

In `daemon/setup.py`, beside `HOSTED_CHOICES`:

```python
@dataclass(frozen=True, slots=True)
class Vendor:
    """One known OpenAI-compatible service, and what to prefill for it."""

    name: str
    label: str
    base_url: str
    model: str
    """Empty when the vendor's catalogue rotates too fast for a default to age
    well - OpenRouter's free ids carry a `:free` suffix and come and go."""
    keys_url: str


COMPATIBLE_VENDORS: tuple[Vendor, ...] = (
    Vendor(
        "qwen",
        "Qwen (Alibaba Model Studio)",
        # International (Singapore). The new-account free quota is only granted
        # on this endpoint, and the China one does not honour it.
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
        "https://bailian.console.alibabacloud.com/",
    ),
    Vendor(
        "kimi",
        "Kimi (Moonshot)",
        "https://api.moonshot.ai/v1",
        "kimi-k2.5",
        "https://platform.moonshot.ai/console/api-keys",
    ),
    Vendor(
        "deepseek",
        "DeepSeek",
        "https://api.deepseek.com/v1",
        "deepseek-chat",
        "https://platform.deepseek.com/api_keys",
    ),
    Vendor(
        "openrouter",
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        "",
        "https://openrouter.ai/keys",
    ),
)
"""The endpoints the wizard can prefill. Not a whitelist - a fifth choice takes
any URL - and not stored: `.env` holds the URL alone, and `vendor_label` reads
the name back out of it. Two stored values that can disagree would leave no way
to tell which is the truth."""

COMPATIBLE_CHOICES: tuple[Choice, ...] = (
    *(Choice(v.name, v.label) for v in COMPATIBLE_VENDORS),
    Choice("custom", "Something else — your own server, or a service not listed."),
)


def vendor_label(base_url: str) -> str:
    """A known endpoint's human name, or the URL unchanged.

    The reverse of the table: `.env` stores only the URL, so this is how
    `daemon doctor` and the admin page say "Qwen" rather than a hostname.
    """
    stripped = base_url.rstrip("/")
    for vendor in COMPATIBLE_VENDORS:
        if vendor.base_url == stripped:
            return vendor.label
    return base_url
```

Add a fourth entry to `HOSTED_CHOICES`:

```python
    Choice(
        "openai_compatible",
        "Something else — Qwen, Kimi, DeepSeek, OpenRouter, or your own server.",
        (
            "Anything that speaks OpenAI's API works here; the next question asks "
            "which, and fills in its address for you.",
            "One name covers them all because they differ by address, not by "
            "protocol - so `daemon doctor` says openai_compatible and the address "
            "says who that is.",
        ),
    ),
```

- [ ] **Step 4: Ask the sub-question**

In `_choose_hosted`, after `_record(updates, "DAEMON_HOSTED_PROVIDER", chosen, current)` and before `return chosen`:

```python
        if chosen == OPENAI_COMPATIBLE:
            self._choose_compatible_endpoint(env, updates)
        return chosen

    def _choose_compatible_endpoint(
        self, env: Mapping[str, str], updates: dict[str, str]
    ) -> None:
        """Which compatible service, and therefore which address to prefill.

        Asked as a second question rather than as four more entries in the
        provider menu, because the vendor is not a provider: the menu would show
        `qwen` as a peer of `anthropic` while `.env` and every log line said
        `openai_compatible`, and a choice that is not the thing stored is the
        confusion DEFAULT_HOSTED_PROVIDER was emptied to end.
        """
        current_url = env.get("DAEMON_OPENAI_COMPATIBLE_BASE_URL", "")
        if current_url:
            self.prompt.say(f"Currently {vendor_label(current_url)}. {KEEP_HINT}")
        picked = self._pick("Service", COMPATIBLE_CHOICES, default=COMPATIBLE_VENDORS[0].name)
        vendor = next((v for v in COMPATIBLE_VENDORS if v.name == picked), None)
        if vendor is None:
            # "custom" - nothing to prefill, so `needs_for` asks for the address
            # with no default and the model question has no list behind it.
            return
        _record(updates, "DAEMON_OPENAI_COMPATIBLE_BASE_URL", vendor.base_url, current_url)
        if vendor.model and not env.get("DAEMON_OPENAI_COMPATIBLE_MODEL"):
            updates["DAEMON_OPENAI_COMPATIBLE_MODEL"] = vendor.model
```

Import `Mapping` if the file does not already have it, and `OPENAI_COMPATIBLE` from `daemon.config`.

- [ ] **Step 5: Ask for the three values**

In `needs_for`, after the `GEMINI in providers` block:

```python
    if OPENAI_COMPATIBLE in providers:
        current_url = env.get("DAEMON_OPENAI_COMPATIBLE_BASE_URL", "")
        needs.append(
            Need(
                key="DAEMON_OPENAI_COMPATIBLE_BASE_URL",
                label="endpoint",
                why="The address of the service, up to the version segment - "
                "https://api.deepseek.com/v1, not the /chat/completions below it.",
                default=current_url,
            )
        )
        needs.append(
            Need(
                key="OPENAI_COMPATIBLE_API_KEY",
                label="API key",
                why="Your key for that service. Your account, your bill - Daemon is "
                "not a reseller.",
                url=next(
                    (v.keys_url for v in COMPATIBLE_VENDORS if v.base_url == current_url.rstrip("/")),
                    "",
                ),
                secret=True,
            )
        )
        needs.append(
            Need(
                key="DAEMON_OPENAI_COMPATIBLE_MODEL",
                label="model id",
                why="Which model answers. Settings has no default here, so an empty "
                "value would refuse to start.",
                default=env.get("DAEMON_OPENAI_COMPATIBLE_MODEL", ""),
                listed=True,
            )
        )
```

The endpoint comes first because the key probe needs an address to probe.

- [ ] **Step 6: Add the probe**

Beside `check_openai`:

```python
def check_openai_compatible(
    key: str, base_url: str, model: str, *, client: httpx.Client | None = None
) -> Verdict:
    """Prove the key against the endpoint, and list its models when it can.

    Two steps rather than one, because `/models` is optional in the compatible
    spec while `/chat/completions` is the thing this install will actually call:

    1. `GET /models` costs no tokens, proves the key, and its ids become the menu.
    2. Anything other than 200 or a definitive 401/403 means the endpoint may
       simply not implement it, so fall back to a one-token chat call. That
       proves the exact path the daemon will use - `/models` succeeding does not
       imply `/chat/completions` will, which is how an unfunded OpenRouter
       account lists a paid model happily and then answers 402.
    """
    root = base_url.rstrip("/")
    headers = {"authorization": f"Bearer {key}"}
    owns = client is None
    http = client or httpx.Client(timeout=HTTP_TIMEOUT)
    try:
        try:
            response = http.get(f"{root}/models", headers=headers)
        except httpx.HTTPError as exc:
            return Verdict(False, f"could not reach {root}: {_redact(str(exc), key)}")

        if response.status_code in (401, 403):
            return Verdict(
                False,
                f"the endpoint rejected the key (HTTP {response.status_code}).",
                hint="Check that the key belongs to the service at that address.",
            )
        if response.status_code == 200:
            ids = _newest_first(response, "created")
            listed = {"DAEMON_OPENAI_COMPATIBLE_MODEL": ids}
            if model and ids and model not in ids:
                return Verdict(
                    True,
                    f"key works, but {model!r} is not in that endpoint's model list",
                    hint="The next question offers the ids that do exist.",
                    models=listed,
                )
            return Verdict(True, "key works", models=listed)

        # No usable list. Prove the key on the path that matters instead.
        try:
            chat = http.post(
                f"{root}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
            )
        except httpx.HTTPError as exc:
            return Verdict(False, f"could not reach {root}: {_redact(str(exc), key)}")

        if chat.status_code == 200:
            return Verdict(
                True,
                "key works (that endpoint lists no models, so the next question "
                "takes an id you type)",
            )
        said = _redact(chat.text[:BODY_LIMIT], key)
        return Verdict(False, f"{root} returned HTTP {chat.status_code}: {said}")
    finally:
        if owns:
            http.close()
```

Register it on `Checks`:

```python
    openai_compatible: Callable[[str, str, str], Verdict] = check_openai_compatible
```

Add to `LISTED_BY`:

```python
    "DAEMON_OPENAI_COMPATIBLE_MODEL": "OPENAI_COMPATIBLE_API_KEY",
```

In `_list_with_saved_key`'s probe dict, add:

```python
            "OPENAI_COMPATIBLE_API_KEY": lambda: self.checks.openai_compatible(
                key, env.get("DAEMON_OPENAI_COMPATIBLE_BASE_URL", ""), need.default
            ),
```

And in `Wizard._verify` — the method holding the `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` branches — add one more, before the final `return Verdict(True, "")`:

```python
        if need.key == "OPENAI_COMPATIBLE_API_KEY":
            # The endpoint question comes first in `needs_for`, so by the time the
            # key is typed its address is already in `updates` and merged into `env`.
            return self.checks.openai_compatible(
                value,
                env.get("DAEMON_OPENAI_COMPATIBLE_BASE_URL", ""),
                env.get("DAEMON_OPENAI_COMPATIBLE_MODEL", ""),
            )
```

- [ ] **Step 6b: Let `daemon doctor` name the service**

`_doctor` builds its `config` line as `preset=… voice=… [task->provider, …]`. A
bare `openai_compatible` there does not say *which* service, and the address is
the only thing that does. In `daemon/cli.py:_doctor`, after `table` is built:

```python
        endpoint = ""
        if any(
            route.provider == OPENAI_COMPATIBLE
            for route in settings.routing_table().values()
        ):
            # The provider name is one word for many services, so the config line
            # would otherwise not say which one is answering.
            endpoint = f" endpoint={vendor_label(settings.openai_compatible_base_url)}"
```

and append `{endpoint}` to the `config` check's detail string. Import
`vendor_label` from `daemon.setup` and `OPENAI_COMPATIBLE` from `daemon.config`
(`cli.py` already imports from both).

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_setup.py -k "compatible or vendor" -v`
Expected: PASS.

Run: `python3 -m pytest -q`
Expected: PASS.

- [ ] **Step 8: Lint and commit**

```bash
python3 -m ruff check .
git add daemon/setup.py daemon/cli.py tests/test_setup.py
git commit -m "setup: the fourth option asks which service, then fills in its address"
```

---

### Task 4: Admin UI

**Files:**
- Modify: `daemon/admin/settings_io.py`
- Test: `tests/test_admin.py`

**Interfaces:**
- Consumes: the `Settings` fields from Task 2.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing test**

`tests/test_admin.py` drives the real HTTP surface with `create_app` and a
`_settings(tmp_path, **kw)` helper. Match that — do not reach into
`settings_io` directly, because the thing worth testing is that the page can
set these.

```python
def test_patch_sets_the_compatible_endpoint_and_model(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=balanced\n", encoding="utf-8")
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.patch(
            "/admin/api/settings",
            json={
                "hosted_provider": "openai_compatible",
                "openai_compatible_base_url": "https://api.deepseek.com/v1",
                "openai_compatible_model": "deepseek-chat",
            },
        )
    assert resp.status_code == 200
    written = env.read_text(encoding="utf-8")
    assert "DAEMON_OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com/v1" in written
    assert "DAEMON_OPENAI_COMPATIBLE_MODEL=deepseek-chat" in written


def test_patch_rejects_an_endpoint_carrying_the_full_path(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("DAEMON_PRESET=balanced\n", encoding="utf-8")
    original = env.read_text(encoding="utf-8")
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.patch(
            "/admin/api/settings",
            json={
                "openai_compatible_base_url": (
                    "https://api.deepseek.com/v1/chat/completions"
                )
            },
        )
    assert resp.status_code == 400
    assert env.read_text(encoding="utf-8") == original, "a rejected patch still wrote"


def test_get_settings_masks_the_compatible_key(tmp_path: Path) -> None:
    app = create_app(
        _settings(tmp_path, openai_compatible_api_key="sk-compat-SUPERSECRET")
    )
    with TestClient(app) as client:
        resp = client.get("/admin/api/settings")
    assert "SUPERSECRET" not in resp.text


def test_settings_offer_openai_compatible_as_a_provider(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        payload = client.get("/admin/api/settings").json()
    assert "openai_compatible" in payload["hosted_providers"]
```

Copy the exact `TestClient` construction and `_settings` call style from the
neighbouring tests in that file — the ones around
`test_patch_with_a_valid_value_writes_only_that_key` — rather than the sketch
above, which may differ in how it opens the client.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_admin.py -k compatible -v`
Expected: FAIL — the patch is rejected as an unknown field, and
`hosted_providers` does not contain the name.

- [ ] **Step 3: Add the fields**

In `daemon/admin/settings_io.py`, add to `STR_FIELDS`:

```python
    "openai_compatible_model": "DAEMON_OPENAI_COMPATIBLE_MODEL",
    # The endpoint belongs beside the model for the same reason the model ids do:
    # a page that lets you choose `openai_compatible` without letting you name its
    # address could only ever answer the choice with an error it cannot fix.
    "openai_compatible_base_url": "DAEMON_OPENAI_COMPATIBLE_BASE_URL",
```

and to `SECRET_FIELDS`:

```python
    "openai_compatible_api_key": "OPENAI_COMPATIBLE_API_KEY",
```

The `hosted_provider` dropdown needs no change — it renders `list(HOSTED_PROVIDERS)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_admin.py -q`
Expected: PASS.

Run: `python3 -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
python3 -m ruff check .
git add daemon/admin/settings_io.py tests/test_admin.py
git commit -m "admin: the compatible endpoint is editable where its model already was"
```

---

### Task 5: Documentation

**Files:**
- Modify: `.env.example`
- Modify: `daemon/CLAUDE.md` (the `llm/` row says "4 providers")
- Modify: `daemon/RECIPES.md` (the add-a-provider recipe)
- Modify: `docs/ARCHITECTURE.md` if it enumerates providers — check with `grep -n "gemini" docs/ARCHITECTURE.md`

- [ ] **Step 1: Add the keys to `.env.example`**

Beside the other hosted keys, commented out, with the vendor addresses as the
comment so nobody has to leave the file to find one:

```bash
# Any service speaking OpenAI's Chat Completions API. Set DAEMON_HOSTED_PROVIDER
# to openai_compatible to use one. The address stops at the version segment -
# not the /chat/completions below it, which startup refuses.
#   Qwen      https://dashscope-intl.aliyuncs.com/compatible-mode/v1
#   Kimi      https://api.moonshot.ai/v1
#   DeepSeek  https://api.deepseek.com/v1
#   OpenRouter https://openrouter.ai/api/v1
# OPENAI_COMPATIBLE_API_KEY=
# DAEMON_OPENAI_COMPATIBLE_BASE_URL=
# DAEMON_OPENAI_COMPATIBLE_MODEL=
```

- [ ] **Step 2: Update the counts and the recipe**

In `daemon/CLAUDE.md`, change `llm/providers/` (4) to (5).

In `daemon/RECIPES.md`, extend the add-a-provider recipe:

```markdown
**A new LLM provider.** Implement `Provider` in `daemon/llm/base.py` under `daemon/llm/providers/`, name
it in `HOSTED_PROVIDERS`, build it in `daemon/app.py`, offer it in `daemon/setup.py`. Missing the last two shipped once.
**A vendor is not always a provider.** Qwen, Kimi, DeepSeek and OpenRouter all speak Chat Completions,
so they share `openai_compatible` and differ by `DAEMON_OPENAI_COMPATIBLE_BASE_URL`. Reach for a new
module only when the *protocol* is new; a new address is configuration.
```

- [ ] **Step 3: Verify documented paths still exist**

Run: `python3 scripts/check_docs.py`
Expected: PASS.

Run: `python3 -m pytest -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add .env.example daemon/CLAUDE.md daemon/RECIPES.md docs/ARCHITECTURE.md
git commit -m "docs: a vendor is not always a provider"
```

---

### Task 6: The live spike

The unit suite may not open a socket, so this is what proves the feature works.
It is run by hand, never in CI.

**Files:**
- Create: `evals/openai_compatible_spike.py`
- Modify: `evals/CLAUDE.md` (list the new spike)

- [ ] **Step 1: Read the existing spike and copy its shape**

Read `evals/m1c_text_tools_spike.py` first. Match how it loads `.env`, how it
reports, and how it refuses to run without a key. Do not invent a second
convention.

- [ ] **Step 2: Write the spike**

`evals/openai_compatible_spike.py` takes the endpoint, key and model from `.env`
(overridable with `--base-url`, `--model`) and runs three checks in order,
printing the raw request and response for each:

1. `GET {base_url}/models` — does this endpoint list anything?
2. A plain Korean turn through `OpenAICompatibleProvider.complete`, asserting a
   non-empty reply.
3. A turn offering one `ToolSpec` that the prompt forces the model to call,
   asserting `completion.tool_calls` is non-empty, then feeding a
   `Message(role="tool", ...)` result back and asserting the follow-up reply
   mentions it.

Step 3 is the one that matters: tools are central to the product and the most
likely thing a compatible endpoint gets wrong.

- [ ] **Step 3: Run it against OpenRouter**

Pick a `:free` model whose `supported_parameters` includes `tools` — the free
catalogue rotates, so list it at run time rather than trusting a hardcoded id:

```bash
python3 -m evals.openai_compatible_spike --base-url https://openrouter.ai/api/v1
```

Expected: all three checks pass. Record the model id used and the date in the
run's output.

- [ ] **Step 4: Run it against Qwen**

```bash
python3 -m evals.openai_compatible_spike \
  --base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 --model qwen-plus
```

Expected: all three pass. This is the run that exercises the `/compatible-mode/v1`
path prefix, which OpenRouter's ordinary `/api/v1` would not.

- [ ] **Step 5: End-to-end, which is the actual acceptance bar**

```bash
daemon setup      # choose balanced → "Something else" → Qwen, paste the key
daemon doctor     # provider and endpoint reported, health green
daemon run
```

Then, over Telegram: send a message and get a reply, and send one that forces a
tool call (e.g. asking it to read a file) and confirm the tool ran via
`daemon tools log`.

**Do not mark this task complete on green unit tests.** The bar agreed in the
spec is a real conversation with a real tool call.

- [ ] **Step 6: Commit and report honestly**

```bash
git add evals/openai_compatible_spike.py evals/CLAUDE.md
git commit -m "evals: a spike that asks a compatible endpoint for a tool call"
```

In the PR description, state plainly which endpoints were exercised and which
were not. Per the spec: **OpenRouter and Qwen are verified; Kimi, DeepSeek and a
custom URL are code-supported and unverified.** Do not describe them as
supported without that qualification.

---

## Done when

- `python3 -m pytest` passes, including `tests/test_reachable.py`.
- `python3 -m ruff check .` is clean.
- `python3 scripts/check_docs.py` passes.
- `daemon setup` offers the fourth provider and the five services behind it.
- A real Telegram conversation runs on Qwen, including one successful tool call.
- `daemon/llm/providers/openai.py` shows **zero** changes in `git diff main`.
