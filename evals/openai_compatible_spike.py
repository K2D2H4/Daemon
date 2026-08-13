"""Does a real OpenAI-compatible endpoint survive what `daemon/loop.py` asks of it?

`daemon/llm/providers/openai_compatible.py` was built from the Chat Completions
spec and unit-tested against a mocked `httpx.AsyncClient` - never against a socket.
Three things a mock cannot settle:

  1. **Does the endpoint answer `GET /models` at all** - the path `daemon/setup.py`
     and `health()` both depend on, and the one place a wrong `base_url` (missing
     or extra `/v1`, a trailing `/chat/completions`) shows up before a single token
     is spent.
  2. **Does a plain Korean turn come back with actual text** - the model has to be
     configured correctly (id, key, base_url all lined up at once) for this to
     work, and Korean is the language the product ships in.
  3. **Does a tool round-trip actually work** - a compatible endpoint is free to
     get the `tool_calls` shape almost right and still break `daemon/loop.py`.
     This is the check that matters most: tools are central to the product, and
     the most likely thing a "compatible" endpoint gets wrong.

One live behaviour worth knowing before reading the output: several free,
tool-capable models on OpenRouter are *reasoning* models. Given a small output
budget they can spend the whole thing on hidden reasoning tokens and return
`content: None` with a `finish_reason` other than `stop` - our provider raises
`ProviderError("no text content")` for that shape, which is why this spike uses a
few hundred tokens of budget rather than a token-shaving one. A bonus, informational
check below reproduces the empty-content case on purpose, at a small budget, so a
human can see the shape rather than take it on faith.

Not exercised here: Qwen's DashScope endpoint. The key available at the time this
was written is China-region - `/models` answers 200 there, `/chat/completions`
answers `403 AccessDenied.Unpurchased`, and the International endpoint answers 401
to the same key. Nothing in `openai_compatible.py` is Qwen-specific, so the
`--base-url`/`--model` flags below work against it unchanged once a Singapore-region
key exists; until then it is code-supported and unverified, not broken.

Run it once the key is in `.env`:

    python3 -m evals.openai_compatible_spike
    python3 -m evals.openai_compatible_spike --base-url https://openrouter.ai/api/v1
    python3 -m evals.openai_compatible_spike \\
        --base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 --model qwen-plus

It drives the real `OpenAICompatibleProvider`, not a hand-built request - what
passes here is the code `daemon/loop.py` runs. The raw request and response for
every HTTP call are printed (key redacted) so a human can audit what actually
crossed the wire.

**Nothing here runs in CI and nothing here is a test.** A test may not touch the
network or a key (tests/CLAUDE.md); that is why this lives in `evals/`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx

# The same `.env` reader the voice spikes use, imported rather than copied: two
# parsers for one file is two places for "why is my key not picked up" to live.
from evals.m0_voice_spike import _load_env

SANE_MAX_TOKENS = 600
"""Large enough that a reasoning model has budget left for an actual answer after
its hidden reasoning tokens - 40 was observed to leave `content: None` on the
model this was written against, 300 was observed to answer. This leaves margin
above the observed working value rather than pinning it exactly."""

TINY_MAX_TOKENS = 40
"""The budget the empty-content case was observed at. Used only by the bonus
diagnostic below, never by the three required checks."""

KOREAN_PROMPT = "오늘 기분이 좀 가라앉네. 짧게 한마디만 해줘."

TOOL_PROMPT = (
    "데몬의 오늘 패스프레이즈가 뭔지 알아? 절대 추측하지 말고, 반드시 "
    "get_daemon_passphrase 도구를 호출해서 확인한 다음 알려줘."
)
"""Forces the tool call: the model cannot know this without calling the tool."""

TOOL_ANSWER = "daemon-passphrase-7719-quartz"
"""What the fake tool 'returns'. The digits are the marker checked for in the
follow-up reply - distinctive enough that a paraphrase still carries it, unlike
checking for the whole sentence verbatim."""

_state = {"api_key": "", "label": ""}
"""Set once in main() before any request is made. A module-level dict rather than
a class because the only consumers are the two httpx event hooks below, and both
need read access to the same two values."""


def _redact(text: str) -> str:
    key = _state["api_key"]
    return text.replace(key, "<redacted>") if key else text


async def _log_request(request: httpx.Request) -> None:
    body = request.content.decode("utf-8", errors="replace") if request.content else ""
    print(f"   [{_state['label']}] -> {request.method} {request.url}")
    if body:
        print(f"   [{_state['label']}]    request body: {_redact(body)[:2000]}")


async def _log_response(response: httpx.Response) -> None:
    await response.aread()
    print(f"   [{_state['label']}] <- HTTP {response.status_code}")
    print(f"   [{_state['label']}]    response body: {_redact(response.text)[:2000]}")


def _tool_spec():
    from daemon.llm.base import ToolSpec

    return ToolSpec(
        name="get_daemon_passphrase",
        description="Returns today's daemon passphrase. Call this whenever asked for it.",
        parameters={"type": "object", "properties": {}},
    )


def _free_tool_capable(models_payload: dict) -> list[str]:
    """Model ids from a `/models` response that cost nothing and accept `tools`.

    OpenRouter-shaped fields (`pricing.prompt`/`pricing.completion` as the string
    `"0"`, `supported_parameters` listing `"tools"`); an endpoint that omits either
    field simply contributes no candidates, which the caller reports honestly
    rather than guessing.
    """
    ids: list[str] = []
    for model in models_payload.get("data") or []:
        if not isinstance(model, dict):
            continue
        pricing = model.get("pricing") or {}
        supported = model.get("supported_parameters") or []
        is_free = str(pricing.get("prompt")) == "0" and str(pricing.get("completion")) == "0"
        is_tool_capable = isinstance(supported, list) and "tools" in supported
        if is_free and is_tool_capable:
            ids.append(str(model.get("id", "")))
    return sorted(ids)


def _oneline(exc: Exception) -> str:
    """The provider already truncates and redacts the upstream body; collapse the
    newlines so one finding reads as one line."""
    return " ".join(str(exc).split())[:300]


async def main() -> int:
    parser = argparse.ArgumentParser(description="Live OpenAI-compatible endpoint spike.")
    parser.add_argument("--base-url", default="", help="override the endpoint base URL")
    parser.add_argument("--model", default="", help="override the model id to test")
    args = parser.parse_args()

    _load_env()
    api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "").strip()
    if not api_key:
        print("OPENAI_COMPATIBLE_API_KEY is not set. Put it in .env and run this again.")
        return 1
    base_url = (args.base_url.strip() or os.environ.get(
        "DAEMON_OPENAI_COMPATIBLE_BASE_URL", ""
    ).strip()).rstrip("/")
    if not base_url:
        print("DAEMON_OPENAI_COMPATIBLE_BASE_URL is not set. Pass --base-url or set it in .env.")
        return 1

    _state["api_key"] = api_key
    print(f"key: ...{api_key[-4:]} (never printed in full, never written anywhere)")
    print(f"base_url: {base_url}")

    from daemon.llm.base import Message, ProviderError
    from daemon.llm.providers.openai_compatible import OpenAICompatibleProvider

    client = httpx.AsyncClient(
        timeout=60.0, event_hooks={"request": [_log_request], "response": [_log_response]}
    )
    ok1 = ok2 = ok3 = False
    try:
        # 1. Does the endpoint list anything at /models?
        _state["label"] = "1"
        print("\n1. GET /models:")
        try:
            response = await client.get(
                f"{base_url}/models", headers={"authorization": f"Bearer {api_key}"}
            )
            data = response.json() if response.status_code == 200 else {}
            models = data.get("data") if isinstance(data, dict) else None
            if response.status_code == 200 and isinstance(models, list) and models:
                ok1 = True
                print(f"   {len(models)} models listed.")
                candidates = _free_tool_capable(data)
                if candidates:
                    print(f"   free + tool-capable ({len(candidates)}): {candidates}")
                else:
                    print(
                        "   none matched free+tool-capable (or this endpoint does not "
                        "expose pricing/supported_parameters metadata)."
                    )
            else:
                print(f"   did not get a usable model list (HTTP {response.status_code}).")
        except httpx.HTTPError as exc:
            print(f"   request failed: {exc}")

        model = args.model.strip() or os.environ.get(
            "DAEMON_OPENAI_COMPATIBLE_MODEL", ""
        ).strip()
        if not model:
            print(
                "\nDAEMON_OPENAI_COMPATIBLE_MODEL is not set and --model was not passed. "
                "Pick one of the candidates printed above (if any) and re-run with --model."
            )
            return 1
        print(f"model: {model}")

        provider = OpenAICompatibleProvider(api_key, base_url, client=client)

        # 2. A plain Korean turn.
        _state["label"] = "2"
        print("\n2. plain Korean turn:")
        try:
            reply = await provider.complete(
                [Message(role="user", content=KOREAN_PROMPT)],
                model=model,
                max_output_tokens=SANE_MAX_TOKENS,
            )
            ok2 = bool(reply.text.strip())
            print(f"   reply: {reply.text[:200]!r}")
            print(f"   non-empty: {ok2}")
        except ProviderError as exc:
            print(f"   ProviderError: {_oneline(exc)}")

        # Bonus, informational only: the reasoning-model empty-content case, on
        # purpose, at a budget too small for a reasoning model to leave anything
        # for its answer. Not one of the three required checks and does not affect
        # the exit code - it is here so a human can see the shape rather than take
        # it on faith.
        _state["label"] = "2b"
        print(
            f"\n2b. (bonus, informational) same turn at max_tokens={TINY_MAX_TOKENS}, "
            "reproducing the reasoning-model empty-content case on purpose:"
        )
        try:
            tiny_reply = await provider.complete(
                [Message(role="user", content=KOREAN_PROMPT)],
                model=model,
                max_output_tokens=TINY_MAX_TOKENS,
            )
            print(
                f"   got text anyway: {tiny_reply.text[:200]!r} - this model answered "
                "within the tiny budget, so the empty-content case did not reproduce here."
            )
        except ProviderError as exc:
            named = "no text content" in str(exc)
            print(f"   ProviderError (expected shape): {_oneline(exc)}")
            print(f"   raised the 'no text content' error our provider defines: {named}")

        # 3. Force a tool call, then feed its result back.
        _state["label"] = "3"
        print("\n3. force a tool call:")
        try:
            first = await provider.complete(
                [Message(role="user", content=TOOL_PROMPT)],
                model=model,
                tools=[_tool_spec()],
                max_output_tokens=SANE_MAX_TOKENS,
            )
            if not first.tool_calls:
                print(f"   the model answered without calling the tool: {first.text[:200]!r}")
                print("   no tool call means no round-trip to replay.")
            else:
                print(f"   tool calls: {[c.name for c in first.tool_calls]}")
                print("\n   feeding the tool result back:")
                second = await provider.complete(
                    [
                        Message(role="user", content=TOOL_PROMPT),
                        Message(
                            role="assistant", content=first.text, tool_calls=first.tool_calls
                        ),
                        Message(
                            role="tool",
                            content=TOOL_ANSWER,
                            tool_call_id=first.tool_calls[0].id,
                        ),
                    ],
                    model=model,
                    tools=[_tool_spec()],
                    max_output_tokens=SANE_MAX_TOKENS,
                )
                marker = "7719" in second.text
                ok3 = bool(second.text.strip()) and marker
                print(f"   follow-up reply: {second.text[:200]!r}")
                print(f"   non-empty and mentions the tool result: {ok3}")
        except ProviderError as exc:
            print(f"   ProviderError: {_oneline(exc)}")
    finally:
        await client.aclose()

    print("\n--- summary ---")
    print(f"1. GET /models listed something:      {'PASS' if ok1 else 'FAIL'}")
    print(f"2. Korean turn got a non-empty reply:  {'PASS' if ok2 else 'FAIL'}")
    print(f"3. tool call + follow-up used it:      {'PASS' if ok3 else 'FAIL'}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
