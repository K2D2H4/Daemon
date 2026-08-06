"""Does our Gemini provider survive a real Gemini 3 tool round-trip? Ask the API.

`daemon/llm/providers/gemini.py` builds a `functionCall` part on the way out and
reads one on the way back, and for Gemini 3 both legs have to carry an opaque
`thoughtSignature` or the *second* call of the turn is rejected 400: the model
issues the signature with the call and requires it echoed back verbatim when the
call is replayed in history (docs: generate-content/thought-signatures). That
contract is invisible to a mock. `tests/test_providers.py` can assert we emit the
field, but only the live API can say the field we emit is the one it wants, and
only the live API enforces it at all - Gemini 2.5 does not, so a suite pinned to
2.5 stays green while the product 400s on every tool turn. That gap shipped once: a
`chat_text` turn on `gemini-3.1-pro-preview` failed with "Function call is missing
a thought_signature", and the owner saw "Something went wrong on my side".

This settles three things a fake cannot:

  1. **Does Gemini 3 attach a `thoughtSignature` to a `functionCall`,** and does our
     `_tool_calls` capture it onto `ToolCall.provider_signature`.
  2. **Does omitting it on replay 400** - is the field load-bearing, or were we
     carrying it for nothing. This reproduces the shipped failure on purpose.
  3. **Does including it succeed** - the same round-trip `daemon/loop.py` runs, end
     to end, so the fix is answered against the live contract, not a mock of it.

Run it once the key is in `.env`:

    python3 -m evals.m1c_text_tools_spike
    python3 -m evals.m1c_text_tools_spike --model gemini-3.1-pro-preview

It drives the real `GeminiProvider`, not a hand-built request, so what passes here
is the code the loop runs. The tool it declares is a fake clock that touches
nothing; the key is only ever read from the environment and never printed in full.

The contract is Gemini-3-only, so on a 2.5 id the demonstration is vacuous and this
says so rather than passing quietly - the failure it guards cannot happen there,
and a green run that proved nothing is the trap `evals/` exists to avoid.

**Nothing here runs in CI and nothing here is a test.** A test may not touch the
network or a key (tests/CLAUDE.md); that is why this lives in `evals/`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# The same `.env` reader the voice spikes use, imported rather than copied: two
# parsers for one file is two places for "why is my key not picked up" to live.
from evals.m0_voice_spike import _load_env

RECOMMENDED_MODEL = "gemini-3.1-pro-preview"
"""The id the incident was reported on, kept as the fallback on purpose: the
signature contract is Gemini-3-only, so the fallback has to be a 3.x model or the
spike proves nothing. `--model` and DAEMON_GEMINI_MODEL override it."""

PROMPT = (
    "오늘 날짜가 어떻게 되지? 너는 오늘 날짜를 알 수 없으니 추측하지 말고, "
    "반드시 get_current_date 도구를 호출해서 확인해줘."
)
"""Forces the first leg: the model cannot answer without the tool, and a spike that
never gets a tool call has no round-trip to replay. If the model answers anyway
that is reported, not passed over."""

TOOL_ANSWER = "2026-08-06"
"""What the fake tool 'returns'. Its value is irrelevant - the signature, not the
date, is what the second and third calls are testing."""


def _spec():
    from daemon.llm.base import ToolSpec

    return ToolSpec(
        name="get_current_date",
        description="Today's date on the owner's machine, as an ISO-8601 date string.",
        parameters={"type": "object", "properties": {}},
    )


def _replay(question: str, calls, *, keep_signature: bool):
    """The messages `daemon/loop.py` assembles after the tools run: the question, the
    model's tool-call turn, and one tool result per call. `keep_signature=False`
    rebuilds the calls with the signature dropped, which is exactly the state that
    400s - reproducing the bug rather than describing it."""
    from daemon.llm.base import Message, ToolCall

    replayed = (
        calls
        if keep_signature
        else tuple(
            ToolCall(id=c.id, name=c.name, arguments=c.arguments, provider_signature=None)
            for c in calls
        )
    )
    messages = [
        Message(role="user", content=question),
        Message(role="assistant", content="", tool_calls=replayed),
    ]
    messages.extend(
        Message(role="tool", content=TOOL_ANSWER, tool_call_id=c.id) for c in calls
    )
    return messages


def _show(sig: str | None) -> str:
    """Presence and length only. A signature is an opaque carrier of the model's
    reasoning state, not a value we own, so its bytes do not belong in a log."""
    return f"present, {len(sig)} chars" if sig else "<none>"


def _oneline(exc: Exception) -> str:
    """The provider already truncates and redacts the upstream body; collapse the
    newlines so one finding reads as one line."""
    return " ".join(str(exc).split())[:200]


async def main() -> int:
    parser = argparse.ArgumentParser(description="Live Gemini tool round-trip spike.")
    parser.add_argument("--model", default="", help="override the model id to test")
    args = parser.parse_args()

    _load_env()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is not set. Put it in .env and run this again.")
        return 1

    configured = args.model.strip() or os.environ.get("DAEMON_GEMINI_MODEL", "").strip()
    model = configured or RECOMMENDED_MODEL
    enforces = "gemini-3" in model
    """Whether the signature contract is expected to bite. On a gemini-3 id steps 2
    and 3 are the real guard and their failure is this spike's failure; elsewhere
    they are narrated as vacuous and the run stays green on purpose."""
    print(f"key: ...{api_key[-4:]} (never printed in full, never written anywhere)")
    print(f"model: {model}{'' if configured else '  (falling back to the incident id)'}")
    if not enforces:
        print(
            "  ! not a gemini-3 id. The thoughtSignature contract is 3-only, so steps "
            "2 and 3 cannot fail here - the run looks green and proves nothing. Pass "
            "--model gemini-3.1-pro-preview."
        )
    print()

    from daemon.llm.base import Message, ProviderError
    from daemon.llm.providers.gemini import GeminiProvider

    provider = GeminiProvider(api_key)
    exit_code = 0
    try:
        # 1. Force a tool call and read what the model attached to it.
        print("1. force a tool call, and read what the model attached to it:")
        first = await provider.complete(
            [Message(role="user", content=PROMPT)], model=model, tools=[_spec()]
        )
        if not first.tool_calls:
            print(f"   the model answered without calling the tool: {first.text[:120]!r}")
            print("   no tool call means no round-trip to replay - re-run, or pass a")
            print("   --model that calls reliably, before reading anything into this.")
            return 1
        signature = first.tool_calls[0].provider_signature
        print(f"   tool calls: {[c.name for c in first.tool_calls]}")
        print(f"   provider_signature on the first call: {_show(signature)}")
        if not signature:
            print("   the model issued NO signature - it is not enforcing the contract")
            print("   (expected on gemini-2.5); steps 2 and 3 are vacuous on this model.")

        # 2. Replay with the signature stripped: the shipped failure, on purpose.
        print("\n2. replay the turn with the signature stripped (the shipped bug):")
        try:
            await provider.complete(
                _replay(PROMPT, first.tool_calls, keep_signature=False),
                model=model,
                tools=[_spec()],
            )
            print("   NO error - the API accepted a stripped turn.")
            if enforces:
                print("   ^ but this IS a gemini-3 id, where a stripped turn MUST 400.")
                print("     The reproduction no longer reproduces - failing the run so")
                print("     that a green result never means 'proved nothing' here.")
                exit_code = 1
            else:
                print("   Expected here: this model never enforced the contract, so")
                print("   nothing below is a real guard.")
        except ProviderError as exc:
            leaked = api_key in str(exc)
            named = "thought_signature" in str(exc) or "thoughtSignature" in str(exc)
            print(f"   ProviderError (expected): {_oneline(exc)}")
            print(f"   names the missing signature: {named}  <- the shipped 400")
            print(f"   key present in the error text: {leaked}  <- must be False")
            if leaked or (enforces and not named):
                # On a gemini-3 id a 400 that does not name the signature is a
                # different failure wearing the same status code, not a clean
                # reproduction - so it does not count as green.
                exit_code = 1

        # 3. Replay with the signature kept: the fix, against the live contract.
        print("\n3. replay the same turn with the signature kept (the fix):")
        try:
            final = await provider.complete(
                _replay(PROMPT, first.tool_calls, keep_signature=True),
                model=model,
                tools=[_spec()],
            )
            answer = final.text or f"(no text; {len(final.tool_calls)} further tool call(s))"
            print(f"   accepted, and the model answered: {answer[:200]!r}")
        except ProviderError as exc:
            print(f"   STILL FAILING with the signature kept: {_oneline(exc)}")
            print("   the fix does not hold against this model - do not ship on it.")
            exit_code = 1
    finally:
        await provider.aclose()

    print(
        "\nWhat this settles: stripping the signature reproduces the shipped 400, and "
        "keeping it - the exact round-trip daemon/loop.py runs - is accepted. Re-run "
        "when the Gemini model changes; the contract is read off the socket, not the docs."
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
