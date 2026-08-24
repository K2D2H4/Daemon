"""Stage 3: the one model call.

The model is a fake, so what is under test is everything around it - what reaches
the prompt, what one call means, and what a hostile reply costs. The reply strings
below are the shapes a real local model produced in a spike against gemma3:4b:
bare JSON, fenced JSON with a preamble, and prose refusing to answer.

What a fake cannot test is whether the *prompt* makes the model decline. It cannot,
by construction: the fake says what the test tells it to. So the tests here prove
the decline path is live and reachable from every wrong reply shape, and whether
the prompt elicits one is a live-model question that belongs in `evals/`.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeProvider

from daemon.config import Route
from daemon.llm.gateway import LLMGateway
from daemon.proactivity.base import Candidate
from daemon.proactivity.judge import (
    MAX_CHARS,
    MAX_OUTPUT_TOKENS,
    SYSTEM,
    Judge,
    _read_reply,
)
from daemon.tasks import Task

SEED = "너는 반말을 쓰고, 말수가 적다. 걱정을 길게 늘어놓지 않는다."

OPEN_LOOP = Candidate(
    kind="open_loop",
    reason=(
        "08월 03일에 '내일 발표' 이야기를 했고, 그 시각(08월 04일 20시)이 지났다. "
        "어떻게 됐는지 아직 듣지 못했다."
    ),
)

VAGUE = Candidate(
    kind="pattern_time",
    reason="최근 21일 중 8일은 이 시간(현지 22시)에 대화를 했는데, 오늘은 아직 한 마디도 없다.",
)
"""A candidate that passes the gate and gives the model nothing concrete: the case
declining exists for. Type D knows only that an hour usually has a conversation."""


def test_the_two_conditions_name_the_same_three_kinds() -> None:
    """SYSTEM's two conditions are AND'ed (`둘 다`): a candidate that satisfies 1
    but not 2 is declined forever. Condition 1 admits 사건/감정/기억 (event, feeling,
    memory - the last added for type E); condition 2 has to ask about the same
    three or a reason that only ever names a memory - never an event or a feeling -
    can pass 1 and always fail 2, which is exactly what shipped in review round 1:
    every type E candidate declined regardless of content, making that generator's
    output permanently inert.

    Also checks the fix did not widen 2 the other way: the exclusion in 1's second
    sentence (time/interval/frequency alone is not content) is 1's job, and 2 must
    not independently start admitting it.
    """
    _, numbered = SYSTEM.split("\n\n1. ", 1)
    condition_1, rest = numbered.split("\n2. ", 1)
    condition_2, _ = rest.split("\n\n", 1)

    for noun in ("사건", "감정", "기억"):
        assert noun in condition_1, f"condition 1 no longer names {noun}"
        assert noun in condition_2, f"condition 2 does not ask about {noun}"

    for excluded in ("시간", "간격", "빈도"):
        assert excluded not in condition_2


def judge_for(
    data_dir: Path,
    reply: str = '{"say": "어제 발표 어떻게 됐어?"}',
    *,
    fail: bool = False,
    seed: str | None = SEED,
) -> tuple[Judge, FakeProvider]:
    if seed is not None:
        (data_dir / "persona" / "seed.md").write_text(seed, encoding="utf-8")
    provider = FakeProvider(reply, fail=fail)
    # Routed for PROACTIVE_JUDGE only, so a call made under any other task would
    # raise ConfigError out of the gateway rather than quietly pass.
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PROACTIVE_JUDGE: Route(provider.name, "gemma3:4b")}
    )
    return Judge(gateway, data_dir), provider


def system_text(provider: FakeProvider) -> str:
    return "\n".join(m.content for m in provider.calls[0] if m.role == "system")


def user_text(provider: FakeProvider) -> str:
    return "\n".join(m.content for m in provider.calls[0] if m.role == "user")


# --- the line it decided to say ---------------------------------------------


async def test_a_good_reply_becomes_the_utterance(data_dir: Path) -> None:
    judge, _ = judge_for(data_dir)

    utterance = await judge.decide(OPEN_LOOP)

    assert utterance  # truthy means it spoke
    assert utterance.text == "어제 발표 어떻게 됐어?"
    assert utterance.why_not == ""


async def test_exactly_one_model_call_per_decision(data_dir: Path) -> None:
    """Non-negotiable 7. A retry loop, or a second "are you sure", violates it."""
    judge, provider = judge_for(data_dir)

    await judge.decide(OPEN_LOOP)

    assert len(provider.calls) == 1
    assert provider.models == ["gemma3:4b"]


async def test_a_declined_candidate_still_costs_only_one_call(data_dir: Path) -> None:
    """The tempting place to retry is the decline, and it is the one that must not:
    asking again is PLAN 6.2's open question with extra steps."""
    judge, provider = judge_for(data_dir, '{"say": ""}')

    await judge.decide(VAGUE)

    assert len(provider.calls) == 1


# --- what reaches the prompt -------------------------------------------------


async def test_the_persona_seed_reaches_the_prompt(data_dir: Path) -> None:
    """PLAN 5: an unsolicited line is the voice at its most exposed. A generic
    assistant saying it is the one failure the product cannot absorb."""
    judge, provider = judge_for(data_dir)

    await judge.decide(OPEN_LOOP)

    assert SEED in system_text(provider)


async def test_no_seed_means_no_call_at_all(data_dir: Path) -> None:
    """Declined before the model, not after: nobody asked for this line, so there is
    nothing to degrade gracefully into."""
    judge, provider = judge_for(data_dir, seed=None)

    utterance = await judge.decide(OPEN_LOOP)

    assert not utterance
    assert "seed" in utterance.why_not
    assert provider.calls == []


async def test_an_empty_seed_file_is_the_same_as_none(data_dir: Path) -> None:
    judge, provider = judge_for(data_dir, seed="   \n")

    assert not await judge.decide(OPEN_LOOP)
    assert provider.calls == []


async def test_learned_rules_reach_the_prompt(data_dir: Path) -> None:
    """The text loop and voice both carry M4's learned rules. Proactivity not
    carrying them meant the same person spoke differently depending on which
    path reached them. Decided 2026-08-11; judge.py had left it open on purpose."""
    (data_dir / "persona" / "learned.md").write_text(
        "- 아침에는 말을 짧게 한다.", encoding="utf-8"
    )
    judge, provider = judge_for(data_dir)

    await judge.decide(OPEN_LOOP)

    assert "아침에는 말을 짧게" in system_text(provider)


async def test_a_missing_seed_still_refuses_to_speak(data_dir: Path) -> None:
    """Unchanged and load-bearing: PLAN 5 says a generic-assistant voice is the
    one thing this product must not have, and nobody asked for this line."""
    judge, _ = judge_for(data_dir, seed=None)

    utterance = await judge.decide(OPEN_LOOP)

    assert not utterance
    assert "seed" in utterance.why_not


async def test_learned_rules_alone_are_not_a_persona(data_dir: Path) -> None:
    """The trap in this task. `load_persona` returns a non-empty string when
    *either* file has content (loader.py:118), so checking its output instead of
    the seed would let an install with no seed.md and a populated learned.md
    speak first in nobody's voice. The seed is the anchor; accumulated rules are
    not a substitute for it."""
    (data_dir / "persona" / "learned.md").write_text(
        "- 아침에는 말을 짧게 한다.", encoding="utf-8"
    )
    judge, provider = judge_for(data_dir, seed=None)

    utterance = await judge.decide(OPEN_LOOP)

    assert not utterance
    assert "seed" in utterance.why_not
    assert provider.calls == []


async def test_the_reason_reaches_the_prompt(data_dir: Path) -> None:
    """It is the whole input: the model is told why this surfaced and asked what to
    say about it. Without it there is nothing to answer but the open question."""
    judge, provider = judge_for(data_dir)

    await judge.decide(OPEN_LOOP)

    assert "발표" in user_text(provider)
    assert "open_loop" in user_text(provider)


async def test_an_enormous_reason_is_bounded(data_dir: Path) -> None:
    """`candidates.py` builds reasons from lexicons and dates, never user text. That
    is an assumption about another module, so it is not the only defence."""
    judge, provider = judge_for(data_dir)

    await judge.decide(Candidate(kind="emotional", reason="힘들다 " * 2000))

    assert len(user_text(provider)) < 600


# --- the reply treated as hostile --------------------------------------------


async def test_an_empty_say_declines(data_dir: Path) -> None:
    """Declining is a first-class answer and this is what it looks like arriving."""
    judge, _ = judge_for(data_dir, '{"say": ""}')

    utterance = await judge.decide(VAGUE)

    assert not utterance
    assert utterance.text == ""
    assert utterance.why_not == "nothing worth saying"


async def test_whitespace_only_say_declines(data_dir: Path) -> None:
    judge, _ = judge_for(data_dir, '{"say": "  \\n "}')

    assert not await judge.decide(VAGUE)


async def test_a_null_say_declines(data_dir: Path) -> None:
    judge, _ = judge_for(data_dir, '{"say": null}')

    utterance = await judge.decide(VAGUE)

    assert not utterance
    assert "say" in utterance.why_not


async def test_prose_instead_of_json_declines_rather_than_being_spoken(
    data_dir: Path,
) -> None:
    """The reply a model gives when it has decided not to speak. Delivered as a
    line it becomes an apology out of the speaker - PLAN 6.4's accident, and the
    reason this file's contract is an object rather than a bare sentence."""
    judge, _ = judge_for(data_dir, "지금은 특별히 할 말이 없어요. 다음에 말씀드릴게요!")

    utterance = await judge.decide(VAGUE)

    assert not utterance
    assert "할 말이 없어요" not in utterance.text


async def test_an_empty_reply_declines(data_dir: Path) -> None:
    judge, _ = judge_for(data_dir, "")

    assert not await judge.decide(OPEN_LOOP)


async def test_an_over_long_reply_is_declined_not_truncated(data_dir: Path) -> None:
    """Half a sentence out of a speaker is worse than the silence it replaced, and
    silence here is free: the candidate is still in the table on the next tick."""
    monologue = "발표 어떻게 됐는지 궁금했어. " * 20
    assert len(monologue) > MAX_CHARS
    judge, _ = judge_for(data_dir, json.dumps({"say": monologue}, ensure_ascii=False))

    utterance = await judge.decide(OPEN_LOOP)

    assert not utterance
    assert str(MAX_CHARS) in utterance.why_not


async def test_a_line_at_the_limit_is_still_spoken(data_dir: Path) -> None:
    """The other side of the cap, so it cannot be tightened into silence unnoticed."""
    line = "가" * MAX_CHARS
    judge, _ = judge_for(data_dir, json.dumps({"say": line}, ensure_ascii=False))

    assert (await judge.decide(OPEN_LOOP)).text == line


async def test_a_fenced_and_prefixed_reply_is_read(data_dir: Path) -> None:
    """What gemma3:4b actually does: a cheerful preamble, then a ```json fence,
    then markdown emphasis inside the sentence it wants spoken."""
    reply = (
        "물론이죠! 아래와 같이 답변드립니다.\n"
        '```json\n{"say": "어제 **발표** 어떻게 됐어?"}\n```'
    )
    judge, _ = judge_for(data_dir, reply)

    assert (await judge.decide(OPEN_LOOP)).text == "어제 발표 어떻게 됐어?"


async def test_a_wrapping_quote_pair_is_stripped(data_dir: Path) -> None:
    """A quoted line reads as a quotation on Telegram and gets read out with the
    quotes by `say`. Only the wrapping pair goes; quotes inside stay."""
    judge, _ = judge_for(data_dir, json.dumps({"say": '"어제 발표 어떻게 됐어?"'}))

    assert (await judge.decide(OPEN_LOOP)).text == "어제 발표 어떻게 됐어?"


async def test_quotes_inside_the_line_survive(data_dir: Path) -> None:
    judge, _ = judge_for(
        data_dir, json.dumps({"say": "어제 '힘들다'고 했잖아, 좀 나아졌어?"}, ensure_ascii=False)
    )

    assert (await judge.decide(OPEN_LOOP)).text == "어제 '힘들다'고 했잖아, 좀 나아졌어?"


async def test_a_multiline_say_becomes_one_line(data_dir: Path) -> None:
    """Two lines on Telegram look like two messages, and to `say` the newline is a
    pause in the middle of one sentence."""
    judge, _ = judge_for(
        data_dir, json.dumps({"say": "어제 발표 어떻게 됐어?\n궁금했는데."}, ensure_ascii=False)
    )

    assert (await judge.decide(OPEN_LOOP)).text == "어제 발표 어떻게 됐어? 궁금했는데."


# --- the model is not there --------------------------------------------------


async def test_a_provider_error_declines_instead_of_raising(data_dir: Path) -> None:
    """The tick must record nothing, not crash: "could not reach the model" and
    "nothing worth saying" have the same consequence."""
    judge, provider = judge_for(data_dir, fail=True)

    utterance = await judge.decide(OPEN_LOOP)

    assert not utterance
    assert "unavailable" in utterance.why_not
    # It tried exactly once. The gateway owns fallback; the judge does not retry.
    assert len(provider.calls) == 1


async def test_an_unrouted_task_is_not_swallowed(data_dir: Path) -> None:
    """A gateway with no route for PROACTIVE_JUDGE is a configuration bug, and
    `ConfigError` is not a `ProviderError`. Silently declining forever would look
    exactly like a quiet week."""
    from daemon.config import ConfigError

    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    provider = FakeProvider()
    judge = Judge(LLMGateway({provider.name: provider}, {}), data_dir)

    with pytest.raises(ConfigError):
        await judge.decide(OPEN_LOOP)


# --- a reply the token limit cut in half (v0.1.54) ----------------------------
# The daemon went days without ever speaking first, and the log said only
# `did not return a JSON object` - which reads as the model misbehaving. It was
# not: `gemini-3.6-flash` spends thinking tokens out of the same output budget,
# so a 300-token cap ran out mid-sentence and left `{"say": "어제 그 미팅은 잘`.
# Every provider already reports why it stopped; the judge just never read it.


def test_a_truncated_reply_says_it_was_truncated() -> None:
    """The diagnostic that was missing. Naming the token limit is the difference
    between "the model is being odd" and "our cap is too small" - and the second
    is a one-line fix nobody made for days because the log never said it."""
    utterance = _read_reply(
        '{"say": "어제 그 미팅은 잘',
        model="gemini-3.6-flash",
        stop_reason="MAX_TOKENS",
    )

    assert not utterance
    assert "cut off" in utterance.why_not
    assert "MAX_OUTPUT_TOKENS" in utterance.why_not, "say which knob to turn"


@pytest.mark.parametrize("reason", ["MAX_TOKENS", "max_tokens", "length", "incomplete"])
def test_every_provider_spelling_of_truncation_is_recognised(reason: str) -> None:
    """gemini says MAX_TOKENS, anthropic max_tokens, openai length or incomplete.
    A spelling this misses degrades to the old unhelpful message, so all four are
    pinned rather than assumed."""
    assert "cut off" in _read_reply("{'say", model="m", stop_reason=reason).why_not


def test_prose_without_truncation_still_reads_as_prose() -> None:
    """The other direction: a model that answered in words rather than JSON has
    not been cut off, and calling that truncation would send the next reader to
    the wrong knob."""
    utterance = _read_reply("지금은 할 말이 없어요", model="m", stop_reason="STOP")

    assert not utterance
    assert "did not return a JSON object" in utterance.why_not


def test_the_token_budget_leaves_room_for_a_thinking_model() -> None:
    """MAX_OUTPUT_TOKENS was 300, sized for the JSON plus 120 characters of
    Korean. That arithmetic was right for a model that only answers, and wrong
    for one that thinks first out of the same allowance."""
    assert MAX_OUTPUT_TOKENS >= 1000


async def test_the_persona_reaching_the_judge_is_dated_when_rules_are_wired(
    db: Any, data_dir: Path
) -> None:
    """Mirrors `test_companion.py`'s equivalent case: the judge builds its own
    persona directly rather than through `Companion` (see the docstring on
    `Judge.__init__` for why it carries learned rules at all), so it needs the
    same wiring or it would re-create the very split that comment describes."""
    from datetime import UTC, datetime

    from daemon.memory.store import Store
    from daemon.persona.rules import LearnedRules, Proposal

    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    rules = LearnedRules(data_dir, Store(db))
    await rules.add(
        [Proposal(body="변명을 싫어한다", evidence=(1, 2, 3))],
        now=datetime(2026, 8, 9, 15, 0, tzinfo=UTC),
    )
    provider = FakeProvider('{"say": "어제 발표 어떻게 됐어?"}')
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PROACTIVE_JUDGE: Route(provider.name, "gemma3:4b")}
    )
    judge = Judge(gateway, data_dir, rules=rules)

    await judge.decide(OPEN_LOOP)

    assert "2026-08-09 (관찰 3건) 변명을 싫어한다" in system_text(provider)


async def test_a_judge_with_no_rules_wired_still_has_a_persona(data_dir: Path) -> None:
    """The production default (`Judge(gateway, data_dir)`, no `rules`) and most
    tests here carry no store. Undated is the same degrade this had before dates
    existed at all - not silence."""
    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    (data_dir / "persona" / "learned.md").write_text(
        "- 변명을 싫어한다\n", encoding="utf-8"
    )
    judge, provider = judge_for(data_dir)

    await judge.decide(OPEN_LOOP)

    assert "- 변명을 싫어한다" in system_text(provider)


async def test_an_unreadable_mirror_costs_the_dates_not_the_judges_voice(
    data_dir: Path,
) -> None:
    """The case that matters most on this surface: unlike `Companion.persona`, a
    persona that fails to build here is not merely a worse prompt - `decide`
    returns `Utterance(why_not=...)` without calling the model at all when
    `_persona()` comes back empty (see `no_seed_means_no_call_at_all` above). If
    a raise from `annotations()` escaped `_persona` uncaught, it would propagate
    out of `decide` - there is no `try` around that call - and silence
    proactivity entirely, not just un-date one rule."""

    class Broken:
        def annotations(self) -> dict[str, tuple[str, int]]:
            raise sqlite3.OperationalError("no such table: persona_rules")

    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    (data_dir / "persona" / "learned.md").write_text(
        "- 변명을 싫어한다\n", encoding="utf-8"
    )
    provider = FakeProvider('{"say": "어제 발표 어떻게 됐어?"}')
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PROACTIVE_JUDGE: Route(provider.name, "gemma3:4b")}
    )
    judge = Judge(gateway, data_dir, rules=Broken())

    utterance = await judge.decide(OPEN_LOOP)

    assert utterance
    assert "변명을 싫어한다" in system_text(provider)
