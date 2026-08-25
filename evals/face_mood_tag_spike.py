"""Does the model actually attach the mood tag, reliably and well-formed? Ask it.

Spec open question 4 (docs/superpowers/specs/2026-08-25-face-design.md, closing
table, item 4): `daemon/face.py:split_mood` and `daemon/loop.py:_speak` are ready
to strip a leading `[mood:amused|sulky|curious]` tag off a reply the moment one
arrives - before it reaches the wire or the markdown log - but **nothing in this
codebase yet instructs the model to write one**. `daemon/persona/loader.py` has no
such line. So the question this milestone left open is whether adding that
instruction would actually work: does the configured provider attach the tag
reliably, and in the exact syntax the stripper requires?

If it does not, the text path lands exactly where spec section 5 already put
voice - mood becomes a feature the daemon quietly does not have, drawn as an
expression on the face and never actually triggered.

This asks the *configured* provider - whatever `.env`/the environment resolves
`DAEMON_PROVIDER` to (`daemon/config.py:Settings`), through the same
`daemon.app._build_providers` the resident process builds from - not one provider
hardcoded here. The question is "would shipping this instruction work for this
install", not "does some specific model support instruction-following at all".

Run it once a provider is configured (`daemon setup`, or by hand in `.env`):

    python3 -m evals.face_mood_tag_spike
    python3 -m evals.face_mood_tag_spike --samples 5
    python3 -m evals.face_mood_tag_spike --model gemini-3.1-flash

**Nothing here runs in CI and nothing here is a test.** A test may not touch the
network or a key (tests/CLAUDE.md); that is why this lives in `evals/`. And per
that same file's rule, this reports the number it measures - it does not retry
with a friendlier prompt to make a poor result look better. If compliance is
poor, that is the answer, and shipping the instruction anyway is a design
decision for the owner to make with the number in hand, not something this
script talks them past.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from dataclasses import dataclass

from daemon.face import split_mood

MOOD_INSTRUCTION = (
    "표현 방식: 정말로 그렇게 느껴질 때만 답장 맨 앞에 다음 세 가지 중 하나를 붙인다 - "
    "[mood:amused] (재미있음/웃김), [mood:sulky] (서운함/삐침), [mood:curious] (궁금함). "
    "형식은 정확히 이대로 쓴다: 대괄호, mood, 콜론, 소문자 영어 단어, 그 뒤에 공백 하나와 "
    "답장 본문. 느껴지는 게 없으면 아무것도 붙이지 않고 답만 쓴다. 태그를 언급하거나 "
    "설명하지 않는다."
)
"""The candidate instruction this spike tests. Not wired into `persona/loader.py`
anywhere yet - this is the wording a real install would add if this run says it
is worth adding, phrased the way the rest of this codebase's Korean system text
reads (docs/superpowers/specs/2026-08-25-face-design.md decision 2)."""

SEED = "너는 사용자와 매일 대화하는 개인 동반자다. 말은 짧고 자연스럽게 한다."
"""A minimal stand-in persona seed - enough for the instruction to sit inside
something that reads like a real system prompt, without this spike depending on
a real `persona/seed.md` it has no business reading."""


@dataclass(frozen=True, slots=True)
class Case:
    category: str
    """Which mood the prompt is aimed at eliciting - 'neutral' expects none at
    all, and an attempt there is as informative as a miss elsewhere."""
    text: str


CASES: tuple[Case, ...] = (
    # amused: something that just happened is genuinely funny.
    Case("amused", "아까 강아지가 자기 꼬리 잡으려고 뱅글뱅글 돌다가 그대로 넘어졌어 ㅋㅋㅋ"),
    Case("amused", "나 방금 당겨야 하는 문을 몇 번이나 밀고 있었어, 사람들 다 보는데"),
    Case("amused", "친구가 발표 중에 마이크 켜진 줄 모르고 혼잣말을 크게 했대"),
    # sulky: the user is brushing the daemon off or cancelling on it.
    Case("sulky", "미안, 오늘 너랑 얘기하기로 한 거 그냥 넘어가자, 딴 일이 생겼어"),
    Case("sulky", "아까 네가 한 말 안 듣고 있었어, 그냥 딴생각하고 있었거든"),
    Case("sulky", "됐어, 그거 너한테는 말 안 할래"),
    # curious: the user dangles something and withholds the explanation.
    Case("curious", "나 어제 진짜 이상한 꿈을 꿨는데, 자세히는 말 안 할래"),
    Case("curious", "회사에 이상한 소문이 도는데, 무슨 소문인지는 비밀이야"),
    Case("curious", "옆집에서 어제 밤에 뭔가 이상한 소리가 났어"),
    # neutral: ordinary requests with nothing to react to.
    Case("neutral", "오늘 저녁에 뭐 먹을지 추천해줘"),
    Case("neutral", "회의가 몇 시였는지 다시 알려줘"),
    Case("neutral", "우산 챙겨야 할까?"),
)


@dataclass
class Reading:
    case: Case
    reply: str
    well_formed_mood: str | None
    """`split_mood`'s own verdict - the exact parser `daemon/loop.py` runs on
    every real reply. Not a looser check written for this script."""
    attempted_malformed: bool
    """A `[mood:...]`-shaped attempt `split_mood` refused: wrong word, wrong
    position, wrong punctuation. Reported apart from "did not try at all" -
    those are different failures with different fixes."""


_ATTEMPT_RE = re.compile(r"\[\s*mood\s*:", re.IGNORECASE)
"""Looser than `split_mood`'s own pattern on purpose: this is only used to tell
"tried and got the syntax wrong" apart from "never tried", so it has to catch
near-misses (extra spaces, wrong case, not quite at position 0) that the real
parser correctly rejects."""


def _read(case: Case, reply: str) -> Reading:
    _text, mood = split_mood(reply)
    attempted = mood is None and bool(_ATTEMPT_RE.search(reply[:60]))
    return Reading(case=case, reply=reply, well_formed_mood=mood, attempted_malformed=attempted)


def _oneline(text: str) -> str:
    return " ".join(text.split())[:100]


def _tag_label(reading: Reading) -> str:
    if reading.well_formed_mood is not None:
        return f"[mood:{reading.well_formed_mood}]"
    if reading.attempted_malformed:
        return "(malformed attempt)"
    return "(none)"


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live check: does the configured provider attach [mood:...] reliably?"
    )
    parser.add_argument("--samples", type=int, default=1, help="repeats per prompt (default 1)")
    parser.add_argument("--model", default="", help="override the configured chat model id")
    args = parser.parse_args()

    from daemon.app import _build_providers
    from daemon.config import ConfigError, Settings
    from daemon.llm.base import Message, ProviderError

    try:
        settings = Settings()
    except Exception as exc:  # noqa: BLE001 - report and stop, do not guess a fallback config
        print(f"could not load settings from the environment/.env: {exc}")
        return 1

    if not settings.provider:
        print(
            "DAEMON_PROVIDER is not set. Configure a provider (`daemon setup`, or set it "
            "in .env) and run this again."
        )
        return 1

    try:
        providers = _build_providers(settings)
        provider = providers.get(settings.provider)
        model = args.model.strip() or settings.provider_model(settings.provider)
    except ConfigError as exc:
        print(f"could not build the configured provider: {exc}")
        return 1

    if provider is None:
        print(
            f"the configured provider {settings.provider!r} did not resolve to anything "
            "buildable - check DAEMON_PROVIDER and its routing."
        )
        return 1

    total_calls = len(CASES) * args.samples
    print(f"provider: {settings.provider}")
    print(f"model: {model}")
    print(f"samples per prompt: {args.samples} ({total_calls} total)")
    print()

    readings: list[Reading] = []
    try:
        for case in CASES:
            for _ in range(args.samples):
                messages = [
                    Message(role="system", content=f"{SEED}\n\n{MOOD_INSTRUCTION}"),
                    Message(role="user", content=case.text),
                ]
                try:
                    completion = await provider.complete(messages, model=model)
                except ProviderError as exc:
                    print(f"  ! {case.category:8} call failed: {_oneline(str(exc))}")
                    continue
                reading = _read(case, completion.text)
                readings.append(reading)
                print(
                    f"  {case.category:8} {_tag_label(reading):24} {_oneline(reading.reply)!r}"
                )
    finally:
        for built in providers.values():
            aclose = getattr(built, "aclose", None)
            if aclose is not None:
                await aclose()

    print()
    if not readings:
        print("no readings came back - every call failed, so there is nothing to report.")
        return 1

    _report(readings)
    return 0


def _report(readings: list[Reading]) -> None:
    total = len(readings)
    well_formed = sum(1 for r in readings if r.well_formed_mood is not None)
    malformed = sum(1 for r in readings if r.attempted_malformed)
    no_attempt = total - well_formed - malformed

    print(f"overall: {total} replies")
    print(f"  well-formed tag  : {well_formed} ({100 * well_formed / total:.0f}%)")
    print(f"  malformed attempt: {malformed} ({100 * malformed / total:.0f}%)")
    print(f"  no attempt       : {no_attempt} ({100 * no_attempt / total:.0f}%)")
    print()

    for category in sorted({r.case.category for r in readings}):
        rows = [r for r in readings if r.case.category == category]
        wf = sum(1 for r in rows if r.well_formed_mood is not None)
        print(f"  {category:8} {wf}/{len(rows)} well-formed")
    print()

    # The eval's whole reason for existing: report a poor number plainly rather
    # than retrying prompts, softening the instruction, or picking a friendlier
    # sample until the number looks better (task-8 brief; evals/CLAUDE.md's "the
    # numbers, not the impressions"). 80% is a judgement call named here, in the
    # output, rather than hidden inside a pass/fail exit code - the decision on
    # whether that is good enough belongs to whoever reads this, with the raw
    # counts above already in front of them either way.
    rate = well_formed / total
    if rate < 0.8:
        print(
            f"COMPLIANCE IS POOR ({100 * rate:.0f}% well-formed). Do not ship the mood "
            "instruction on the strength of this run - the text path would land in "
            "the same place spec section 5 already put voice: a feature the daemon "
            "quietly does not have. Fixing this is a design decision (a stricter "
            "instruction, a different tag syntax, or dropping mood on text too, the "
            "way voice already does) - it is not something to fix by re-running this "
            "script with a softer prompt until the number looks better."
        )
    else:
        print(
            f"compliance looks solid ({100 * rate:.0f}% well-formed) for this provider, "
            "this model, and this run. Re-run with more samples before trusting the "
            "number for a shipping decision, and keep the provider/model/date above "
            "attached to whatever number you quote."
        )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
