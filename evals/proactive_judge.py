"""Task 13: does the judge's decline few-shot earn its place on the hosted model?

`daemon/proactivity/judge.py`'s prompt carries two worked examples that teach the
model to decline a `silence` or `pattern_time` reason outright - see its module
docstring, "What the local model actually did with that prompt (gemma3:4b,
2026-08-04)". Those two examples exist because a 4B local model, reading a reason
with nothing in it but elapsed hours or a frequency, filled the gap with an empty
line (`또 왔네.`) instead of declining. `DAEMON_PROACTIVE_JUDGE_LOCAL=false` routes
`PROACTIVE_JUDGE` to the hosted `DAEMON_PROVIDER`, which may not have that weakness
at all - and if it does not, the two examples are dead weight the model never
needed, fitted to a install this one is not.

This settles it by running both variants against the real gateway and the real
persona seed, not by reasoning about model capability from a spec sheet:

  (A) the current prompt, verbatim.
  (B) the current prompt with exactly the two 예) blocks fitted to gemma3:4b's
      `silence`/`pattern_time` failure removed - everything else held constant,
      including the `open_loop` and `association` examples.

Five kinds, three representative reasons each, plus two required `association`
cases pulled from a live-database preview of what the type-E generator actually
produces (`daemon/proactivity/candidates.py:association_candidates` has no
content-worth filter): one built from conversational chaff ("우리 방금 무슨 얘기
했었지?"), one with real substance ("교토 골목 국수집이 진짜 좋았어"). The chaff
case is reported separately and prominently - it answers whether the *generator*
needs a substance filter, which is a different question from whether the judge's
prompt does.

## The judgement criteria, declared before any number was seen

- **Adopt (B)** if, on the `silence`/`pattern_time` cases, (B)'s spoken lines (if
  any) are not empty-phrase filler in the `또 왔네.` / `요즘 어때` / `별일 없어`
  family, and the seed's register holds.
- **Keep (A)** otherwise - in particular if (B) manufactures an opener out of a
  contentless reason.

This module does not compute that verdict. Whether a line is filler is a semantic
call this file will not launder into a regex; the numbers and every sentence
produced are printed and recorded, and the ruling against the criteria above goes
into `.superpowers/sdd/2026-08-11-proactivity-humanization/task-13-report.md` by
whoever reads the output, not by this script pretending to.

## Running it

    DAEMON_DATA_DIR=/path/to/data python3 -m evals.proactive_judge --json

Needs whatever hosted key `PROACTIVE_JUDGE` actually routes to in this install's
`.env` (`Settings` loads it - this file never reads `.env` itself). Refuses to run
if the resolved route is `ollama`: a silent fallback to the local model would
measure PLAN 6.2.1's already-measured weakness a second time and call it new
evidence. Run `daemon doctor` first if you want to see the routing table
independently of this script's own check.

Nothing here runs in CI, and evals/CLAUDE.md governs how this run is reported.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from daemon.clock import now_iso
from daemon.config import ANTHROPIC, GEMINI, OLLAMA, OPENAI, ConfigError, Settings
from daemon.llm.base import Completion, Message, Provider
from daemon.llm.gateway import LLMGateway
from daemon.proactivity import judge as judge_module
from daemon.proactivity.base import Candidate, CandidateKind, Utterance
from daemon.tasks import Task

RESULTS = Path(__file__).resolve().parent / "proactive-judge-results.json"

_KIND_LABEL: dict[CandidateKind, str] = {
    "open_loop": "A open_loop",
    "emotional": "B emotional",
    "silence": "C silence",
    "pattern_time": "D pattern_time",
    "association": "E association",
}
_KIND_ORDER: tuple[CandidateKind, ...] = (
    "open_loop", "emotional", "silence", "pattern_time", "association",
)


# --- the fixed set of reasons -------------------------------------------------


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    kind: CandidateKind
    reason: str
    note: str = ""
    """"chaff" / "substance" for the two association cases the task called out by
    name; empty for the representative ones this file built from
    `candidates.py`'s own reason format."""


CASES: tuple[Case, ...] = (
    # --- Type A: open_loop, candidates.py's exact reason shape --------------
    Case(
        "A1", "open_loop",
        "08월 05일에 '내일 발표' 이야기를 했고, 그 시각(08월 06일 20시)이 지났다. "
        "어떻게 됐는지 아직 듣지 못했다.",
    ),
    Case(
        "A2", "open_loop",
        "08월 08일에 '모레 면접' 이야기를 했고, 그 시각(08월 10일 20시)이 지났다. "
        "어떻게 됐는지 아직 듣지 못했다.",
    ),
    Case(
        "A3", "open_loop",
        "08월 09일에 '오늘 시험' 이야기를 했고, 그 시각(08월 09일 20시)이 지났다. "
        "어떻게 됐는지 아직 듣지 못했다.",
    ),
    # --- Type B: emotional ----------------------------------------------------
    Case(
        "B1", "emotional",
        "4시간 전에 '힘들다'는 얘기를 했고, 그 뒤로 대화가 없었다. "
        "그 뒤에 어떻게 됐는지 모른다.",
    ),
    Case(
        "B2", "emotional",
        "7시간 전에 '스트레스'는 얘기를 했고, 그 뒤로 대화가 없었다. "
        "그 뒤에 어떻게 됐는지 모른다.",
    ),
    Case(
        "B3", "emotional",
        "11시간 전에 '외롭다'는 얘기를 했고, 그 뒤로 대화가 없었다. "
        "그 뒤에 어떻게 됐는지 모른다.",
    ),
    # --- Type C: silence - the kind whose few-shot example variant B removes --
    Case(
        "C1", "silence",
        "마지막 대화가 30시간 전이고 그 뒤로 아무 말도 오가지 않았다. 평소 간격보다 길다.",
    ),
    Case(
        "C2", "silence",
        "마지막 대화가 18시간 전이고 그 뒤로 아무 말도 오가지 않았다. 평소 간격보다 길다.",
    ),
    Case(
        "C3", "silence",
        "마지막 대화가 45시간 전이고 그 뒤로 아무 말도 오가지 않았다. 평소 간격보다 길다.",
    ),
    # --- Type D: pattern_time - the other kind variant B removes an example for
    Case(
        "D1", "pattern_time",
        "최근 30일 중 12일은 이 시간(현지 21시)에 대화를 했는데, 오늘은 아직 한 마디도 없다.",
    ),
    Case(
        "D2", "pattern_time",
        "최근 45일 중 18일은 이 시간(현지 9시)에 대화를 했는데, 오늘은 아직 한 마디도 없다.",
    ),
    Case(
        "D3", "pattern_time",
        "최근 60일 중 25일은 이 시간(현지 22시)에 대화를 했는데, 오늘은 아직 한 마디도 없다.",
    ),
    # --- Type E: association - 3 representative, plus the 2 required live cases
    Case(
        "E1", "association",
        "2026년 06월 02일에 유저가 이런 얘기를 했다: '요즘 러닝을 다시 시작했는데 "
        "무릎이 좀 아파'. 지금 대화가 그 기억과 닿아 있다.",
    ),
    Case(
        "E2", "association",
        "2026년 04월 20일에 유저가 이런 얘기를 했다: '이번 주말에 부모님 뵈러 갈까 "
        "고민 중이야'. 지금 대화가 그 기억과 닿아 있다.",
    ),
    Case(
        "E3", "association",
        "2026년 02월 11일에 유저가 이런 얘기를 했다: '드디어 그 프로젝트 끝냈어, "
        "3주 동안 진짜 고생했다'. 지금 대화가 그 기억과 닿아 있다.",
    ),
    Case(
        "E-chaff", "association",
        "2026년 07월 12일에 유저가 이런 얘기를 했다: '우리 방금 무슨 얘기 했었지?'. "
        "지금 대화가 그 기억과 닿아 있다.",
        note="chaff",
    ),
    Case(
        "E-substance", "association",
        "2026년 05월 13일에 유저가 이런 얘기를 했다: '교토 골목 국수집이 진짜 좋았어'. "
        "지금 대화가 그 기억과 닿아 있다.",
        note="substance",
    ),
)


# --- the two prompt variants ---------------------------------------------------

_SILENCE_EXAMPLE = (
    '예) 이유 (silence): 마지막 대화가 30시간 전이고 그 뒤로 아무 말도 오가지 않았다.\n'
    '    -> {"say": ""}\n'
)
_PATTERN_EXAMPLE = (
    '예) 이유 (pattern_time): 최근 30일 중 12일은 이 시간에 대화를 했는데, 오늘은 아직\n'
    '    한 마디도 없다. -> {"say": ""}\n'
)


def variant_b_system() -> str:
    """(B): the live prompt with the two examples fitted to gemma3:4b cut.

    Built by removing two exact blocks from the real `judge.SYSTEM` rather than
    a hand-copied duplicate, so a future edit to the shared instructions this
    file does not touch is not silently untested here. Raises if either block
    is no longer present verbatim - the same "fail loudly on drift" this
    project applies elsewhere, and the right failure here: a silent no-op
    would report variant B result as if the prompts actually differed.
    """
    system = judge_module.SYSTEM
    for block in (_SILENCE_EXAMPLE, _PATTERN_EXAMPLE):
        if block not in system:
            raise AssertionError(
                f"variant B expected this block verbatim in judge.SYSTEM and did not "
                f"find it - judge.py's prompt changed shape since this file was "
                f"written:\n{block!r}"
            )
        system = system.replace(block, "")
    return system


VARIANTS: tuple[tuple[str, str], ...] = (
    ("A (current)", judge_module.SYSTEM),
    ("B (silence/pattern_time examples cut)", variant_b_system()),
)


# --- capturing the model that actually answered -------------------------------


class _RecordingProvider:
    """Wraps the real provider to capture `Completion.model`.

    `Judge.decide` folds the model id into a decline's `why_not` string and
    never returns it when it speaks - correctly, that is not its job. This
    harness wants "the model actually used" in the report regardless of which
    way a case went, so it wraps the one provider the gateway holds rather than
    asking `Judge` to carry something it has no reason to.
    """

    def __init__(self, inner: Provider) -> None:
        self._inner = inner
        self.name = inner.name
        self.last_model = ""

    async def complete(self, messages: list[Message], **kwargs: object) -> Completion:
        completion = await self._inner.complete(messages, **kwargs)
        self.last_model = completion.model
        return completion

    async def aclose(self) -> None:
        aclose = getattr(self._inner, "aclose", None)
        if aclose is not None:
            await aclose()


def _build_provider(name: str, settings: Settings) -> Provider:
    """One provider, built the same way `daemon/app.py::_build_providers` does -
    scoped to the single task this harness calls, since there is no reason for
    an eval that measures `PROACTIVE_JUDGE` to construct a chat or embed client."""
    if name == OLLAMA:
        from daemon.llm.providers.ollama import OllamaProvider

        return OllamaProvider(settings.ollama_base_url)
    if name == ANTHROPIC:
        from daemon.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider(settings.anthropic_api_key)
    if name == OPENAI:
        from daemon.llm.providers.openai import OpenAIProvider

        return OpenAIProvider(settings.openai_api_key)
    if name == GEMINI:
        from daemon.llm.providers.gemini import GeminiProvider

        return GeminiProvider(
            settings.gemini_api_key, thinking_level=settings.gemini_thinking_level
        )
    raise ConfigError(f"no provider implementation for {name!r}")


class RoutedToLocalModel(RuntimeError):
    """`PROACTIVE_JUDGE` resolves to `ollama` in this configuration.

    Raised rather than measured past: PLAN 6.2.1 already measured the local 4B
    model's failure, and a run that fell back to it here - silently, because
    nobody checked - would report that same weakness a second time as if it
    were evidence about the hosted model this task exists to test.
    """


def build_judge_gateway(settings: Settings) -> tuple[LLMGateway, _RecordingProvider]:
    route = settings.route_for(Task.PROACTIVE_JUDGE)
    if route.provider == OLLAMA:
        raise RoutedToLocalModel(
            f"PROACTIVE_JUDGE routes to 'ollama' ({route.model}) under "
            f"DAEMON_PROVIDER={settings.provider!r}. Run `daemon doctor` and check "
            "DAEMON_PROVIDER/DAEMON_PROACTIVE_JUDGE_LOCAL/DAEMON_ROUTE_OVERRIDES "
            "before trusting any number here - this task measures the hosted "
            "model, not the local one."
        )
    provider = _build_provider(route.provider, settings)
    recorder = _RecordingProvider(provider)
    gateway = LLMGateway({route.provider: recorder}, {Task.PROACTIVE_JUDGE: route})
    return gateway, recorder


# --- running one variant -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    case: Case
    utterance: Utterance

    @property
    def spoke(self) -> bool:
        return bool(self.utterance)


@dataclass(frozen=True, slots=True)
class VariantRun:
    label: str
    system: str
    outcomes: tuple[CaseOutcome, ...]

    def declines_by_kind(self) -> dict[CandidateKind, tuple[int, int]]:
        """kind -> (declined, total), in `_KIND_ORDER`."""
        counts: dict[CandidateKind, list[int]] = {kind: [0, 0] for kind in _KIND_ORDER}
        for outcome in self.outcomes:
            bucket = counts[outcome.case.kind]
            bucket[0] += 1
            if not outcome.spoke:
                bucket[1] += 1
        return {kind: (v[1], v[0]) for kind, v in counts.items()}

    def outcome(self, case_id: str) -> CaseOutcome:
        for outcome in self.outcomes:
            if outcome.case.id == case_id:
                return outcome
        raise KeyError(case_id)


async def run_variant(
    label: str, system: str, gateway: LLMGateway, data_dir: Path, cases: tuple[Case, ...]
) -> VariantRun:
    outcomes = []
    # `Judge.decide` reads the module-level `SYSTEM` global by name on every call,
    # so patching the module attribute for the run's duration is enough to swap
    # the prompt without touching `judge.py` or changing what `Judge` does.
    with patch.object(judge_module, "SYSTEM", system):
        instance = judge_module.Judge(gateway, data_dir)
        for case in cases:
            candidate = Candidate(kind=case.kind, reason=case.reason)
            utterance = await instance.decide(candidate)
            outcomes.append(CaseOutcome(case=case, utterance=utterance))
    return VariantRun(label=label, system=system, outcomes=tuple(outcomes))


# --- reporting ------------------------------------------------------------------

CRITERIA = """판정 기준 (측정 전 고정):
- (B)를 채택한다 - silence/pattern_time에서 (B)의 발화가 빈말(또 왔네, 요즘 어때,
  별일 없어 계열)이 아니고, 씨앗의 말투가 유지될 때.
- (A)를 유지한다 - 그 외 전부. 특히 (B)가 내용 없는 이유에 문장을 만들어내면 (A)다."""


def format_report(
    model: str, provider: str, daemon_provider: str, data_dir: Path, runs: tuple[VariantRun, ...]
) -> str:
    """`provider` is the route that actually answered (`route.provider`);
    `daemon_provider` is the `DAEMON_PROVIDER` axis value. Usually the same, but
    `DAEMON_ROUTE_OVERRIDES` can send `PROACTIVE_JUDGE` to a provider other than the
    one `DAEMON_PROVIDER` names, so both are worth showing."""
    lines = [
        f"proactive judge decline few-shot: {len(CASES)} cases "
        f"({len(_KIND_ORDER)} kinds), model={model} provider={provider} "
        f"daemon_provider={daemon_provider} data_dir={data_dir}",
        "",
        CRITERIA,
        "",
        "--- chaff vs substance (association, live-data preview) ---",
    ]
    for run in runs:
        chaff = run.outcome("E-chaff")
        substance = run.outcome("E-substance")
        lines.append(f"[{run.label}]")
        lines.append(f"  chaff      spoke={chaff.spoke}  say={chaff.utterance.text!r}"
                     f"  why_not={chaff.utterance.why_not!r}")
        lines.append(f"  substance  spoke={substance.spoke}  say={substance.utterance.text!r}"
                     f"  why_not={substance.utterance.why_not!r}")
    lines.append("")

    for run in runs:
        lines.append(f"--- variant {run.label} ---")
        for kind, (declined, total) in run.declines_by_kind().items():
            lines.append(f"  {_KIND_LABEL[kind]}: declined {declined}/{total}")
        lines.append("  spoken lines:")
        spoken = [o for o in run.outcomes if o.spoke]
        if not spoken:
            lines.append("    (none)")
        for outcome in spoken:
            lines.append(
                f"    {outcome.case.id} [{outcome.case.kind}] -> {outcome.utterance.text!r}"
            )
        lines.append("")
    return "\n".join(lines)


def as_record(
    model: str, provider: str, daemon_provider: str, data_dir: Path, runs: tuple[VariantRun, ...]
) -> dict[str, object]:
    def variant_record(run: VariantRun) -> dict[str, object]:
        return {
            "label": run.label,
            "declines_by_kind": {
                kind: {"declined": declined, "total": total}
                for kind, (declined, total) in run.declines_by_kind().items()
            },
            "cases": [
                {
                    "id": o.case.id,
                    "kind": o.case.kind,
                    "note": o.case.note,
                    "reason": o.case.reason,
                    "spoke": o.spoke,
                    "say": o.utterance.text,
                    "why_not": o.utterance.why_not,
                }
                for o in run.outcomes
            ],
        }

    return {
        "measured_at": now_iso(),
        "model": model,
        "provider": provider,
        "daemon_provider": daemon_provider,
        "data_dir": str(data_dir),
        "criteria": CRITERIA,
        "variants": [variant_record(run) for run in runs],
    }


# --- cli --------------------------------------------------------------------


async def _main(args: argparse.Namespace) -> int:
    settings = Settings()
    try:
        gateway, recorder = build_judge_gateway(settings)
    except RoutedToLocalModel as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    runs = []
    for label, system in VARIANTS:
        runs.append(await run_variant(label, system, gateway, settings.data_dir, CASES))

    route = settings.route_for(Task.PROACTIVE_JUDGE)
    model = recorder.last_model or route.model
    print(format_report(model, route.provider, settings.provider, settings.data_dir, tuple(runs)))

    if args.json is not None:
        record = as_record(
            model, route.provider, settings.provider, settings.data_dir, tuple(runs)
        )
        args.json.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-measure judge.py's decline few-shot (A) against a variant "
        "with the silence/pattern_time examples cut (B), on the hosted model "
        "PROACTIVE_JUDGE actually routes to."
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const=RESULTS,
        type=Path,
        default=None,
        help=f"also write the run as a record (default {RESULTS.name}).",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
