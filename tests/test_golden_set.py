"""The golden-set harness. Offline: the deterministic embedder, no network.

A harness that only ever reports a number is worse than no harness, because the
number gets quoted. So most of what is tested here is whether the pass rate
*moves*: it has to collapse when recall is broken, and it has to be sensitive to
ranking, or it cannot be M1b's gate.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from daemon.memory.base import RecalledItem
from evals.golden_set import (
    Case,
    CharGramEmbedder,
    Spec,
    broken_cases,
    evaluate,
    format_report,
    load_spec,
    main,
    run,
)


@pytest.fixture(scope="module")
def spec() -> Spec:
    return load_spec()


# --- the set itself ----------------------------------------------------------


def test_the_shipped_set_is_big_enough_to_mean_something(spec: Spec) -> None:
    """50 is the size docs/PLAN.md 8.3 asks of layer 1, and the reason is arithmetic:
    at 30 cases one question is 3.3 percentage points, so a single case flipping
    looks like a trend."""
    assert len(spec.cases) >= 50
    assert len({case.id for case in spec.cases}) == len(spec.cases)


def test_every_expected_answer_really_is_in_the_log_it_names(spec: Spec) -> None:
    """A typo in the golden set otherwise reads as a regression in the code."""
    assert broken_cases(spec) == []


def test_the_fixtures_cover_several_days(spec: Spec) -> None:
    assert len(list(spec.logs.glob("*.md"))) >= 3


def test_a_wrong_expectation_is_reported_as_broken_not_as_a_failure(spec: Spec) -> None:
    bogus = Case(id="qx", question="아무거나", expect=("없는 문구",), log="2026-07-28.md")
    assert broken_cases(Spec(spec.top_n, spec.now, spec.logs, (bogus,))) == [bogus]


# --- the deterministic embedder ----------------------------------------------


async def test_chargram_embedder_is_reproducible_and_normalised() -> None:
    first = await CharGramEmbedder().embed(["어제 저녁에 김치찌개 먹었어"])
    second = await CharGramEmbedder().embed(["어제 저녁에 김치찌개 먹었어"])

    assert first == second
    assert math.isclose(math.sqrt(sum(v * v for v in first[0])), 1.0, rel_tol=1e-6)


async def test_chargram_embedder_sees_a_substring_fts5_cannot() -> None:
    """The property that makes it a usable floor: `찌개` is inside `김치찌개`."""
    embedder = CharGramEmbedder()
    related, unrelated, probe = await embedder.embed(
        ["어제 저녁에 김치찌개 먹었어", "맥북 배터리가 부풀었어", "찌개"]
    )

    assert _dot(related, probe) > _dot(unrelated, probe)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# --- the run -----------------------------------------------------------------


async def test_the_harness_produces_a_pass_rate(spec: Spec) -> None:
    report = await run(spec, embedder=CharGramEmbedder())

    assert report.vectors > 0, "the fixtures were not indexed"
    assert 0.0 < report.pass_rate <= 1.0
    assert len(report.scored) == len(spec.cases)


async def test_both_lanes_carry_passes(spec: Spec) -> None:
    """If either column were always zero, one lane would be dead weight and the
    two-lane design would be unsupported by measurement."""
    report = await run(spec, embedder=CharGramEmbedder())
    counts = report.lane_counts()

    assert counts["keyword"] + counts["both"] > 0
    assert counts["vector"] + counts["both"] > 0


async def test_the_vector_lane_earns_its_place(spec: Spec) -> None:
    """docs/PLAN.md 4.3 pulled vectors forward from M2 into M1b. This is the
    claim behind that decision, measured: hybrid beats keyword-only in Korean."""
    keyword_only = await run(spec, embedder=None)
    hybrid = await run(spec, embedder=CharGramEmbedder())

    assert keyword_only.vectors == 0
    assert hybrid.pass_rate > keyword_only.pass_rate


async def test_a_broken_recall_collapses_the_pass_rate(spec: Spec) -> None:
    """Does the harness have teeth? Recall that returns nothing must score zero."""
    results = await evaluate(_SilentRecall(), spec)

    assert all(not result.passed for result in results)
    assert all(result.rank is None for result in results)


async def test_the_pass_rate_is_sensitive_to_ranking(spec: Spec) -> None:
    """Only the top N counts, so a harness that ignored order would report the
    same number for top-1 and top-5."""
    strict = Spec(1, spec.now, spec.logs, spec.cases)
    loose = Spec(8, spec.now, spec.logs, spec.cases)

    tight = await run(strict, embedder=CharGramEmbedder())
    wide = await run(loose, embedder=CharGramEmbedder())

    assert tight.pass_rate < wide.pass_rate


async def test_failures_are_attributed_to_a_lane(spec: Spec) -> None:
    report = await run(spec, embedder=CharGramEmbedder())
    failures = [result for result in report.scored if not result.passed]

    assert failures, "the fixtures are meant to include cases recall still misses"
    for failure in failures:
        assert failure.reason in {"keyword", "vector", "both", "-"}
        assert (failure.rank is None) == (failure.reason == "-")


async def test_the_report_names_the_lane_split_and_the_failures(spec: Spec) -> None:
    text = format_report(await run(spec, embedder=CharGramEmbedder()))

    assert "passes by lane" in text
    assert "failures" in text
    for line in ("keyword=", "vector=", "both="):
        assert line in text


async def test_the_report_shouts_when_there_is_no_vector_lane(spec: Spec) -> None:
    """A keyword-only pass rate quoted as the real one would be a lie by omission."""
    text = format_report(await run(spec, embedder=None))
    assert "KEYWORD-ONLY" in text


def test_the_module_runs_as_a_command(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """`python3 -m evals.golden_set` - the offline default must need no network."""
    assert main(["--embedder", "chargram"]) == 0
    assert "recall golden set" in capsys.readouterr().out


class _SilentRecall:
    """Recall that finds nothing. The floor the harness has to be able to detect."""

    async def search(self, query: str, *, limit: int = 8) -> list[RecalledItem]:
        return []

    async def index(self, message_id: int, text: str) -> None:
        return None
