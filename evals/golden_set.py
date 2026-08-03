"""Golden set for Lane 1 recall — a pass rate that moves when recall changes.

Run it:

    python3 -m evals.golden_set                    # offline, deterministic embedder
    python3 -m evals.golden_set --embedder ollama  # the real vector lane
    python3 -m evals.golden_set --embedder none    # keyword lane only

docs/PLAN.md 8.3 layer 1. The M1b gate is "it quotes yesterday accurately", and
without a number that is a matter of opinion after three anecdotes. This harness
loads fixture logs into a throwaway data dir, indexes them, asks each question,
and checks whether the answer is in the top N.

## Golden set format

`evals/fixtures/questions.json`:

```
{
  "top_n": 5,                          how many recalled items count as a hit
  "now": "2026-08-01T09:00:00Z",       pinned clock, so recency decay - and
                                       therefore the pass rate - is reproducible
  "logs": "logs",                      fixture log dir, relative to this file
  "cases": [
    {
      "id":       "q01",               stable handle, used in the failure report
      "question": "어제 저녁에 뭐 먹었지?",
      "expect":   ["김치찌개"],         every phrase must appear in ONE recalled
                                       item; substring match, no normalisation
      "log":      "2026-07-28.md"      which fixture day holds the answer
    }
  ]
}
```

`log` is not used to search - it is checked. A case whose phrases appear in no
message of the named file is reported as BROKEN rather than as a recall failure,
because a typo in the golden set otherwise reads as a regression in the code.

## Where the real questions go

The fixtures are synthetic: M1a only just started running, so there is no real
history to draw on yet. Once the owner has been talking to Daemon for a few
weeks, the questions they actually ask - the ones recall fails on in practice -
belong in this same file alongside the synthetic ones, with a log day copied out
of the real `memory/log/`. Synthetic cases are a floor, not the target: they were
written by the same mind that wrote the retrieval code, and they cannot surprise
it.

## Reading the report

Each failure is attributed to a lane by re-asking with a large limit and reading
the `reason` on the item that should have won:

    rank=17 reason=keyword    both lanes exist, the vector lane missed it,
                              and bm25 buried it at 17
    not recalled at all       neither lane produced the message
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from daemon.clock import parse_iso
from daemon.llm.base import Embedder
from daemon.memory import log as memory_log
from daemon.memory.base import Recall, RecalledItem
from daemon.memory.recall import MemoryRecall
from daemon.memory.reindex import reindex
from daemon.memory.store import Store

FIXTURES = Path(__file__).parent / "fixtures"
QUESTIONS = FIXTURES / "questions.json"

DEEP_LIMIT = 100
"""Limit used only to attribute a failure: was the answer recalled at all, and by
which lane? Never what the daemon itself asks for."""


# --- the deterministic offline embedder --------------------------------------


class CharGramEmbedder:
    """Hashed character n-grams. Deterministic, offline, no model.

    Not a semantic embedder and not meant to be one. It exists so this harness
    produces the same number on every machine with no network, and so the
    vector lane is exercised in tests at all. What it does capture is exactly the
    thing FTS5 cannot: `찌개` inside `김치찌개`, `동생` inside `동생인데`. So the
    pass rate it reports is a floor - a real embedder (`--embedder ollama`,
    bge-m3) should beat it, and if it does not, something is wrong.

    blake2b rather than `hash()` because Python's string hash is salted per
    process, which would make "reproducible" false in the most confusing way.
    """

    name = "chargram"
    dimensions = 256

    def __init__(self, dimensions: int = 256, grams: tuple[int, ...] = (2, 3)) -> None:
        self.dimensions = dimensions
        self.model = f"chargram-{dimensions}"
        self._grams = grams

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        compact = "".join(character for character in text.lower() if not character.isspace())
        for n in self._grams:
            for start in range(max(len(compact) - n + 1, 0)):
                gram = compact[start : start + n]
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=4).digest()
                vector[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return [float(value) for value in vector]


# --- the set ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Case:
    id: str
    question: str
    expect: tuple[str, ...]
    log: str


@dataclass(frozen=True, slots=True)
class Spec:
    top_n: int
    now: str
    logs: Path
    cases: tuple[Case, ...]


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: Case
    passed: bool
    broken: bool
    """The golden set is wrong, not the code: no message in `case.log` matches."""
    rank: int | None
    """1-based position of the answer in a deep search, or None if never recalled."""
    reason: str
    """'keyword' / 'vector' / 'both' from the deep search, '-' when not recalled."""
    top: tuple[RecalledItem, ...]


@dataclass(frozen=True, slots=True)
class Report:
    results: tuple[CaseResult, ...]
    top_n: int
    embedder: str
    vectors: int

    @property
    def scored(self) -> tuple[CaseResult, ...]:
        return tuple(result for result in self.results if not result.broken)

    @property
    def passed(self) -> tuple[CaseResult, ...]:
        return tuple(result for result in self.scored if result.passed)

    @property
    def pass_rate(self) -> float:
        return len(self.passed) / len(self.scored) if self.scored else 0.0

    def lane_counts(self) -> dict[str, int]:
        """Which lane carried each pass. The measurement that says whether the
        vector lane earned its place in M1b."""
        counts = {"keyword": 0, "vector": 0, "both": 0}
        for result in self.passed:
            counts[result.reason] = counts.get(result.reason, 0) + 1
        return counts


def load_spec(path: Path = QUESTIONS) -> Spec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Spec(
        top_n=int(raw["top_n"]),
        now=str(raw["now"]),
        logs=path.parent / str(raw.get("logs", "logs")),
        cases=tuple(
            Case(
                id=str(case["id"]),
                question=str(case["question"]),
                expect=tuple(str(phrase) for phrase in case["expect"]),
                log=str(case["log"]),
            )
            for case in raw["cases"]
        ),
    )


# --- running ----------------------------------------------------------------


def install_fixtures(spec: Spec, data_dir: Path) -> None:
    """Copy the fixture logs in as if the daemon had written them there."""
    target = data_dir / "memory" / "log"
    target.mkdir(parents=True, exist_ok=True)
    for source in sorted(spec.logs.glob("*.md")):
        shutil.copyfile(source, target / source.name)


def broken_cases(spec: Spec) -> list[Case]:
    """Cases whose expected phrases are in no message of the log they name."""
    broken = []
    for case in spec.cases:
        records = memory_log.read(spec.logs / case.log)
        if not any(all(phrase in record.content for phrase in case.expect) for record in records):
            broken.append(case)
    return broken


async def build_recall(
    spec: Spec, data_dir: Path, embedder: Embedder | None
) -> tuple[MemoryRecall, int]:
    """Fixtures on disk -> mirrored -> embedded, returning the vector count.

    Uses the daemon's own code paths: `reindex` is what rebuilds the mirror from
    markdown, and `backfill` is what fills the vector index after such a rebuild.
    """
    install_fixtures(spec, data_dir)
    store = Store.open(data_dir / "daemon.sqlite3")
    reindex(data_dir, store)
    recall = MemoryRecall(store, embedder, now=parse_iso(spec.now))
    return recall, await recall.backfill(limit=10_000)


async def evaluate(
    recall: Recall, spec: Spec, *, broken: frozenset[str] = frozenset()
) -> tuple[CaseResult, ...]:
    results = []
    for case in spec.cases:
        top = await recall.search(case.question, limit=spec.top_n)
        hit = _find(top, case)
        if hit is not None:
            rank: int | None = hit + 1
            reason = top[hit].reason
        else:
            # Only to attribute the failure: was it recalled at all, by which lane?
            deep = await recall.search(case.question, limit=DEEP_LIMIT)
            deep_hit = _find(deep, case)
            rank = None if deep_hit is None else deep_hit + 1
            reason = "-" if deep_hit is None else deep[deep_hit].reason
        results.append(
            CaseResult(
                case=case,
                passed=hit is not None,
                broken=case.id in broken,
                rank=rank,
                reason=reason,
                top=tuple(top),
            )
        )
    return tuple(results)


def _find(items: list[RecalledItem], case: Case) -> int | None:
    for index, item in enumerate(items):
        if all(phrase in item.content for phrase in case.expect):
            return index
    return None


async def run(
    spec: Spec | None = None,
    *,
    embedder: Embedder | None = None,
    data_dir: Path | None = None,
) -> Report:
    spec = spec or load_spec()
    broken = frozenset(case.id for case in broken_cases(spec))
    with tempfile.TemporaryDirectory(prefix="daemon-golden-") as tmp:
        recall, vectors = await build_recall(spec, data_dir or Path(tmp), embedder)
        results = await evaluate(recall, spec, broken=broken)
    label = "none (keyword lane only)" if embedder is None else embedder.model
    return Report(results=results, top_n=spec.top_n, embedder=label, vectors=vectors)


# --- reporting --------------------------------------------------------------


def format_report(report: Report) -> str:
    lines = [
        f"recall golden set: {len(report.scored)} cases, top-{report.top_n}",
        f"embedder: {report.embedder}  ({report.vectors} vectors indexed)",
    ]
    if report.vectors == 0:
        lines.append("WARNING: no vectors indexed - this is a KEYWORD-ONLY pass rate")
    lines.append(
        f"pass: {len(report.passed)}/{len(report.scored)} = {report.pass_rate:.1%}"
    )
    counts = report.lane_counts()
    lines.append(
        "passes by lane: "
        + ", ".join(f"{lane}={counts.get(lane, 0)}" for lane in ("keyword", "vector", "both"))
    )

    failures = [result for result in report.scored if not result.passed]
    if failures:
        lines.append("")
        lines.append(f"failures ({len(failures)}):")
        for result in failures:
            where = (
                f"rank={result.rank} reason={result.reason} ({_missed(result.reason)})"
                if result.rank is not None
                else "not recalled at all - neither lane found it"
            )
            lines.append(f"  {result.case.id}  {result.case.question}")
            lines.append(f"        expect {list(result.case.expect)} in {result.case.log}")
            lines.append(f"        {where}")

    broken = [result for result in report.results if result.broken]
    if broken:
        lines.append("")
        lines.append(f"BROKEN cases ({len(broken)}) - the golden set is wrong, not the code:")
        for result in broken:
            lines.append(
                f"  {result.case.id}  {list(result.case.expect)} appears in no message "
                f"of {result.case.log}"
            )
    return "\n".join(lines)


def _missed(reason: str) -> str:
    if reason == "keyword":
        return "vector lane missed it"
    if reason == "vector":
        return "keyword lane missed it"
    return "both lanes found it, ranking put it too low"


# --- cli --------------------------------------------------------------------


def _build_embedder(kind: str, base_url: str, model: str) -> Embedder | None:
    if kind == "none":
        return None
    if kind == "chargram":
        return CharGramEmbedder()
    from daemon.llm.embedders.ollama import OllamaEmbedder

    return OllamaEmbedder(base_url, model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Lane 1 recall against the golden set.")
    parser.add_argument(
        "--embedder",
        choices=("chargram", "ollama", "none"),
        default="chargram",
        help="chargram (default): deterministic, offline, a floor. "
        "ollama: the real vector lane. none: keyword only.",
    )
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    args = parser.parse_args(argv)

    spec = load_spec(args.questions)
    embedder = _build_embedder(args.embedder, args.ollama_url, args.model)
    report = asyncio.run(run(spec, embedder=embedder))
    print(format_report(report))
    return 0 if report.pass_rate > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
