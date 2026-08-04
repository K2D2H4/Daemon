# evals/ — the numbers, not the impressions

## Owns

`docs/PLAN.md` §8.3 splits evaluation three ways and only one of them can be
automated. This is that one, plus the spike that needed a real key.

| | |
|---|---|
| `golden_set.py` | recall quality as a pass rate, repeatable |
| `fixtures/` | four days of Korean fixture conversation, 79 messages, 30 questions |
| `m0_voice_spike.py` | the six things about Gemini Live only a live key could settle |
| `evals/agent-results.json` | the last run as data — score *with* its conditions |

## golden_set.py

```bash
python3 -m evals.golden_set                    # offline stand-in embedder
python3 -m evals.golden_set --embedder ollama  # the real vector lane
python3 -m evals.golden_set --embedder none    # keyword lane only
python3 -m evals.golden_set --embedder ollama --json   # ...and record the run
```

Measured, top-5: **keyword only 50.0% · stub embedder 56.7% · bge-m3 93.3%.**

The number that mattered was the middle one, not the last. Keyword-only does not
improve from top-5 to top-8 — 15 of 30 cases are invisible to FTS5 at any limit,
because `unicode61` matches whole tokens and Korean inflects. That is what moved
the vector index from M2 into M1b.

Two things to know before quoting a pass rate:

- **It depends on the index being complete.** The 93.3% was measured with
  `backfill(limit=10_000)`. Production defaulted to 500, once, oldest-first — so
  a large history sat mostly unembedded while `/health` said recall was ready.
  Check the vector count, not just the score.
- **A case whose phrases appear in no message of its stated file is reported
  BROKEN, not failed.** A typo in the set must never read as a regression.

The fixtures are a floor, not the target. The owner's own failed questions belong
in the same file once real logs exist.

## m0_voice_spike.py

Needs `GEMINI_API_KEY`. Sends text and reads the reply, so no microphone.

It corrected three things we had inferred from SDK source and docs: an invalid key
closes with **1007**, not 1008; **1008 is not permanent** (`The operation was
aborted` is an idle abort); and `receive()` never ended at the turn boundary,
which is what those aborts actually were. It also settled the seam recall needed —
`clientContent` with `turnComplete: false` produces no audio and no transcript,
while `true` produces a full answer.

That is what this file is for. When a doc and a socket disagree, the socket wins,
and the way to find out is to ask it.

## Common changes

**Adding a golden case.** A question, the file whose messages answer it, and the
phrases that prove it. Put your own failed questions here — the fixtures are a
floor, not the target.

**Changing the recall algorithm.** Run all three embedder modes above and quote
all three numbers. **Why:** the keyword-only column is the one that carries an
argument; a single hybrid number cannot tell you whether vectors earned their
place.

**Adding a spike.** Live keys are read from the environment, never written to a
file, and a spike lives here rather than in `tests/` precisely because tests may
not touch the network.

## Depends on

[daemon/](../daemon/CLAUDE.md)'s memory and llm packages, for the real recall path
— the whole point is that these exercise the product rather than a mock. Nothing
depends on `evals/`, and CI does not run it, because it needs Ollama or a key.
Why the vector lane is measured at all: [ADR
0005](../docs/adr/0005-vectors-belong-in-m1b.md).
