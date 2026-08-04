# 0005 — The vector index moved into M1b

**Status:** accepted · 2026-08-03

## Context

The plan had M1b shipping recall on FTS5 alone with vectors deferred to M2, on the
evidence that Hermes runs on FTS5 with no embeddings at all.

## Decision

Move the vector index into M1b. Hermes is English.

SQLite's `unicode61` tokenizer matches whole tokens, and Korean inflects: `김치찌개`
matches, `김치찌` does not, and `어제는` is a different token from `어제`. Measured on
the golden set, top-5:

| | |
|---|---|
| keyword only | **50.0%** — and 50.0% at top-8 |
| hybrid, stub embedder | 56.7% |
| hybrid, bge-m3 | **93.3%** |

The decisive column is the first, not the last: keyword-only **does not improve
from top-5 to top-8**, because 15 of 30 cases are invisible to FTS5 at any limit.
Of the 28 hybrid passes, zero are keyword-only wins. Without vectors, M1b's own
gate — quote yesterday accurately — is unreachable in Korean.

Implementation: float32 blobs searched brute-force with numpy, **not** a SQLite
extension. Measured 0.18 ms per query over 10k messages, 1.07 ms over 50k. The
decisive point is portability, not speed: this Python build ships with
`enable_load_extension` disabled, so `sqlite-vec` cannot load at all, and
self-hosters on such a build would hit the same wall.

## Consequences

M1b grew by a few days and M2 shrank. Recall now depends on an embedder, which
costs ~117 ms per query — almost all fixed overhead, so a smaller model does not
help (`paraphrase-multilingual` was slower *and* dropped the gate to 83.3%). Voice
hides it by embedding the partial transcript while the user is still speaking.

A pass rate is only meaningful with the index complete. Production defaulted to a
single 500-row backfill, oldest-first, which left a large history mostly unembedded
while health reported ready — the 50% ceiling with nothing failing.

## What would change our mind

A morphological analyser cheap enough to install would change the mix, not the
conclusion; PLAN §4.3 declined one for dependency weight.
