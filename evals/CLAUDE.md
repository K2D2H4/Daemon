# evals/ — the numbers, not the impressions

## Owns

`docs/PLAN.md` §8.3 splits evaluation three ways and only one of them can be
automated. This is that one, plus the spike that needed a real key.

| | |
|---|---|
| `golden_set.py` | recall quality as a pass rate, repeatable |
| `fixtures/` | four days of Korean fixture conversation, 79 messages, 50 questions |
| `m0_voice_spike.py` | the six things about Gemini Live only a live key could settle |
| `m1c_voice_tools_spike.py` | whether answering a voice tool call costs the answer — it does not |
| `evals/agent-results.json` | the last run as data — score *with* its conditions |

## golden_set.py

```bash
python3 -m evals.golden_set                    # offline stand-in embedder
python3 -m evals.golden_set --embedder ollama  # the real vector lane
python3 -m evals.golden_set --embedder none    # keyword lane only
python3 -m evals.golden_set --embedder ollama --json   # ...and record the run
```

Measured 2026-08-05 on 50 questions, top-5: **keyword only 56.0% · stub embedder
60.0% · bge-m3 94.0%.** The 30-question set this grew out of read 50.0 · 56.7 ·
93.3 on 2026-08-03; nearly doubling the set moved every column by less than four
points, which is the first evidence that the shape was not an artefact of 30
questions.

The number that mattered was the middle one, not the last. Keyword-only does not
improve from top-5 to top-8 — re-measured at 50 questions both are **56.0%, to the
case**, so 22 of 50 are invisible to FTS5 at any limit, because `unicode61` matches
whole tokens and Korean inflects. That is what moved the vector index from M2 into
M1b, and the lane split still says it: of 47 hybrid passes, **0 were carried by the
keyword lane alone** (18 vector, 29 both).

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

## m1c_voice_tools_spike.py

```bash
python3 -m evals.m1c_voice_tools_spike
```

Needs `GEMINI_API_KEY`. Sends text and reads the reply, so no microphone; the tool
it declares is a fake clock that touches nothing. Four sessions — one blocking call,
then `NON_BLOCKING` once per `scheduling` value.

Measured 2026-08-05 on `gemini-3.1-flash-live-preview`, one session each:

| | audio before the answer | after | interrupts |
|---|---|---|---|
| blocking (what ships) | 0.0s | **13.69s** | 0 |
| `NON_BLOCKING` + `INTERRUPT` | 10.16s | 0.0s | 0 |
| `NON_BLOCKING` + `WHEN_IDLE` | 8.89s | 0.0s | 0 |
| `NON_BLOCKING` + `SILENT` | 13.84s | 0.0s | 0 |

**A `toolResponse` does not interrupt generation.** That was the question worth
asking, because `clientContent` does — 2.2s of audio against 46.7s — and "different
message type, therefore safe" is exactly the inference this directory replaces. The
blocking reply ran 13.69s past our answer and spoke the value we returned. The
`0.0s before` is the second half of it: a blocking `toolCall` arrives *before any
audio*, so there is no generation for the response to land in the middle of. The
`clientContent` failure needed a mid-answer arrival to exist at all.

**`NON_BLOCKING` was accepted and then ignored,** on a model whose docs say
asynchronous function calling is unsupported. All three scheduling values: the model
talked for 9–14s while it waited, we answered, and nothing followed — no audio, no
`interrupted`, no second turn inside 60s. `INTERRUPT` is documented as making the
model break off and report; it did not. This run cannot separate "inert here" from
"the answer landed after the turn boundary, so scheduling had nothing to schedule" —
the call arrived at the end of the model's own turn. Both readings say the same
thing, and **a field the server accepts and ignores is worse than one it rejects**,
because a rejection fails loudly and this fails while looking configured.

Two smaller corrections. Native audio and function calling do compose — the answer
shaped a spoken Korean reply. And **Live issues its own call ids** (`fc_<19
digits>`), unlike the REST half of the same API, so `synthesise_call_id` is a
fallback that never fires here.

`daemon/voice/gemini_live.py` sends neither field, which is now measured rather than
cautious, and warns anyone who sets one.

## Common changes

**Adding a golden case.** A question, the file whose messages answer it, and the
phrases that prove it. Put your own failed questions here — the fixtures are a
floor, not the target.

q01–q30 all ask what a single stated fact was. q31–q50 are the shapes that could
not ask: a fact a later day replaced, a question sharing no token with the message
that answers it, an answer the daemon said rather than the user. **Gotcha when
writing one:** `_find` checks whether the phrase is in a recalled item, not which
day the item came from — so a phrase that also appears on another date gives you a
case that can pass off the wrong message. Pick a phrase that exists once.

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
