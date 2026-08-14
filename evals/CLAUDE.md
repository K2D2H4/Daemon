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
| `voice_write_nudge_spike.py` | why voice fakes a write it never performs — the wall is nested tool-argument **schema**, not crowding or the contract wording |
| `m1c_text_tools_spike.py` | whether our provider survives a real Gemini 3 tool round-trip — the `thoughtSignature` contract |
| `openai_compatible_spike.py` | whether a real OpenAI-compatible endpoint answers `/models`, a Korean turn, and a tool round-trip |
| `openai_compatible_loop_spike.py` | whether the *assembled* app survives a real turn on that endpoint — loop, recall, tool policy, audit table |
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

## m1c_text_tools_spike.py

```bash
python3 -m evals.m1c_text_tools_spike
python3 -m evals.m1c_text_tools_spike --model gemini-3.1-pro-preview
```

Needs `GEMINI_API_KEY`. Drives the real `GeminiProvider` over `generateContent` —
the text tool path `daemon/loop.py` runs, not a hand-built request — so what passes
here is the code the loop runs.

The gap it exists for is the sibling of the voice one, on the REST half of the same
API. Gemini 3 attaches an opaque `thoughtSignature` to a `functionCall` and rejects
the turn 400 on replay if it is not echoed back; Gemini 2.5 does not. So a mock
suite pinned to 2.5 stayed green while a `gemini-3.1-pro-preview` `chat_text` turn
400'd on *every* tool call — "Function call is missing a thought_signature",
surfaced to the owner as "Something went wrong on my side". The spike reproduces
that by stripping the signature (expect the 400), then keeps it (expect the
answer), so the field is *shown* to be load-bearing rather than asserted to be.

The contract is Gemini-3-only. On a 2.5 id the spike says so and treats its own
green as vacuous — the `evals/` rule that a run proving nothing is worse than a red
one. It is why this is a spike and not a `tests/` case: only a live key can settle
whether the field we emit is the field the API wants.

## openai_compatible_spike.py

```bash
python3 -m evals.openai_compatible_spike
python3 -m evals.openai_compatible_spike --base-url https://openrouter.ai/api/v1
```

Needs `OPENAI_COMPATIBLE_API_KEY`, `DAEMON_OPENAI_COMPATIBLE_BASE_URL` and a model
id. Drives the real `OpenAICompatibleProvider`, not a hand-built request, over
three checks in order: `GET /models`, a plain Korean turn, and a forced tool call
whose result is fed back on a second turn. Also lists the endpoint's models that
are both free and tool-capable, since the free catalogue rotates and a hardcoded
id goes stale.

Measured 2026-08-11 on OpenRouter, model `openai/gpt-oss-20b:free`: all three
checks passed. That model is a reasoning one, and at `max_tokens: 40` it spent its
entire budget on reasoning tokens (37 of 40) and returned `content: None` with
`finish_reason: "length"` — our provider correctly raised `ProviderError("no text
content")` rather than treating that as an empty answer. At `max_tokens: 600` the
same prompt answered normally with room to spare, which is the budget the spike
uses for its three real checks; a bonus, informational check reproduces the tiny
budget on purpose so the shape is visible rather than assumed. This is a real
model behaviour, not a provider defect — a small output budget on a reasoning
model is genuinely insufficient for it to reason and answer, and raising rather
than returning empty text is the correct call for `daemon/loop.py`, which cannot
distinguish "the model refused" from "the model was cut off empty" any other way.

One more thing worth knowing before reading a raw response body: **observed
2026-08-11**, an OpenRouter non-streaming response arrived with blank-line padding
in front of the JSON, which shows up in the raw dump this spike prints. Nothing
was consumed by it — `httpx`'s and the standard library's JSON parsing both skip
leading whitespace — so it is cosmetic here. The keep-alive padding OpenRouter
*documents* is for streaming responses, so treat the non-streaming case as one
run's observation rather than as vendor-documented behaviour: it has been seen
once, not explained.

**Not exercised**: Qwen's DashScope endpoint (`--base-url
https://dashscope-intl.aliyuncs.com/compatible-mode/v1 --model qwen-plus`).
Nothing in `daemon/llm/providers/openai_compatible.py` is Qwen-specific, so this is expected to work
once a Singapore-region key exists — the China-region key available when this was
last run gets `403 AccessDenied.Unpurchased` from `/chat/completions` and `401`
from the International endpoint. Code-supported, unverified; do not call it
working until a run against it is recorded here. Kimi, DeepSeek and a custom
self-hosted URL are unverified for the same reason: no key, no run, no claim.

## openai_compatible_loop_spike.py

```bash
python3 -m evals.openai_compatible_loop_spike
```

Needs the same three `.env` values as the spike above. Reuses
`tests/test_acceptance.py`'s assembly — `Store` + `FileMemoryWriter` +
`MemoryRecall` + `LLMGateway` + the real tool stack + `Companion` +
`ConversationLoop` — with exactly one substitution: the fake `Provider` becomes
the real `OpenAICompatibleProvider`. `DAEMON_DATA_DIR` is a temporary directory,
torn down at exit, so the owner's real `data/` is never touched. The embedder
stays the offline fake: recall's *quality* is `golden_set.py`'s job, and what is
asked here is whether recall's *wiring* survives a real provider.

Measured 2026-08-11 on OpenRouter, model `openai/gpt-oss-20b:free`, all four
PASS:

| | |
|---|---|
| 1 | a Korean turn through `ConversationLoop.handle` gets a non-empty reply |
| 2 | the exchange lands in the conversation markdown *and* the SQLite mirror |
| 3 | a tool is decided, executed and written to the `tool_calls` audit table — the real registry and the real `ToolPolicy` origin gate, driven by this provider's own `tool_calls` shape |
| 4 | recall carries turn 1's fact into turn 4's prompt, as a `recalled-memory:` block |

`PACING_SECONDS = 5.0` between turns, because the free tier answers rapid
back-to-back calls with `429` — found by hitting it mid-run.

**One real defect came out of this run, in `daemon/loop.py`, and it is not
fixed.** The first attempt phrased turn 1 as "...잊지 말고 기억해줘" ("don't
forget, remember it"). With the production-default full tool access, the model
read that as an instruction to *use a tool* to persist the fact, spent all six
`MAX_TOOL_ROUNDS` trying, and the loop's forced final-answer call — the safety
valve that exists to get *some* reply out of a turn that hit the limit — then hit
the same reasoning-emptied-content case, so `ProviderError` propagated uncaught.
In production that reaches the user as the generic `FAILURE_NOTICE` for an
entirely ordinary message. The safety valve can fail the same way the thing it
guards fails. Filed as follow-up work, not fixed on this branch.

### The acceptance bar this replaced

State it plainly, because it was substituted rather than met. `docs/PLAN.md`'s
"Done when" for this work was: **a real Telegram conversation on Qwen, including
one successful tool call.** What was actually run is this spike, on OpenRouter.

Deliberate, for two reasons that do not go away by being restated: a second
poller on the owner's bot token would `409`-conflict with their running daemon,
and no working Qwen key exists (the available one is China-region — `403
AccessDenied.Unpurchased` there, `401` on the International endpoint).

So what is proven is the real loop, the real recall path, the real tool registry,
the real origin gate and the real audit table, against a real endpoint. What
remains **unproven**: `create_app`, the Telegram channel, `daemon run` as a
resident process, and any Qwen endpoint at all. Nobody may read the four PASSes
above as "a real Telegram conversation ran on Qwen".

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
