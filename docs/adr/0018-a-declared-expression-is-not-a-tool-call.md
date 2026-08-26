# 0018 — A declared expression is not a tool call

**Status:** accepted · 2026-08-26 · measured

## Context

The face ships three mood clips — `amused`, `sulky`, `curious`. They fire on the
text path, where the model prepends `[mood:...]` to its reply and
`daemon/face.py:split_mood` cuts it off before the wire and before the markdown log.
Measured in production on 2026-08-26: Telegram turns published four one-shots in the
same window a live 6-turn voice session published **zero**.

Voice cannot use that mechanism. Audio, transcripts and tool calls are all that
arrive, and there is no text we own — a tag in the transcript is a tag the model
reads out loud. [The face design spec](../superpowers/specs/2026-08-25-face-design.md)
§5 said so, ruled out inferring emotion from words (already measured dead: 463 owner
utterances, 0 lexicon matches), and named the one mechanism left: a flat
`set_mood(mood)` the model calls.

Which [non-negotiable 12](../CONTRACTS.md) blocks:

> **Every executed tool call leaves a `tool_calls` audit row.** That row is the
> owner's ground-truth record of what touched the machine […] a tool that ran
> without one is a defect.

A mood touched nothing. Filling that row means `verdict=allow` with nothing to
allow, `ran=1` with nothing having run, `origin=owner` proving nothing — six to nine
of them per voice conversation, on top of the `run_command` and `write_file` rows
`daemon tools log` exists to show.

## The measurement that had to come first

Changing a contract on an assumption is how you end up holding the exception and not
the feature, so `evals/voice_set_mood_spike.py` asked the live socket before anything
here was written. `gemini-3.1-flash-live-preview`, 81 flat-filtered declarations,
Korean TTS over the real audio path, 48 sessions:

| | |
|---|---|
| call rate on turns carrying a mood | 24/24 |
| …and the mood picked was right | 24/24 |
| false positives on neutral turns | 0/8 |
| **said it out loud** | **0/32** |

The last row was a veto, not a metric: §5's whole objection to the transcript was
that the model narrates the mechanism. It did not, once.

Worth recording next to it: the text tag over-fires `curious` on **11 of 15**
deliberately neutral prompts, and this over-fires on **0 of 8**. Plausibly because a
call is a decision where a prepended tag is nearly free — but that is a hypothesis
this run did not test.

## Decision

**Rule 12 is split, not weakened**, in the shape
[ADR 0015](0015-code-may-search-where-the-model-may-not.md) used on rule 10: the
protective half keeps its exact wording, and a narrow capability is carved out, named,
and pinned by a test that fails if it grows.

*Every executed tool call still leaves an audit row.* What changes is that **a
value the model declares which touches nothing outside this process is not a tool
call.** Today that is exactly one thing, `set_mood`, and the boundary is mechanical
rather than a promise:

- **It is not in the registry.** `daemon/tools/` has no entry for it, so no policy
  decision, no execution path and no audit row exist to be skipped.
- **It never reaches `ToolRunner`.** `VoiceConversation._run_tool_call` answers it
  inline and returns before `Companion.run_tools`. The exemption is that the runner
  never sees it — *not* a runner that sometimes omits a row, which would make rule 12
  conditional and unauditable.
- **Its argument is validated, not trusted.** An enum in a declaration is a request.
  The value is checked against the `Mood` type and dropped if it fails; the call is
  still answered, because the session blocks until it is and refusing would cost the
  answer rather than the expression.
- **It is only declared when a face is attached**, in `daemon/app.py`, the one module
  allowed to assemble (rule 4). No face, no switch.

## Consequences

Voice gets its expressions with **no new model call** — the mood rides a turn that was
happening anyway, so non-negotiable 2 is untouched — and `daemon tools log` keeps
meaning what it says.

**The real cost is that "every" now has a named exception, and exceptions are doors.**
The next `set_x` that does touch something will look like it belongs on this list. That
is what the guard tests are for, and they are the load-bearing part of this decision
rather than routine coverage: `set_mood` must never appear in the registry, and a
`set_mood` call must never reach `run_tools`. If a future capability needs an entry
here, it needs its own ADR and its own measurement, not a line added to a list.

One thing this does not do: the mood is published when the answer's first audio
arrives, not when the call does. A blocking tool call reaches us before any audio, and
`speaking` is the one transition allowed to cut a one-shot — so publishing on arrival
put the expression on screen for about 0ms. The text path learned this first and the
fix is the same one.

## Alternative rejected

A post-turn classification call: ask a text model, after the answer, which mood it
carried. It needs no contract change at all — non-negotiable 2 is about recall Lane 1
on the latency path, and this runs after the turn. It was rejected because the spike
removed its only advantage. Against 24/24 with nothing spoken aloud, its costs — one
model call per voice turn, the face design's "adds no model call" claim withdrawn, and
an expression landing a beat after the sentence it belongs to — buy nothing.
