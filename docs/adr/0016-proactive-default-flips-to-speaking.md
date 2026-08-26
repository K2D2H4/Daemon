# 0016 — The proactive default flips from silence to speaking

**Status:** accepted · 2026-08-26 · overturns ADR 0008 in part

## Context

`daemon/proactivity/judge.py`'s `SYSTEM` prompt has never once produced a line.
**572 judge calls, 0 utterances, all time.** [ADR 0008](0008-three-stages-one-model-call.md)
and [PLAN.md](../PLAN.md) §6.2 explain why the prompt was built to make that likely:
measured against local `gemma3:4b`, a prompt that permitted declining and called it
correct declined 0 of 15 times, and answered `특별한 일은 없다.` ("nothing in
particular is happening") with `별일 없어.` while drifting out of the seed's 반말.
A second prompt that stated silence as the *default* and speaking as the exception,
gated on two AND'd conditions (a named event/feeling/memory in the reason, and
something to ask about it), declined 3 of 15 - and it declined exactly the
contentless reasons. **That measurement was sound.** A 4B model asked an open
question about its own permission to interrupt does grant it almost every time, and
narrowing the question to "is there content" measurably fixed that.

The owner has since asked, three times, to loosen this judge:

> 그냥 아무말이나 좋으니 좀 말을 걸었으면 좋겠어, 뭐하세요 이런것도 괜찮음 지금
> 조건이 너무 엄격한거같아

The first two readings of this got the fix wrong (loosening budgets and gate
thresholds - deterministic stage 2, not stage 3), because both assumed the
complaint was "it declines too much." The owner corrected that directly:

> 내가 싫어하던건 그런게 아니고, 매번 말걸때마다 무슨 재밌는일 없어요? 처럼
> 부자연스러운 질문을 하는거였어, 일상적인 질문들은 전혀 문제없음

## Why the old conclusion does not survive this

§6.2's test asked one question: *does the judge decline when the reason carries
no content?* It answered that question correctly, and the two-condition gate that
came out of it did what it was built to do. What that test never asked, because
nobody had reason to yet, is a second, separate question: **is "no content" the
right definition of what makes an opener bad?**

The owner's correction answers that second question, and the answer is no. `무슨
재밌는 일 없어요?` and `뭐하세요?` are both content-free in exactly the sense §6.2
means - neither names an event, a feeling or a memory. Only one of them bothered
him. The difference is not content; it is who the question puts to work. `무슨
재밌는 일 없어요?` hands the owner an assignment - produce something interesting,
right now, or the exchange has nothing. `뭐하세요?` does not; it is a question a
person asks another person with nothing in particular to report, and it is
answerable with nothing at all. The old prompt's list of banned "empty phrases"
(`"오랜만이야"`, `"요즘 어때"`, `"별일 없어"`, `"시간이 많이 흘렀네"`,
`"오늘도 변함없네"`) is a list of exactly the ordinary, answerable kind - banned
for being contentless, when contentless was never the defect.

So §6.2's finding stands - a model given an open question about permission will
grant it - and its remedy does not, because the remedy's classification
(*content-free = a defect to decline*) is the thing this task's evidence shows was
wrong, not the model's willingness to talk.

## Decision

Three changes to `SYSTEM`, all in `daemon/proactivity/judge.py`:

1. **The default flips.** `say`에 문장을 넣는 것이 대부분의 정답이다, not the
   reverse. The two-condition AND'd gate is gone. Declining is not deleted - a
   reason with nothing in it and nothing worth asking about can still return
   `{"say": ""}`, and the prompt still says so - but it is no longer the thing
   the model has to earn its way out of by default.
2. **The banned-phrase list is replaced.** Out: the five ordinary check-ins named
   above. In: the demanding shape the owner actually named - `"무슨 재밌는 일
   없어요?"`, `"오늘은 어떤 얘기 해주실 거예요?"`, `"재밌는 얘기 좀 해주세요"` -
   plus one generalisable line (*상대에게 화제를 내놓으라고 요구하지 않는다*) so
   the rule reaches shapes not on the list. This is not new phrasing:
   `daemon/voice/conversation.py`'s `CALLED_BY_NAME` already tells the model "do
   not ask what they want, and do not offer to help" for the wake-word turn, and
   the owner's own `data/persona/seed.md` states the same rule for the persona
   as a whole. `daemon/proactivity/judge.py` was the one place in this repo
   that rule had not reached.
3. **The `silence` worked example now speaks.** It used to be this file's only
   demonstration of the behaviour being reversed - a reason with nothing but
   elapsed hours, answered `{"say": ""}`. It now answers `{"say": "뭐하세요?"}`.
   `pattern_time`'s example is left declining on purpose, so the few-shot set
   still shows silence as a real, reachable answer rather than a dead branch.

Also (secondary, and load-bearing that it stays secondary): `Judge` gained an
optional `store` dependency so `daemon/persona/tics.py`'s `verbal_tics` - already
measured on the text and voice loops at 18/30 → 6/30 repeats, p=0.0017 - can name
the daemon's own recently overused proactive phrases and ask it to say the same
thing a different way. This changes *how* the model phrases a line it has already
decided to say; it does not add a second reason to decline, and the owner was
explicit that repetition was never his complaint - the fix on offer here is for
the register of the question, not for how often it repeats itself.

## What this does not touch

Stage 1 (`daemon/proactivity/candidates.py`) and stage 2 (`daemon/proactivity/gate.py`)
are unchanged: timing, budgets, cooldown and quiet hours still decide *whether now*
deterministically, exactly as
[ADR 0008](0008-three-stages-one-model-call.md) requires. This ADR only changes
what stage 3 does with a reason that already passed the gate. `docs/CONTRACTS.md`
non-negotiable 7 (silence is the default *system*, one model call, only after the
gate) is unchanged in shape; what changed is which answer that one call reaches
for first.

## Consequences

- Every existing test asserting the old two-condition gate or the old banned-list
  had to change; see `tests/test_judge.py`'s report on this task for exactly which
  assertions moved and why each one no longer describes the intended behaviour.
- A model that used to fill a contentless `silence`/`pattern_time` reason with a
  generic line and get declined for it (`또 왔네.`, `전혀 변한 게 없어.` -
  PLAN §6.2.1's own recorded gap) will now have that line accepted, on purpose:
  that is the reversal, not a side effect of it.
- The daily budget and cooldown ([ADR 0015](0015-code-may-search-where-the-model-may-not.md):
  5/day, 90-minute cooldown) are the only remaining ceiling on volume. They were
  never measured against a judge that actually speaks, because none ever had.

## What would overturn this

The tuning instrument this repo already has and has never been able to use:
👍/👎 labels on delivered utterances (`docs/PLAN.md` §8.1's label clock,
`proactive_utterances.label`). Zero utterances means zero labels, ever - every
number this repo has about proactivity's *content* is a spike against a fake
harness or an inference from a small offline test, never the owner's own reaction
to a real line. If, once this ships and actually speaks, 👎 clusters on lines that
match the new default (a contentless reason answered with an ordinary opener)
rather than on the demanding shape this ADR targets, that is the label data §6.2
never had, and it should move the default back.

This is deliberately a single prompt block in one file, not a new mechanism -
`SYSTEM`'s constant, the two changed examples, and the tests that pin its wording.
Reverting it is one text change and a test update, the same size as the change
itself. `docs/adr/README.md`'s note about a recurring shape in this file applies
here too: several of these decisions were correct at the time they were measured
and were overturned only when new evidence arrived, and this one names in advance
what that evidence would look like.
