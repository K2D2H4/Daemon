# Time awareness for the conversation path — design

- Date: 2026-08-18
- Status: approved for planning
- Branch of origin: `claude/daemon-time-date-awareness-19a4d6`

## Problem

The daemon resumes a finished conversation as if no time had passed. Observed on
Telegram: a Friday (2026-08-14) thread arranged a 16:40 meeting reminder; the owner
said nothing over the weekend; on Tuesday morning (2026-08-18, 09:28) they wrote a
bare "벨라" and the daemon answered

> 오후 4시 40분 회의 5분 전(4시 35분) 알림 잘 챙겨드리려고 옆에서 대기 중이었습니다.

Two failures in one sentence: it read a four-day-old thread as the live one, and it
reported an **expired commitment as still pending**. The owner reports this is
frequent, not a corner case.

## Evidence (what the code does today)

**1. "Now" is nowhere in the prompt.** `Companion.context` (`daemon/companion.py:248`)
returns exactly three blocks — persona, tool rules, recall. No current date, time, or
weekday. `system_state` (`daemon/tools/builtin.py:677`) does report local time, but it
is a tool the model must decide to call; it will not call it to answer a greeting.

**2. The recent window carries no time at all.** `ConversationLoop._assemble`
(`daemon/loop.py:540`):

```python
messages.extend(Message(role=item.role, content=item.content) for item in history)
```

Twenty messages as bare role+content. Friday 17:24 sits on the line directly above
Tuesday 09:28, so to the model it is one unbroken conversation. The observed reply is
the *correct* reading of that context.

**3. Recall is timestamped, but in a form the model cannot use.** `render_recall`
(`daemon/companion.py:496`) emits `2026-08-14T07:32:00.000Z`. Converting that to
"지난주 금요일 오후" requires knowing what "now" is (failure 1) and doing UTC→KST
arithmetic. Both are things this model does badly.

**4. The other two paths already model time correctly — only the conversation loop is
blind.**

| path | time sense today |
|---|---|
| `proactivity` | parses day markers and stated hours, computes `due`, expires follow-ups after `OPEN_LOOP_TTL_HOURS` (`daemon/proactivity/candidates.py:333`) |
| `voice` continuity | 120-minute freshness cutoff on the handed-over tail (`daemon/companion.py:415`) |
| conversation loop | **none** |

Replaying the observed case through `open_loop_candidates`: `_stated_hour("오후 4시
40분") = 16`, and `_due_at` takes `max(FOLLOWUP_HOUR=20, 16 + EVENT_LAG_HOURS)` → due
Friday 20:00 local, expiring Saturday 20:00 (verified by running the real functions).
**By Tuesday proactivity had already judged that commitment dead and was correctly
silent.** The conversation turn could not see that judgement.

So this is not "invent time awareness". It is "expose the time sense that already
exists to the one path that lacks it".

## Decisions (approved)

1. **The now block carries time + weekday + position in the day.** `지금은 2026년
   8월 18일 화요일 오전 10시입니다. 평일 오전입니다.` — position in the day is what
   makes "이 시간까지 안 자고 계시네요" possible.
2. **A session boundary is drawn when the local date changes *and* the gap is ≥6
   hours.** A three-hour afternoon gap is still one conversation; sleeping is not.
   The marker is placed inline, at the boundary, so the model never counts lines -
   true on `ollama` and `openai_compatible`, which keep the message where it was
   spliced. **Corrected after implementation:** `gemini`, `anthropic` and `openai`
   hoist every `role="system"` message out of the turn array and concatenate it
   into a top-level field before the model sees it (verified against each
   provider's payload builder), so on those three the position is gone and a line
   that said "위는 ..., 아래는 ..." arrived somewhere with no 위 and no 아래. The
   line's wording therefore cannot lean on its position: it names both dates and
   the gap directly and says outright which side is finished, so it reads true
   whether it lands inline or hoisted. See `daemon/MEASURED.md`.
3. **Recall timestamps render relative *and* absolute.** `지난주 금요일 오후 4시 32분
   (4일 전)`. Relative alone is ambiguous when two hits land on the same weekday.
4. **The commitment block reports both live and expired commitments.** Expired:
   `이미 지났습니다. 대기 중인 일이 아닙니다.` Live: `아직 오지 않았습니다.`
   Accepted cost: false positives are now user-visible (see Limitations).
5. **Both paths get all of it** — text via `Companion.context`, voice via the block
   channel `send_context` already uses.

Scope decided earlier in the same conversation: the daemon must know the time, speak
about it naturally, and judge whether a past commitment is alive or dead — but it
**does not raise expired commitments on its own initiative** (that would contradict
the measured `OPEN_LOOP_TTL_HOURS = 24` decision, whose comment reads "사흘 늦은 건
백로그를 처리하는 기계다"). Passive correctness only: know it is dead, do not treat it
as pending, answer accurately when the owner brings it up.

## Non-goals (v1, YAGNI)

- **No proactive raising of stale commitments.** Decided above; `OPEN_LOOP_TTL_HOURS`
  is untouched.
- **No owner-timezone config setting.** The resident runs on the owner's machine;
  `.astimezone()` resolves the zone in force on the date in question, which is also
  the only form that survives a DST boundary. A travelling-owner setting is a separate
  request.
- **No widening of `_DAY_OFFSETS`.** See Limitations.
- **No restructuring of the text path's message list.** Rejected as approach C: the
  comments at `daemon/loop.py:500` document why the list ends on a user turn (three of
  four providers hoist a trailing system note, and Anthropic reads a trailing assistant
  turn as a prefill to continue). Not worth disturbing for this.

## Architecture

One new module, `daemon/timesense.py`, holding every decision about how time is
*spoken*. Three injection points consume it. Nothing else learns to format a date.

```
daemon/timesense.py
  now_block(now)                -> str        decision 1
  session_breaks(history, now)  -> list[(index, str)]   decision 2
  relative(ts, now)             -> str        decision 3
  commitments(messages, now)    -> str        decision 4
  # extraction primitives moved here from proactivity/candidates.py:
  day_marker / stated_hour / due_at / EVENTS / EVENT_CANCELLED / ...

daemon/companion.py   context()        += now_block, commitments   (decisions 1, 4)
                      render_recall()   uses relative()            (decision 3)
                      continuity_block() += the same blocks         (decision 5)
daemon/loop.py        _assemble()       inserts session_breaks      (decision 2)
daemon/proactivity/candidates.py  imports the moved primitives     (no behaviour change)
```

### Components

**`now_block(now)`** — local time, weekday, and a bucket for position in the day
(새벽 00-05 / 아침 06-08 / 오전 09-11 / 점심 12-13 / 오후 14-17 / 저녁 18-21 / 밤
22-23), plus 평일/주말. Pure function of `now`.

**`session_breaks(history, now)`** — returns the indices in the history window where
the previous message's local date differs from the next one's *and* the gap is ≥6
hours, with the line to insert there. `_assemble` splices them in as
`Message(role="system")`, which is still the point on `ollama`/`openai_compatible`:
the boundary's position states the fact with no counting or arithmetic. **Corrected
after implementation:** three of the five providers hoist every system turn out of
the array before the position can be read (see decision 2), so the line's own
wording has to state the fact instead of relying on where it sits.

**`relative(ts, now)`** — 오늘 / 어제 / 그저께 for the nearest three days; `이번주
X요일` inside the current ISO week; `지난주 X요일` for the one before; `M월 D일`
beyond that. `(N일 전)` appended whenever N ≥ 1. Local time throughout.

**`commitments(messages, now)`** — scans **only the messages already in this prompt**
(the history window plus the recalled items), not a fresh database sweep. Two
consequences, both wanted: the block can only ever annotate text the model can
actually see, and the turn adds no query. Each owner utterance carrying a day marker
and an event word yields one line, `due` in the past → expired, in the future → live.

### Reuse seams (verified in this session)

- `_day_marker`, `_stated_hour`, `_due_at`, `_EVENTS`, `_EVENT_CANCELLED`,
  `_TENSE_NEUTRAL`, `_PAST_MARKERS`, `_HOUR_RE`, `FOLLOWUP_HOUR`, `EVENT_LAG_HOURS`
  (`daemon/proactivity/candidates.py`) — moved to `timesense.py` and imported back.
  The conversation path asks a **different predicate** of the same primitives:
  proactivity asks "is `now` inside `[due, due + TTL]`", this asks "is `due` past".
- `_is_owner_utterance` (`candidates.py:630`) — reused unchanged. Only the owner's own
  words can create a commitment; relayed text cannot.
- `clock.now` / `clock.to_iso` / `clock.parse_iso` — unchanged. One new display-only
  helper, `clock.local()`.
- `Companion.continuity_block` / `send_context` — the existing voice block channel
  carries the new blocks; no new voice plumbing.

## Trust invariants (the reason this cannot become an injection)

**Every new block is built from computed values and fixed lexicon words only. No
substring of any message reaches the prompt through them.** `now_block` and
`session_breaks` derive purely from timestamps. `commitments` follows the discipline
`candidates.py:364` already states for its `reason` field — "Only lexicon words and
clock times: `surface` and `event` both came from the tuples above, never from the
message." So a message crafted to look like a boundary marker cannot ride into a new
block, and the new blocks need no nonce.

`render_recall`'s change is confined to the timestamp field of a line whose content
already passes through `_one_line` (marker-stripped, length-capped). The nonce
framing, `_label`'s origin annotation, and the untrusted-origin marking on curated
facts are untouched.

## Error handling

Never fail a turn. Each helper is wrapped at its injection point the way
`_send_continuity` (`daemon/voice/conversation.py:420`) already treats continuity: a
missing time block costs some naturalness, a raised exception costs the turn. An
unparseable stored timestamp drops that one line, not the block.

## Testing

- `timesense` unit tests with `clock.now` patched: the day buckets at each edge hour,
  weekday and week-boundary rendering (including the same weekday one and two weeks
  back), and the ≥6h-plus-date-change rule against a same-day 5h gap and an
  across-midnight 2h gap.
- **The observed case as a regression test.** Friday 16:40-commitment thread + Tuesday
  09:28 "벨라" through `_assemble`, asserting a session boundary lands between them and
  the commitment renders expired.
- `commitments` false-positive guards: `EVENT_CANCELLED` in the same message, a past
  tense with `TENSE_NEUTRAL`, and a non-owner-origin utterance — each yields nothing.
  **Each guard needs a control assertion showing the same sentence WITHOUT the
  guard-tripping word does produce a commitment, and each must be verified by removing
  the guard and confirming its test fails.** These guards overlap, and the overlap hid a
  hole: the natural cancellation phrasing ("오늘 회의 취소됐어") is filtered by the
  past-tense guard instead, so `EVENT_CANCELLED` could be deleted outright with every
  test still passing. Without a control, an empty result cannot be told apart from a
  sentence that parsed as nothing.
- Injection: a message containing a boundary marker and event words produces a block
  containing no substring of it.
- **Proactivity must not move.** The existing `candidates` suite passes unchanged after
  the primitives are relocated — that is the whole guarantee of the move.
- Live verification (per repo practice): drive the real resident over Telegram, not
  just the suite. Reproduce the bare-greeting-after-a-gap case and read the reply.

## Frozen-contract notes

Two, neither done quietly:

1. **`daemon/clock.py`** documents itself as the one place the wall clock is read, in
   UTC. Adding `local()` keeps storage, ordering and recency decay on UTC and confines
   the local value to display, but it does add a second way to read the clock, and the
   module docstring must say when each is correct.
2. **`candidates.py` internals move.** Not a frozen contract, but proactivity's
   measured behaviour depends on them and four ADRs in `docs/adr/` were corrected by
   measurement. The relocation is mechanical and pinned by the existing suite.

## Limitations (state these, do not fix them here)

**The commitment block only sees `오늘 / 이따 / 내일 / 모레`.** `_DAY_OFFSETS`
(`candidates.py:252`) excludes `다음주`, `주말` and bare weekday names on purpose:
resolving "금요일" is a guess about whether it means today or in seven days, and a
wrong `due` makes the daemon ask how something went before it happened. The observed
case is caught ("**오늘** 오후 4시40분에 **회의**"); "금요일에 회의 있어" is not.
Widening the lexicon is a separate, measurement-backed change.

**A commitment older than the prompt window is invisible.** `commitments` deliberately
scans only what is already in front of the model. A live commitment from thirty
messages back appears only if recall surfaces it — and raising it on time is
proactivity's job, which already does it.

**`origin == "owner"` is not forgery-proof after a reindex.** `candidates.py`'s module
docstring records this: `daemon reindex` re-derives `origin` from `role` alone, because
the markdown it rebuilds from carries no provenance, so a forward stored as
`role='user'` comes back `origin='owner'`. `MemoryRecall.associate` handles it by also
excluding `reindexed` rows; `commitments` cannot, because neither `LoggedMessage` nor
`RecalledItem` carries the column. The exposure is bounded by the invariant above — the
block quotes no message text — so the worst case is a phantom "아직 오지 않았습니다"
line derived from a forwarded message after a rebuild, which falls inside the
false-positive cost decision 4 already accepts.

**Decision 4's false positives are now visible.** An event word plus a day marker in a
sentence that was not a commitment produces a stray "아직 오지 않았습니다". The
lexicon is conservative and `_EVENT_CANCELLED` catches the expensive direction, but the
boundary cannot be set correctly without measurement. If it reads as noise in use, the
fallback is decision 4(가) — expired only.

## Measured outcome (2026-08-18, after implementation)

Live against `gemini` with an isolated data directory, replaying the reported case: the
assembled prompt carries all three blocks correctly, and the model still treats the
expired commitment as live in **5 of 26 runs (~19%)**. Block order was A/B'd — commitment
block at the top versus immediately before the final user turn — and both arms gave 0/10,
which settles open question 1 no better than not asking it. Details and the caution about
batch-to-batch variance are in `daemon/MEASURED.md`.

**What this means for the decisions above.** Decisions 1-3 and 5 are delivered and hold.
Decision 4's block is present and correct, but the *outcome* it exists for — the daemon
never describing a past commitment as pending — is not reliably achieved by putting the
fact in the prompt. The remaining failures all have one shape: the window's last assistant
turn is an enthusiastic promise about the reminder, and the model continues it. Closing the
gap is prompt-effectiveness work, needs n well above 10 per arm, and is not in this spec.

## Open questions for planning

1. Where the now block sits in the block order — `context()` returns persona, tools,
   recall. Time probably belongs first (it is a fact about the world, not an
   instruction), but the ordering has never been measured.
2. Whether the now block should be re-rendered per turn in a long voice session, or
   sent once at wake. Sending once is cheaper; a two-hour session then drifts.
