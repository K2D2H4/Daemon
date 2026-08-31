# daemon/ — measured, not assumed

Every line here is something this project believed until it ran it. Read it before
optimising, before trusting a documented probe, and before repeating a measurement
that is already here. Orientation: [CLAUDE.md](CLAUDE.md).

- **Reflection's starvation was two problems and only one of them was the tool results.**
  Measured on the live database (2026-08-18: 890 messages, 7 observations, 1 persona rule,
  307 tool calls). Re-running the *unchanged* pass over five real days - 664 messages - with
  the hosted model produced **0 facts and 0 observations**. So the input was not the only
  thing missing; the conversation call was returning empty on days that had plenty to say.
  Feeding it a **usage summary** - which tools ran, roughly when, how often, and whether they
  were refused, with no output text at all - fixed the yield: 2026-08-10, 20 runs per arm,
  **0.85 -> 1.75 observations per run, p = 0.0012** (two-sided permutation test, 200k
  shuffles). The number that matters for M4 is not the mean: **nights that produced zero
  observations went from 9/20 to 1/20**.
- **One run of a reflection pass tells you nothing about a reflection pass.** The first
  measurement of the above was a single run per arm, 0 -> 2, and it looked convincing.
  Twenty runs put the pre-change arm at 0.85 with 9 zeros - so "0 observations" was a coin
  flip that had come up tails, not a fixed property of the day. Any A/B on this pass needs
  a p-value; the pass is a model call and the model is not deterministic.
- **The tool-content path works, is accurate, and on this history added nothing** - which is
  a result, not a failure. Reading 2026-08-09..13's calendar and Notion output, the second
  call correctly extracted the owner's birthday (off a `google__get_events` "Happy birthday!"
  entry), their name and email, and their job title (off a Notion JD). **All three were
  already in `memory_entries`**, learned from the same day's conversation. Its prompt had no
  idea, because it was handed the day's material and nothing else. Giving it the curated tier
  read-only took it to 0 new facts on all four days - the honest number. The value shows on a
  day the daemon reads something the owner never mentions; this history has none.
- **Web search was in `CONTENT_TOOLS` until it was measured, and a two-word fragment is why
  it is not.** On 2026-08-10 01:45, in a voice session, `tavily__tavily_search` ran the query
  "김대현 resume english" - the owner's own name - and got back **a different 김대현's CV**
  (cv.hatemogi.com: Rust/Scala/Clojure, against an owner who is an AI/LLM engineer). The
  owner had not asked for a web search. The user turn immediately before it was
  **"그냥 영어로"**.

  **Correction, and the way it was got wrong is the point.** This was first written up here
  as "an `open_path` failed, so the model fell back to searching the web with the same
  words". That was inferred from the *order of the tool_calls rows* - a failed open at
  01:45:19, a search at 01:45:57 - and it did not survive reading the conversation those
  rows sat in: the user asked for Chrome in between, and the utterance before the search was
  a fragment about the filename being in English. Tool rows record what the machine did, not
  why; reconstructing intent from them alone produces a confident story with nothing holding
  it up. Read the messages beside them.

  Dropping it cut the digest by roughly 60% on the days that had it (2026-08-09: 2788 -> 1080
  chars, 08-10: 3443 -> 1308) while the second call still fires on every one of them - the
  volume was noise and the fact contribution was zero. The tool itself is untouched: the
  allowlist governs only whether an output may become a *permanent* fact.
- **The daemon used to run tools on wake-word misrecognitions. `CALLED_BY_NAME` (v0.1.36,
  2026-08-10) already fixed it, and measuring across that date is the only way to see so.**
  Over the whole live history, executed tool calls whose nearest preceding user turn is 6
  characters or shorter look alarming - voice 13/73 (17.8%) against text's 20/224 (8.9%), the
  voice ones largely the recognizer mangling the wake alias 벨라: `open_path` after "Allah",
  "Oops.", "el la"; `read_page` after "¿Ah?"; `see_screen` and `start_screen_share` after
  "Bella." / "Ella.". Split at the day the fix landed, it is a different picture:

  | | text | voice |
  |---|---|---|
  | on/before 2026-08-10 | 20/141 (14.2%) | 12/61 (19.7%) |
  | after 2026-08-10 | **0/83** | **1/12**, and that one is "응." → a Notion fetch, an ordinary yes |

  Not one wake-noise tool call after the fix. What actually ran before it was harmless
  anyway - the single `run_command` was a read-only `icalbuddy` query that failed.

  **The reusable part is the near-miss.** This was written up here as a live defect worth
  opening an investigation into, on a whole-history aggregate, when the fix had already
  shipped inside the measured window. A log that spans a release measures two different
  programs. Date the fix first, then split the data on it - and note that `origin='owner'`
  is honest on these turns either way (the owner did make a sound), so rule 10 was never
  what would have caught it.

  **Do not read this as "voice tool calls are safe now".** `CALLED_BY_NAME` covers only the
  case where *the recognizer decided the segment is the name and nothing else*
  (`_opening_text`, `daemon/voice/conversation.py`). The utterance that actually fired the
  namesake web search - "그냥 영어로" - was not wake noise but an ambiguous mid-conversation
  fragment, which that fix does not touch. A model reaching for a tool on a fragment is still
  open, and only 12 voice tool calls fall after the fix, so the data cannot separate the two
  cases either.
- **Running reflection for real found two defects unit tests did not.** Entity notes were
  stamped with the day of the *run*, so a months-long catch-up reads as all-today; and two
  facts sharing a supersession key retired the wrong half (`data/memory/core.md` kept the 3).
- **A 4B local model is unstable at reflection.** Same 6-message Korean fixture, two
  runs: 1 fact / 1 entity / 0 observations, then 2 / 2 / 1. Point it hosted if anything.
- **Two of the plan's three macOS presence probes were wrong as documented.** `HIDIdleTime`
  is **nanoseconds**; every `ioreg` audio class is absent on Apple Silicon, so CoreAudio via
  ctypes is the only one that answers (input too); `name of ... application process` gives
  the *executable* — `stable`, not `Warp`.
- **Narrowing the question is not enough to make a model decline.** Merely permitting it:
  **0 declines of 15**. Stating silence as the default and speaking as the exception: 3.
- **Do not use a reasoning model for local chat.** gemma3:4b answers in 1.7 s, qwen3:4b
  in 11.8 s, and the gap is chain-of-thought nobody sees; it cannot be turned off
  (`think: false` stops Ollama *separating* it, so it becomes the reply).
- **The vector lane is the cheap half of recall** (0.22 ms vs FTS5's 1.9 ms at
  10k). What dominates is the embedder round trip, ~117 ms, mostly fixed overhead.
- **FTS5 with `unicode61` cannot carry Korean recall alone** — whole-token matching,
  so an inflected word is a different token: 50% keyword-only vs 93% hybrid.
- **Voice mode interrupted itself on every single turn, and echo was only half of
  it.** Two independent causes, both measured. (1) The speaker leaks into the open
  microphone: 80.1% of mic frames read as speech through PortAudio while the speaker
  played, 0.0% through macOS voice processing, and 81.6% for speech the canceller
  does not know about - so the echo goes and the user does not
  (`daemon/voice/apple_audio.py`). (2) Gemini delivers `inputTranscription` **at the
  turn boundary, in the same server event as the answer's first audio chunk**, and
  `daemon/voice/conversation.py` inferred a barge-in from the transcript growing while audio
  played. So the question's own transcript condemned its own answer: a complete,
  fluent reply generated and **0.0s of it played**. The barge-in is now the server's
  `interrupted`, which is what `daemon/voice/base.py` always said the authority was. Same
  measurement retires a documented claim: the recall prefetch cannot be free against
  this provider, because the partial it fires on arrives with the answer.
- **Recall was killing the answer it was fetched for, and this was the biggest of the
  three.** `send_context` is `clientContent`, and the Live API says plainly that "a
  message here will interrupt any current model generation". The prefetch landed
  mid-answer, so seeding a memory cut the reply off at "아..." - the owner's log paired
  every barge-in with the embed call immediately before it, 1:1. Measured, one
  conversation, same room and microphone: **2.2s of audio with recall on, 46.7s with
  it off, 38.8s with it deferred to the turn boundary.** Deferring costs nothing that
  was not already lost, because the prefetch fires on a partial that arrives with the
  answer anyway. And note what `serverContent.interrupted` actually means - "a client
  message has interrupted current model generation", *not* "the user spoke" - so it is
  two failures wearing one flag, and reading it as pure user-VAD is what let the daemon
  mistake its own memory for the owner talking over it.
- **An interruption arriving after `generationComplete` is not an interruption.**
  Measured four times: it lands ~0.25 s later on a turn nobody touched, and acted on
  it empties the speaker of an answer that was fully delivered.
- **Two Telegram traps.** The inbound poll needs a floor — left to the long poll alone, a
  transport that returns immediately spins at ~16,000 requests/second. And `allowed_updates`
  is **server-side**: at `["message"]` a 👍 press is never delivered at all.
- **The hygiene rule that fed nothing to reflection cost 29 of 38 real messages**, and the
  29 were the persona-relevant lines while the 9 survivors were wake-word noise: 0 facts, 0
  entities, 0 observations out. It also never blocked the loop it named — recall injects its
  hits as a system block, and only the user turn and the reply become rows. Rule 2 is
  retired; the same real day then gave **110 read → 3 facts, 1 entity, 3 observations**. The
  flag is still written and nothing reads it; see `Store.mark_recalled`.
- **`daemon doctor` and `daemon reflect` disagreed about the same day.** Doctor's backlog
  counted today; `Reflection.catch_up` deliberately excludes it. So doctor said "run `daemon
  reflect`" every day and that command answered "nothing to reflect on." Today is dropped
  from the backlog now — two commands contradicting each other teaches you to read neither.
- **A real weekly evolution pass, run against Gemini with 30 seeded observations:** 30 read
  -> 7 proposed -> 3 added, 10.8 s, with the 4 the per-cycle cap dropped reported rather than
  discarded. A same-week rerun skipped in 0.64 s and made no model call.
- **Printing a real turn's assembled prompt found a defect.** `load_persona` was injecting
  all of `learned.md`, including its human-facing header (`daemon persona forget <id>`, a
  repeat of the sentence the loader already prefixes). Only the rule bullets go in now;
  `seed.md` still goes in verbatim.
- **`learned.md` was being rewritten from the mirror, so deleting the rebuildable sqlite file
  cost 5 rules out of 5 on the next ordinary write** — non-negotiable 1, measured, not argued.
  `add`/`retire` now refuse on divergence (`LearnedFileDiverged`), `daemon reindex` restores
  rows from the file additively, and `doctor` reports the divergence as a blocker. A crash
  between the two writes has the same shape: the instant after is allowed, the *next* write
  was what destroyed.
- **`write_private_replace` used one fixed temp filename**, so two writers on the same path
  (the Monday job and a hand-run `evolve`) raced: one `O_TRUNC`ed the other's bytes, then the
  loser's `os.replace` raised. Random suffix now. Two concurrent writers still do not merge —
  the later `replace` wins outright, and that is not fixed.
- **A voice `toolResponse` does not interrupt generation, and the reason is more
  useful than the fact.** Measured on `gemini-3.1-flash-live-preview`: a blocking
  reply ran **13.69 s past** our answer and spoke the value we returned, with
  **0.0 s of audio before** it — the `toolCall` arrives before the model says
  anything, so there is no generation for the response to land inside. That is what
  makes it unlike `clientContent`, which killed a 46.7 s turn down to 2.2 s; that
  failure *needed* a mid-answer arrival. Meanwhile `behavior: NON_BLOCKING` was
  **accepted and then ignored** — all three `scheduling` values gave 0.0 s of audio
  after the answer, no `interrupted`, and no second turn in 60 s, including the
  `INTERRUPT` that is documented to make the model break off. A field the server
  accepts and ignores is worse than one it rejects, so neither is sent.
- **Which tools `mode=allowlist` could never run was guessed wrong in both
  directions.** The guess was `write_file`, `open_path` and `notify`. Enumerated
  instead: `notify` is `risk="safe"` and never reaches the mode check at all, and
  `open_path` *does* implement `argv`, so of the ten built-in and browser tools
  exactly **one** — `write_file` — was stuck. The group it actually cost is MCP:
  `McpTool` has no argv by construction and is `guarded` unless `data/mcp.json` names it
  safe, so `allowlist` refused every remote tool an owner added. `tool_grants` is
  that second axis, and it is read only for tools that are *not* `Executable` —
  a tool-level grant on `run_command` would be `mode=full` wearing one table row.
- **A headphone probe was designed, approved, and implemented, then killed by
  one `system_profiler` call.** The plan: read the default output device's
  `Transport:` from `system_profiler SPAudioDataType` and treat `usb` /
  `bluetooth` / `headphone` / `displayport` / `thunderbolt` as point-to-point.
  Running it for real on the development machine found the default output is
  *always* the virtual `MacBook Pro Speakers (eqMac)`, and it reports
  `Transport: USB` — indistinguishable from real USB headphones — for the
  laptop's own built-in speakers, whatever is actually plugged in:
  ```
  MacBook Pro Speakers (eqMac)  -> USB       (the default output, always)
  MacBook Pro Speakers          -> Built-in
  LG FULL HD                    -> HDMI
  Microsoft Teams Audio         -> Virtual
  eqMac Export                  -> USB
  ```
  Not a miscalibration to tune — the transport field cannot see past the virtual
  device to the real hardware behind it, on this machine, ever. `headphones` is
  the one presence signal that only *widens* what the speaker may do, so under
  PLAN 6.4's asymmetry a probe that answers wrong in that direction is worse than
  one that never answers. Removed rather than fixed; `Reading.headphones` stays
  `None` (`daemon/proactivity/presence.py`, `daemon/proactivity/base.py`
  unchanged). It was also the most expensive probe in the file at 0.21-0.30 s,
  more than the `osascript` foreground fallback (168-233 ms) — so removing the
  wrong signal also removed the biggest cost.
- **`output_muted`'s two `osascript` calls, measured on this machine:** `MUTED`
  134-178 ms, `VOLUME` (asked only when not muted) 145-247 ms. A reading now
  costs ~155-200 ms muted and ~300-445 ms not muted, on top of the ~20 ms
  `idle_seconds`/`foreground_app` already cost when `lsappinfo` answers.
  `screen_locked`'s Quartz call is 0.09-0.15 ms warm (~18 ms the first time in a
  process) — cheap enough that it now runs before `output_muted` in `read()`.
- **"The SDK handles refresh on its own" was true only until the process restarted.**
  `OAuthContext` derives `token_expiry_time` from `expires_in` at the moment a token
  arrives and keeps it *in memory*; `TokenStorage` persists neither it nor the arrival
  time, and `is_token_valid()` reads a falsy expiry as "no expiry declared". So every
  boot loaded Notion's eight-hour token, judged it valid, sent it, took a 401, and
  escalated to a **full** re-authorization — never once trying the refresh token sitting
  beside it (the failing log has no `POST /token` at all). Any restart more than eight
  hours after the last auth made the owner click through Notion's consent screen again.
  Stamping `obtained_at` at write and priming the context at build puts it back on the
  refresh branch: `POST /token 200`, no 401, 28 tools, verified live.
- **The judge's decline few-shot, fitted to gemma3:4b, showed no measurable effect on
  the hosted model that now also runs it.** PLAN 6.2.1 found a 4B local model reading a
  `silence`/`pattern_time` reason — elapsed hours or a frequency, nothing else — filling
  the gap with `또 왔네.`/`전혀 변한 게 없어.`; `daemon/proactivity/judge.py`'s two
  worked examples exist to muzzle exactly that pair of kinds. Re-measured 2026-08-11
  against the model `PROACTIVE_JUDGE` actually resolves to under `DAEMON_PRESET=quality`
  — `gemini-3.6-flash`, confirmed off `daemon doctor`'s routing line and off
  `Completion.model` on the real response, not off config — with `evals/proactive_judge.py`:
  17 reasons across the five kinds, run twice, once against the current prompt (A) and
  once with only the `silence`/`pattern_time` examples cut (B), nothing else touched.
  Declines were identical on every kind, both variants: `silence` 3/3, `pattern_time`
  3/3, `open_loop` 0/3, `emotional` 0/3, `association` 1/5. B never produced a single
  spoken line on `silence` or `pattern_time` to check for filler — it declined all six
  the same as A — so the pre-declared bar for adopting B ("B's lines on those two kinds
  are not filler, and the seed's voice holds") has nothing to be judged against. Kept
  (A): a tie is not evidence for removing a muzzle that costs six lines and no latency,
  and the task this measurement was for says plainly that changing code because it was
  expected to is worse than reporting no change needed with evidence.
  **Superseded 2026-08-26 by docs/adr/0016-proactive-default-flips-to-speaking.md.**
  Both worked examples now answer with a line, so the muzzle this entry argued for
  keeping no longer exists. Nothing above was overturned - the 2026-08-11 tie stands,
  and it was never the reason the muzzle came off. What removed it was the owner
  saying the shape he had objected to was the *demanding* question
  (`무슨 재밌는 일 없어요?`), not the ordinary check-in the examples were muzzling.
- **A live-data preview of the type-E generator turned up conversational chaff among
  its own candidates, and the judge caught it without being told to.**
  `association_candidates` (`daemon/proactivity/candidates.py`) quotes the owner's own
  words with no content-worth filter of its own; previewing it against the real database
  found two of three quoted associations were meta-utterances - `'우리 방금 무슨 얘기
  했었지?'` among them. Fed to the judge verbatim on 2026-08-11 (`gemini-3.6-flash`,
  both prompt variants), it declined - `{"say": ""}`, `why_not="nothing worth saying"` -
  while a substance reason from the same run (`'교토 골목 국수집이 진짜 좋았어'`)
  produced a natural line in both. One instance is not proof the generator never needs a
  filter of its own, but on this model the judge's own content criterion ("구체적인
  사건·감정·기억이 내용으로 적혀 있다") already screens chaff before anyone hears it, so
  nothing in this run argues for adding one now.
  **The mechanism this conclusion rests on was removed 2026-08-26** (ADR 0016): that
  content criterion is the clause the flip deleted, so "the judge already screens
  chaff" stopped being a reason to leave `association_candidates` unfiltered. A
  narrower rule was carved back out in its place, and re-measured on this same
  material - a reason built from the owner's command history declines **0/30 spoke**
  (n=30, `gemini-3.6-flash`, 2026-08-26). The conclusion holds; the thing holding it
  up is now a named carve-out rather than a general criterion.
- **The C rhythm was accepted on the machine, not in the suite (2026-08-11).** Every
  routing rule was driven against a live `Reading` from this Mac, with the daemon's
  own microphone hold declared the way the wake listener declares it:
  ```
  editor in front, unmuted     -> both      ok
    + muted                    -> telegram  output muted
    + screen locked            -> telegram  screen locked
    + somebody else on the mic -> telegram  microphone in use
  Slack in the foreground      -> telegram  Slack is in the foreground
  ```
  `both` is the line that matters: it is the first time the local speaker has been
  reachable at all. Before the split, `mic_busy` was `True` forever on any install
  with `DAEMON_WAKE_ENABLED` on, so the speaker branch could not be selected —
  switching voice on was what switched the voice route off. With the hold declared,
  the same probe reads `mic_busy=False` while the device is genuinely running.
  The 👎 brake was driven against the real `Settings` (budget 8/day, ceilings
  association 3 · emotional 2 · open_loop 2 · silence 1 · pattern_time 1): one press
  gives `thumbs down: association is resting for 6h` and leaves every other kind at
  `ok`.
- **Type E cannot fire on this install for about another month, and that is the
  design (2026-08-11).** `ASSOCIATION_MIN_AGE_DAYS` is 30 and the whole conversation
  history spans 2026-08-06..2026-08-11, so there are zero owner messages old enough
  to be an association rather than a conversation. Same shape as `pattern_time`
  needing 14 distinct days. Worth writing down because "the generator produces
  nothing" and "the generator is broken" look identical from outside, and this one
  is the former.
- **A thinking model bills its thinking out of `max_output_tokens`, and that kept
  the daemon silent for days (2026-08-15).** `daemon/proactivity/judge.py` capped the proactive reply
  at 300 tokens, sized as "the JSON plus 120 characters of Korean, with room to
  spare". Correct for a model that only answers. `gemini-3.6-flash` reasons first
  and `candidatesTokenCount` bills both, so the budget ran out mid-sentence:
  ```
  run 1: parsed=False len=19  '{"say": "어제 그 미팅은 잘'
  run 2: parsed=True  len=46  '{"say": "오늘 미팅은 무사히 끝났어요? ..."}'
  ```
  1 failure in 5 on a synthetic reason, and **4 in 4** against the reasons the
  owner's own history produced - so every `open_loop` candidate, the kind that
  measurably does have something to say, died on JSON parsing. At 1500 tokens:
  6 of 6, every one `stop_reason=STOP`.
- **The log said the wrong thing about it, which is why it survived.** A truncated
  reply and a model answering in prose both arrive as "no JSON", and the judge
  reported both as `did not return a JSON object` - which reads as the model
  misbehaving, not as our own cap being too small. All four providers already
  report `stop_reason` in `Completion.meta`; nothing read it. They now say
  `was cut off at the token limit (stop_reason=MAX_TOKENS); raise
  MAX_OUTPUT_TOKENS`. The lesson is not the number: it is that two failures with
  different fixes must not share a message.
- **The time-awareness blocks fix the reported defect most of the time, not every time
  (2026-08-18).** Driven against live `gemini` with an isolated `DAEMON_DATA_DIR`, replaying
  the exact reported case — a Friday thread that arranged a 16:40 reminder, four days of
  silence, then a bare "벨라" — the assembled prompt carries all three blocks correctly
  (`[현재 시각]`, `[약속 상태]` saying the meeting's time has passed, `[대화 단절]` between the
  finished thread and today). The model still answered "오후 4시 35분에 회의 알림 잊지 않고
  챙겨드릴 테니" in **5 of 26 runs (~19%)**, and the rate is unstable across batches: 3/10 in
  one batch, 0/10 in another with the same code. So a single batch cannot tell you whether a
  prompt change helped. **Moving the `[약속 상태]` block from the top of the prompt to just
  before the final user turn — the recency hypothesis, and `docs/superpowers/specs/2026-08-18-time-awareness-design.md`'s
  open question 1 — produced 0/10 in both arms and settled nothing at that sample size.** The
  failures are all the same shape: the window's last assistant turn is an enthusiastic promise
  about the reminder, and the model continues it. Anyone tuning this needs n well above 10 per
  arm and should measure against that shape specifically.
- **The reason that A/B settled nothing: on every hosted provider there is no such thing as
  block position (2026-08-18).** The three provider files under `daemon/llm/providers/` -
  gemini, anthropic and openai - each do
  `"\n\n".join(m.content for m in messages if m.role == "system")` into one top-level field
  and drop those messages from the turn array. So `[대화 단절]`, spliced into the window at the
  index where the conversation broke, arrives concatenated with the persona and the tool rules
  — and its original wording, "위는 8월 14일 금요일, 아래는 …", pointed at a position the model
  could not see. Feeding the reported case's message list to gemini's `_contents` returns
  `[user, model, user]` with the line absent. Only `ollama` and `openai_compatible` keep system
  turns in place. Two consequences worth keeping: a `FakeProvider` that receives the raw
  `list[Message]` cannot see this class of defect, and any future reasoning about where a
  system block sits is meaningless unless the provider is one of those two.
- **Rewording that line to be position-independent measured 1/20, against 5/26 before — and
  that is NOT a demonstrated improvement (2026-08-18).** Fisher's exact on 5/26 vs 1/20 gives
  p = 0.21. The reword is worth having because the old line was *provably false* on the
  configured provider, not because the failure rate is proven lower. Anyone claiming this fixed
  the rate needs a much larger n; see the batch instability above.
- **"Inline placement is the whole point" was true for two providers out of five
  (2026-08-18).** `docs/superpowers/specs/2026-08-18-time-awareness-design.md`'s decision 2
  said the `[대화 단절]` line's position states the fact so the model does no counting -
  written without checking what a hosted provider does with a `role="system"` message
  planted mid-list. Fed the reported case's message list to each provider's payload
  builder: `gemini._contents`, `anthropic._turns` and `openai._input_items` all drop every
  system message from the turn array and `complete()` hoists the joined text into a
  top-level field (`systemInstruction`, `system`, `instructions`) instead - confirmed by
  running each function directly and checking the break line is absent from the array and
  present in the hoisted field. Only `ollama` and `openai_compatible` leave it where it was
  spliced. So on `gemini` - this owner's configured provider - the line "위는 8월 14일
  금요일, 아래는 4일 뒤인 오늘 8월 18일 화요일입니다. 위쪽은 이미 끝난 대화입니다." landed in
  the system prompt next to the persona and tool rules, referring to a 위/아래 that provider
  never showed the model. Reworded to name both dates and the gap directly instead of
  pointing at a position (`daemon/timesense.py::_break_line`), and pinned against the actual
  `gemini` payload builder in `tests/test_providers.py` so this cannot regress silently.
- **The proactive path reaches a labelable row, and it took an isolated data dir to
  find out (2026-08-18).** `proactive_utterances` had been empty since the daemon
  became able to speak first in v0.1.46, so nothing had ever confirmed the half of
  the pipeline *after* the judge. Driven against live `gemini` on a copy of the
  owner's data dir, with one substitution — a capturing channel in place of Telegram,
  so nothing was sent — one armed `open_loop` candidate produced:
  ```
  kind=open_loop  route=telegram  label=None
  text=아까 말했던 그 회의는 무사히 끝났어요? 피 터지는 전쟁은 아니었길 바라는데.
  gate=ok - telegram: DAEMON_VOICE_ENABLED is off
  ```
  The row carries its `gate_snapshot`, and the outbound message carries
  `labelable=True` with the `utterance_id`. A second tick in the same run was
  refused by the 30-minute global cooldown, which is the gate working. Not verified
  here: the Telegram render and the button press itself.
- **The judge's refusals are deterministic per reason *shape*, not noisy
  (2026-08-18).** 20 live calls per reason, using the reason strings the owner's own
  history produced:

  | reason | outcome |
  |---|---|
  | `open_loop` (08-14 '오늘 회의') | **20/20 spoke** |
  | `silence` ("마지막 대화가 12시간 전…") | **20/20 nothing worth saying** |
  | `association` at `min_age_days=7` (two candidates) | **20/20 and 20/20 nothing worth saying** |

  With the PR's own 6/6, `open_loop` is 26/26. The `silence` column explains the
  production tally exactly — 159 of 163 logged declines were `silence`
  (`nothing worth saying`), and the remaining 4 were the `open_loop` truncation
  above, all before v0.1.54. **A generator whose reason is nothing but elapsed time
  cannot produce speech**, so the 288 ticks a day that generate one are spending a
  model call to be refused. That is PLAN 6.2's asymmetric default working, and it is
  also the whole of type C's contribution. **Superseded 2026-08-26** —
  [docs/adr/0016](../docs/adr/0016-proactive-default-flips-to-speaking.md) flips
  this: `silence` and `pattern_time` now speak on an elapsed-time-only reason by
  design, so this bullet's "cannot produce speech" is what the old prompt did, not
  what the current one does.
- **Lowering `ASSOCIATION_MIN_AGE_DAYS` was proposed and the measurement killed it
  (2026-08-18).** At floors 7/10/14 days against 884 embedded messages, type E
  surfaced 3/2/0 candidates. What it surfaced at 7 days was the owner's own
  command history: `'오늘 날짜가 어떻게됨?'` (score 0.795), `'오늘 날짜좀 알려줄래?'`
  (0.766), `'이내용들 옵시디언 위키에도 좀 넣어줄래?'` (0.708) — matched against a
  query built from the last three owner utterances, which were
  `'우리가 생성했던 옵시디언 배치 기억나?'` and two about a job that did not run. The
  similarity is real and the association is worthless, and the judge said so 20/20
  on both. So E's floor is not what stops E; **this install's conversation is almost
  entirely tool commands**, and E quotes owner messages. The constant was left at 30.
  **Still relevant 2026-08-26, narrowed** —
  [docs/adr/0016](../docs/adr/0016-proactive-default-flips-to-speaking.md) flips the
  judge's default to speaking and deleted the clause that produced this 20/20; a
  narrower clause was carved back out specifically for a quoted line that is an
  instruction to the daemon, so this finding is expected to still hold, but has not
  been re-measured against the new prompt.
- **Type B is starved by the same fact, and its lexicon is not the problem
  (2026-08-18).** 463 owner utterances, **0** matching `_EMOTIONS`. Widening the scan
  to 30 emotion words that are deliberately *not* in the lexicon (피곤·귀찮·실망·후회·
  부담·최악·망했…) finds **3 hits in 579 lines**. The owner talks to this daemon in
  imperatives — "오늘 서울시 날씨좀 알려줘", "notion 메인 화면 띄워 줘", "파일좀 열어줘".
  So of the five generators, exactly one (A) can currently produce a reason the judge
  will speak on, which is the shape PLAN 6.2 warns about: a competent assistant
  rather than a companion. That is not a tuning gap.
  **Both halves of that sentence changed 2026-08-26.** There are now six generators
  (`topic`, ADR 0015), and `silence` and `pattern_time` speak rather than decline
  (ADR 0016) - `silence` measured **30/30 spoke, 0/30 demanding shape** (n=30,
  `gemini-3.6-flash`). The scarcity this entry measured was real and is why both ADRs
  exist; it is no longer a description of what the judge does.
- **The proactive search buys content on two entity names out of six, and invented a
  whole new failure on the other four (2026-08-26).** ADR 0015's reversal test was first
  run before ADR 0016 flipped the judge's default, so it measured a prompt that no longer
  ships; the re-run is the number that counts. Live search results for six of this owner's
  real entities, `gemini-3.6-flash`, n=30, every line hand-audited because
  `_carries_concrete_fact` scores any Latin run as content and would have passed a
  contentless opener that merely names the entity. **The predicted failure barely
  happened**: obvious chaff was still declined with no help from any instruction —
  `Sendbird` (a job posting, an Instagram page, a salary table) silent 5/5, `ReadyTalk`
  (`Breakfast is ready talk to ya later.`) silent 5/5, `Kiwi` (the bird, the fruit)
  silent 5/5. **A different failure did**: `Daemon` returns House of the Dragon's Daemon
  Targaryen, and 3 of 5 lines asked the owner — whose `Daemon` is this project — whether
  he was waiting for season 3. Chaff is declined because it reads as chaff; a namesake
  reads as material, and the model has plenty to say about the wrong subject. Naming the
  shapes in `topics.render` (a namesake person or character, a same-named product, a
  dictionary entry for the word) put `Daemon` at 5/5 on the project and 0/5 on the
  character, and `Kiwi` at 5/5 on the owner's own (it had been silent 5/5 - the
  same-name rule turns a decline into a usable check-in, not just a wrong line
  into a right one). Two lessons worth more than the fix:
  a confidently wrong line is a worse failure mode than an empty one and no instruction
  to be quiet in general catches it, and **an entity name is a search query only for
  names the web knows the way its owner does** — for the rest the search is discarded
  work and the line is an ordinary check-in.
- **The first proactive utterance ever spoken went to Telegram while the owner sat at
  the keyboard, and neither half was wrong (2026-08-26).** `gate_snapshot` on the row:
  `"why": "ok"`, `"delivery": "both"`, `idle_seconds: 0.06`, screen unlocked, mic and
  output free - the gate read presence correctly and asked for the speaker. The `route`
  column says `telegram`, and the log says why: `speaker: refusing to speak while this
  process holds the microphone`. `Speaker.say`'s rule is right (a speaker talking into a
  live gate is the gate hearing the daemon), the gate's reading was right, and the two
  had no way to say either thing to each other - so the failure was invisible from both
  sides and from the utterance row, which records the achieved route but not that a
  route was achievable. Fixed by `daemon/mic_floor.py`; live-verified on this Mac with a
  real capture stream and a real `say`, gate standing down and frames climbing again
  afterwards (290 -> 636). **Two things worth keeping from it.** A verdict field and an
  outcome column that disagree is not a lie either one told, and it is the shape to look
  for when a feature works in tests and does nothing in the room. And the acceptance
  tests written for the fix passed on a build where `_wake_round` never called the code
  they exercised - they drove the helper directly - which is `tests/CLAUDE.md`'s
  documented blind spot arriving on schedule; four seams were mutation-checked before
  the fix was believed.
- **Half-duplex was leaking the daemon's own voice into memory as the owner's words,
  and the tell is that every leak is a *tail* (2026-08-19).** `DAEMON_VOICE_BARGE_IN=
  false` was set and `apple audio: ... echo cancellation on` was in the log, yet
  `data/memory/log/2026-08-19.md` has user turns nobody said. They are not whole sentences. Against
  the assistant line before them:

  | spoken | filed as the owner |
  |---|---|
  | "당연하죠! **저도 응원하고 있을게요. 잘하고 오세요!**" | "원할 때 있을게요. 잘하고 오세요." |
  | "…알고 계시죠? **테크니컬 인터뷰니까 …오시면 돼요. 화이팅!**" | "테크니컬 인터뷰니까 …오세요. 파이팅!" |

  The opening clause is missing every time, and the rest is mangled the way a residual
  that got past echo cancellation is mangled — one leak came back as Spanish. So the
  microphone was not open for the whole answer; it opened *partway through* it. The
  gate was `self._generating`, which clears when the last audio chunk **arrives**, and
  `_on_audio` already measured the model generating faster than real time (28.4 s of
  audio in ~19 s). Seconds of answer were still queued at the speaker with the room
  live. `_playback_until` had tracked the drain all along for the idle budget; the
  microphone gate now reads it too.

  **The cost was not the stray rows.** They land under `inputTranscription`, i.e. as
  the owner, so the recent window fed the daemon its own last sentence as something to
  answer — and it answered by parroting. The owner's complaint that arrived first was
  about *tone*, not about echo: `재미난` in 8 of 17 assistant turns that day, and a
  filler word the daemon coined while apologising ("담백하게 가볼까요?") reused as a new
  tic 35 minutes later. An audio defect surfaced as a personality defect.
- **`realtimeInput.text` silently fails to generate a third of the time, and one
  trial said it was fine (2026-08-19).** The owner's report was "she stopped
  answering". `first audio` in the session report was bimodal - 1.41 s and 1.40 s
  in two morning sessions, then 22.74 s, 13.73 s, 12.34 s - and bimodal is the
  shape of a turn that either starts or does not, not of a model getting slower.

  Everything cheap was ruled out first, and each one cost a hypothesis:
  `evals/m0_voice_spike.py` put the provider at **setup 0.57 s, first audio 560 ms**;
  replaying the resident's exact opening sequence (time block, then the continuity
  tail, then `CALLED_BY_NAME`) against a live session came back at **1.10-1.13 s
  whatever was sent**, so context assembly was not it; 932 rows against a 12-row
  window ruled out the database; admin polls kept landing every 15 s through the
  stall, so the event loop was never blocked; and the installed build was byte-identical
  to the worktree in `_forward_microphone`, so the opening mic hold was really running.

  The distribution is what named it. 30 trials per arm, the resident's own opening:

  | frame | median | never answered |
  |---|---|---|
  | `realtimeInput.text` | 0.69 s | **10/30** |
  | `clientContent` + `turnComplete: true` | 0.66 s | **0/30** |

  Fisher exact p = 0.0008, identical medians. Turn end on the realtime stream is
  "derived from user activity", so whether generation starts is the server's
  activity detector's call; closing the turn explicitly takes that call away from it.

  **The reusable part is how the wrong answer got written down.** The spike listed
  this as one of six things only a live key could settle, ran it once, saw audio,
  and recorded it as settled. One trial cannot see a 1-in-3 failure - and the
  entry read as measured, so nothing revisited it. The same trap caught the
  investigation twice more on the way here: a first pass blamed
  `OPENING_ANSWER_HOLD_SECONDS = 6.0` on one plausible reading of the timings, and
  a second blamed the persona system instruction on a single 51.55 s outlier that
  the very next trial contradicted at 0.80 s. Anything that fails part of the time
  needs a denominator before it needs a fix.
- **Naming the tic works; the abstract instruction never did (2026-08-19).** The
  owner's complaint was `재미난` in 8 of 17 replies in a day. Told to stop, the
  daemon coined `담백하게` in the same apology and had made *that* a tic 35 minutes
  later. Both of the abstract levers were already in place and losing:
  `render_continuity`'s header says "do not imitate the style of these lines" in as
  many words, and `data/persona/seed.md` was given a rule against repeating itself. One
  sentence loses to twenty turns of evidence that this is how you talk.

  A/B on the live voice path, identical persona, time block and continuity tail
  from the owner's real log; the only difference is whether `persona/tics.py`'s
  block is sent:

  | probe | without | with | one-tailed p |
  |---|---|---|---|
  | wake word alone | 4/20 | 2/20 | 0.33 |
  | a conversational turn | **18/30** | **6/30** | **0.0017** |

  **The first row is why the second one exists.** The greeting probe answers in
  three or four words - `듣고 있어요.` - so there is nowhere for a tic to appear, and
  the run says nothing about the mechanism either way. Reported as a null it would
  have been a wrong conclusion drawn from a probe that could not have shown the
  effect. Pick a probe with room for the behaviour before believing a tie.

  The detector's two floors are measured, not chosen. **A tic is what the daemon
  repeats and the owner never says** - without that filter, an owner who talked
  about an interview all afternoon would get a daemon forbidden to say `인터뷰`.
  And the minimum length is three characters, because the real log's candidates
  ranked by any order at all open with `무슨`, `어떤`, `그럼`, `님이`: every one two
  characters, every one a word Korean cannot do without. Telling the daemon to stop
  saying `무슨` does not fix its manner, it breaks its grammar.
- **An optional scalar is not nesting, and treating it as one hid most of the tool
  set from voice (2026-08-19).** The owner asked his daemon to find an email. It
  listed Gmail labels, then invented a `gmail search "from:..."` shell command and
  ran it twice - `run_command ok=False` both times - and told him "메일을 검색하는
  도구에 문제가 생겼어요". The audit table has no `search_gmail_messages` row at
  all, because the tool was never offered.

  `is_flat_schema` refused any property carrying `anyOf`. That is what FastMCP and
  pydantic emit for **every defaulted argument**: `Optional[str]` becomes
  `anyOf: [{"type": "string"}, {"type": "null"}]`. `search_gmail_messages` is
  `query`/`user_google_email`/`page_size`/`include_headers` - all scalar - plus one
  `page_token=None`, and that single argument hid the whole tool.

  | server | tools | reached voice before | after |
  |---|---|---|---|
  | google | 27 | 9 | **17** |
  | git | 12 | 8 | **11** |
  | time | 2 | 2 | 2 |

  **The wall itself is real and was re-measured before being moved.** 20 trials per
  arm against the live model, offered `search_gmail_messages` with its real schema:
  the audio model emitted a correct call with both required arguments **20/20**,
  and **20/20** again with the `anyOf` folded to a plain type. The nesting the
  original measurement found is still refused - `send_gmail_message`'s `to` is
  `anyOf: [string, array, null]`, one address or a list, and it stays on the
  delegate path with every `*_batch` and every array argument.

  **What made this invisible for so long is that the failure was articulate.** A
  tool that is not offered produces a model that explains, plausibly, why the tool
  did not work - and then reaches for a shell. `daemon/tools/schema.py`'s gate had
  a test suite that agreed with it, because the tests were written from the same
  belief as the code. It took reading an actual MCP server's schemas to see it.

  **Still open:** `delegate_task` is the designed escape hatch for what stays
  nested, and it has been called **0 times in the whole history**. Voice hits the
  wall and confabulates instead of delegating.
- **"텔레그램으로 보내줘" had no tool behind it, and the model stalled politely rather
  than lying (2026-08-24).** Telegram is a door the daemon answers, not a place it can
  address: the only senders are the loop replying to whoever wrote, the proactive
  delivery, and the delegation report. Asked by voice to send a link, the model had
  nothing to call. The expectation, from
  [voice_write_nudge_spike.py](../evals/voice_write_nudge_spike.py), was a
  confabulated "보냈어". **Measured (`evals/voice_send_message_spike.py`,
  `gemini-3.1-flash-live-preview`, 80/81 tools, N=4 per arm, Korean TTS over the live
  audio path): it never faked the send. It asked a clarifying question every single
  run** - "어떤 링크를 말씀하시는 건가요?", once "텔레그램 정보가 필요해요, 아이디를
  알려주시면". So a missing capability does not always produce a lie; here it produced
  an unanswerable question, which reads to the owner as "못 하네" and is why this went
  unreported as a bug for so long. With a flat `send_message(text)` offered:
  **0/4 -> 4/4 called, 4/4 carrying the real content in `text`.**
- **`recipient_id=None` reached nobody, and 397 proactive utterances were dropped
  saying so (2026-08-24).** `channels/base.py` documents None as "no request to
  answer, so the channel delivers it to its configured owner". `TelegramChannel.send`
  implements that as `sorted(self._allowed)`, and `_allowed` is only ever the
  constructor's env list. Under `dm_policy=pairing` - this install - that list is
  **empty** and the approved owner lives in `channel_pairing` instead, so every
  unaddressed send raised `TelegramNoRecipient`. Counted in the resident's own log:
  **397 × "proactive: channel refused the utterance" between 2026-08-14 and
  2026-08-20**, plus the delegation reports, all swallowed by their callers'
  `except`/`logger` and therefore silent. `send_message` names the owner explicitly
  (`Store.owner_id`) rather than inheriting the bug; the two older callers still have
  it, and fixing them turns proactive Telegram messages back **on** at roughly that
  rate, which is the owner's call and not a refactor.
- **The 397 was a retry storm, not suppressed volume: the brakes had never once
  engaged (2026-08-24).** The entry above predicted that fixing the two older callers
  turns proactive Telegram messages back on "at roughly that rate". Measured against
  the resident's own database before shipping the fix: **`proactive_utterances` holds
  zero rows, and always has** (`MIN(spoken_at)`/`MAX(spoken_at)` are both NULL).
  `ProactiveDelivery.deliver` inserts the utterance, and when nothing reaches the user
  it calls `delete_utterance` and leaves the candidate live to try again - correct for
  a dead network, but it means a delivery that can *never* succeed erases its own
  evidence. Both rate limits read that table: `_budget_block` through
  `utterances_since()`, `_cooldown_block` through `last_utterance_at()`. An empty table
  means neither has ever fired on this install, which is why one day shows **167
  attempts against a daily budget of 8**. So 397/week is the signature of an
  unthrottled retry loop, not the volume the owner would receive. Post-fix the ceiling
  is `proactive_daily_budget` **8/day** (~56/week), with a 30-minute cooldown and
  23:00-09:00 quiet hours - and those three knobs start working for the first time,
  because a delivered utterance finally survives to be counted. The same emptiness
  explains the stalled label clock (PLAN 8.1): **zero labelable messages have ever been
  delivered**, so there was never anything to label.
- **Voice was answering "what's on my screen" from nothing, and the frame was the
  wrong transport rather than the wrong resolution (2026-08-24).** The owner's report
  was that Telegram answers screen questions accurately and voice answers them
  *plausibly*. `_deliver_images` had already fixed the version where voice sent the
  caption and no pixels; this was the layer under it. A `realtimeInput.video` frame
  sent between a `toolCall` and its `toolResponse` **never reaches the model at all** -
  read off `usageMetadata.promptTokensDetails`, which lists an `IMAGE` entry for a
  picture the prompt holds and nothing for one it does not. In a tool round it listed
  nothing at every gap tried, and the model named digits that were never on screen
  **25 times out of 25**. Outside a tool round the same frame does arrive (60 tokens,
  and it read a 24px code correctly) but needs **~1s** before the next client message -
  at 0.0/0.2/0.5s it silently does not land, which is why the live-share pump is fine
  and `see_screen` was not. Two fixes failed first, and both are worth keeping.
  (1) `mediaResolution: MEDIA_RESOLUTION_HIGH` raises a frame from 60 tokens to 522 and
  raises its *ingest* cost from ~1s to ~3s, so at a 1.0s gap the frame stopped arriving
  entirely - **a sharper frame that lands after the answer is worth less than a coarse
  one that lands before it**, and a field the socket accepts can still be a regression.
  (2) Widening the gap: 0.0s and 1.5s both scored 0/4. What worked is the JPEG as a
  `clientContent` image part - priced as an *image*, **1092 tokens against 60**, next
  to the 1120 the Telegram path gets - sent **after** the `toolResponse`. Order is the
  contract in both directions: sent *before* it, the `clientContent` cancels the
  pending call and the session goes silent, 4/4 at every gap. Measured through the
  product's own code, **0/20 → 19/20** (two runs, 12/12 then 7/8); restating the
  question in the image turn does not help. **Three wrong numbers were reported for this fix before the right one, and
  every one of them was a measurement bug rather than a behaviour** - the reason this
  file exists. (a) `Transcript.role` is `"assistant"`, and reading it as the wire's
  `"model"` scored five empty transcripts as five wrong answers. (b) Scoring the
  *first* transcript after the image scores the answer the image turn exists to
  interrupt: the real sequence is `[interrupted] -> "6423입니다" -> "114170입니다"`, the
  invention then the correction, so stopping at the first read the invention. That
  reported 2/10, then 8/15, against an actual 12/12 - and it was believable, because
  "a real but weak improvement" is exactly what a half-working transport looks like.
  **A plausible partial result is the dangerous one; only the raw socket disagreeing
  with the harness (5/6 against 2/10) exposed it.** What is left after all that is one
  genuine defect and it is not accuracy: the owner *hears* the invented answer before
  the correction in **5 of 12 turns**, because a `toolResponse` starts generation
  server-side. Telling the model to wait for the image changed that to 5 of 12, the
  same count to the trial, so the wording was dropped rather than kept as decoration -
  the lever is playback on our side, not a prompt. A published GitHub issue claiming
  `clientContent` + `inline_data` closes this model with 1007 did not reproduce on a
  correctly-shaped raw payload; it was the library's bug, not the API's, which is again
  the whole reason to ask the socket. `evals/screen_frame_arrival_spike.py`.
- **Dating a learned rule by its age and observation count was reverted: three
  independent measurements against the live model found no detectable effect,
  and the one run that looked significant did not replicate.** The idea
  (`daemon/persona/loader.py::rule_line`, `e34785d`/`d53be62`/`f7593fc`/
  `22255b9`/`586d804`, all reverted) was that a rule formed from a single
  terse-QA exchange should carry less weight in the prompt than one built
  from many repeated observations, so a stale one-off correction fades while
  a real, repeatedly-confirmed preference holds. It looked right in the first
  hand-audited three-arm run (the spike script, n=30 each, removed here but
  recoverable from `ee2801a`): the stale rule's dominance trended down
  (17/30 → 13/30) while the real preference held - **30/30 → 28/30, and it
  never dropped below that in this run or any later one, so the mechanism was
  never a regression to weigh against. It was inert, not harmful.**

  **Two defects in the probe had to be found before arm 1's number meant
  anything.** First, its judge demanded a reply clear four clauses at once (no
  jokes, no affection, no self-disclosure, no question back) to count as
  "terse", and this persona never clears all four together even while
  visibly shortening - so it answered "no" unconditionally regardless of
  which arm was dated. Replacing that with reply length against the two arms'
  *pooled* median fixed the floor/ceiling problem but created a second one:
  the two hit-counts became structurally near-complementary (they sum to
  about n), so a Fisher exact over them was scoring roughly one coin flip
  about which arm landed on the short side, not a real difference - caught
  only because an identical second n=30 run reversed the sign (13/30 → 19/30
  against the first run's 17/30 → 13/30). `ee2801a` replaced it with the two
  reply-length distributions compared directly: median and mean per arm, a
  rank-sum test, and a two-sided permutation p-value over 20000 shuffles.
  That is the metric every number below used.

  With the corrected metric: n=30, the dated arm was shorter (p=0.40); n=60,
  it was longer (p=0.00065); a second n=60 replication came back shorter
  again (p=0.34). Pooled, 120 vs 120, the medians are identical at **99.0**
  (p=0.107). No effect survives pooling, and the single run that cleared
  conventional significance did not replicate - it was noise a large enough
  one-off sample can produce, not a real effect too small to see at n=30.

  **Arm 3 (does a manner remark leak from `facts` into `observations` under
  reflection's old prompt) never reproduced on the real 2026-08-19 incident
  day: `facts` 0/30 under both the old and new prompt.** So `70c6a37`'s
  `facts`/`observations` boundary closes a path rarer than the incident that
  prompted it implied - and it stays anyway, being one paragraph of prompt
  wording whose cost was not measured: `facts` was 0/30 under both prompts,
  so the run had no positive `facts` control to weigh a cost against, unlike
  the dating mechanism it shipped alongside and which this entry retires.

  **A real defect the same hand audit turned up: on 2 of 60 records, the
  model returned `observations` as a bare list of strings instead of
  `{"body": ..., "confidence": ...}` objects.** No test caught this - it
  surfaced only from reading the spike's raw replies by hand - and
  `reflection.py::_items` treated every such entry as "not an object" and
  dropped it, discarding that whole night's persona signal silently. Fixed by
  making `_items` recover a bare string as `body`, falling back to the
  schema's own defaults for everything else. The `facts`/`observations`
  boundary above routes more content into `observations`, so this parsing gap
  gets more consequential, not less.

  **What a retry would need first, if anyone repeats this.** The spike runs
  all of one arm and then all of the other, so arm is confounded with
  whatever drifted between the two blocks - model routing, load, anything
  else that changes over the run's wall-clock span. Every number above
  inherits that confound. Interleaving the two arms trial-by-trial, not a
  larger n, is what a retry needs before its p-value means anything either.
- **The wake->voice handover had a 107 ms hole, and CoreAudio deadlocked in it
  (2026-08-26).** The owner's report was "she suddenly stopped answering, and
  calling again does nothing". The resident had logged `wake: heard '벨라'` and
  `opening a voice session` at 15:07:26 and then never another voice line - while
  Telegram kept answering, the scheduler kept ticking and `/health` kept saying
  `status: ok`, `wake_gate: running`. The tell was one missing line: every working
  wake round logs `apple audio: engine at 48000 Hz ...` within 0.3-0.6 s of opening
  the session, and this one never did, so the wedge was inside `_build`, before the
  log.

  `sample` of the resident named it. Four threads, all at `__psynch_mutexwait` for
  100% of a 3 s sample, in a closed cycle:

  | thread | doing | waiting on |
  |---|---|---|
  | `com.apple.audio.IOThread.client` | PortAudio `startStopCallback` -> `AudioUnitGetProperty` | the AudioUnit recursive mutex |
  | `AVAudioIOUnit` queue | property listener -> `_GetHWFormat` -> `GetPropertyDataSize` | the HAL mutex, held by the row above |
  | `engine` queue | `AVAudioEngineImpl::IOBindingChanged` | the AVAudioEngine mutex |
  | `voice-mic-release` (Python) | `Pa_StopStream` -> `AudioDeviceStop` | the HAL mutex |

  So the wake gate's PortAudio stop and the session's VoiceProcessing engine were
  running **at the same time** and took the two locks in opposite orders. v0.1.45
  and v0.1.47 had moved the stop and the open onto detached threads, which is why
  the daemon stayed up instead of freezing - it converted a total freeze into a
  daemon that is alive and permanently deaf, which `/health` cannot tell from a
  quiet house.

  **Why they overlapped is the part worth keeping.** `_wake_round` broke out of
  `async for event in gate.listen()`, and breaking out of an `async for` does not
  finalise the generator - CPython drops the last reference and schedules `aclose()`
  for a later loop turn. `close_gate()` returned in **1 ms** (measured: the two log
  lines are 15:07:26,499 and ,500) because `SoundDeviceAudio.close()` only ever
  closed the *speaker*. The microphone was still being let go, by a thread that had
  not started yet, while `run_voice` was already building the engine.

  **The hole is 107 ms wide.** Measured on the owner's Mac against real PortAudio,
  with the resident holding the device too: `aclose()` returns at the same
  millisecond the first block arrives, and `wait_for_input_release()` then takes
  **0.107 s** for the stop and close to actually finish. That is the whole race
  window, and nothing was waiting on it. With no competing client a stop is that
  fast, which is also why the 2 s bound on the wait is a wedge detector rather than
  a latency cost.

- **An ordinary spoken turn had the same unguarded window the opening and the tool
  round had each already been given (2026-08-26).** The owner's report was "she
  goes quiet, and if I ask again she answers". Counted off the day's own
  conversation log rather than from feel: **5 of 47 spoken turns got no answer at
  all** (11%), and 6 more answers were cut off mid-sentence. The distribution is why
  it reads as random - per session 1, 1, 0, 0, 3, 0 - and why a good session proves
  nothing: at 11% a clean six-turn session is a coin flip (0.89^6 = 50%), which is
  exactly the session that arrived while this was being investigated, with the owner
  reporting it as fixed when nothing had been changed.

  Between the owner's transcript settling and the model's first audio, the
  microphone was still streaming the room to the server, which reads it as the owner
  opening a *new* turn and cancels the one it was composing. Two shapes, one cause:
  cancelled **before** the first chunk it is silent and leaves no trace at all -
  `gemini_live._decode_content` only raises `Interrupted` while it is already
  generating - and the 06:00 session proves that half, **7 turns, 3 unanswered, and
  its own report saying `0 interruption(s)`**. Cancelled **during** playback it
  truncates the answer and does log a barge-in. The leaked tails then land under
  `inputTranscription`, i.e. as the owner: `los ladros`, `the lock`, `ella` - the
  same mangled-residual signature as the 2026-08-19 entry above, Spanish and all.

  `_answer_hold_until` (was `_opening_answer_until`) is now armed on every settled
  owner transcript, not only the wake-word opening - except on a turn the owner
  *barged in* with, where the answer to the previous turn is still playing. That
  exception is a limit, not a gap left open by accident: one microphone cannot be
  both open for the interruption and shut for the answer that follows it. So
  half-duplex (`DAEMON_VOICE_BARGE_IN=false`, what these numbers were taken under)
  is fully covered and the default is not, and **nobody has measured what the
  uncovered case costs** - a barge-in-on day would need its own count.

  **Not yet measured live**: the
  fix is argued from the two cases already fixed the same way, and the honest check
  is the same 30-trial-per-arm shape the `realtimeInput.text` entry above needed -
  count unanswered turns, do not run one session and call it settled.
- **The admin's restart button hung 6 of 8 times on one day, and what it left behind
  was not a stopped daemon but a live one with no HTTP surface.** Read off the
  resident's own log for 2026-08-26: eight `Shutting down` lines, and six of them
  reached `Waiting for connections to close.` and never printed another word. The
  gaps to the next `Application startup complete` were 16s, 27s, 29s, 29s, 3m and
  **40m** — every one of them ended by something external, never by the process.
  Meanwhile the stuck process kept working: at 17:03:32, eighty seconds into a
  shutdown it never finished, it woke on '연락' and opened a full voice session.
  The wake loop, the embedder and Telegram's long poll all survive a graceful
  shutdown; only the listening socket does not. So the symptom the owner sees is
  the console frozen on "applying…" (its `pollBack` retries `/health` forever, and
  `/health` stopped listening), while the daemon is still talking to them.

  **Cause: one endpoint that never ends, plus a wait with no bound.** `/face/stream`
  is server-sent events, so its response is open for as long as the face page is.
  Uvicorn's `connection.shutdown()` cannot close a connection whose response is
  still open — it clears `keep_alive` and waits — and `timeout_graceful_shutdown`
  defaults to `None`, i.e. `while self.server_state.connections: await sleep(0.1)`,
  forever. One open face page is enough. Reproduced deterministically against real
  uvicorn and the real handler: no stream open, exit in **0.18s**; one stream open,
  **still alive after 15s**; `timeout_graceful_shutdown=3`, exit in **3.18s**.

  **The bound alone was a fix that logged an error on every success, so it is the
  backstop and not the mechanism.** Reaching it makes uvicorn print `Cancel 1
  running task(s), timeout graceful shutdown exceeded` at **ERROR** — and
  `daemon/cli.py::_LOG_NOISE` does not filter it (checked), correctly, since that
  filter exists to never hide a real error. So the owner's normal settings save
  would have printed an ERROR line meaning "working as designed", every time,
  which is the fastest way to teach someone to stop reading them. Measured on the
  assembled app with a face page open: bound only → **3.19s** and that ERROR line;
  `FaceBus.close()` from `admin/restart.py::schedule_exit` before the signal →
  **0.40s**, and `Waiting for connections to close.` never appears at all. The
  bound stays for the SIGTERMs no endpoint sees coming — `launchctl`, logout,
  `daemon update` — and for the next endless response somebody adds.

  **Where the close check goes is a two-sided constraint, and both sides were
  measured.** In `FaceBus._events`, after the wait it deadlocks (the wake that
  carried the close has already been cleared and nothing will set it again); after
  the yielded batch it drops whatever was published in the same tick as the close
  — a one-shot queued just before `close()` never reached the page, caught by a
  test written before the placement was. It goes at the top of the loop, guarded on
  an empty mailbox.

  **The tell, for next time: a shutdown that stops logging is not a shutdown that
  finished.** `Waiting for connections to close.` is the last line either way, and
  nothing downstream of it says which. `Application shutdown complete.` is the line
  that means the process is actually leaving; grep for its *absence*.
- **The resident cannot see homebrew, so `shutil.which` is not binary discovery.**
  Measured 2026-08-26 on the live `ai.daemon.default` job: `PATH` is
  `/usr/bin:/bin:/usr/sbin:/sbin` while `ollama` is at `/opt/homebrew/bin/ollama`.
  `_render_plist` omits `EnvironmentVariables` deliberately (`service.py:227` - the
  working directory is how the process finds `.env`) and `launchctl getenv PATH` is
  unset, so nothing puts homebrew's bin back. `which("ollama")` therefore resolves
  in every terminal test and returns `None` in the service the code actually runs
  in - a soft dependency that then degrades silently instead of failing.
  `ollama_process.py:BINARY_FALLBACKS` is the consequence.
