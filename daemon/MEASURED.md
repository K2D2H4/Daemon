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

  Residual, stated because the number is small: only 12 voice tool calls fall after the fix,
  so "zero" here is weak evidence rather than a clean bill of health.
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
