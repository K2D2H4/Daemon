# daemon/ — measured, not assumed

Every line here is something this project believed until it ran it. Read it before
optimising, before trusting a documented probe, and before repeating a measurement
that is already here. Orientation: [CLAUDE.md](CLAUDE.md).

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
