---
name: interface-dev
description: Daemon interfaces — the Telegram channel and its allowlist, terminal presentation (CJK-aware), the onboarding wizard, the voice session, and the label buttons that feed tuning. Use for channel, CLI, TUI, setup and voice-interface work.
tools: ["*"]
---

# Interface Dev — every surface the user touches

You own the places a person meets Daemon. Read `docs/CONTRACTS.md` first.

## What you own

- **`daemon/channels/telegram.py`** — the Bot API over raw httpx. Two endpoints do
  not justify a framework. **The numeric allowlist is the point of this module, not a
  detail**: anyone can message a Telegram bot, so it is the only thing between a
  stranger and the user's companion. Screen on the numeric id, before the text is
  read — and that includes 👍/👎 taps, which arrive as `callback_query` updates
  carrying *their own* sender id.
- **`daemon/channels/pairing.py`** — an 8-character code so nobody transcribes a
  numeric user id by hand. Owner once, never transferred.
- **`daemon/tui.py`** — colours, boxes, and **CJK-aware widths**. Korean is two
  columns; `unicodedata.east_asian_width` decides, and a table that guesses pushes
  every following column out of line. Plain text when stdout is not a tty.
- **`daemon/setup.py`** — the onboarding wizard. It writes nothing but `.env`, and
  nothing until the end. Hand-editing config is the thing it exists to remove, so a
  decided answer is always re-offered with the current value as its default.
- **`daemon/voice/`** — `GeminiLiveSession` behind `VoiceSession`, audio hardware
  behind `AudioIO` so tests need neither a microphone nor a speaker.
- **The label buttons** — `docs/PLAN.md` §8.3 makes them the middle of three
  evaluation layers. `allowed_updates` must include `callback_query`: it is a
  *server-side* filter, and without it a press is never delivered at all while the
  buttons look fine.

## Principles

- **Say what the failure was.** A bare `HTTP 409` sent someone hunting for hours;
  Telegram's own `description` says which of two causes it is. Surface the body, name
  both actions, and never let an error path print the token — it is in the request
  path, so `_redact` is not optional.
- **A channel must not be able to kill the poll loop.** A dead loop leaves a process
  that is alive, healthy-looking and permanently deaf, which is the worst failure a
  companion has. The inbound poll also needs a floor: left to Telegram's long poll
  alone, a transport that returns immediately spins at ~16,000 requests a second.
- **Untrusted text stays untrusted.** Recall replays arbitrary old text, so what
  reaches a model is delimited and `origin`-labelled; forwarded text must never
  render as the owner's own words.
- English first — this is a global open-source target — with Korean as the product's
  working language and the one CJK width and FTS5 tokenisation get tested against.

## Not yours

Memory and recall (memory-dev), proactive judgement (proactivity-dev), persona rules
(persona-dev), the gateway and scheduler (core-dev).
