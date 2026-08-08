# 0009 — Images in the Message contract

**Status:** accepted · 2026-08-07

## Context

Screen sharing needs the model to see the owner's screen, not just be told about
it. Two shapes were on the table: describe-via-submodel, where a vision model
turns the screenshot into a text caption and only the caption reaches
`daemon/llm/gateway.py`'s chosen provider; or true multimodal, where the JPEG
itself travels as part of the `Message` history and the model that answers is
the model that looks.

The describe path keeps `Message` untouched, which is the whole appeal - no
frozen file moves. But it costs a second model call on every screen turn, adds
its latency to a path voice already measures in milliseconds (0004), and
replaces the actual screen with somebody's summary of it - the owner asking
"what does this error dialog say" gets a caption's guess at the text, not the
text.

## Decision

**Option 1: the raw image reaches the model's own context.** Fidelity is the
argument that wins outright - the model sees exactly what is there, including
layout and which button the owner is pointing at, not a lossy paraphrase - and
there is no second call to place on the latency path or fail independently of
the first.

Two frozen files carry it, both additively:

- `daemon/llm/base.py`: `Message` gains `images: tuple[ImageBlock, ...] = ()`
  and a new `ImageBlock` dataclass, defaulted exactly like `tool_calls` and
  `ToolCall.provider_signature` already are on the same class. Every existing
  `Message(...)` call keeps compiling unchanged - that default is what makes
  touching a frozen file here safe rather than a rewrite of every call site.
- `daemon/voice/base.py`: `VoiceSession` gains `async def send_frame(jpeg:
  bytes) -> None`. A new protocol method, not a change to an existing one; the
  only implementation obliged to satisfy it yet is the Gemini Live session.

## The offline degrade

`daemon/config.py`'s `offline` preset routes `Task.CHAT_TEXT` to Ollama - a
local model, chosen for the privacy promise in PLAN §7, with no guarantee it is
a VLM. Sending it images it cannot read is not a smaller version of the
feature; it is silent failure wearing the feature's clothes, either an error
the preset never expected or a model quietly answering as if it saw nothing.
Screen sharing is a hosted-model / VLM capability. On a non-VLM local model it
is unavailable, and the code must say so - "I can't see the screen" - rather
than attach a frame and hope. This ADR is what holds that line when the later
task writes the code that enforces it.

## Security stance

Every screenshot is preceded by `screen_note` (`daemon/tools/screen.py`,
Task 0.2), the image-equivalent of `daemon/tools/browser.py`'s `fence`: a
plain statement that what follows is DATA, not instruction, so on-screen text
addressing the model gets reported rather than obeyed. It cannot do what
`fence` does for page text - nonce-delimit the content itself - because there
is no way to embed a matching end marker inside a JPEG for the model to read
back out as a boundary. That gap is accepted with mitigation, not assumed
away: on-screen text could still try to steer a `safe` tool, and
`screen_note`'s docstring says so in the same voice
`daemon/tools/screen.py`'s module docstring already uses for the capture's
own residual risk.

## Consequences

`daemon/llm/providers/` gains one job per provider that supports vision:
translate `ImageBlock` into that provider's wire shape. A provider that never
sees a non-empty `images` tuple - Ollama on `offline`, and any provider before
its own task lands - needs no changes at all, which is the point of a
defaulted field on a frozen contract.

## What would change our mind

A measured describe-path that was materially cheaper or safer at fidelity the
owner could not tell apart from the original - PLAN's habit of trusting the
socket over the inference applies here too, so this is a claim to re-check
against a real VLM-vs-caption comparison, not a closed question. Also: a
future contract that let an image-carrying `Message` leak into a place images
were never meant to reach (recall's markdown log, the reflection pass) would
mean the field needs a narrower home than `Message` itself.
