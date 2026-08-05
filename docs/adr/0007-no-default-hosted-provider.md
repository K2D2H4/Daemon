# 0007 — The hosted provider has no default

**Status:** accepted · 2026-08-03

## Context

The presets named Anthropic directly for hosted conversation. So "provider-agnostic
gateway" was true of the config surface and false of the product: someone who
wanted GPT or Gemini had no way to say so.

Fixing that by adding a `DAEMON_HOSTED_PROVIDER` setting with `anthropic` as its
default fixed less than it looked. A person who had run setup before the question
existed had no provider in their file, silently got Claude, and asked why it was
still hard-wired. **Someone who was never asked cannot tell a choice from a
fallback.**

## Decision

Two axes, not multiplied: a **preset** answers *where work runs*,
`DAEMON_HOSTED_PROVIDER` answers *whose model*. Three presets, not nine.

And no default. A preset that needs a hosted model without one **fails at startup**
naming `daemon setup`. `offline` needs none. Anything enumerating what a
configuration requires drops hosted tasks rather than guessing — guessing is how
someone who chose GPT gets asked for an Anthropic key.

`Task.CHAT_VOICE` is exempt and pinned to Gemini: in native audio the model is both
the brain and the voice, so pointing it at a provider without a voice session would
fail on the first voice turn instead of at startup.

## Consequences

Consistent with the rest of the configuration, which dies loudly rather than
degrading quietly — a silent switch to a weaker model is worse than an error.
Onboarding always runs, so the answer always exists by the time it is needed.

`Task.PROACTIVE_JUDGE` stays local whatever is chosen: it runs every five minutes
whether or not it ever speaks, so hosted cost would accumulate for nothing.

## What would change our mind

If a preset ever ships that is obviously tied to one provider's capability. Not the
case today; all three answer text.
