"""Can voice actually see the screen it was just asked about? Ask the socket.

The owner's report: Telegram answers questions about the screen accurately, voice
answers them *plausibly* - roughly right, wrong in the details. `_deliver_images`
(daemon/voice/conversation.py) had already fixed the version where voice sent the
caption and no pixels at all. This is the layer under that one, and it is the
regression measurement for the ordering fix that came out of it.

**What was wrong.** A `realtimeInput.video` frame sent between a `toolCall` and its
`toolResponse` never reaches the model at all. Measured against the raw socket by
`usageMetadata.promptTokensDetails`, which lists an `IMAGE` entry for a picture the
prompt actually holds and nothing for one it does not - so "the model never saw it"
can be told apart from "the model saw it and misread it". In a tool round it listed
nothing, at every gap tried, and the model answered a "what number is on my screen"
question with fluent Korean naming digits that were never there, 4/4. Outside a
tool round the same frame does arrive (60 tokens; it read a 24px code correctly),
which is why this looked verified and why the live-share pump still uses it.

**Two fixes failed before the third worked, and both failures are worth keeping.**

  1. `generationConfig.mediaResolution: MEDIA_RESOLUTION_HIGH` raises a frame from
     60 tokens to 522 - and raises its *ingest* cost from ~1s to ~3s, so at a 1.0s
     gap the frame stopped arriving at all. A sharper frame that lands after the
     answer is worth less than a coarse one that lands before it.
  2. Closing the gap. 0.0s and 1.5s both scored 0/4, both with confident wrong
     digits. The gap is not the variable; the tool round is.

**What worked.** The pixels as a `clientContent` image part - priced as an *image*,
1092 tokens against 60, the same order of magnitude as the 1120 the Telegram path
gets - sent *after* the `toolResponse`. Order is not a detail: sent before it, a
`clientContent` cancels the pending call and the session says nothing at all (4/4,
every gap). The interrupt that makes `clientContent` dangerous for recall is wanted
here: it cuts off the answer the model was about to invent from a caption alone.

Measured 2026-08-24 through the product's own code, two runs of 12 and 8 trials an
arm: **0/20 against 19/20**. Restating the question in the image turn does not help -
the model answers the question it is already holding. Quote the pooled number, not
the 12/12 the first run happened to give.

With `all_displays`, every captured monitor is one part of that single turn - four
of four two-display trials read both codes, and `_deliver_images` explains why a
turn per image would not.

**What is left is not accuracy.** In 5 of those 12 turns the model spoke an invented
answer before the correction, because a `toolResponse` starts generation server-side
and the image turn arrives to interrupt something already being said. Counted here
as `spoke-a-wrong-answer-first`, because it is what the owner hears even when the
answer that survives is right. `conversation.SCREENSHOT_FOLLOWS` records the caption
wording that failed to suppress it.

So the arms below are the old order and the new one, both driving the product's own
`_deliver_images` / `_caption_only`. Ground truth is a 6-digit code at ordinary UI
size, so "read it" is a fact and not an impression.

    python3 -m evals.screen_frame_arrival_spike              # 4 trials an arm
    python3 -m evals.screen_frame_arrival_spike --trials 15

**`_deliver_images` is called unbound, with `None` for `self`**, so this measures
the product's own handover rather than a copy of it. That works because the method
uses no instance state, and its docstring says so - if it ever needs `self`, fix
this call in the same change.

**Read `Trial.after_image` before changing the scoring.** The image turn interrupts
the answer the model composed from the caption alone, so a reader that scores the
first transcript scores the invented answer and never sees the corrected one. That bug
reported this fix as 2/10 and then 8/15 against an actual 12/12, and it was
believable both times, because "a real but weak improvement" is what a half-working
transport looks like. What exposed it was the raw socket disagreeing with the
harness. It is the second measurement mistake in this investigation, after reading
`Transcript.role` as `"model"` when it is `"assistant"`.

**Nothing here runs in CI and nothing here is a test.** A test may not touch the
network or a key (tests/CLAUDE.md). The frame is synthetic, so no screenshot of the
owner's machine leaves it, and the key is only ever read from the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import random
import re
import sys

CODE_LABEL = "CODE"
FRAME_SIZE = (1536, 960)
"""`DAEMON_SCREEN_MAX_PX`'s default long edge, 16:10 - what `see_screen` hands over."""

FONT_PX = 24
"""Ordinary UI label size. Deliberately not smaller: at Pillow's ~11px default
bitmap font the default resolution cannot read the code *even when the frame
arrives*, which measures the resolution question and hides the arrival one. 24px
is legible at 60 tokens once the frame is in the prompt - proven by the gap=1.0s
row above - so a miss here is an arrival failure and not an acuity failure."""

QUESTION = "내 화면에 CODE 라고 적힌 숫자가 뭐야? 숫자만 말해줘."
"""Korean, because that is the language the owner asks in and the one the report
was made about. It also forces the tool: the model has no other way to answer."""

CAPTION = f"captured the main display ({FRAME_SIZE[0]}x{FRAME_SIZE[1]})"
"""Verbatim what `SeeScreen.run` returns, so the model reads the same caption in
this spike as in production - including the part that asserts a capture happened."""

TURN_BUDGET_SECONDS = 60.0
MAX_TURNS = 4


def _frame(code: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", FRAME_SIZE, (28, 30, 34))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", FONT_PX)
    draw.rectangle([(0, 0), (FRAME_SIZE[0], 44)], fill=(46, 49, 56))
    draw.text((16, 14), "Account settings", fill=(210, 214, 220), font=font)
    draw.text((60, 300), CODE_LABEL, fill=(150, 156, 165), font=font)
    draw.text((60, 340), code, fill=(245, 245, 245), font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _spec():
    from daemon.llm.base import ToolSpec

    return ToolSpec(
        name="see_screen",
        description=(
            "Look at the owner's screen right now - a screenshot of what is on "
            "their display - so you can talk about what they are looking at."
        ),
        parameters={"type": "object", "properties": {}},
    )


ORDERS = ("frame-then-response (the old bug)", "response-then-image (what ships)")


class Trial:
    """One `see_screen` turn under one ordering. Printed, not consumed."""

    def __init__(self, order: str, code: str) -> None:
        self.order = order
        self.code = code
        self.error: str | None = None
        self.called = False
        self.said = ""
        self.after_image: list[str] = []
        """Every transcript that arrived after the image went out, in order, because
        there is normally more than one and only the last is the answer.

        The image turn interrupts the answer the model was already composing from
        the caption alone - that is the point of it - and the interrupted fragment
        is itself delivered as a transcript. So the real sequence is
        `[interrupted] -> "6423입니다" -> "114170입니다"`: the invented answer, then
        the corrected one. A reader that stops at the first post-image transcript
        scores the invention. That mistake reported this fix as 2/10, then 8/15,
        while reading the whole tail put it at 5/6."""
        self.spoke_first = ""
        """The invented answer the owner *hears* before the correction, when there
        is one. Not a scoring failure - a real defect, one layer up from this
        one."""
        self.read_it = False
        self.blind = False


async def _one_trial(api_key: str, model: str, order: str, rng: random.Random) -> Trial:
    from daemon.llm.base import ImageBlock, ToolCall
    from daemon.tools.base import ToolResult
    from daemon.voice.base import Transcript
    from daemon.voice.conversation import VoiceConversation, _caption_only
    from daemon.voice.gemini_live import GeminiLiveSession

    code = "".join(rng.choice("0123456789") for _ in range(6))
    trial = Trial(order, code)
    jpeg = _frame(code)

    delivered: bool | None = None
    session = GeminiLiveSession(api_key=api_key, model=model, tools=[_spec()])
    try:
        async with session:
            await session.send_text(QUESTION)
            for _ in range(MAX_TURNS):
                got_anything = False
                async with asyncio.timeout(TURN_BUDGET_SECONDS):
                    async for event in session.receive():
                        got_anything = True
                        if isinstance(event, Transcript) and event.role == "assistant":
                            trial.said += event.text
                            if delivered is not None:
                                trial.after_image.append(event.text)
                        elif isinstance(event, ToolCall):
                            trial.called = True
                            results = [
                                ToolResult(
                                    call_id=event.id,
                                    name=event.name,
                                    content=CAPTION,
                                    images=(ImageBlock(jpeg, "image/jpeg"),),
                                )
                            ]
                            if order is ORDERS[0]:
                                # What used to ship: pixels as a realtime frame,
                                # before the response. Reconstructed here rather
                                # than read from the product, which no longer does
                                # it - an arm has to be able to reproduce the bug.
                                await session.send_frame(jpeg)
                                await session.send_tool_response(results)
                                delivered = True
                            else:
                                # The product's own code, and its order.
                                await session.send_tool_response(_caption_only(results))
                                await VoiceConversation._deliver_images(None, session, results)
                                delivered = True
                # Only a genuinely empty `receive()` ends this. Stopping on the
                # first post-image transcript reads the answer the image turn
                # exists to interrupt - see `Trial.after_image`.
                if not got_anything:
                    break
    except Exception as exc:  # noqa: BLE001 - a spike reports, it does not raise
        trial.error = f"{type(exc).__name__}: {exc}"
        if api_key and api_key in str(exc):
            trial.error = f"{type(exc).__name__}: <redacted, the error carried the key>"

    tail = trial.after_image[-1] if trial.after_image else trial.said
    trial.read_it = code in re.sub(r"[^0-9]", "", tail)
    if trial.read_it and len(trial.after_image) > 1:
        earlier = " ".join(trial.after_image[:-1])
        digits = re.sub(r"[^0-9]", "", earlier)
        if digits and code not in digits:
            trial.spoke_first = " ".join(earlier.split())[:40]
    lowered = tail.lower()
    trial.blind = any(
        phrase in lowered
        for phrase in ("can't see", "cannot see", "don't see", "안 보", "보이지 않", "못 보")
    )
    return trial


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=4, help="trials per arm")
    parser.add_argument("--model", default="")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is not set; nothing to ask.", file=sys.stderr)
        return 2

    from evals.m0_voice_spike import RECOMMENDED_MODEL

    configured = os.environ.get("DAEMON_GEMINI_LIVE_MODEL", "").strip()
    model = args.model or configured or RECOMMENDED_MODEL

    rng = random.Random(args.seed)
    print(f"model: {model}")
    print(f"key:   ...{api_key[-4:]} (never printed in full, never written anywhere)")
    print(f"frame: {FRAME_SIZE[0]}x{FRAME_SIZE[1]}, {FONT_PX}px code, {args.trials} trials an arm")
    print()

    results: list[Trial] = []
    # Alternating rather than blocked: a socket that gets slower over the run must
    # not land its whole effect on one arm.
    for round_number in range(args.trials):
        for order in ORDERS:
            trial = await _one_trial(api_key, model, order, rng)
            results.append(trial)
            if trial.error:
                mark = f"ERROR {trial.error}"
            elif not trial.called:
                mark = "NO TOOL CALL"
            elif trial.read_it:
                mark = "read"
            elif trial.blind:
                mark = "said-blind"
            else:
                mark = "WRONG"
            tail = trial.after_image[-1] if trial.after_image else trial.said
            said = " ".join(tail.split())[:44]
            if trial.spoke_first:
                said = f"(said {trial.spoke_first!r} first) {said}"
            print(f"  {round_number + 1} {order:36} want {trial.code}  {mark:12} {said!r}")

    print()
    for order in ORDERS:
        arm = [t for t in results if t.order == order]
        read = sum(1 for t in arm if t.read_it)
        blind = sum(1 for t in arm if t.blind and not t.read_it)
        wrong = sum(1 for t in arm if not t.read_it and not t.blind and t.called and not t.error)
        no_call = sum(1 for t in arm if not t.called and not t.error)
        print(
            f"{order:36} read {read}/{len(arm)}  said-blind {blind}  "
            f"wrong-digits {wrong}  no-tool-call {no_call}  "
            f"errors {sum(1 for t in arm if t.error)}  "
            f"spoke-a-wrong-answer-first {sum(1 for t in arm if t.spoke_first)}"
        )
    print()
    print(
        "`wrong-digits` is the failure the owner reported: digits spoken confidently "
        "that were never on the screen. `said-blind` is the honest version of the "
        "same miss. A `no-tool-call` trial measures nothing - discount it rather "
        "than scoring it. The first arm is expected to fail; it is here so a "
        "regression in the second one is legible as a return to it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
