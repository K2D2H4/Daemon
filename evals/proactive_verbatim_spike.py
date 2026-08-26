"""Will the live voice model say a given sentence **exactly**, or reword it?

The daemon speaks first through `/usr/bin/say` (`daemon/proactivity/speaker.py`),
which is a different engine and a different voice from the one the owner picked for
conversation - so a proactive line arrives in a system TTS voice that is not hers.
The owner's report, 2026-08-27: `선제발화가 tts처럼 나오는데 이거맞음...?`

The cheapest fix is the session that already opens right after she speaks (PR #115):
let *it* say the line, in her own voice, at no extra cost. PR #115 rejected that on
the grounds that `opening_text` is a prompt the model answers, so the line would come
out as the model's paraphrase - and the sentence the judge length-capped and refused
for URLs would not be the sentence the room heard.

**That objection is about wording, and wording is measurable.** This measures it.

## What is actually at risk

Not URL injection: the search titles never reach the voice session (they exist only
in the judge's prompt), so the model cannot speak a pointer it has never seen. What
is at risk is that `proactive_utterances.text` - the row the owner's 👍/👎 attaches
to, and the line every later measurement quotes - stops being what was said. That is
the same verdict-and-outcome mismatch this branch spent PR #115 removing, and it is
why this is measured rather than argued.

## Design

Two cells, same line, same persona, N sessions each. The only thing that varies is
what the opening asks for:

  A. baseline - `CALLED_BY_NAME`-shaped instruction to speak in character
  B. verbatim - the same, plus "say this sentence exactly, changing nothing"

Scored on the assistant transcript against the intended line, three ways, because
"the same" has three useful meanings here:

  exact       - identical after stripping whitespace and trailing punctuation
  near        - identical after also removing all spaces and punctuation
  substantive - the line's content words all survive, in order

`near` is the bar that matters. Korean TTS transcripts routinely differ from the
sent text by a space or a comma, and a spike that demanded byte equality would
report failure for a session that said the sentence perfectly.

    cd ~/Daemon && python3 -m evals.proactive_verbatim_spike            # N=8 per cell
    cd ~/Daemon && python3 -m evals.proactive_verbatim_spike --runs 15

Needs GEMINI_API_KEY (+ DAEMON_GEMINI_LIVE_MODEL). No microphone, no speaker: the
line is handed over with `send_text`, which is the production path a proactive
opening would use. Not a test - it needs a key and the network (tests/CLAUDE.md).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TURN_BUDGET_S = 25.0
MAX_TURNS = 4

LINE = (
    "요즘 llm-wiki 쪽은 잘 돼가고 있어요? "
    "AI가 알아서 지식베이스 정리해주는 거 보니까 문득 생각나서요."
)
"""A real one. Taken verbatim from the first proactive utterance this daemon ever
produced (2026-08-26, `topic`/`llm-wiki`), because a line invented here would be
shorter, plainer and easier to repeat than what the judge actually writes."""

SEED = (
    "너는 벨라다. 대현의 AI 동반자이고, 장난기 있고 담백하게 말한다. "
    "존댓말을 쓰되 딱딱하지 않다."
)

BASELINE = (
    "너는 방금 먼저 말을 걸기로 했다. 아래 문장의 내용을 상대에게 말하고, 그 다음 "
    f"상대의 대답을 기다려라. 이 지시문 자체는 읽지 마라.\n\n{LINE}"
)

VERBATIM = (
    "아래 문장을 **그대로** 소리 내어 말해라. 한 글자도 바꾸지 말고, 더하지도 빼지도 "
    "마라. 인사도, 설명도, 다른 말도 붙이지 마라. 문장을 말한 뒤에는 멈추고 상대의 "
    f"대답을 기다려라. 이 지시문 자체는 읽지 마라.\n\n{LINE}"
)

_PUNCT = re.compile(r"[\s.,!?~…·「」“”\"'()]+")


def _fold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _near_key(text: str) -> str:
    """Everything a listener would call the same sentence, collapsed to one string.

    Korean transcripts of the same utterance differ by spacing and punctuation far
    more often than by words, and the question here is whether the *sentence*
    survived, not whether the transcriber and the sender agree about commas.
    """
    return _PUNCT.sub("", _fold(text))


def _content_words(text: str) -> list[str]:
    return [w for w in _PUNCT.split(_fold(text)) if w]


def _substantive(said: str, intended: str) -> bool:
    """Every content word of the intended line appears in the transcript, in order.

    Deliberately loose: it passes a model that added a word and fails one that
    dropped or reordered the sentence's substance. What it is for is separating "she
    said it plus 네," from "she said something else about the same topic".
    """
    want = _content_words(intended)
    have = _content_words(said)
    i = 0
    for word in have:
        if i < len(want) and word == want[i]:
            i += 1
    return i == len(want)


async def _one_session(api_key: str, model: str, opening: str, voice: str) -> str:
    """One live session, opened with `opening`. Returns what it said, joined."""
    from daemon.voice.base import Transcript
    from daemon.voice.gemini_live import GeminiLiveSession

    said: list[str] = []
    session = GeminiLiveSession(
        api_key=api_key,
        model=model,
        system_instruction=SEED,
        tools=[],
        voice_name=voice,
        start_sensitivity="high",
        end_sensitivity="high",
    )
    async with session:
        await session.send_text(opening)
        for _ in range(MAX_TURNS):
            got = False
            try:
                async with asyncio.timeout(TURN_BUDGET_S):
                    async for event in session.receive():
                        got = True
                        if isinstance(event, Transcript) and event.role == "assistant":
                            said.append(event.text)
            except TimeoutError:
                break
            if not got or said:
                break
    return "".join(said)


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=8, help="sessions per cell")
    args = parser.parse_args(argv)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("DAEMON_GEMINI_LIVE_MODEL", "")
    voice = os.environ.get("DAEMON_GEMINI_LIVE_VOICE", "Despina")
    if not api_key or not model:
        print("needs GEMINI_API_KEY and DAEMON_GEMINI_LIVE_MODEL", file=sys.stderr)
        return 2

    print(f"model={model} voice={voice} runs={args.runs}/cell")
    print(f"intended: {LINE}\n")

    want_near = _near_key(LINE)
    for name, opening in (("A baseline", BASELINE), ("B verbatim", VERBATIM)):
        exact = near = subst = spoke = 0
        samples: list[str] = []
        for _ in range(args.runs):
            try:
                said = await _one_session(api_key, model, opening, voice)
            except Exception as exc:  # noqa: BLE001 - a dead socket is a datum
                print(f"  (session failed: {exc})")
                continue
            if not said.strip():
                continue
            spoke += 1
            exact += _fold(said).rstrip(".!?~") == _fold(LINE).rstrip(".!?~")
            near += _near_key(said) == want_near
            subst += _substantive(said, LINE)
            if len(samples) < 3:
                samples.append(said)
        print(f"[{name}] spoke {spoke}/{args.runs} · "
              f"exact {exact}/{max(spoke, 1)} · near {near}/{max(spoke, 1)} · "
              f"substantive {subst}/{max(spoke, 1)}")
        for s in samples:
            print(f"    - {s[:110]}")
        print()
    print("Hand-audit the samples before trusting the counts: `near` is a string "
          "comparison and cannot tell a good paraphrase from a bad one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
