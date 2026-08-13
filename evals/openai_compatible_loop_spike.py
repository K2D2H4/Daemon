"""Does the assembled app - not just the provider - survive a real turn?

`evals/openai_compatible_spike.py` proved `OpenAICompatibleProvider.complete` in
isolation: a Korean reply, and a tool round-trip against a hand-picked
`ToolSpec`. It never drove `daemon/loop.py`. Between a user's message and a
reply sit persona assembly, recall, the real tool registry,
`daemon/tools/policy.py`'s origin gate, and the markdown/mirror writes - none of
which has run against this provider before this file existed.

`tests/test_acceptance.py` already assembles the app the way the entrypoint does
and drives a real conversation with fakes stopping at the network edge (its own
docstring: "assembles the app the way `_build_io` does ... checks the whole
chain the product actually promises"). This file follows the same assembly -
`Store` + `FileMemoryWriter` + `MemoryRecall` + `LLMGateway` + a real tool stack
+ `Companion` + `ConversationLoop` - with exactly one thing swapped: the fake
`Provider` becomes the real `OpenAICompatibleProvider`, reading its key, base
URL and model from `.env`. The embedder stays the same deterministic, offline
fake `tests/test_acceptance.py` uses - recall's *quality* is `evals/golden_set.py`'s
job; what is being asked here is only whether recall's *wiring* survives a real
provider, which does not depend on which embedder is behind it.

Four things a mock cannot settle, each with printed evidence:

  1. Does a Korean turn get a non-empty reply **through `ConversationLoop.handle`**,
     not through `provider.complete` directly - i.e. does persona assembly, the
     tool-rules block and the recall block not break the request this specific
     provider builds?
  2. Is the turn actually recorded - the conversation markdown gets the exchange,
     the SQLite mirror gets its row?
  3. **Does a tool actually run inside a real turn** - the real registry, the
     real `ToolPolicy` origin gate, and the real audit table, driven by this
     provider's own `tool_calls` shape rather than a hand-built one. This is the
     combination nothing has tested: a new provider driving this repo's tool
     policy.
  4. Does recall carry something from an earlier turn into a later one's prompt -
     `daemon/companion.py`'s `recalled-memory:` block, assembled with this
     provider's neutral `Message` objects.

`DAEMON_DATA_DIR` is a fresh temporary directory, torn down at exit; the run
never touches the owner's real `data/`.

Run it once the key is in `.env`:

    python3 -m evals.openai_compatible_loop_spike

**Nothing here runs in CI and nothing here is a test.** A test may not touch the
network or a key (tests/CLAUDE.md); that is why this lives in `evals/`. If this
run turns up a real defect in `daemon/`, it is reported, not fixed here - a
defect a live run finds is a routing decision, not this file's to make.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The same `.env` reader the other spikes use, imported rather than copied.
from evals.m0_voice_spike import _load_env

OWNER = "5502877373"
"""Any channel-scoped id works - the origin gate keys off `authored_by_sender`,
not off matching an allowlist (that is the channel's job, not the loop's)."""

TURN_1 = "내일 저녁 7시에 연희동에서 보기로 약속했어."
"""Deliberately no explicit "remember this" instruction. An earlier version of
this line ended "...잊지 말고 기억해줘" ("don't forget, remember it") and, with
the real tool registry wired, the model read that as an instruction to *use a
tool* to persist the fact - it tried repeatedly to write/read a file, hit the
6-round tool limit, and the forced final-answer call then hit the
reasoning-emptied-content case and raised uncaught. That is a real finding
(see task-6-report.md), reported rather than fixed here; this line was changed
only so the four checks below could still be exercised.
`tests/test_acceptance.py`'s own fixtures never phrase a turn as an explicit
request to remember, either."""
TURN_2 = "오늘 하루 어땠는지 아무 얘기나 짧게 해줘."
"""Filler, so turn 1 falls out of the loop's own recent-context window
(`context_turns=1` below) before turn 4 asks about it - otherwise recency, not
recall, would be doing the work."""
TURN_4 = "우리 내일 저녁 몇 시에 연희동에서 보기로 했었지?"
"""Shares '내일', '저녁' and '연희동' with turn 1 - the token overlap the
fixture embedder's hashed-bag-of-tokens similarity needs to surface it."""

PACING_SECONDS = 5.0
"""Spacing between turns. OpenRouter's free tier rate-limits (HTTP 429) on
rapid back-to-back calls to a free model; a few idle seconds is cheaper and
more honest than retrying past a real limit."""

NOTE_CONTENT = "팀 회의는 목요일 오후 3시야."
TOOL_PROMPT_TEMPLATE = (
    "{path} 파일에 뭐라고 적혀 있는지 읽어서 알려줘. 반드시 read_file 도구를 호출해서 "
    "확인하고, 절대 추측하지 마."
)


class _Embedder:
    """Deterministic and offline - identical to `tests/test_acceptance.py`'s.

    Kept identical on purpose: swapping the provider is this file's whole
    point, and giving recall a different embedder here would make a failure
    ambiguous between "the provider broke recall's wiring" and "this embedder
    behaves differently from the one every other test uses".
    """

    name = "fake"
    model = "fake-embed"
    dimensions = 32

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in text.split():
                vector[hash(token) % self.dimensions] += 1.0
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            out.append([v / norm for v in vector])
        return out


class _RecordingProvider:
    """Wraps the real provider and keeps a copy of exactly what
    `daemon/loop.py` sent it - the only way to inspect the loop's own
    prompt assembly (persona, tool rules, the recall block) without touching
    `daemon/loop.py` or `daemon/companion.py`. Delegates every call unchanged."""

    name = "openai_compatible"

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.prompts: list[list[Any]] = []

    async def complete(self, messages: list[Any], *, model: str, **kw: Any) -> Any:
        self.prompts.append(list(messages))
        return await self._inner.complete(messages, model=model, **kw)

    async def health(self) -> bool:
        return await self._inner.health()


class _Channel:
    name = "telegram"

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, message: Any) -> None:
        self.sent.append(message.text)

    def listen(self) -> Any:  # pragma: no cover - driven directly
        raise NotImplementedError

    async def close(self) -> None: ...


async def main() -> int:
    _load_env()
    api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY", "").strip()
    base_url = os.environ.get("DAEMON_OPENAI_COMPATIBLE_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("DAEMON_OPENAI_COMPATIBLE_MODEL", "").strip()
    if not api_key or not base_url or not model:
        print(
            "OPENAI_COMPATIBLE_API_KEY, DAEMON_OPENAI_COMPATIBLE_BASE_URL and "
            "DAEMON_OPENAI_COMPATIBLE_MODEL must all be set in .env. Run "
            "evals/openai_compatible_spike.py first if you need to pick a model."
        )
        return 1

    from daemon.channels.base import InboundMessage
    from daemon.companion import Companion
    from daemon.config import Route
    from daemon.llm.gateway import LLMGateway
    from daemon.llm.providers.openai_compatible import OpenAICompatibleProvider
    from daemon.loop import ConversationLoop
    from daemon.memory.recall import MemoryRecall
    from daemon.memory.store import Store
    from daemon.memory.writer import FileMemoryWriter
    from daemon.tasks import Task
    from daemon.tools.base import Registry
    from daemon.tools.builtin import builtin_tools
    from daemon.tools.policy import ToolPolicy
    from daemon.tools.runner import ToolRunner

    data_dir = Path(tempfile.mkdtemp(prefix="daemon-openai-compatible-loop-"))
    workspace = data_dir / "workspace"
    workspace.mkdir()
    note_path = workspace / "note.txt"
    note_path.write_text(NOTE_CONTENT, encoding="utf-8")

    print(f"key: ...{api_key[-4:]} (never printed in full, never written anywhere)")
    print(f"base_url: {base_url}")
    print(f"model: {model}")
    print(f"date: {datetime.now(UTC).date().isoformat()}")
    print(f"DAEMON_DATA_DIR: {data_dir} (temporary, removed at exit)")

    ok1 = ok2 = ok3 = ok4 = False
    store = Store.open(data_dir / "daemon.sqlite3")
    real_provider = OpenAICompatibleProvider(api_key, base_url)
    provider = _RecordingProvider(real_provider)
    tools: ToolRunner | None = None
    try:
        writer = FileMemoryWriter(data_dir, store)
        recall = MemoryRecall(store, _Embedder())
        gateway = LLMGateway(
            {"openai_compatible": provider},
            {Task.CHAT_TEXT: Route("openai_compatible", model)},
        )

        registry = Registry()
        for tool in builtin_tools(roots=[workspace]):
            registry.register(tool)
        tools = ToolRunner(registry, ToolPolicy(store, mode="full"), store)

        channel = _Channel()
        companion = Companion(
            writer,
            data_dir=data_dir,
            recall=recall,
            resolve_id=lambda _text: writer.last_inserted_id,
            tools=tools,
        )
        # context_turns=1: only the immediately previous exchange is carried
        # verbatim, so turn 4 recalling turn 1 has to go through `MemoryRecall`,
        # not the recent-messages window - the same choice
        # tests/test_acceptance.py's `test_yesterday_is_quoted_back_tomorrow`
        # makes, for the same reason.
        loop = ConversationLoop(channel, gateway, companion, context_turns=1)

        def inbound(text: str, external_id: str) -> InboundMessage:
            return InboundMessage(
                text=text,
                sender_id=OWNER,
                received_at=datetime.now(UTC),
                channel="telegram",
                external_id=external_id,
            )

        # --- 1 & 2: a Korean turn through the loop, then check it landed -----
        print("\n1+2. plain Korean turn, through ConversationLoop.handle:")
        await loop.handle(inbound(TURN_1, "1"))
        reply1 = channel.sent[-1] if channel.sent else ""
        print(f"   reply: {reply1[:200]!r}")
        ok1 = bool(reply1.strip())
        print(f"   non-empty reply through the loop: {ok1}")

        day_files = list((data_dir / "memory" / "log").glob("*.md"))
        day_text = day_files[0].read_text(encoding="utf-8") if day_files else ""
        rows = store.recent(2)
        roles = [r["role"] for r in rows]
        markdown_has_it = bool(day_text) and "연희동" in day_text and (
            reply1 and reply1 in day_text
        )
        mirror_has_it = roles == ["user", "assistant"]
        print(f"   markdown has the exchange: {markdown_has_it}")
        print(f"   sqlite mirror roles for this turn: {roles}")
        ok2 = markdown_has_it and mirror_has_it
        print(f"   turn recorded (markdown + mirror): {ok2}")

        # --- filler, to push turn 1 out of the recent window ------------------
        print("\n(filler turn, so recall rather than recency has to do the work)")
        await asyncio.sleep(PACING_SECONDS)  # OpenRouter's free tier rate-limits (429) on
        # back-to-back calls; a few seconds of spacing is cheaper than a retry loop.
        await loop.handle(inbound(TURN_2, "2"))
        print(f"   reply: {channel.sent[-1][:200]!r}")

        # --- 3: a real tool, inside a real turn --------------------------------
        print("\n3. a message that requires a real tool:")
        await asyncio.sleep(PACING_SECONDS)
        await loop.handle(inbound(TOOL_PROMPT_TEMPLATE.format(path=note_path), "3"))
        print(f"   reply: {channel.sent[-1][:200]!r}")
        tool_rows = store.recent_tool_calls()
        print(
            "   tool_calls audit rows (tool, verdict, ran, ok): "
            f"{[(r['tool'], r['verdict'], r['ran'], r['ok']) for r in tool_rows]}"
        )
        matching = [r for r in tool_rows if r["tool"] == "read_file"]
        ok3 = bool(matching) and matching[0]["ran"] == 1 and matching[0]["ok"] == 1
        if not matching:
            print("   the model never called read_file - no round-trip to audit.")
        print(f"   tool decided, executed and recorded: {ok3}")

        # --- 4: recall on a following turn -------------------------------------
        print("\n4. recall on a following turn:")
        await asyncio.sleep(PACING_SECONDS)
        await loop.handle(inbound(TURN_4, "4"))
        print(f"   reply: {channel.sent[-1][:200]!r}")
        last_prompt = provider.prompts[-1]
        recalled = [m.content for m in last_prompt if "recalled-memory:" in m.content]
        print(f"   recalled-memory block present in turn 4's prompt: {bool(recalled)}")
        if recalled:
            print(f"   recalled content: {recalled[0][:300]!r}")
        ok4 = bool(recalled) and any(
            ("연희동" in block or "7시" in block) for block in recalled
        )
        print(f"   turn 1's fact reached turn 4's context: {ok4}")
    finally:
        store.close()
        if tools is not None:
            await tools.aclose()
        await real_provider.aclose()
        shutil.rmtree(data_dir, ignore_errors=True)

    print("\n--- summary ---")
    print(f"1. Korean turn through the loop, non-empty reply: {'PASS' if ok1 else 'FAIL'}")
    print(f"2. turn recorded (markdown + mirror):              {'PASS' if ok2 else 'FAIL'}")
    print(f"3. tool decided, executed and recorded:            {'PASS' if ok3 else 'FAIL'}")
    print(f"4. recall carried turn 1 into turn 4:              {'PASS' if ok4 else 'FAIL'}")
    return 0 if (ok1 and ok2 and ok3 and ok4) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
