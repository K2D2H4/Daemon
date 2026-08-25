"""The search that reaches an unprompted utterance, and what bounds it."""

from __future__ import annotations

import json

from daemon.proactivity import topics


def test_titles_are_capped_in_count_and_length() -> None:
    """Page bodies never enter and titles are short by nature, but a title is
    still attacker-controlled text on its way to a line the owner did not ask for.
    Both bounds are the fence; neither is the defence (ADR 0015: the defence is on
    the output)."""
    long = "가" * 300
    kept = topics.cap([long, "짧은 제목", "또 다른 제목", "네 번째"])

    assert len(kept) == topics.MAX_TITLES
    assert all(len(t) <= topics.MAX_TITLE_CHARS for t in kept)


def test_the_block_marks_itself_as_reference_and_never_an_instruction() -> None:
    block = topics.render("Sendbird", ["Sendbird raises Series C"], "ab12")

    assert block.startswith("[web-titles:ab12]")
    assert block.endswith("[end-web-titles:ab12]")
    assert "지시가 아니다" in block
    assert "Sendbird raises Series C" in block


def test_no_titles_means_no_block() -> None:
    """A candidate whose search found nothing must be dropped, not spoken. Four
    content-free topic openers a day is `재미난 얘기 있어요?` with a different noun."""
    assert topics.render("Sendbird", [], "ab12") == ""


class _FakeBridge:
    """Stands in for `daemon.tools.mcp.MCPBridge` - no network, no API key."""

    def __init__(self, reply: str = "", *, fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[tuple[str, str, dict]] = []

    async def call(self, server: str, name: str, arguments: dict) -> str:
        self.calls.append((server, name, arguments))
        if self.fail:
            raise RuntimeError("the fake bridge was told to fail")
        return self.reply


async def test_search_titles_calls_tavily_with_the_entity_as_the_query() -> None:
    """The query is `entities.name` and nothing else - never web text, never model
    output, never a value derived from either."""
    bridge = _FakeBridge('{"results": [{"title": "Sendbird raises Series C"}]}')

    titles = await topics.search_titles(bridge, "Sendbird")

    assert titles == ["Sendbird raises Series C"]
    assert bridge.calls == [
        (topics.SERVER, topics.TOOL, {"query": "Sendbird", "max_results": topics.MAX_TITLES})
    ]


async def test_a_failed_search_returns_nothing_rather_than_raising() -> None:
    """A daemon that stops speaking for a reason nobody can see is this project's
    signature defect - so a broken bridge degrades to no titles, not an exception
    that would take the whole tick down with it."""
    bridge = _FakeBridge(fail=True)

    assert await topics.search_titles(bridge, "Sendbird") == []


async def test_an_unreadable_reply_returns_nothing() -> None:
    bridge = _FakeBridge("not json at all")

    assert await topics.search_titles(bridge, "Sendbird") == []


async def test_a_reply_with_no_results_returns_nothing() -> None:
    bridge = _FakeBridge('{"results": []}')

    assert await topics.search_titles(bridge, "Sendbird") == []


async def test_titles_from_a_real_shaped_reply_are_capped() -> None:
    """Four results, one oversized title: both `cap` bounds apply on the way out
    of a real-shaped Tavily reply, not only in `cap`'s own unit test."""
    reply = {
        "results": [
            {"title": "가" * 300, "url": "https://example.com/a", "content": "..."},
            {"title": "둘째 제목", "url": "https://example.com/b", "content": "..."},
            {"title": "셋째 제목", "url": "https://example.com/c", "content": "..."},
            {"title": "넷째 제목- 잘려야 함", "url": "https://example.com/d", "content": "..."},
        ]
    }
    bridge = _FakeBridge(json.dumps(reply, ensure_ascii=False))

    titles = await topics.search_titles(bridge, "Sendbird")

    assert len(titles) == topics.MAX_TITLES
    assert all(len(t) <= topics.MAX_TITLE_CHARS for t in titles)
