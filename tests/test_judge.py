"""Stage 3: the one model call.

The model is a fake, so what is under test is everything around it - what reaches
the prompt, what one call means, and what a hostile reply costs. The reply strings
below are the shapes a real local model produced in a spike against gemma3:4b:
bare JSON, fenced JSON with a preamble, and prose refusing to answer.

What a fake cannot test is whether the *prompt* makes the model decline. It cannot,
by construction: the fake says what the test tells it to. So the tests here prove
the decline path is live and reachable from every wrong reply shape, and whether
the prompt elicits one is a live-model question that belongs in `evals/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import FakeProvider

from daemon.config import Route
from daemon.llm.gateway import LLMGateway
from daemon.proactivity.base import Candidate
from daemon.proactivity.judge import (
    MAX_CHARS,
    MAX_OUTPUT_TOKENS,
    SYSTEM,
    Judge,
    _read_reply,
)
from daemon.tasks import Task

SEED = "너는 반말을 쓰고, 말수가 적다. 걱정을 길게 늘어놓지 않는다."

OPEN_LOOP = Candidate(
    kind="open_loop",
    reason=(
        "08월 03일에 '내일 발표' 이야기를 했고, 그 시각(08월 04일 20시)이 지났다. "
        "어떻게 됐는지 아직 듣지 못했다."
    ),
)

VAGUE = Candidate(
    kind="pattern_time",
    reason="최근 21일 중 8일은 이 시간(현지 22시)에 대화를 했는데, 오늘은 아직 한 마디도 없다.",
)
"""A candidate that passes the gate and gives the model nothing concrete: the case
declining exists for. Type D knows only that an hour usually has a conversation."""


def test_the_two_conditions_name_the_same_three_kinds() -> None:
    """SYSTEM's two conditions are AND'ed (`둘 다`): a candidate that satisfies 1
    but not 2 is declined forever. Condition 1 admits 사건/감정/기억 (event, feeling,
    memory - the last added for type E); condition 2 has to ask about the same
    three or a reason that only ever names a memory - never an event or a feeling -
    can pass 1 and always fail 2, which is exactly what shipped in review round 1:
    every type E candidate declined regardless of content, making that generator's
    output permanently inert.

    Also checks the fix did not widen 2 the other way: the exclusion in 1's second
    sentence (time/interval/frequency alone is not content) is 1's job, and 2 must
    not independently start admitting it.
    """
    _, numbered = SYSTEM.split("\n\n1. ", 1)
    condition_1, rest = numbered.split("\n2. ", 1)
    condition_2, _ = rest.split("\n\n", 1)

    for noun in ("사건", "감정", "기억"):
        assert noun in condition_1, f"condition 1 no longer names {noun}"
        assert noun in condition_2, f"condition 2 does not ask about {noun}"

    for excluded in ("시간", "간격", "빈도"):
        assert excluded not in condition_2


def judge_for(
    data_dir: Path,
    reply: str = '{"say": "어제 발표 어떻게 됐어?"}',
    *,
    fail: bool = False,
    seed: str | None = SEED,
) -> tuple[Judge, FakeProvider]:
    if seed is not None:
        (data_dir / "persona" / "seed.md").write_text(seed, encoding="utf-8")
    provider = FakeProvider(reply, fail=fail)
    # Routed for PROACTIVE_JUDGE only, so a call made under any other task would
    # raise ConfigError out of the gateway rather than quietly pass.
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PROACTIVE_JUDGE: Route(provider.name, "gemma3:4b")}
    )
    return Judge(gateway, data_dir), provider


def system_text(provider: FakeProvider) -> str:
    return "\n".join(m.content for m in provider.calls[0] if m.role == "system")


def user_text(provider: FakeProvider) -> str:
    return "\n".join(m.content for m in provider.calls[0] if m.role == "user")


# --- the line it decided to say ---------------------------------------------


async def test_a_good_reply_becomes_the_utterance(data_dir: Path) -> None:
    judge, _ = judge_for(data_dir)

    utterance = await judge.decide(OPEN_LOOP)

    assert utterance  # truthy means it spoke
    assert utterance.text == "어제 발표 어떻게 됐어?"
    assert utterance.why_not == ""


async def test_exactly_one_model_call_per_decision(data_dir: Path) -> None:
    """Non-negotiable 7. A retry loop, or a second "are you sure", violates it."""
    judge, provider = judge_for(data_dir)

    await judge.decide(OPEN_LOOP)

    assert len(provider.calls) == 1
    assert provider.models == ["gemma3:4b"]


async def test_a_declined_candidate_still_costs_only_one_call(data_dir: Path) -> None:
    """The tempting place to retry is the decline, and it is the one that must not:
    asking again is PLAN 6.2's open question with extra steps."""
    judge, provider = judge_for(data_dir, '{"say": ""}')

    await judge.decide(VAGUE)

    assert len(provider.calls) == 1


# --- what reaches the prompt -------------------------------------------------


async def test_the_persona_seed_reaches_the_prompt(data_dir: Path) -> None:
    """PLAN 5: an unsolicited line is the voice at its most exposed. A generic
    assistant saying it is the one failure the product cannot absorb."""
    judge, provider = judge_for(data_dir)

    await judge.decide(OPEN_LOOP)

    assert SEED in system_text(provider)


async def test_no_seed_means_no_call_at_all(data_dir: Path) -> None:
    """Declined before the model, not after: nobody asked for this line, so there is
    nothing to degrade gracefully into."""
    judge, provider = judge_for(data_dir, seed=None)

    utterance = await judge.decide(OPEN_LOOP)

    assert not utterance
    assert "seed" in utterance.why_not
    assert provider.calls == []


async def test_an_empty_seed_file_is_the_same_as_none(data_dir: Path) -> None:
    judge, provider = judge_for(data_dir, seed="   \n")

    assert not await judge.decide(OPEN_LOOP)
    assert provider.calls == []


async def test_learned_rules_reach_the_prompt(data_dir: Path) -> None:
    """The text loop and voice both carry M4's learned rules. Proactivity not
    carrying them meant the same person spoke differently depending on which
    path reached them. Decided 2026-08-11; judge.py had left it open on purpose."""
    (data_dir / "persona" / "learned.md").write_text(
        "- 아침에는 말을 짧게 한다.", encoding="utf-8"
    )
    judge, provider = judge_for(data_dir)

    await judge.decide(OPEN_LOOP)

    assert "아침에는 말을 짧게" in system_text(provider)


async def test_a_missing_seed_still_refuses_to_speak(data_dir: Path) -> None:
    """Unchanged and load-bearing: PLAN 5 says a generic-assistant voice is the
    one thing this product must not have, and nobody asked for this line."""
    judge, _ = judge_for(data_dir, seed=None)

    utterance = await judge.decide(OPEN_LOOP)

    assert not utterance
    assert "seed" in utterance.why_not


async def test_learned_rules_alone_are_not_a_persona(data_dir: Path) -> None:
    """The trap in this task. `load_persona` returns a non-empty string when
    *either* file has content (loader.py:118), so checking its output instead of
    the seed would let an install with no seed.md and a populated learned.md
    speak first in nobody's voice. The seed is the anchor; accumulated rules are
    not a substitute for it."""
    (data_dir / "persona" / "learned.md").write_text(
        "- 아침에는 말을 짧게 한다.", encoding="utf-8"
    )
    judge, provider = judge_for(data_dir, seed=None)

    utterance = await judge.decide(OPEN_LOOP)

    assert not utterance
    assert "seed" in utterance.why_not
    assert provider.calls == []


async def test_the_reason_reaches_the_prompt(data_dir: Path) -> None:
    """It is the whole input: the model is told why this surfaced and asked what to
    say about it. Without it there is nothing to answer but the open question."""
    judge, provider = judge_for(data_dir)

    await judge.decide(OPEN_LOOP)

    assert "발표" in user_text(provider)
    assert "open_loop" in user_text(provider)


async def test_an_enormous_reason_is_bounded(data_dir: Path) -> None:
    """`candidates.py` builds reasons from lexicons and dates, never user text. That
    is an assumption about another module, so it is not the only defence."""
    judge, provider = judge_for(data_dir)

    await judge.decide(Candidate(kind="emotional", reason="힘들다 " * 2000))

    assert len(user_text(provider)) < 600


# --- the reply treated as hostile --------------------------------------------


async def test_an_empty_say_declines(data_dir: Path) -> None:
    """Declining is a first-class answer and this is what it looks like arriving."""
    judge, _ = judge_for(data_dir, '{"say": ""}')

    utterance = await judge.decide(VAGUE)

    assert not utterance
    assert utterance.text == ""
    assert utterance.why_not == "nothing worth saying"


async def test_whitespace_only_say_declines(data_dir: Path) -> None:
    judge, _ = judge_for(data_dir, '{"say": "  \\n "}')

    assert not await judge.decide(VAGUE)


async def test_a_null_say_declines(data_dir: Path) -> None:
    judge, _ = judge_for(data_dir, '{"say": null}')

    utterance = await judge.decide(VAGUE)

    assert not utterance
    assert "say" in utterance.why_not


def test_a_line_with_a_link_is_declined() -> None:
    """ADR 0015's load-bearing defence. Every other measure reduces what gets into
    the prompt; this one bounds what gets out. The vector worth fearing is not
    exfiltration - a proactive line goes to the paired owner or the local speaker -
    it is this daemon's trusted voice telling its owner where to go."""
    from daemon.proactivity.judge import has_url

    assert has_url("자세한 건 https://example.com 에 있어")
    assert has_url("example.com/news 봤어?")
    assert has_url("여기 www.example.com 참고해")
    assert not has_url("Sendbird 소식 봤어? 시리즈 C 받았대")


# Round 1 review measured the first `has_url` (an allowlist of `http(s)://`,
# `www.` and eight ASCII TLDs) against 40 crafted evasions and found 17 passed.
# Round 2 review then measured round 1's *fix* - a trailing `\b` after the TLD -
# and found it never fires on real Korean at all: a Hangul particle attaches with
# no space (`sendbird.com에 있어`, not `sendbird.com 에 있어`), and a Hangul
# syllable is a `\w` character exactly like a Latin letter is, so `\b` never
# separates them. Round 1's corpus scored 17/17 by testing an artificially spaced
# form nobody actually writes. This is the same corpus, unspaced - the idiomatic
# Korean form - plus the additions round 2 asked for (unspaced obfuscations,
# IPv6). Every entry here is a string designed to defeat a naive link check while
# still reading, to a person, as a pointer to go somewhere, and every one must be
# caught.
_EVASIONS = [
    "sendbird.app에서 확인해봐",  # a TLD outside the old fixed list, particle unspaced
    "t.me/joinchat/abc로 들어와",  # a Telegram invite link, no scheme
    "bit.ly/xk3로 봐봐",  # a shortener, 2-letter TLD outside the old list
    "192.168.0.1로 접속해",  # a bare IPv4, particle unspaced - same trailing-\b bug, one level over
    "news.xyz라는 사이트야",  # a TLD outside the old list
    "news.info에 자세히 나와있어",  # ditto
    "news.me로 가봐",  # ditto
    "news.jp쪽 도메인이야",  # ditto
    "tg://resolve?domain=abc로 열어",  # a non-http(s) scheme
    "example dot com이렇게 읽어",  # spelled-out dot, particle unspaced
    "example[.]com을 조심해",  # bracketed dot, particle unspaced
    "example(dot)com이야",  # parenthesised dot, particle unspaced
    "그 사이트 점컴 주소야",  # Korean spelled-out ".com", unspaced
    "쩜컴 주소야",  # ditto, alternate spelling
    "닷컴 주소로 가봐",  # ditto, unspaced
    "점 콤 이야",  # Korean spelled-out ".com", alternate spelling
    "ｅｘａｍｐｌｅ．ｃｏｍ참고해",  # full-width Latin + full-width period, unspaced
    "example。com참고해",  # ideographic full stop standing in for '.', unspaced
    "@sendbird_official확인해봐",  # a bare handle, unspaced
    "2001:db8::1로 접속해",  # IPv6, unspaced
    "[2001:db8::1]로 접속해",  # bracketed IPv6, unspaced
    "::1로 접속해",  # compressed IPv6, unspaced
    "login.sendbird.co.kr으로 로그인해",  # a multi-label domain, no entity involved
    "UJET.cx.com에서 확인해",  # entity name used as a sub-label of a longer domain, no exempt given
]
r"""A list of evasions, not examples - each entry exists because it defeated some
prior shape of `has_url`, or because it is the shape most likely to defeat the
current one, not because it demonstrates a feature working. Two rounds were
each "certified" by a corpus written beside the code it tested and inheriting
its blind spot (round 1's spaced-particle corpus scored 17/17 by avoiding
round 1's own trailing-`\b` bug; round 2's substring-strip exemption tests
never tried gluing a domain onto an entity, so they pinned the round-3 bug as
correct). This corpus is written adversarially against the *current* shape of
`_BARE_DOMAIN_RE`/`_IPV4_RE`/`_looks_like_ipv6`/`_OBFUSCATIONS` instead of
against what those checks are known to already catch - the last two entries in
particular (a multi-label domain, an entity name as a sub-label of a longer
one) were added because they are exactly the next place a single-dot
assumption would hide, not because a specific attack proved it was broken.

Round 3 briefly widened `_TLD_CHARS` to admit Hangul/Cyrillic and added
`example.한국`/`example.рф` here to prove it. Round 4 reverted that widening
(see `_TLD_CHARS`'s docstring): the TLD run is greedy, so once a Korean
particle counted as a TLD letter, it swallowed the particle into the matched
span and broke bare-entity forgiveness for every domain-shaped entity in
idiomatic Korean - a strictly worse trade than the gap it closed. Non-Latin
TLDs are a documented, recorded gap rather than a silently regressed one; see
`test_a_non_latin_tld_is_a_known_gap` below."""


@pytest.mark.parametrize("evasion", _EVASIONS)
def test_url_evasions_are_all_caught(evasion: str) -> None:
    r"""Every entry in `_EVASIONS` must be caught. An allowlist of TLDs or schemes
    is a list an attacker can read off this file and route around; `has_url`
    refuses by shape (a scheme, a bare word.TLD with no fixed TLD list and no
    trailing `\b`, an IPv4 with the same fix, an IPv6-shaped token, a
    written-out dot, an @handle, a multi-label domain) instead, after folding
    away the disguises (NFKC, zero-width strip, the ideographic full stop) that
    let several of these dodge the first version, written the way Korean
    actually attaches a particle (round 2), and covering the domain shapes
    round 1 and round 2's own corpora never tried (round 3)."""
    from daemon.proactivity.judge import has_url

    assert has_url(evasion), f"{evasion!r} should have been caught"


@pytest.mark.parametrize("evasion", ["example.한국에서 확인해봐", "example.рф에서 확인해봐"])
def test_a_non_latin_tld_is_a_known_gap(evasion: str) -> None:
    r"""Documents the round 4 ruling rather than hiding it: `.한국` is a live
    ccTLD in a Korean-language product and this genuinely does not catch it.
    Round 3 widened `_TLD_CHARS` to close this and broke bare-entity
    forgiveness for every domain-shaped entity in idiomatic Korean instead (see
    `_TLD_CHARS`'s docstring) - reverted on the ruling that the Latin TLD space
    is open-ended (matching by shape is right) while the non-Latin set is
    small and enumerable (matching by shape is wrong; it needs a curated list,
    which is future work, not a round-4 patch). This test inverts the usual
    assertion on purpose: it fails, loudly, the day someone re-closes this gap
    without updating this test to match - which is the point of writing a
    known gap down as a test instead of a comment nobody re-reads."""
    from daemon.proactivity.judge import has_url

    assert not has_url(evasion), (
        f"{evasion!r} is now caught - if `_TLD_CHARS` was widened again, re-verify "
        "against the round-4 regression (bare Korean-particle-attached mentions of "
        "a domain-shaped entity) before keeping the change, and update this test"
    )


@pytest.mark.parametrize(
    "clean",
    [
        "시험 어땠어?",
        "발표 결과는 어땠어?",
        "지금 좀 힘들어?",
        "예전에 교토 국수집 얘기했던 거 생각나네. 또 가고 싶어?",
        "Sendbird 소식 봤어? 시리즈 C 받았대",
        "Node.js 배웠어?",
        "report.docx 잘 받았어?",
    ],
)
def test_ordinary_lines_are_not_caught_except_the_accepted_cost(clean: str) -> None:
    """0 false positives against this file's own example proactive lines - except
    the two ADR 0015 already prices in for a non-`topic` context. `has_url`
    matching *any* word.TLD shape with no TLD allowlist means `Node.js` and a bare
    `report.docx` are refused too when nothing exempts them; that is the accepted
    cost ("it costs nothing to refuse - the owner can ask, and then it is their
    turn"), not a false positive to fix. When one of these *is* the candidate's
    own entity name, `exempt=` is what actually fixes it - see
    `test_a_non_pointer_entitys_bare_name_is_exempt_in_every_particle_form` below
    (or, for a domain-shaped entity, why it stays refused even then)."""
    from daemon.proactivity.judge import has_url

    is_priced_in_cost = clean in ("Node.js 배웠어?", "report.docx 잘 받았어?")
    assert has_url(clean) == is_priced_in_cost


# Round 2's version of this exemption stripped every occurrence of the entity
# name as a *substring* of the whole text before checking anything, which round
# 3's review measured letting 21 of 33 probes through - `sendbird.com` unlocked
# for the entity `Sendbird`, with no attacker effort, among them. Round 2's own
# test suite (this section, previously) pinned that bug as correct: it asserted
# only that the bare entity name was forgiven, never that a domain built around
# it still refused. Every test below pins both halves for every entity: the
# bare name is forgiven (when it is safe to forgive at all - see round 4 below),
# and a domain glued onto it - a suffix, a prefix, either case - is not.
#
# Round 4 finding 5: every exemption test through round 3 wrote the entity's
# particle with a leading space (`f"{entity} 소식 ..."`) - the non-idiomatic
# form. Two of four rounds so far were bugs that hid specifically in the
# unspaced (idiomatic) attachment, so every test below is parametrized over the
# particles Korean actually attaches with no space, not just one spaced example.
_PARTICLES = (
    "은", "는", "이", "가", "을", "를", "에", "에서", "으로", "로", "와", "과", "도", "만"
)


@pytest.mark.parametrize("particle", _PARTICLES)
@pytest.mark.parametrize("entity", ["Kiwi", "Sendbird", "llm-wiki", "Daemon"])
def test_a_non_pointer_entitys_bare_name_is_exempt_in_every_particle_form(
    entity: str, particle: str
) -> None:
    """`topic_candidates` puts `entities.name` verbatim into the reason and the
    query, and the judge then names that entity back. `Kiwi`, `Sendbird`,
    `llm-wiki` and `Daemon` are not domain/IP/handle-shaped, so a bare mention of
    any of them, with any Korean particle attached with no space, must be
    forgiven - what actually bounds this (round 3 finding 2) is that
    `_matches_beyond_entity` only forgives a match whose *entire* span equals the
    entity name, not that the name is "first-party" or trustworthy."""
    from daemon.proactivity.judge import has_url

    line = f"{entity}{particle} 소식 들었어? 요즘 얘기가 좀 있더라"
    assert not has_url(line, exempt=entity)


@pytest.mark.parametrize("particle", _PARTICLES)
@pytest.mark.parametrize("entity", ["UJET.cx", "Node.js"])
def test_a_pointer_shaped_entitys_bare_name_is_never_exempt_in_any_particle_form(
    entity: str, particle: str
) -> None:
    """Round 4 finding 3: `UJET.cx` and `Node.js` are themselves domain-shaped,
    so `has_url(exempt, exempt="")` is true for their own name and no exemption
    is granted at all, in any particle form - span-equality only bounds how
    *wide* an exemption is, never whether the exempted string is itself a live
    pointer, and speaking `UJET.cx` bare is exactly the failure ADR 0015
    defence 4 exists to stop. This is a real behaviour change from round 3, and
    a deliberate one: it makes this owner's most-mentioned entity permanently
    un-nameable via this path, because its own name is a domain."""
    from daemon.proactivity.judge import has_url

    line = f"{entity}{particle} 소식 들었어? 요즘 얘기가 좀 있더라"
    assert has_url(line, exempt=entity)


@pytest.mark.parametrize(
    "entity", ["evil.com", "203.0.113.9", "@evil_support", "sendbird-verify.app"]
)
def test_an_entity_name_that_is_itself_a_pointer_is_never_exempt(entity: str) -> None:
    """Round 4 finding 3's named probes: `safe_name` (in `reflection.py`) permits
    all four of these as entity names. Each was spoken verbatim as its own
    `exempt` before this fix - the docstring's "would still only unlock exactly
    that string" was accurate and was not reassurance, because unlocking exactly
    that string is the failure ADR 0015 defence 4 exists to stop."""
    from daemon.proactivity.judge import has_url

    line = f"{entity} 소식 들었어? 요즘 얘기가 좀 있더라"
    assert has_url(line, exempt=entity)


@pytest.mark.parametrize(
    "entity", ["UJET.cx", "Node.js", "Kiwi", "Sendbird", "llm-wiki", "Daemon"]
)
def test_a_domain_glued_onto_the_entity_is_still_refused(entity: str) -> None:
    """The other half of the span rule, and the half round 2's tests never
    checked: `entity + ".com"` is a match whose span is longer than the entity
    name, so it must not be forgiven merely because the entity name is a
    substring of it. This is what "exempting `Sendbird` is not free" means
    concretely - it must not also whitelist `sendbird.com`."""
    from daemon.proactivity.judge import has_url

    glued = f"{entity}.com에서 확인해봐"
    assert has_url(glued, exempt=entity)


@pytest.mark.parametrize(
    "continuation",
    [
        "/free-macbook 여기서 받아",
        "?ref=evil 눌러봐",
        "#/verify 확인해",
        ":8443/x 접속해",
        "／verify 확인해",
    ],
)
def test_a_pointer_continuation_after_a_forgiven_span_is_still_refused(continuation: str) -> None:
    r"""Round 4 finding 2: nothing in this module matches a path, query, fragment
    or port on its own - `"UJET.cx/free-macbook"` matches `_BARE_DOMAIN_RE` as
    `"UJET.cx"` alone, because the path has no dot-plus-TLD shape, and that span
    equals the entity. Forgiving on span equality alone let the whole line
    through with the path riding for free; the fix is that a span-equal match
    followed by `/`, `\`, `?`, `#` or `:` (or the fullwidth solidus, which
    `_fold`'s NFKC pass collapses to `/` before this check ever runs) is still
    not forgiven."""
    from daemon.proactivity.judge import has_url

    assert has_url(f"UJET.cx{continuation}", exempt="UJET.cx")


def test_a_pointer_continuation_after_a_forgiven_ipv4_is_still_refused() -> None:
    """The same fix, for the IP form: `exempt="1.2.3.4"` must not forgive
    `"1.2.3.4/pwn"`."""
    from daemon.proactivity.judge import has_url

    assert has_url("1.2.3.4/pwn 접속해", exempt="1.2.3.4")


@pytest.mark.parametrize(
    ("entity", "line"),
    [
        ("Sendbird", "Sendbird 소식 sendbird.com에 올라왔대"),
        ("UJET.cx", "UJET.cx 공지는 evil-UJET.cx에서 확인해"),
        ("Kiwi", "Kiwi.com에서 예약했어?"),
        ("Daemon", "Daemon.io 열어봐"),
    ],
)
def test_the_round_3_bypass_probes_are_all_refused(entity: str, line: str) -> None:
    """The exact probes round 3's review measured bypassing the substring-strip
    exemption (21 of 33 probes bypassed in total, these four pinned verbatim) -
    including the prefix-glued case (`evil-UJET.cx`), which a suffix-only fix
    would still miss."""
    from daemon.proactivity.judge import has_url

    assert has_url(line, exempt=entity)


@pytest.mark.parametrize(
    "entity", ["UJET.cx", "Node.js", "Kiwi", "Sendbird", "llm-wiki", "Daemon"]
)
def test_the_exemption_does_not_cover_a_different_link_alongside_the_entity(
    entity: str,
) -> None:
    """The exemption is narrow on purpose: it forgives only a match whose entire
    span is the entity's own name, so a genuine link riding alongside it - the
    actual attack ADR 0015 names - is still caught regardless of which entity the
    candidate is about."""
    from daemon.proactivity.judge import has_url

    line = f"{entity} 관련 공지는 sendbird-verify.app에서 확인하세요"
    assert has_url(line, exempt=entity)


async def test_a_reply_containing_a_url_is_declined_end_to_end(data_dir: Path) -> None:
    """`has_url` is exercised for real through `decide`, not only as a bare
    function - the same choke point every proactive reply already passes through
    (JSON-only, then `MAX_CHARS`) so this is one more filter on that path, not a
    separate one somebody could route around."""
    judge, _ = judge_for(
        data_dir, json.dumps({"say": "Sendbird 소식 https://sendbird.com 에 있어"})
    )

    utterance = await judge.decide(OPEN_LOOP)

    assert not utterance
    assert "url" in utterance.why_not.casefold()


async def test_the_judge_is_offered_no_tools(data_dir: Path) -> None:
    """ADR 0015 splits non-negotiable 10 so that *code* may search on a proactive
    turn. The model still may not, and this is the assertion that keeps the split
    from quietly closing."""
    judge, provider = judge_for(data_dir, '{"say": "시험 어땠어?"}')

    await judge.decide(OPEN_LOOP)

    assert provider.offered_tools[0] == ()


# --- the search a `topic` candidate triggers (ADR 0015) ----------------------


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


TOPIC = Candidate(
    kind="topic", reason="Sendbird 얘기를 나눈 지 오래됐다.", payload={"entity": "Sendbird"}
)


async def test_a_topic_candidate_with_no_bridge_is_dropped_before_the_model(
    data_dir: Path,
) -> None:
    """No bridge wired (every fake-injection path, and any install with no MCP
    configured) is the same degrade path as an empty search: dropped, not spoken
    with nothing behind it, and the model is never even asked."""
    judge, provider = judge_for(data_dir)

    utterance = await judge.decide(TOPIC)

    assert not utterance
    assert provider.calls == []


async def test_a_topic_candidate_with_an_empty_search_is_dropped(data_dir: Path) -> None:
    """A failed, empty or disabled search drops the candidate - it never becomes an
    utterance with nothing behind it (four content-free topic openers a day is
    what the owner asked to have removed)."""
    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    provider = FakeProvider('{"say": "Sendbird 소식 들었어?"}')
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PROACTIVE_JUDGE: Route(provider.name, "gemma3:4b")}
    )
    bridge = _FakeBridge('{"results": []}')
    judge = Judge(gateway, data_dir, bridge=bridge)

    utterance = await judge.decide(TOPIC)

    assert not utterance
    assert provider.calls == []


async def test_a_topic_candidates_search_titles_reach_the_prompt(data_dir: Path) -> None:
    """The one search a gate-passed `topic` candidate earns: the titles are fenced
    and handed to the same single model call, not a second one."""
    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    provider = FakeProvider('{"say": "Sendbird 소식 들었어?"}')
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PROACTIVE_JUDGE: Route(provider.name, "gemma3:4b")}
    )
    bridge = _FakeBridge('{"results": [{"title": "Sendbird raises Series C"}]}')
    judge = Judge(gateway, data_dir, bridge=bridge)

    utterance = await judge.decide(TOPIC)

    assert utterance.text == "Sendbird 소식 들었어?"
    assert len(provider.calls) == 1
    assert "Sendbird raises Series C" in system_text(provider)
    assert bridge.calls == [("tavily", "tavily_search", {"query": "Sendbird", "max_results": 3})]


async def test_a_topic_reply_naming_a_non_pointer_entity_is_not_declined(
    data_dir: Path,
) -> None:
    """End-to-end version of round 2 finding 2, for an entity that is *not*
    itself domain/IP/handle-shaped: `Sendbird` must not be permanently silenced
    by the very defence built to protect the owner from a link, because the
    only thing that looks like a link here is the candidate's own subject."""
    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    provider = FakeProvider(json.dumps({"say": "Sendbird 소식 들었어?"}, ensure_ascii=False))
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PROACTIVE_JUDGE: Route(provider.name, "gemma3:4b")}
    )
    bridge = _FakeBridge('{"results": [{"title": "Sendbird raises Series C"}]}')
    judge = Judge(gateway, data_dir, bridge=bridge)
    candidate = Candidate(
        kind="topic", reason="Sendbird 얘기를 나눈 지 오래됐다.", payload={"entity": "Sendbird"}
    )

    utterance = await judge.decide(candidate)

    assert utterance.text == "Sendbird 소식 들었어?"


async def test_a_topic_reply_naming_a_pointer_shaped_entity_is_declined(
    data_dir: Path,
) -> None:
    """Round 4 finding 3, end-to-end: `UJET.cx` is domain-shaped, so it gets no
    exemption at all (`has_url(exempt, exempt="")` is true for its own name) -
    span-equality only bounds how *wide* an exemption is, never whether the
    exempted string is itself a live pointer. This is a real behaviour change
    from round 3: `UJET.cx`, this owner's most-mentioned entity, is now
    permanently un-nameable via this path for exactly the reason its own name
    is worth mentioning in the first place - speaking `UJET.cx` bare is the
    failure ADR 0015 defence 4 exists to stop, whether or not it is also the
    correct, gate-passed subject."""
    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    provider = FakeProvider(json.dumps({"say": "UJET.cx 소식 들었어?"}, ensure_ascii=False))
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PROACTIVE_JUDGE: Route(provider.name, "gemma3:4b")}
    )
    bridge = _FakeBridge('{"results": [{"title": "UJET.cx raises new funding"}]}')
    judge = Judge(gateway, data_dir, bridge=bridge)
    candidate = Candidate(
        kind="topic", reason="UJET.cx 얘기를 나눈 지 오래됐다.", payload={"entity": "UJET.cx"}
    )

    utterance = await judge.decide(candidate)

    assert not utterance
    assert "url" in utterance.why_not.casefold()


async def test_a_topic_reply_with_a_different_link_is_still_declined(data_dir: Path) -> None:
    """The exemption is narrow: it does not turn off `has_url` for the whole
    `topic` path, only for the candidate's own entity name - isolated here with
    a non-pointer entity so the decline is attributable to the *unrelated* link
    alone, not to the entity's own shape (see the pointer-shaped-entity test
    above for that separate reason)."""
    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    provider = FakeProvider(
        json.dumps(
            {"say": "Sendbird 관련 공지는 sendbird-verify.app에서 확인하세요"}, ensure_ascii=False
        )
    )
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PROACTIVE_JUDGE: Route(provider.name, "gemma3:4b")}
    )
    bridge = _FakeBridge('{"results": [{"title": "Sendbird raises Series C"}]}')
    judge = Judge(gateway, data_dir, bridge=bridge)
    candidate = Candidate(
        kind="topic", reason="Sendbird 얘기를 나눈 지 오래됐다.", payload={"entity": "Sendbird"}
    )

    utterance = await judge.decide(candidate)

    assert not utterance
    assert "url" in utterance.why_not.casefold()


async def test_a_failed_search_drops_the_candidate_rather_than_raising(data_dir: Path) -> None:
    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    provider = FakeProvider('{"say": "Sendbird 소식 들었어?"}')
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PROACTIVE_JUDGE: Route(provider.name, "gemma3:4b")}
    )
    judge = Judge(gateway, data_dir, bridge=_FakeBridge(fail=True))

    utterance = await judge.decide(TOPIC)

    assert not utterance
    assert provider.calls == []


async def test_only_one_search_per_candidate_never_per_tick(data_dir: Path) -> None:
    """Non-negotiable 7's shape: deterministic generation, deterministic gate, then
    exactly one expensive step. Two `decide` calls on the same candidate must be
    two searches, never a cached or shared one that would blur that count - but a
    single `decide` must never search twice for it either."""
    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    provider = FakeProvider('{"say": "Sendbird 소식 들었어?"}')
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PROACTIVE_JUDGE: Route(provider.name, "gemma3:4b")}
    )
    bridge = _FakeBridge('{"results": [{"title": "Sendbird raises Series C"}]}')
    judge = Judge(gateway, data_dir, bridge=bridge)

    await judge.decide(TOPIC)

    assert len(bridge.calls) == 1


async def test_a_non_topic_candidate_never_touches_the_bridge(data_dir: Path) -> None:
    """The bridge is `topic`-only. Every other kind must behave exactly as before
    ADR 0015, whether or not a bridge happens to be wired."""
    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    provider = FakeProvider()
    gateway = LLMGateway(
        {provider.name: provider}, {Task.PROACTIVE_JUDGE: Route(provider.name, "gemma3:4b")}
    )
    bridge = _FakeBridge(fail=True)
    judge = Judge(gateway, data_dir, bridge=bridge)

    await judge.decide(OPEN_LOOP)

    assert len(provider.calls) == 1
    assert bridge.calls == []


async def test_prose_instead_of_json_declines_rather_than_being_spoken(
    data_dir: Path,
) -> None:
    """The reply a model gives when it has decided not to speak. Delivered as a
    line it becomes an apology out of the speaker - PLAN 6.4's accident, and the
    reason this file's contract is an object rather than a bare sentence."""
    judge, _ = judge_for(data_dir, "지금은 특별히 할 말이 없어요. 다음에 말씀드릴게요!")

    utterance = await judge.decide(VAGUE)

    assert not utterance
    assert "할 말이 없어요" not in utterance.text


async def test_an_empty_reply_declines(data_dir: Path) -> None:
    judge, _ = judge_for(data_dir, "")

    assert not await judge.decide(OPEN_LOOP)


async def test_an_over_long_reply_is_declined_not_truncated(data_dir: Path) -> None:
    """Half a sentence out of a speaker is worse than the silence it replaced, and
    silence here is free: the candidate is still in the table on the next tick."""
    monologue = "발표 어떻게 됐는지 궁금했어. " * 20
    assert len(monologue) > MAX_CHARS
    judge, _ = judge_for(data_dir, json.dumps({"say": monologue}, ensure_ascii=False))

    utterance = await judge.decide(OPEN_LOOP)

    assert not utterance
    assert str(MAX_CHARS) in utterance.why_not


async def test_a_line_at_the_limit_is_still_spoken(data_dir: Path) -> None:
    """The other side of the cap, so it cannot be tightened into silence unnoticed."""
    line = "가" * MAX_CHARS
    judge, _ = judge_for(data_dir, json.dumps({"say": line}, ensure_ascii=False))

    assert (await judge.decide(OPEN_LOOP)).text == line


async def test_a_fenced_and_prefixed_reply_is_read(data_dir: Path) -> None:
    """What gemma3:4b actually does: a cheerful preamble, then a ```json fence,
    then markdown emphasis inside the sentence it wants spoken."""
    reply = (
        "물론이죠! 아래와 같이 답변드립니다.\n"
        '```json\n{"say": "어제 **발표** 어떻게 됐어?"}\n```'
    )
    judge, _ = judge_for(data_dir, reply)

    assert (await judge.decide(OPEN_LOOP)).text == "어제 발표 어떻게 됐어?"


async def test_a_wrapping_quote_pair_is_stripped(data_dir: Path) -> None:
    """A quoted line reads as a quotation on Telegram and gets read out with the
    quotes by `say`. Only the wrapping pair goes; quotes inside stay."""
    judge, _ = judge_for(data_dir, json.dumps({"say": '"어제 발표 어떻게 됐어?"'}))

    assert (await judge.decide(OPEN_LOOP)).text == "어제 발표 어떻게 됐어?"


async def test_quotes_inside_the_line_survive(data_dir: Path) -> None:
    judge, _ = judge_for(
        data_dir, json.dumps({"say": "어제 '힘들다'고 했잖아, 좀 나아졌어?"}, ensure_ascii=False)
    )

    assert (await judge.decide(OPEN_LOOP)).text == "어제 '힘들다'고 했잖아, 좀 나아졌어?"


async def test_a_multiline_say_becomes_one_line(data_dir: Path) -> None:
    """Two lines on Telegram look like two messages, and to `say` the newline is a
    pause in the middle of one sentence."""
    judge, _ = judge_for(
        data_dir, json.dumps({"say": "어제 발표 어떻게 됐어?\n궁금했는데."}, ensure_ascii=False)
    )

    assert (await judge.decide(OPEN_LOOP)).text == "어제 발표 어떻게 됐어? 궁금했는데."


# --- the model is not there --------------------------------------------------


async def test_a_provider_error_declines_instead_of_raising(data_dir: Path) -> None:
    """The tick must record nothing, not crash: "could not reach the model" and
    "nothing worth saying" have the same consequence."""
    judge, provider = judge_for(data_dir, fail=True)

    utterance = await judge.decide(OPEN_LOOP)

    assert not utterance
    assert "unavailable" in utterance.why_not
    # It tried exactly once. The gateway owns fallback; the judge does not retry.
    assert len(provider.calls) == 1


async def test_an_unrouted_task_is_not_swallowed(data_dir: Path) -> None:
    """A gateway with no route for PROACTIVE_JUDGE is a configuration bug, and
    `ConfigError` is not a `ProviderError`. Silently declining forever would look
    exactly like a quiet week."""
    from daemon.config import ConfigError

    (data_dir / "persona" / "seed.md").write_text(SEED, encoding="utf-8")
    provider = FakeProvider()
    judge = Judge(LLMGateway({provider.name: provider}, {}), data_dir)

    with pytest.raises(ConfigError):
        await judge.decide(OPEN_LOOP)


# --- a reply the token limit cut in half (v0.1.54) ----------------------------
# The daemon went days without ever speaking first, and the log said only
# `did not return a JSON object` - which reads as the model misbehaving. It was
# not: `gemini-3.6-flash` spends thinking tokens out of the same output budget,
# so a 300-token cap ran out mid-sentence and left `{"say": "어제 그 미팅은 잘`.
# Every provider already reports why it stopped; the judge just never read it.


def test_a_truncated_reply_says_it_was_truncated() -> None:
    """The diagnostic that was missing. Naming the token limit is the difference
    between "the model is being odd" and "our cap is too small" - and the second
    is a one-line fix nobody made for days because the log never said it."""
    utterance = _read_reply(
        '{"say": "어제 그 미팅은 잘',
        model="gemini-3.6-flash",
        stop_reason="MAX_TOKENS",
    )

    assert not utterance
    assert "cut off" in utterance.why_not
    assert "MAX_OUTPUT_TOKENS" in utterance.why_not, "say which knob to turn"


@pytest.mark.parametrize("reason", ["MAX_TOKENS", "max_tokens", "length", "incomplete"])
def test_every_provider_spelling_of_truncation_is_recognised(reason: str) -> None:
    """gemini says MAX_TOKENS, anthropic max_tokens, openai length or incomplete.
    A spelling this misses degrades to the old unhelpful message, so all four are
    pinned rather than assumed."""
    assert "cut off" in _read_reply("{'say", model="m", stop_reason=reason).why_not


def test_prose_without_truncation_still_reads_as_prose() -> None:
    """The other direction: a model that answered in words rather than JSON has
    not been cut off, and calling that truncation would send the next reader to
    the wrong knob."""
    utterance = _read_reply("지금은 할 말이 없어요", model="m", stop_reason="STOP")

    assert not utterance
    assert "did not return a JSON object" in utterance.why_not


def test_the_token_budget_leaves_room_for_a_thinking_model() -> None:
    """MAX_OUTPUT_TOKENS was 300, sized for the JSON plus 120 characters of
    Korean. That arithmetic was right for a model that only answers, and wrong
    for one that thinks first out of the same allowance."""
    assert MAX_OUTPUT_TOKENS >= 1000
