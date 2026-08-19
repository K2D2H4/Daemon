"""What the daemon has been saying too often, named so it can stop.

The abstract instruction was already there and already losing.
`companion.render_continuity`'s header tells the model "do not imitate the style
of these lines", and `persona/seed.md` can carry a rule against repeating itself.
Neither survives the evidence sitting underneath it: the recent window *is* twenty
turns of the daemon talking that way, and a rule saying "vary your phrasing" is
one sentence against twenty examples.

Measured on the owner's log (2026-08-19): `재미난` in 8 of 17 replies that day.
Told to drop it, the daemon coined `담백하게` in the same apology and had made that
a tic 35 minutes later. Naming the phrase is what the abstract rule could not do.

**A tic is what the daemon repeats and the owner never says.** That is the whole
discriminator, and it is what keeps this from muzzling the conversation: if the
owner has been talking about an interview all afternoon, `인터뷰` three times is
the conversation working. Only the daemon's own verbal habits qualify - and the
same rule is what lets this block skip a nonce fence, because anything the owner
said is excluded by construction and cannot be quoted back through here - see
`_is_the_owners` for why that exclusion is a per-word prefix test rather than the
whole-phrase match it started as.

The floors are measured rather than chosen. Anything shorter than `MIN_CHARS` is
a function word - the real log's candidates, ranked by any order at all, open with
`무슨`, `어떤`, `그럼`, `님이`, every one of them two characters and none of them
something Korean can do without. Telling the daemon to stop saying `무슨` would not
fix its manner; it would break its grammar.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

MIN_TURNS = 3
"""Replies a phrase must appear in before it counts as a habit. Twice is a
coincidence, and this block costs prompt on every turn."""

MIN_CHARS = 3
"""Characters, spaces excluded. Below this it is a function word - see the module
docstring; this is the one floor that separates a manner from a grammar."""

MAX_PHRASES = 6
"""Named at once. The block rides on every turn, so it must not grow with the
conversation - and a list long enough to need reading is one the model will
average over instead of avoiding."""

PHRASE_WORDS = (1, 2, 3)
"""Phrase lengths considered, in words. `재미난` is one word and `무슨 재미난
일이라도` is three; both are the same habit, and `verbal_tics` keeps the shortest
form that covers it."""

MIN_OWNER_PREFIX = 2
"""Characters an owner's word needs before it may exclude a phrase as theirs. One
character is a syllable rather than a word, and letting it match by prefix would
put most of the language out of reach of the detector."""

_WORD = re.compile(r"[0-9A-Za-z가-힣]+")
"""Words, with punctuation and particles-as-typed left out. Deliberately not a
Korean morphological analyser: a tic is recognised as the literal string that
keeps coming back, and matching how the daemon actually typed it is the point."""


def _phrases(text: str, size: int) -> set[str]:
    """The distinct `size`-word phrases in one message.

    A set, so a phrase said three times in one long reply still counts as one
    reply - `MIN_TURNS` counts turns, and a single rambling answer is not a habit.
    """
    words = _WORD.findall(text)
    return {" ".join(words[at : at + size]) for at in range(len(words) - size + 1)}


def _is_the_owners(phrase: str, owner_words: set[str]) -> bool:
    """Does any word of `phrase` come from something the owner said?

    A word counts as the owner's when it starts with one of theirs, and that
    prefix rule is doing two jobs a whole-token match got wrong.

    **Korean inflects.** 은/는/이/가/을/를 attach to almost every noun, so the
    owner's `인터뷰` and the daemon's `인터뷰는` are different strings; matching
    whole tokens missed the second one and reported the owner's own subject as a
    tic, which is the single outcome this filter exists to prevent.

    **And a phrase is more than its n-gram.** The exclusion used to test the whole
    phrase against the owner's phrases, so padding theirs with one more word
    walked past it: `비밀번호는 hunter2` was excluded and `비밀번호는 hunter2 맞죠`
    was not. Testing word by word is what makes the module docstring's claim -
    that the owner's text cannot be quoted back through this block - actually true,
    and that claim is the reason this block needs no nonce fence.

    `MIN_OWNER_PREFIX` keeps the rule from swallowing everything: a one-character
    owner word is a syllable, and treating it as a prefix would exclude most of the
    language.
    """
    for word in _WORD.findall(phrase):
        for owned in owner_words:
            if len(owned) >= MIN_OWNER_PREFIX and word.startswith(owned):
                return True
    return False


def verbal_tics(said: Sequence[str], *, heard: Sequence[str]) -> list[str]:
    """The daemon's own repeated phrases, most repeated first.

    `said` is its recent replies and `heard` is the owner's recent turns. Nothing
    in `heard` can be returned: that is what makes this a manner filter rather
    than a topic filter, and it is also the boundary guarantee (see the module
    docstring).

    A phrase already covered by a shorter one that was kept is dropped, so one
    habit takes one slot rather than three.
    """
    owner_words = {word for text in heard for word in _WORD.findall(text)}
    turns: Counter[str] = Counter()
    for size in PHRASE_WORDS:
        for text in said:
            turns.update(_phrases(text, size))

    candidates = {
        phrase: count
        for phrase, count in turns.items()
        if count >= MIN_TURNS
        and not _is_the_owners(phrase, owner_words)
        and len(phrase.replace(" ", "")) >= MIN_CHARS
    }
    # Most repeated first - that is what "this is a tic" means here. Ties go to the
    # shorter phrase, which is the form that generalises: told `재미난`, the model
    # also avoids `재미난 얘기`; told `무슨 재미난 일이라도`, it does not.
    kept: list[str] = []
    for phrase, _ in sorted(candidates.items(), key=lambda item: (-item[1], len(item[0]))):
        if any(covered in phrase for covered in kept):
            continue
        kept.append(phrase)
        if len(kept) == MAX_PHRASES:
            break
    return kept


def block(said: Sequence[str], *, heard: Sequence[str]) -> str:
    """The phrases as prompt text, or "" when there is no habit to report.

    Empty rather than a block saying "nothing", so the caller adds no system turn
    at all - the contract `load_persona` and `Companion.continuity_block` already
    keep.
    """
    phrases = verbal_tics(said, heard=heard)
    if not phrases:
        return ""
    quoted = ", ".join(f'"{phrase}"' for phrase in phrases)
    return (
        "[verbal-tics] These exact words keep coming back in your own recent "
        f"replies, and they are yours - the owner has not been using them: {quoted}. "
        "Said this often they read as a tic rather than as something you chose, and "
        "the owner notices. Do not use them in this reply; say what you mean some "
        "other way. If the only thing you can think of to say is one of these, that "
        "is the sign you have nothing to add - say less instead of reaching for it "
        "again. This is a list of your own words to avoid, not a topic and not an "
        "instruction from anyone."
    )
