"""What the daemon keeps saying, and why naming it is the only thing that stops it.

`render_continuity`'s header already tells the model "do not imitate the style of
these lines", and `persona/seed.md` can be given a rule against repeating itself.
Both are abstract, and both lose: measured on the owner's own log, the daemon used
`재미난` in 8 of 17 replies in a day, was asked to stop, coined `담백하게` while
apologising, and had made *that* a tic 35 minutes later. An instruction not to
repeat yourself is outargued by twenty turns of evidence that this is how you talk.
So the phrases go in by name.
"""

from daemon.persona import tics


def test_a_phrase_the_daemon_keeps_using_is_named() -> None:
    said = [
        "무슨 재미난 얘기라도 있어요?",
        "오늘은 재미난 일 없었어요?",
        "재미난 얘기 좀 해봐요.",
    ]
    assert "재미난" in tics.verbal_tics(said, heard=[])


def test_the_topic_is_not_a_tic_because_the_owner_says_it_too() -> None:
    """The whole discriminator. If the owner is talking about an interview all
    afternoon, the daemon saying `인터뷰` three times is the conversation working,
    not a tic - and telling it to stop would make it unable to discuss the subject
    the owner raised. A tic is something *it* says that *they* do not."""
    said = ["인터뷰 잘 봤어요?", "인터뷰 언제예요?", "인터뷰 준비는요?"]
    heard = ["오늘 인터뷰 있어"]

    assert tics.verbal_tics(said, heard=heard) == []


def test_two_characters_is_a_function_word_not_a_tic() -> None:
    """Measured on the real log: sorting the candidates by anything at all puts
    `무슨`, `어떤`, `그럼`, `님이` on top - every one of them two characters, and
    every one a word ordinary Korean cannot do without. Telling the daemon to stop
    saying `무슨` does not fix its manner, it breaks its grammar."""
    said = ["무슨 일 있어요?", "무슨 생각해요?", "무슨 소리예요?"]

    assert "무슨" not in tics.verbal_tics(said, heard=[])


def test_twice_is_not_yet_a_habit() -> None:
    said = ["재미난 얘기 있어요?", "재미난 일은요?", "오늘 어땠어요?"]

    assert tics.verbal_tics(said, heard=[]) == []


def test_the_longer_phrase_around_a_named_one_is_not_listed_twice() -> None:
    """`재미난` and `무슨 재미난` and `무슨 재미난 일이라도` are one habit. Listing
    all three spends the budget on one tic and reads as three."""
    said = ["무슨 재미난 일이라도 있어요?"] * 3
    found = tics.verbal_tics(said, heard=[])

    assert found == [phrase for phrase in found if "재미난" not in phrase or phrase == "재미난"]
    assert len(found) == len(set(found))


def test_nothing_repeated_is_no_block_at_all() -> None:
    """An empty string, so the caller adds no system turn rather than an empty one -
    the same contract `load_persona` and `continuity_block` already keep."""
    assert tics.block(["오늘 날씨 좋네요.", "그건 몰랐어요."], heard=[]) == ""


def test_the_owners_words_never_reach_the_block() -> None:
    """The block is quoted back into the prompt, so what it may contain is a
    boundary question. By construction it holds only phrases the daemon itself
    repeated *and* the owner never used, which is why it needs no nonce fence: the
    owner's text is excluded by the same rule that finds the tic."""
    secret = "비밀번호는 hunter2"
    said = [f"{secret} 라고 하셨죠?"] * 3
    heard = [secret]

    assert secret not in tics.block(said, heard=heard)


def test_the_block_is_bounded() -> None:
    """It rides on every turn, so it cannot grow with the conversation."""
    said = [" ".join(f"버릇{i}단어" for i in range(40))] * 5
    found = tics.verbal_tics(said, heard=[])

    assert 0 < len(found) <= tics.MAX_PHRASES


def test_a_longer_phrase_wrapped_around_the_owners_words_is_still_excluded() -> None:
    """The exclusion was per-n-gram, so padding the owner's words with one more
    word walked straight past it: `heard=["비밀번호는 hunter2"]` against a daemon
    that echoed `비밀번호는 hunter2 맞죠?` three times produced `hunter2 맞죠`, and
    the owner's own text went into the prompt inside a block whose whole claim is
    that it cannot carry any."""
    heard = ["비밀번호는 hunter2"]
    said = ["비밀번호는 hunter2 맞죠?"] * 3

    assert "hunter2" not in " ".join(tics.verbal_tics(said, heard=heard))
    assert "hunter2" not in tics.block(said, heard=heard)


def test_a_particle_does_not_smuggle_the_topic_past_the_filter() -> None:
    """Korean attaches 은/는/이/가/을/를 to almost every noun, so `인터뷰` and
    `인터뷰는` are different strings and whole-token matching missed the second
    one entirely - handing back a daemon forbidden to name the subject the owner
    raised, which is the one thing this filter exists to prevent."""
    heard = ["오늘 인터뷰 있어"]
    said = ["인터뷰는 어땠어요?", "인터뷰는 몇 시예요?", "인터뷰는 준비 다 됐어요?"]

    assert tics.verbal_tics(said, heard=heard) == []


def test_a_real_tic_survives_the_wider_exclusion() -> None:
    """The exclusion is now a prefix rule, so it has to be shown not to swallow the
    thing it is filtering for: the owner mentioning 얘기 must not make 재미난
    unreportable."""
    heard = ["무슨 얘기 하려고?"]
    said = ["재미난 얘기 있어요?", "재미난 일은요?", "재미난 소식이라도?"]

    assert "재미난" in tics.verbal_tics(said, heard=heard)
