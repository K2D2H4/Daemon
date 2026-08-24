"""Persona assembly: `seed.md` + `learned.md`, read fresh every turn."""

from __future__ import annotations

from pathlib import Path

from daemon.persona.loader import learned_path, load_persona, seed_path


async def test_empty_when_neither_file_exists(data_dir: Path) -> None:
    assert await load_persona(data_dir) == ""


async def test_seed_alone_is_returned_verbatim(data_dir: Path) -> None:
    seed_path(data_dir).write_text("나는 다정하다.", encoding="utf-8")
    assert await load_persona(data_dir) == "나는 다정하다."


async def test_learned_alone_still_loads(data_dir: Path) -> None:
    """Neither file is required - a fresh install with no rules yet, or a
    learned.md written before a seed exists, must both work."""
    learned_path(data_dir).write_text("- 아침엔 짧게 말한다", encoding="utf-8")

    persona = await load_persona(data_dir)

    assert "아침엔 짧게 말한다" in persona
    assert persona.startswith("What I've worked out")


async def test_both_files_are_combined_seed_first(data_dir: Path) -> None:
    seed_path(data_dir).write_text("나는 다정하다.", encoding="utf-8")
    learned_path(data_dir).write_text("- 아침엔 짧게 말한다", encoding="utf-8")

    persona = await load_persona(data_dir)

    assert persona.index("다정하다") < persona.index("아침엔 짧게 말한다")


async def test_learned_is_marked_as_separate_from_the_seed(data_dir: Path) -> None:
    """The anchor only works if the model can tell the fixed identity from what
    it accumulated (docs/CONTRACTS.md non-negotiable 5) - a rule pasted in with
    no marker would read with the same authority as the seed above it."""
    seed_path(data_dir).write_text("나는 다정하다.", encoding="utf-8")
    learned_path(data_dir).write_text("- 아침엔 짧게 말한다", encoding="utf-8")

    persona = await load_persona(data_dir)

    assert "worked out about dealing with you" in persona


async def test_the_learned_file_header_does_not_reach_the_prompt(data_dir: Path) -> None:
    """Measured on a real turn: the whole of `learned.md` was being injected, so
    the model's system prompt carried the file's header - a notice written for
    the person reading the file, including `daemon persona forget <id>` and a
    sentence LEARNED_PREFIX already says. Only the rule bullets belong in a
    prompt. `seed.md` is the asymmetric case and stays verbatim: every line of it
    was written to be prompt.
    """
    from daemon.persona.rules import render

    learned_path(data_dir).write_text(render(["아침엔 짧게 말한다"]), encoding="utf-8")

    persona = await load_persona(data_dir)

    assert "아침엔 짧게 말한다" in persona
    assert "daemon persona forget" not in persona
    assert "# learned" not in persona
    assert persona.count("worked out about dealing with you") == 1


async def test_an_unreadable_seed_is_swallowed_not_raised(data_dir: Path) -> None:
    """A conversation turn must not die because a persona file has an I/O
    problem. Made a directory rather than chmod(0o000), so the assertion does
    not depend on the test not running as root."""
    path = seed_path(data_dir)
    path.mkdir()

    assert await load_persona(data_dir) == ""


def test_a_rule_line_carries_its_date_and_how_often_it_was_seen() -> None:
    """A rule with no date is read at full weight forever, which is what made one
    remark on 2026-08-19 still be governing the daemon five days later. The date
    is absolute rather than relative because the model is already told
    `[현재 시각]` on every turn and can do the subtraction, while a stored "어제"
    is a lie by the following week."""
    from daemon.persona.loader import rule_line

    assert rule_line("변명을 싫어한다", formed="2026-08-09", observations=3) == (
        "2026-08-09 (관찰 3건) 변명을 싫어한다"
    )
    assert rule_line("짧은 답을 선호했다", formed="2026-08-23", observations=1) == (
        "2026-08-23 (관찰 1건) 짧은 답을 선호했다"
    )


async def test_the_learned_block_is_dated_when_the_mirror_can_say_so(
    data_dir: Path,
) -> None:
    (data_dir / "persona").mkdir(parents=True, exist_ok=True)
    (data_dir / "persona" / "seed.md").write_text("나는 벨라다.\n", encoding="utf-8")
    (data_dir / "persona" / "learned.md").write_text(
        "# learned\n\n- 변명을 싫어한다\n- 짧은 답을 선호했다\n", encoding="utf-8"
    )

    block = await load_persona(
        data_dir,
        annotations={
            "변명을 싫어한다": ("2026-08-09", 3),
            "짧은 답을 선호했다": ("2026-08-23", 1),
        },
    )

    assert "- 2026-08-09 (관찰 3건) 변명을 싫어한다" in block
    assert "- 2026-08-23 (관찰 1건) 짧은 답을 선호했다" in block
    # The seed is still the anchor and still verbatim, above the learned half.
    assert block.index("나는 벨라다.") < block.index("2026-08-09")


async def test_an_unannotated_rule_still_reaches_the_prompt(data_dir: Path) -> None:
    """The degrade path, and it is the common one: no mirror, a diverged mirror, or
    a body the rows do not know about. Today's behaviour - the plain bullet - is
    the fallback, because a rule dropped for want of a date is a personality
    quietly losing a piece of itself."""
    (data_dir / "persona").mkdir(parents=True, exist_ok=True)
    (data_dir / "persona" / "learned.md").write_text(
        "# learned\n\n- 변명을 싫어한다\n", encoding="utf-8"
    )

    assert "- 변명을 싫어한다" in await load_persona(data_dir)
    assert "- 변명을 싫어한다" in await load_persona(data_dir, annotations={})


async def test_the_learned_header_tells_the_model_to_weigh_not_obey(
    data_dir: Path,
) -> None:
    """`LEARNED_PREFIX` used to introduce these as things worked out about the
    owner - a flat claim, which is how a five-day-old single remark kept full
    force. It now says they are dated observations and that older, thinner ones
    count for less; without that sentence the dates are decoration."""
    from daemon.persona.loader import LEARNED_PREFIX

    assert "날짜" in LEARNED_PREFIX
    assert "관찰" in LEARNED_PREFIX
