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
