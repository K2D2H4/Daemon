"""`create_app`: the assembly itself, not any one piece of it.

Six tasks built a face bus, its routes, and the page that plays them - each with
its own passing tests and none of them reachable, because nothing constructed the
bus or mounted the routes on the app the product actually serves. This is the one
test that watches for exactly that gap: a `FaceBus` on `app.state` and `/face/stream`
in the assembled route table, the same failure shape `tests/test_reachable.py`
exists for, caught here at the one seam it cannot see (an argument, not a class).
"""

from __future__ import annotations

from pathlib import Path

from daemon.app import create_app
from daemon.config import Settings
from daemon.face import FaceBus


def _settings(tmp_path: Path, **kw: object) -> Settings:
    """Same base as `tests/test_admin.py`: `provider="ollama"` needs no key, and
    `_env_file=None` keeps the worktree's own `.env` out."""
    kw.setdefault("provider", "ollama")
    return Settings(_env_file=None, data_dir=tmp_path, **kw)


def test_create_app_exposes_a_face_bus_and_the_routes(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))

    assert isinstance(app.state.face, FaceBus)
    # Off the OpenAPI path table rather than `app.routes`, and that is not a
    # stylistic choice. `pyproject.toml` pins only `fastapi>=0.115`, so CI resolves
    # a newer FastAPI than a given checkout may have - and from 0.141 on,
    # `include_router` leaves an `_IncludedRouter` in `app.routes` that has neither
    # `.path` nor a `.routes` to descend into. Reading `.path` off every entry
    # passed locally on 0.115 and raised `AttributeError` on CI's 0.141. The path
    # table is public API and answers the question this test actually asks: does
    # the assembled app serve this route.
    assert "/face/stream" in app.openapi()["paths"]
