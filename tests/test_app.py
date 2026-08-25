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
    assert "/face/stream" in {r.path for r in app.routes}
