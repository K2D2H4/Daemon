"""Daemon - a self-hosted AI companion.

`__version__` is the single source of truth for the version. Hatchling reads it at
build time (`[tool.hatch.version]` in pyproject.toml), so the installed wheel's
metadata and `daemon --version` always agree, and a source checkout reads the same
number without any package metadata present.
"""

__version__ = "0.1.50"
