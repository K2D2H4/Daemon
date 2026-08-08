"""The admin web — a local, loopback-only control plane (M5, Phase 1).

Four things and no fifth: read health, run a side-effect-free chat test, edit
settings (applied by a self-restart), and - Phase 2, behind DAEMON_MCP_ENABLED -
add an MCP server. It is **not a fourth conversation channel**; the real
conversation lives on Telegram and voice (docs/design/2026-08-07-m5-admin-web-design.md).

Mounted onto the existing FastAPI app by `daemon/app.py`. Every route reads its
handles off `app.state` and talks to protocols, so the layering exception stays
where it belongs (CONTRACTS 4).
"""

from __future__ import annotations

from daemon.admin.routes import router

__all__ = ["router"]
