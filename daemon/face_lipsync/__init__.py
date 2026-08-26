"""The lip-sync render path: the model boundary, the clip cache, and compositing.

Nothing here imports anything else in `daemon/` (CONTRACTS 4). `daemon/app.py`
builds a renderer and injects it; no other module knows this package exists.
"""
