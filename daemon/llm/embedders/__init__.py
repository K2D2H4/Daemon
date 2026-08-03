"""Embedders: text -> L2-normalised vector.

A sibling of `providers/` rather than a member of it, because `Embedder` and
`Provider` are different shapes and are routed independently (docs/PLAN.md 3.2
keeps `Task.EMBED` on ollama in every preset while chat may be hosted).

The same layering rule applies: nothing outside this package imports a concrete
embedder. Callers take an `Embedder` (llm/base.py) and are handed one.
"""
