"""Configuration loading and validation (Layer 1 — Foundation).

Exposes :class:`ConfigLoader` so ``from acdp.config import ConfigLoader`` works
while keeping the package layout intact.
"""

from __future__ import annotations

from acdp.config.loader import ConfigLoader

__all__ = ["ConfigLoader"]
