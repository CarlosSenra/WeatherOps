"""Compatibilidade legada para src.tools.professor_pipeline.

Use src.tools.user_pipeline como modulo principal.
"""
from __future__ import annotations

import warnings

from . import user_pipeline as _user_pipeline
from .user_pipeline import *  # noqa: F403


warnings.warn(
    "src.tools.professor_pipeline esta depreciado; use src.tools.user_pipeline.",
    DeprecationWarning,
    stacklevel=2,
)


def main(argv: list[str] | None = None) -> int:
    return _user_pipeline.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

