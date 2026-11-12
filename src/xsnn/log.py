"""Logging setup shared by scripts and the library."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logging to stderr and, optionally, a file.

    Calling this twice replaces the previous handlers.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
