"""Logging setup for the whole application.

Rules this module enforces:

* Libraries (everything under ``src/churnguard``) only ever call
  ``logging.getLogger(__name__)``. They never configure handlers.
* Handlers are attached exactly once, by the *entry point* (training script,
  API startup, or the pytest fixture) calling :func:`configure_logging`.
* Console output stays human-readable; the rotating file handler keeps a
  machine-greppable audit trail that survives a container restart.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from churnguard.config import SETTINGS

_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-38s | %(message)s"
_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
_configured = False


def configure_logging(
    level: str | None = None,
    log_file: Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Attach console and rotating-file handlers to the root logger.

    Idempotent: calling it twice does not duplicate log lines, which matters
    because uvicorn imports the app module more than once with ``--reload``.
    """
    global _configured
    root = logging.getLogger()
    if _configured and not force:
        return root
    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    root.setLevel(getattr(logging, (level or SETTINGS.log_level).upper(), logging.INFO))

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    target = log_file or SETTINGS.paths.log_file
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            target, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(file_handler)
    except OSError as exc:
        # A read-only filesystem must degrade to console-only, never crash.
        root.warning("File logging disabled (%s): %s", target, exc)

    logging.getLogger("churnguard").debug("Logging configured at level %s", root.level)
    _configured = True
    return root
