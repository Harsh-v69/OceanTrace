"""
Logging configuration.

Plain stdlib logging to stdout - no external sinks, in keeping with the
offline-first constraint. ``configure_logging`` is idempotent so importing it
from both ``main`` and the test suite is safe.
"""
from __future__ import annotations

import logging
import logging.config
import sys

from backend.core.config import settings

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | None = None) -> None:
    """Install a single stdout handler on the root logger. Runs at most once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or ("DEBUG" if settings.DEBUG else "INFO")).upper()
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"standard": {"format": _FORMAT, "datefmt": _DATEFMT}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "standard",
                    "level": resolved,
                }
            },
            "root": {"handlers": ["console"], "level": resolved},
            "loggers": {
                "sqlalchemy.engine": {"level": "WARNING"},
                "uvicorn.error": {"level": "INFO"},
                "uvicorn.access": {"level": "WARNING"},
            },
        }
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
