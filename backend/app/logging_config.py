"""Shared logging configuration for backend runtimes."""

from __future__ import annotations

import logging

from .config import settings

logger = logging.getLogger(__name__)

_AZURE_LOGGER_NAMES = (
    "azure",
    "azure.core",
    "azure.cosmos",
    "azure.identity",
    "azure.storage",
    "azure.search",
)


def _resolve_level(raw_level: str, default_level: int) -> int:
    normalized = raw_level.strip().upper()
    level = getattr(logging, normalized, None)
    if isinstance(level, int):
        return level
    logger.warning("Invalid log level %r. Falling back to %s.", raw_level, default_level)
    return default_level


def configure_logging() -> None:
    """Configure app and SDK logging levels."""
    app_level = _resolve_level(settings.app_log_level, logging.INFO)
    azure_level = _resolve_level(settings.azure_sdk_log_level, logging.WARNING)

    logging.basicConfig(level=app_level, format="%(message)s")
    logging.getLogger().setLevel(app_level)

    for logger_name in _AZURE_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(azure_level)
