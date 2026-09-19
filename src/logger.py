"""
Centralized logging configuration using loguru.
"""
from __future__ import annotations

import sys
from loguru import logger

from src.config import get_settings


def setup_logging() -> None:
    """Configure loguru for the application."""
    settings = get_settings()
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
        ),
        level=settings.log_level,
        colorize=True,
    )
    logger.add(
        "logs/app.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time} | {level} | {name}:{line} — {message}",
    )


# Initialize on import
setup_logging()

__all__ = ["logger"]
