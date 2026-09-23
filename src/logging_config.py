from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOGGER_ROOT = "birdclef"
_CONFIGURED = False


def configure_logging() -> logging.Logger:
    """Configure BirdCLEF console and rotating-file logs once per process."""
    global _CONFIGURED
    root = logging.getLogger(_LOGGER_ROOT)
    if _CONFIGURED:
        return root

    level_name = os.getenv("BIRDCLEF_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    project_root = Path(__file__).resolve().parents[1]
    log_path = Path(
        os.getenv("BIRDCLEF_LOG_FILE", str(project_root / "logs" / "birdclef.log"))
    )
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        root.exception("Could not initialize file logging at %s", log_path)

    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True
    root.info("Logging initialized: level=%s file=%s", level_name, log_path)
    return root


def get_logger(component: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"{_LOGGER_ROOT}.{component}")
