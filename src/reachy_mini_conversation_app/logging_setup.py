"""Logging setup for reachy_mini_conversation_app.

Rules:
- App name derived from current working directory (not env vars, not __file__)
- Log base: ~/log/<appname>/
- One file per run: YYYY-MM-DD_HHMMSS_<appname>.log
- Directory size printed/logged on startup before first write
- No log rotation
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


def _get_app_name() -> str:
    """Derive app name from the current working directory name."""
    return Path(os.getcwd()).name


def _dir_size_bytes(path: Path) -> int:
    """Return total bytes of all files under path (non-recursive-safe)."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def setup_logging(level: int = logging.DEBUG) -> Path:
    """Configure logging for the current run.

    - Writes to ~/log/<appname>/YYYY-MM-DD_HHMMSS_<appname>.log
    - Also writes to stderr so existing console output continues working
    - Returns the Path of the created log file

    Call once at application startup, before any other logging calls.
    """
    app_name = _get_app_name()
    log_dir = Path.home() / "log" / app_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Measure existing directory size before we write anything
    existing_bytes = _dir_size_bytes(log_dir)

    # Build timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_filename = f"{timestamp}_{app_name}.log"
    log_path = log_dir / log_filename

    # ---------- formatters ----------
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---------- file handler ----------
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)

    # ---------- stderr handler ----------
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    stderr_handler.setLevel(level)

    # ---------- root logger ----------
    root = logging.getLogger()
    root.setLevel(level)
    # Remove any handlers added before setup_logging() was called
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stderr_handler)

    # First log line — report existing directory size
    startup_logger = logging.getLogger("logging_setup")
    startup_logger.info(
        "Log directory: %s  (existing size: %s)",
        log_dir,
        _human_size(existing_bytes),
    )
    startup_logger.info("Run log file: %s", log_path)

    return log_path


def get_telemetry_path(suffix: str = "telemetry") -> Path:
    """Return a sibling path for a per-run telemetry CSV.

    Example:
        ~/log/reachy_mini_conversation_app/2026-02-20_143822_reachy_mini_conversation_app_telemetry.csv

    Call after setup_logging() has been called (or pass your own timestamp).
    """
    app_name = _get_app_name()
    log_dir = Path.home() / "log" / app_name
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return log_dir / f"{timestamp}_{app_name}_{suffix}.csv"