"""Logging setup for reachy_mini_conversation_app.

Rules:
- App name derived from current working directory (not env vars, not __file__)
- Log base: ~/log/<appname>/
- One file per run: YYYY-MM-DD_HHMMSS_<appname>.log
- Directory size printed/logged on startup before first write
- No log rotation

Session ID:
- 8 hex chars generated once per process at import time
- Shared across all three telemetry files for cross-file joining
- Accessible via get_session_id()
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Session ID — generated once at import, stable for the process lifetime
# ---------------------------------------------------------------------------
_SESSION_ID: str = uuid.uuid4().hex[:8]


def get_session_id() -> str:
    """Return the 8-char hex session ID for this process run."""
    return _SESSION_ID


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

    existing_bytes = _dir_size_bytes(log_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_filename = f"{timestamp}_{app_name}.log"
    log_path = log_dir / log_filename

    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(fmt)
    stderr_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stderr_handler)

    startup_logger = logging.getLogger("logging_setup")
    startup_logger.info(
        "Log directory: %s  (existing size: %s)",
        log_dir,
        _human_size(existing_bytes),
    )
    startup_logger.info("Run log file: %s", log_path)
    startup_logger.info("Session ID: %s", _SESSION_ID)

    return log_path


def _telemetry_path(suffix: str) -> Path:
    """Internal helper: build a timestamped telemetry CSV path."""
    app_name = _get_app_name()
    log_dir = Path.home() / "log" / app_name
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return log_dir / f"{timestamp}_{_SESSION_ID}_{app_name}_{suffix}.csv"


def get_kinematics_telemetry_path() -> Path:
    """Path for kinematics (motor) telemetry CSV."""
    return _telemetry_path("kinematics")


def get_conversation_telemetry_path() -> Path:
    """Path for conversation turn telemetry CSV."""
    return _telemetry_path("conversation")


def get_anima_telemetry_path() -> Path:
    """Path for anima processing telemetry CSV.

    The anima library writes here directly; the conversation app passes this
    path when constructing AnimaTelemetryWriter.
    """
    return _telemetry_path("anima_processing")


# ---------------------------------------------------------------------------
# Back-compat alias (existing code calls get_telemetry_path())
# ---------------------------------------------------------------------------
def get_telemetry_path(suffix: str = "telemetry") -> Path:
    """Legacy helper — kept for back-compat. New code should call the
    specific get_*_telemetry_path() helpers instead."""
    return _telemetry_path(suffix)