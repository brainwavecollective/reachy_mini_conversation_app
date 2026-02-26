"""Conversation turn telemetry writer.

One row per user/assistant turn. Lightweight — pure text events, no motion data.

Columns:
  session_id     — shared 8-char hex session identifier
  wall_time_iso  — ISO timestamp
  monotonic_s    — time.monotonic() elapsed since writer construction
  role           — "user" | "assistant"
  content        — message text
  utterance_id   — populated for assistant turns (links to anima_processing.csv)
                   blank for user turns
"""

from __future__ import annotations

import csv
import io
import logging
import threading
import time
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

HEADERS = [
    "session_id",
    "wall_time_iso",
    "monotonic_s",
    "role",
    "content",
    "utterance_id",   # links to anima_processing.csv utterance_id (assistant turns only)
]


@dataclass
class ConversationRecord:
    session_id: str
    wall_time_iso: str
    monotonic_s: float
    role: str
    content: str
    utterance_id: str  # empty string when not applicable


assert len(HEADERS) == len(fields(ConversationRecord))


class ConversationTelemetryWriter:
    """Simple synchronous CSV writer for conversation turns.

    Volume is very low (one row per turn) so a ring buffer is overkill —
    writes happen directly under a lock.
    """

    def __init__(self, output_path: Path, session_id: str):
        self._path = output_path
        self.session_id = session_id
        self._lock = threading.Lock()
        self._file: Optional[io.TextIOWrapper] = None
        self._csv_writer = None
        self._rows_written = 0
        self._monotonic_start = time.monotonic()

    def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._file)
        self._csv_writer.writerow(HEADERS)
        self._file.flush()
        logger.info("Conversation telemetry writer started → %s", self._path)

    def stop(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.flush()
                self._file.close()
                self._file = None
        logger.info("Conversation telemetry writer stopped. rows=%d", self._rows_written)

    def write_turn(
        self,
        role: str,
        content: str,
        utterance_id: str = "",
    ) -> None:
        """Write one conversation turn. Thread-safe."""
        if self._csv_writer is None or self._file is None:
            return
        now = datetime.now().isoformat(timespec="milliseconds")
        elapsed = time.monotonic() - self._monotonic_start
        rec = ConversationRecord(
            session_id=self.session_id,
            wall_time_iso=now,
            monotonic_s=elapsed,
            role=role,
            content=content,
            utterance_id=utterance_id,
        )
        with self._lock:
            if self._csv_writer is None or self._file is None:
                return
            self._csv_writer.writerow([
                rec.session_id,
                rec.wall_time_iso,
                f"{rec.monotonic_s:.4f}",
                rec.role,
                rec.content,
                rec.utterance_id,
            ])
            self._file.flush()
            self._rows_written += 1
