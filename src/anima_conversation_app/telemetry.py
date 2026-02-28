"""Kinematics telemetry ring buffer and CSV writer.

Captures physical robot state at a tunable sample rate (default 10 Hz).
Motor readbacks at a further-reduced rate (default 5 Hz).

Anima VADCC state has been moved to anima_processing.csv (written by the
anima library directly). This file contains only physical/motion state.

Fields added vs previous version:
  session_id — 8-char hex shared identifier for cross-file joining

Fields removed vs previous version:
  anima_valence, anima_arousal, anima_dominance, anima_complexity, anima_coherence
  baseline_valence, baseline_arousal, baseline_dominance, baseline_complexity, baseline_coherence
  burst_influence
  (all moved to anima_processing.csv)
"""

from __future__ import annotations

import csv
import io
import logging
import threading
import time
from dataclasses import dataclass, fields, astuple
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

HEADERS = [
    # --- Identity ---
    "session_id",

    # --- Timestamps ---
    "wall_time_iso",
    "monotonic_s",

    # --- Emotional offsets produced by MovementAdapter ---
    "emo_offset_head_x",
    "emo_offset_head_y",
    "emo_offset_head_z",
    "emo_offset_head_roll",
    "emo_offset_head_pitch",
    "emo_offset_head_yaw",
    "emo_offset_body_yaw",

    # --- Antenna adapter output ---
    "ant_left_base_rad",
    "ant_right_base_rad",
    "ant_amp_mult",
    "ant_freq_mult",

    # --- Dominant emotional anchor (what's driving motion) ---
    "dominant_anchor_name",
    "dominant_anchor_weight",

    # --- Commanded pose (what we sent to robot) ---
    "cmd_head_z",
    "cmd_head_pitch",
    "cmd_head_roll",
    "cmd_head_yaw",
    "cmd_ant_left",
    "cmd_ant_right",
    "cmd_body_yaw",

    # --- Actual motor state (from robot sensor readback, reduced rate) ---
    "actual_ant_left",
    "actual_ant_right",
    "actual_head_z",
    "actual_head_pitch",
    "actual_head_roll",
    "actual_head_yaw",
    "actual_body_yaw",
    "ik_failed",
]


@dataclass
class KinematicsTelemetryRecord:
    """One kinematics telemetry sample. Field order must match HEADERS exactly."""
    # Identity
    session_id: str

    # Timestamps
    wall_time_iso: str
    monotonic_s: float

    # Emotional head/body offsets
    emo_offset_head_x: float
    emo_offset_head_y: float
    emo_offset_head_z: float
    emo_offset_head_roll: float
    emo_offset_head_pitch: float
    emo_offset_head_yaw: float
    emo_offset_body_yaw: float

    # Antenna adapter output
    ant_left_base_rad: float
    ant_right_base_rad: float
    ant_amp_mult: float
    ant_freq_mult: float

    # Dominant anchor
    dominant_anchor_name: str
    dominant_anchor_weight: float

    # Commanded pose
    cmd_head_z: float
    cmd_head_pitch: float
    cmd_head_roll: float
    cmd_head_yaw: float
    cmd_ant_left: float
    cmd_ant_right: float
    cmd_body_yaw: float

    # Actual motor state (optional — empty string when not sampled)
    actual_ant_left: str
    actual_ant_right: str
    actual_head_z: str
    actual_head_pitch: str
    actual_head_roll: str
    actual_head_yaw: str
    actual_body_yaw: str

    ik_failed: bool


assert len(HEADERS) == len(fields(KinematicsTelemetryRecord)), (
    f"HEADERS ({len(HEADERS)}) and KinematicsTelemetryRecord fields "
    f"({len(fields(KinematicsTelemetryRecord))}) are out of sync!"
)

# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------

class TelemetryBuffer:
    """Fixed-size ring buffer for KinematicsTelemetryRecord objects."""

    def __init__(self, capacity: int = 1024):
        self._buf: list[Optional[KinematicsTelemetryRecord]] = [None] * capacity
        self._capacity = capacity
        self._write_idx = 0
        self._read_idx = 0
        self._lock = threading.Lock()
        self._dropped = 0

    def push(self, record: KinematicsTelemetryRecord) -> None:
        with self._lock:
            next_write = (self._write_idx + 1) % self._capacity
            if next_write == self._read_idx:
                self._dropped += 1
                self._read_idx = (self._read_idx + 1) % self._capacity
            self._buf[self._write_idx] = record
            self._write_idx = next_write

    def drain(self) -> list[KinematicsTelemetryRecord]:
        with self._lock:
            result = []
            while self._read_idx != self._write_idx:
                rec = self._buf[self._read_idx]
                if rec is not None:
                    result.append(rec)
                self._read_idx = (self._read_idx + 1) % self._capacity
            return result

    @property
    def dropped(self) -> int:
        return self._dropped


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class TelemetryWriter:
    """Owns the ring buffer, drain thread, and CSV file."""

    def __init__(
        self,
        output_path: Path,
        drain_interval_s: float = 0.5,
        buffer_capacity: int = 2048,
    ):
        self._path = output_path
        self._drain_interval = drain_interval_s
        self._buffer = TelemetryBuffer(buffer_capacity)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file: Optional[io.TextIOWrapper] = None
        self._csv_writer = None
        self._rows_written = 0

    def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._file)
        self._csv_writer.writerow(HEADERS)
        self._file.flush()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._drain_loop,
            name="kinematics-telemetry-drain",
            daemon=True,
        )
        self._thread.start()
        logger.info("Kinematics telemetry writer started → %s", self._path)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._flush()
        if self._file is not None:
            self._file.close()
            self._file = None
        dropped = self._buffer.dropped
        logger.info(
            "Kinematics telemetry writer stopped. rows=%d dropped=%d path=%s",
            self._rows_written, dropped, self._path,
        )

    def push(self, record: KinematicsTelemetryRecord) -> None:
        self._buffer.push(record)

    def _drain_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self._drain_interval)
            self._flush()

    def _flush(self) -> None:
        records = self._buffer.drain()
        if not records or self._csv_writer is None or self._file is None:
            return
        for rec in records:
            self._csv_writer.writerow(_record_to_row(rec))
        self._rows_written += len(records)
        self._file.flush()
        if self._buffer.dropped:
            logger.warning(
                "Kinematics telemetry buffer overrun: %d records dropped total",
                self._buffer.dropped,
            )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: float, precision: int = 5) -> str:
    if v is None:
        return ""
    return f"{v:.{precision}f}"


def _record_to_row(rec: KinematicsTelemetryRecord) -> list:
    return [
        rec.session_id,
        rec.wall_time_iso,
        _fmt(rec.monotonic_s, 4),

        _fmt(rec.emo_offset_head_x),
        _fmt(rec.emo_offset_head_y),
        _fmt(rec.emo_offset_head_z),
        _fmt(rec.emo_offset_head_roll),
        _fmt(rec.emo_offset_head_pitch),
        _fmt(rec.emo_offset_head_yaw),
        _fmt(rec.emo_offset_body_yaw),

        _fmt(rec.ant_left_base_rad),
        _fmt(rec.ant_right_base_rad),
        _fmt(rec.ant_amp_mult),
        _fmt(rec.ant_freq_mult),

        rec.dominant_anchor_name,
        _fmt(rec.dominant_anchor_weight),

        _fmt(rec.cmd_head_z),
        _fmt(rec.cmd_head_pitch),
        _fmt(rec.cmd_head_roll),
        _fmt(rec.cmd_head_yaw),
        _fmt(rec.cmd_ant_left),
        _fmt(rec.cmd_ant_right),
        _fmt(rec.cmd_body_yaw),

        rec.actual_ant_left,
        rec.actual_ant_right,
        rec.actual_head_z,
        rec.actual_head_pitch,
        rec.actual_head_roll,
        rec.actual_head_yaw,
        rec.actual_body_yaw,
        str(rec.ik_failed),
    ]


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_record(
    *,
    session_id: str,
    monotonic_s: float,
    motion,                          # MotionTarget from adapter
    ant_left_base: float,
    ant_right_base: float,
    ant_amp_mult: float,
    ant_freq_mult: float,
    dominant_anchor_name: str,
    dominant_anchor_weight: float,
    cmd_head_z: float,
    cmd_head_pitch: float,
    cmd_head_roll: float,
    cmd_head_yaw: float,
    cmd_ant_left: float,
    cmd_ant_right: float,
    cmd_body_yaw: float,
    actual_ant_left: Optional[float] = None,
    actual_ant_right: Optional[float] = None,
    actual_head_z: Optional[float] = None,
    actual_head_pitch: Optional[float] = None,
    actual_head_roll: Optional[float] = None,
    actual_head_yaw: Optional[float] = None,
    actual_body_yaw: Optional[float] = None,
    ik_failed: bool = False,
) -> KinematicsTelemetryRecord:
    """Construct a KinematicsTelemetryRecord from the values available in the control loop."""
    import datetime

    def opt_fmt(v: Optional[float]) -> str:
        return _fmt(v) if v is not None else ""

    return KinematicsTelemetryRecord(
        session_id=session_id,
        wall_time_iso=datetime.datetime.now().isoformat(timespec="milliseconds"),
        monotonic_s=monotonic_s,

        emo_offset_head_x=motion.head_x,
        emo_offset_head_y=motion.head_y,
        emo_offset_head_z=motion.head_z,
        emo_offset_head_roll=motion.head_roll,
        emo_offset_head_pitch=motion.head_pitch,
        emo_offset_head_yaw=motion.head_yaw,
        emo_offset_body_yaw=motion.body_yaw,

        ant_left_base_rad=ant_left_base,
        ant_right_base_rad=ant_right_base,
        ant_amp_mult=ant_amp_mult,
        ant_freq_mult=ant_freq_mult,

        dominant_anchor_name=dominant_anchor_name,
        dominant_anchor_weight=dominant_anchor_weight,

        cmd_head_z=cmd_head_z,
        cmd_head_pitch=cmd_head_pitch,
        cmd_head_roll=cmd_head_roll,
        cmd_head_yaw=cmd_head_yaw,
        cmd_ant_left=cmd_ant_left,
        cmd_ant_right=cmd_ant_right,
        cmd_body_yaw=cmd_body_yaw,

        actual_ant_left=opt_fmt(actual_ant_left),
        actual_ant_right=opt_fmt(actual_ant_right),
        actual_head_z=opt_fmt(actual_head_z),
        actual_head_pitch=opt_fmt(actual_head_pitch),
        actual_head_roll=opt_fmt(actual_head_roll),
        actual_head_yaw=opt_fmt(actual_head_yaw),
        actual_body_yaw=opt_fmt(actual_body_yaw),
        ik_failed=ik_failed,
    )
