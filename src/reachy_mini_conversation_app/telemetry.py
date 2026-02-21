"""Telemetry ring buffer and CSV writer for behavioral analytics.

Design goals:
- Zero-allocation writes in the 100 Hz control loop (ring buffer, pre-allocated slots)
- All I/O happens in a background drain thread, fully decoupled from the hot path
- Easy to add new fields: extend TelemetryRecord and HEADERS together
- Tunable sample rate via config (TELEMETRY_SAMPLE_RATE_HZ)

Data captured per sample:
  Timestamps           | wall clock + monotonic elapsed
  Anima VADCC          | the raw emotional state vector from the engine
  Anima influence      | the blend influence scalar applied to burst
  Anima baseline       | the slow-drift baseline VADCC
  Emotional offsets    | the head/body deltas the adapter produced
  Commanded pose       | what we actually sent to the robot (post-fusion)
  Actual motor state   | what the robot reports back (sampled at reduced rate)
  Antenna params       | base angles + amp/freq multipliers
  Dominant anchor      | which emotional anchor is dominating (name + weight)
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
# Record schema
# ---------------------------------------------------------------------------
# To add a new field: add it here AND add the matching header string to HEADERS.
# The order of HEADERS must match the order of fields in TelemetryRecord.

HEADERS = [
    # --- Timestamps ---
    "wall_time_iso",          # ISO-8601 wall clock string
    "monotonic_s",            # time.monotonic() at sample point

    # --- Anima emotional state ---
    "anima_valence",
    "anima_arousal",
    "anima_dominance",
    "anima_complexity",
    "anima_coherence",

    # --- Anima baseline (slow drift) ---
    "baseline_valence",
    "baseline_arousal",
    "baseline_dominance",
    "baseline_complexity",
    "baseline_coherence",

    # --- Anima burst influence scalar (None if no burst this sample) ---
    "burst_influence",

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

    # --- Dominant emotional anchor ---
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

    # --- Actual motor state (from robot sensor readback, sampled at reduced rate) ---
    # None/empty when not sampled this tick
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
class TelemetryRecord:
    """One telemetry sample.  Field order must match HEADERS exactly."""
    # Timestamps
    wall_time_iso: str
    monotonic_s: float

    # Anima VADCC
    anima_valence: float
    anima_arousal: float
    anima_dominance: float
    anima_complexity: float
    anima_coherence: float

    # Anima baseline
    baseline_valence: float
    baseline_arousal: float
    baseline_dominance: float
    baseline_complexity: float
    baseline_coherence: float

    # Burst influence
    burst_influence: Optional[float]

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


# Sanity-check at import time that headers and dataclass fields stay in sync.
assert len(HEADERS) == len(fields(TelemetryRecord)), (
    f"HEADERS ({len(HEADERS)}) and TelemetryRecord fields "
    f"({len(fields(TelemetryRecord))}) are out of sync!"
)


# ---------------------------------------------------------------------------
# Ring buffer (lock-free writer, locked reader)
# ---------------------------------------------------------------------------

class TelemetryBuffer:
    """Fixed-size ring buffer for TelemetryRecord objects.

    Writers (control loop thread) call push() — O(1), no allocation.
    The drain thread calls drain() — returns a list of pending records.

    Because there is exactly one writer and one reader, a single lock on
    drain() is sufficient; push() never contends.
    """

    def __init__(self, capacity: int = 1024):
        self._buf: list[Optional[TelemetryRecord]] = [None] * capacity
        self._capacity = capacity
        self._write_idx = 0          # owned by writer
        self._read_idx = 0           # owned by drain thread, protected by lock
        self._lock = threading.Lock()
        self._dropped = 0

    def push(self, record: TelemetryRecord) -> None:
        """Write a record.  Called from the control loop thread."""
        next_write = (self._write_idx + 1) % self._capacity
        if next_write == self._read_idx:
            # Buffer full — overwrite oldest and count the drop
            self._dropped += 1
            self._read_idx = (self._read_idx + 1) % self._capacity
        self._buf[self._write_idx] = record
        self._write_idx = next_write

    def drain(self) -> list[TelemetryRecord]:
        """Return all unread records and advance read pointer.  Called from drain thread."""
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
# Telemetry writer (drain thread + CSV output)
# ---------------------------------------------------------------------------

class TelemetryWriter:
    """Owns the ring buffer, drain thread, and CSV file.

    Usage:
        writer = TelemetryWriter(path, drain_interval_s=0.5)
        writer.start()
        writer.push(record)   # from control loop, cheap
        writer.stop()         # flushes remaining records, closes file
    """

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
        self._csv_writer: Optional[csv.writer] = None
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
            name="telemetry-drain",
            daemon=True,
        )
        self._thread.start()
        logger.info("Telemetry writer started → %s", self._path)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        # Final flush
        self._flush()
        if self._file is not None:
            self._file.close()
            self._file = None
        dropped = self._buffer.dropped
        logger.info(
            "Telemetry writer stopped. rows=%d dropped=%d path=%s",
            self._rows_written, dropped, self._path,
        )

    def push(self, record: TelemetryRecord) -> None:
        """Called from the hot path (control loop thread)."""
        self._buffer.push(record)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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
                "Telemetry buffer overrun: %d records dropped total",
                self._buffer.dropped,
            )


def _fmt(v: float, precision: int = 5) -> str:
    """Format float with fixed precision, or empty string for None."""
    if v is None:
        return ""
    return f"{v:.{precision}f}"


def _record_to_row(rec: TelemetryRecord) -> list:
    """Convert a TelemetryRecord to a CSV row list."""
    return [
        rec.wall_time_iso,
        _fmt(rec.monotonic_s, 4),

        _fmt(rec.anima_valence),
        _fmt(rec.anima_arousal),
        _fmt(rec.anima_dominance),
        _fmt(rec.anima_complexity),
        _fmt(rec.anima_coherence),

        _fmt(rec.baseline_valence),
        _fmt(rec.baseline_arousal),
        _fmt(rec.baseline_dominance),
        _fmt(rec.baseline_complexity),
        _fmt(rec.baseline_coherence),

        _fmt(rec.burst_influence) if rec.burst_influence is not None else "",

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
# Convenience: build a TelemetryRecord from live data
# ---------------------------------------------------------------------------

def build_record(
    *,
    monotonic_s: float,
    vadcc: tuple,                    # 5-tuple from anima
    baseline: tuple,                 # 5-tuple baseline
    burst_influence: Optional[float],
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
) -> TelemetryRecord:
    """Construct a TelemetryRecord from the values available in the control loop."""
    import datetime

    def opt_fmt(v: Optional[float]) -> str:
        return _fmt(v) if v is not None else ""

    return TelemetryRecord(
        wall_time_iso=datetime.datetime.now().isoformat(timespec="milliseconds"),
        monotonic_s=monotonic_s,

        anima_valence=vadcc[0],
        anima_arousal=vadcc[1],
        anima_dominance=vadcc[2],
        anima_complexity=vadcc[3],
        anima_coherence=vadcc[4],

        baseline_valence=baseline[0],
        baseline_arousal=baseline[1],
        baseline_dominance=baseline[2],
        baseline_complexity=baseline[3],
        baseline_coherence=baseline[4],

        burst_influence=burst_influence,

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