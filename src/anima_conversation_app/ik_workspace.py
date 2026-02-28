"""IK workspace hull for Reachy Mini head pose validation.

Loads the convex hull characterization produced by probe_ik_limits.py and
exposes a cheap point-in-hull query for runtime observation or clamping.

The hull covers the three axes that compound to hit IK limits during
emotional expression: z (metres), pitch (radians), roll (radians).
Yaw, x, and y are logged as context but not hull-checked — they stay
well within limits across all defined emotional anchors.

Usage:
    from anima_conversation_app.ik_workspace import IKWorkspace

    ws = IKWorkspace()               # loads hull once at import/init
    result = ws.check(z, pitch, roll)
    # result.inside: bool | None
    # result.tag:    "IN_HULL" | "OUT_OF_HULL" | "NO_HULL"
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default path relative to project root (where the app is invoked from)
_DEFAULT_HULL_PATH = "tests/ik_workspace_hull.json"


@dataclass(frozen=True)
class WorkspaceCheckResult:
    """Result of a single hull query."""
    z: float
    pitch: float
    roll: float
    inside: Optional[bool]       # None if hull not loaded

    @property
    def tag(self) -> str:
        if self.inside is None:
            return "NO_HULL"
        return "IN_HULL" if self.inside else "OUT_OF_HULL"


class IKWorkspace:
    """Convex hull query for the Reachy Mini head pose workspace.

    Loads the hull equations from a JSON file produced by probe_ik_limits.py.
    The hull is loaded once at construction and held in memory — the query
    is a single matrix multiply, suitable for use at 100 Hz if needed.

    Hull axes (in order): [z_m, pitch_rad, roll_rad, 1.0]
    A point is inside when: equations @ point <= 0  (all rows)
    """

    def __init__(self, hull_path: str = _DEFAULT_HULL_PATH) -> None:
        self._equations: Optional[np.ndarray] = None
        self._hull_path = hull_path
        self._load(hull_path)

    def _load(self, path: str) -> None:
        p = pathlib.Path(path)
        if not p.exists():
            logger.warning(
                "IK hull file not found at %s — workspace checks disabled. "
                "Run tests/probe_ik_limits.py to generate it.",
                path,
            )
            return
        try:
            data = json.loads(p.read_text())
            self._equations = np.array(data["equations"], dtype=np.float64)
            logger.info(
                "Loaded IK workspace hull: %d halfspaces, %d vertices, "
                "%d/%d samples achievable (%.1f%%) — axes: %s",
                len(self._equations),
                len(data.get("vertices", [])),
                data.get("n_achievable", 0),
                data.get("n_samples", 0),
                100.0 * data.get("n_achievable", 0) / max(data.get("n_samples", 1), 1),
                data.get("axes", ["z_m", "pitch_rad", "roll_rad"]),
            )
        except Exception as e:
            logger.warning("Failed to load IK hull from %s: %s", path, e)

    @property
    def available(self) -> bool:
        """True if hull was loaded successfully."""
        return self._equations is not None

    def check(self, z: float, pitch: float, roll: float) -> WorkspaceCheckResult:
        """Test whether (z, pitch, roll) lies inside the achievable workspace.

        Args:
            z:     Head z translation in metres.
            pitch: Head pitch in radians.
            roll:  Head roll in radians.

        Returns:
            WorkspaceCheckResult with inside=True/False/None and a log tag.
        """
        if self._equations is None:
            return WorkspaceCheckResult(z=z, pitch=pitch, roll=roll, inside=None)

        point = np.array([z, pitch, roll, 1.0])
        inside = bool(np.all(self._equations @ point <= 0))
        return WorkspaceCheckResult(z=z, pitch=pitch, roll=roll, inside=inside)

    def log_observation(
        self,
        z: float,
        pitch: float,
        roll: float,
        yaw: float = 0.0,
        emotional_z: float = 0.0,
        emotional_pitch: float = 0.0,
        emotional_roll: float = 0.0,
        emotional_yaw: float = 0.0,
    ) -> WorkspaceCheckResult:
        """Check workspace and emit a structured INFO log line.

        Logs both the total composed offsets (z, pitch, roll, yaw) and the
        raw emotional component separately, so speech/face-tracking
        contributions are visible.

        Returns the WorkspaceCheckResult for the caller to use if needed.
        """
        result = self.check(z, pitch, roll)

        logger.debug(
            "[WORKSPACE] %s  "
            "z=%+.4fm(%+.1fmm)  pitch=%+.4frad(%+.1f°)  roll=%+.4frad(%+.1f°)  yaw=%+.4frad(%+.1f°)"
            "  | emotional: z=%+.4f  pitch=%+.4f  roll=%+.4f  yaw=%+.4f",
            result.tag,
            z,     z * 1000,
            pitch, np.rad2deg(pitch),
            roll,  np.rad2deg(roll),
            yaw,   np.rad2deg(yaw),
            emotional_z,
            emotional_pitch,
            emotional_roll,
            emotional_yaw,
        )

        return result
