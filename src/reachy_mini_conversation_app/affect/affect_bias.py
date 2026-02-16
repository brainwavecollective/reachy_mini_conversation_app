"""
AffectBias
Transforms VADCC into continuous motion bias using AffectManifold.

This class is a pure stateless converter:
- No envelope
- No attack/release
- No lingering memory

The AffectEngine owns all temporal dynamics. By the time VADCC arrives
here it is already envelope-shaped. This class simply maps it to a pose offset.

    bias = strength * axis_weights * target

Where:
    strength     = peak RBF weight from manifold (already modulated by AffectEngine)
    axis_weights = per-DOF scaling from the blended anchor
    target       = blended pose offset from the manifold anchors

Note: current pose is NOT used. This is a direct additive offset, not a
proportional attractor. Using (target - current) causes the bias to decay
toward zero as the robot approaches the target, producing microscopic movement.
"""
from __future__ import annotations

import threading
import logging
from typing import Tuple

from reachy_mini_conversation_app.affect.affect_manifold import AffectManifold

logger = logging.getLogger(__name__)

Vec6 = Tuple[float, float, float, float, float, float]
Vec5 = Tuple[float, float, float, float, float]

_ZERO6: Vec6 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class AffectBias:
    """
    Continuous emotional bias field.

    The AffectEngine provides a streaming, envelope-adjusted VADCC trajectory.
    This class converts that trajectory into a direct pose offset.
    """

    def __init__(self) -> None:
        self._manifold = AffectManifold()

        # Lock guards all fields written by update_vadcc() (async thread)
        # and read by compute_bias() (100 Hz control loop thread).
        self._lock = threading.Lock()

        self._target: Vec6 = _ZERO6
        self._strength: float = 0.0
        self._axis_weights: Vec6 = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        self._antenna_amp_deg: float = 6.0
        self._antenna_freq_hz: float = 0.5

    # ------------------------------------------------------------------
    # External update — called by MovementAdapter from AffectEngine stream
    # ------------------------------------------------------------------

    def update_vadcc(self, vadcc: Vec5) -> None:
        """Map incoming VADCC to a pose target via the manifold. Thread-safe."""
        vadcc = tuple(float(v) for v in vadcc)
        motion = self._manifold.compute_motion(vadcc)

        t = motion["target"]
        target: Vec6 = (
            float(t["x"]),
            float(t["y"]),
            float(t["z"]),
            float(t["roll"]),
            float(t["pitch"]),
            float(t["yaw"]),
        )
        strength = float(motion["strength"])
        axis_weights: Vec6 = tuple(float(a) for a in motion["axis_weights"])

        b = motion["breathing"]
        antenna_amp = float(b["antenna_amplitude_deg"])
        antenna_freq = float(b["antenna_frequency_hz"])

        with self._lock:
            self._target = target
            self._strength = strength
            self._axis_weights = axis_weights
            self._antenna_amp_deg = antenna_amp
            self._antenna_freq_hz = antenna_freq

    # ------------------------------------------------------------------
    # Bias computation — called at 100 Hz by the control loop
    # ------------------------------------------------------------------

    def compute_bias(self, current: Vec6) -> Vec6:
        """Return a direct additive pose offset driven by current VADCC state.

        Args:
            current: Unused. Kept for API compatibility with MovementManager.
                     The bias is absolute, not relative to current pose.

        Returns:
            (x, y, z, roll, pitch, yaw) offset in metres / radians.
        """
        with self._lock:
            k = self._strength
            target = self._target
            axis_weights = self._axis_weights

        if k <= 1e-6:
            return _ZERO6

        return tuple(
            k * target[i] * axis_weights[i]
            for i in range(6)
        )

    # ------------------------------------------------------------------
    # Breathing access — called by BreathingMove via callback
    # ------------------------------------------------------------------

    def get_breathing_params(self) -> Tuple[float, float]:
        """Return (antenna_amplitude_deg, antenna_frequency_hz)."""
        with self._lock:
            return self._antenna_amp_deg, self._antenna_freq_hz
