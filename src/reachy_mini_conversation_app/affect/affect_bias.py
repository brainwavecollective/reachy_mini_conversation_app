"""
AffectBias

Transforms VADCC into motion bias using AffectManifold.
Pure runtime signal — no file polling.
"""

import time
from typing import Tuple

from reachy_mini_conversation_app.affect.affect_manifold import AffectManifold


Vec6 = Tuple[float, float, float, float, float, float]
Vec5 = Tuple[float, float, float, float, float]


class AffectBias:
    """
    Converts VADCC into continuous motion bias.

    - update_vadcc() sets new emotional target.
    - compute_bias() is called inside 100Hz loop.
    """

    def __init__(self) -> None:
        self._manifold = AffectManifold()

        self._vadcc: Vec5 = (0.5, 0.5, 0.5, 0.5, 0.5)

        self._target: Vec6 = (0, 0, 0, 0, 0, 0)
        self._strength = 0.0
        self._envelope = 0.0
        self._last_time = time.monotonic()

        self._base_gain = 0.6
        self._attack_time = 1.5
        self._release_time = 3.0
        self._axis_weights = (1, 1, 1, 1, 1, 1)

        self._antenna_amp_deg = 15.0
        self._antenna_freq_hz = 0.5

    # -----------------------------------------------------
    # External Update (from MovementAdapter)
    # -----------------------------------------------------

    def update_vadcc(self, vadcc: Vec5) -> None:
        """
        Receive live VADCC from AffectEngine.
        Computes new motion manifold target.
        """
        self._vadcc = tuple(float(v) for v in vadcc)

        motion = self._manifold.compute_motion(self._vadcc)

        t = motion["target"]

        self._target = (
            float(t["x"]),
            float(t["y"]),
            float(t["z"]),
            float(t["roll"]),
            float(t["pitch"]),
            float(t["yaw"]),
        )

        b = motion["breathing"]
        self._antenna_amp_deg = float(b["antenna_amplitude_deg"])
        self._antenna_freq_hz = float(b["antenna_frequency_hz"])

        self._strength = float(motion["strength"])
        self._base_gain = float(motion["base_gain"])
        self._attack_time = float(motion["attack_time"])
        self._release_time = float(motion["release_time"])
        self._axis_weights = tuple(float(a) for a in motion["axis_weights"])

    # -----------------------------------------------------
    # Bias Computation (100Hz loop calls this)
    # -----------------------------------------------------

    def compute_bias(self, current: Vec6) -> Vec6:
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now

        self._update_envelope(dt)

        k = self._base_gain * self._strength * self._envelope

        if k <= 1e-6:
            return (0, 0, 0, 0, 0, 0)

        bias = []
        for i in range(6):
            delta = self._target[i] - current[i]
            weighted = delta * self._axis_weights[i]
            bias.append(k * weighted)

        return tuple(bias)

    # -----------------------------------------------------
    # Envelope Dynamics
    # -----------------------------------------------------

    def _update_envelope(self, dt: float) -> None:
        if self._strength > 0:
            rate = dt / max(self._attack_time, 1e-6)
            self._envelope = min(1.0, self._envelope + rate)
        else:
            rate = dt / max(self._release_time, 1e-6)
            self._envelope = max(0.0, self._envelope - rate)

    # -----------------------------------------------------
    # Breathing Access (used by MovementManager)
    # -----------------------------------------------------

    def get_breathing_params(self) -> tuple[float, float]:
        return self._antenna_amp_deg, self._antenna_freq_hz
