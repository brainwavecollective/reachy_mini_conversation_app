"""
AffectBias

Transforms VADCC into continuous motion bias using AffectManifold.

This version is fully responsive:
- No event system
- No attack/release envelope
- No lingering emotional memory
- No baseline folding

The affect engine owns temporal smoothing.
This class simply produces a continuously updated motion attractor.
"""

from typing import Tuple
import logging

from reachy_mini_conversation_app.affect.affect_manifold import AffectManifold

logger = logging.getLogger(__name__)

Vec6 = Tuple[float, float, float, float, float, float]
Vec5 = Tuple[float, float, float, float, float]


class AffectBias:
    """
    Continuous emotional bias field.

    The affect engine provides a streaming VADCC trajectory.
    This class converts that trajectory into a motion attractor.

    bias = strength * axis_weight * (target - current)

    Strength is derived directly from the manifold blend.
    """

    def __init__(self) -> None:
        self._manifold = AffectManifold()

        self._vadcc: Vec5 = (0.5, 0.5, 0.5, 0.5, 0.5)

        # Current manifold outputs
        self._target: Vec6 = (0, 0, 0, 0, 0, 0)
        self._strength: float = 0.0
        self._axis_weights: Vec6 = (1, 1, 1, 1, 1, 1)

        # Breathing parameters
        self._antenna_amp_deg: float = 0.0
        self._antenna_freq_hz: float = 0.0

    # -----------------------------------------------------
    # External Update (called by MovementManager)
    # -----------------------------------------------------

    def update_vadcc(self, vadcc: Vec5) -> None:
        """
        Update internal manifold state from live VADCC stream.
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

        # Strength now comes directly from manifold (RBF dominance)
        self._strength = float(motion["strength"])

        # Axis weights shape influence per DOF
        self._axis_weights = tuple(float(a) for a in motion["axis_weights"])

        # Breathing parameters (max values defined in anchors)
        b = motion["breathing"]
        self._antenna_amp_deg = float(b["antenna_amplitude_deg"])
        self._antenna_freq_hz = float(b["antenna_frequency_hz"])

    # -----------------------------------------------------
    # Bias Computation (called at 100Hz)
    # -----------------------------------------------------

    def compute_bias(self, current: Vec6) -> Vec6:
        """
        Compute continuous motion bias.

        current: (x, y, z, roll, pitch, yaw)
        """

        k = self._strength

        if k <= 1e-6:
            return (0, 0, 0, 0, 0, 0)

        bias = []
        for i in range(6):
            delta = self._target[i] - current[i]
            weighted = delta * self._axis_weights[i]
            bias.append(k * weighted)

        return tuple(bias)

    # -----------------------------------------------------
    # Breathing Access
    # -----------------------------------------------------

    def get_breathing_params(self) -> tuple[float, float]:
        """
        Returns (antenna_amplitude_deg, antenna_frequency_hz)
        """
        return self._antenna_amp_deg, self._antenna_freq_hz

