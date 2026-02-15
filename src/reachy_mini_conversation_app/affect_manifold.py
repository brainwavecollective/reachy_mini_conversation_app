# affect_manifold.py

import math
import numpy as np
from typing import Dict, Tuple, List


Vec5 = Tuple[float, float, float, float, float]


class AffectManifold:

    def __init__(self, sigma: float = 0.25):
        self.sigma = sigma
        self.anchors = self._create_anchors()

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def compute_motion(self, vadcc: Vec5) -> Dict:
        weights = self._compute_weights(vadcc)

        blended = self._blend_motion(weights)

        return blended

    # -----------------------------------------------------
    # RBF
    # -----------------------------------------------------

    def _compute_weights(self, x: Vec5) -> List[float]:
        weights = []
        for anchor in self.anchors:
            dist2 = sum((x[i] - anchor["input"][i]) ** 2 for i in range(5))
            w = math.exp(-dist2 / (2 * self.sigma ** 2))
            weights.append(w)

        total = sum(weights)
        if total <= 1e-8:
            return [1.0 / len(weights)] * len(weights)

        return [w / total for w in weights]

    # -----------------------------------------------------
    # Blending
    # -----------------------------------------------------

    def _blend_motion(self, weights: List[float]) -> Dict:
        result = self._zero_motion()

        for w, anchor in zip(weights, self.anchors):
            self._accumulate(result, anchor["motion"], w)

        return result

    # -----------------------------------------------------
    # Motion Accumulation
    # -----------------------------------------------------

    def _zero_motion(self):
        return {
            "target": {
                "x": 0.0, "y": 0.0, "z": 0.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "body_yaw": 0.0
            },
            "breathing": {
                "antenna_amplitude_deg": 0.0,
                "antenna_frequency_hz": 0.0
            },
            "strength": 0.0,
            "base_gain": 0.0,
            "attack_time": 0.0,
            "release_time": 0.0,
            "axis_weights": [0.0]*6
        }

    def _accumulate(self, result, motion, weight):
        for k in result["target"]:
            result["target"][k] += motion["target"][k] * weight

        for k in result["breathing"]:
            result["breathing"][k] += motion["breathing"][k] * weight

        result["strength"] += motion["strength"] * weight
        result["base_gain"] += motion["base_gain"] * weight
        result["attack_time"] += motion["attack_time"] * weight
        result["release_time"] += motion["release_time"] * weight

        for i in range(6):
            result["axis_weights"][i] += motion["axis_weights"][i] * weight

    # -----------------------------------------------------
    # Anchor Definitions
    # -----------------------------------------------------

    def _create_anchors(self):
        return [

            # Neutral
            self._anchor(
                (0.5,0.5,0.5,0.5,0.5),
                pitch=0.0, z=0.0,
                amp=6, freq=0.6,
                strength=0.6
            ),

            # Joy
            self._anchor(
                (0.9,0.85,0.8,0.7,0.8),
                pitch=-0.2, z=0.01,
                amp=10, freq=0.7,
                strength=0.7
            ),

            # Sadness
            self._anchor(
                (0.1,0.2,0.3,0.6,0.55),
                pitch=0.3, z=-0.015,
                amp=2, freq=0.25,
                strength=0.75
            ),

            # Anger
            self._anchor(
                (0.1,0.9,0.9,0.8,0.9),
                pitch=-0.1, z=0.0,
                amp=4, freq=1.1,
                strength=0.8
            ),

            # Anxiety
            self._anchor(
                (0.2,0.85,0.4,0.85,0.85),
                pitch=0.05, z=-0.01,
                amp=3, freq=1.2,
                strength=0.7
            ),

            # Calm
            self._anchor(
                (0.7,0.2,0.4,0.6,0.6),
                pitch=0.0, z=0.005,
                amp=4, freq=0.2,
                strength=0.5
            ),

            # Pride
            self._anchor(
                (0.85,0.65,0.85,0.65,0.65),
                pitch=-0.3, z=0.015,
                amp=12, freq=0.4,
                strength=0.75
            ),

            # Curiosity
            self._anchor(
                (0.7,0.75,0.4,0.8,0.75),
                pitch=-0.05, z=0.005,
                amp=9, freq=0.55,
                strength=0.65
            ),

            # Boredom
            self._anchor(
                (0.35,0.15,0.5,0.6,0.55),
                pitch=0.15, z=-0.01,
                amp=3, freq=0.18,
                strength=0.6
            ),

        ]

    def _anchor(self, input_vec, pitch, z, amp, freq, strength):
        return {
            "input": input_vec,
            "motion": {
                "target": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": z,
                    "roll": 0.0,
                    "pitch": pitch,
                    "yaw": 0.0,
                    "body_yaw": 0.0
                },
                "breathing": {
                    "antenna_amplitude_deg": amp,
                    "antenna_frequency_hz": freq
                },
                "strength": strength,
                "base_gain": 0.6,
                "attack_time": 1.5,
                "release_time": 3.0,
                "axis_weights": [1,1,1,0.6,1,1]
            }
        }

