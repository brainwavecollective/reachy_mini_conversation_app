import time
import json
import os
from typing import Tuple

from reachy_mini_conversation_app.affect_manifold import AffectManifold


Vec6 = Tuple[float, float, float, float, float, float]
Vec5 = Tuple[float, float, float, float, float]


class AffectBias:

    CONFIG_PATH = "affect_config.json"
    CONFIG_POLL_INTERVAL = 0.5  # seconds

    def __init__(self) -> None:
        self._manifold = AffectManifold()

        self._vadcc: Vec5 = (0.5, 0.5, 0.5, 0.5, 0.5)

        self._target = (0, 0, 0, 0, 0, 0)
        self._strength = 0.0
        self._envelope = 0.0
        self._last_time = time.monotonic()

        self._base_gain = 0.6
        self._attack_time = 1.5
        self._release_time = 3.0
        self._axis_weights = (1, 1, 1, 1, 1, 1)

        self._antenna_amp_deg = 15.0
        self._antenna_freq_hz = 0.5

        self._last_config_check = 0.0
        self._last_mtime = 0.0


    # -----------------------------------------------------
    # MAIN BIAS COMPUTATION
    # -----------------------------------------------------

    def compute_bias(self, current: Vec6) -> Vec6:
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now

        self._maybe_reload_config(now)
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
    # ENVELOPE
    # -----------------------------------------------------

    def _update_envelope(self, dt: float) -> None:
        if self._strength > 0:
            rate = dt / max(self._attack_time, 1e-6)
            self._envelope = min(1.0, self._envelope + rate)
        else:
            rate = dt / max(self._release_time, 1e-6)
            self._envelope = max(0.0, self._envelope - rate)


    # -----------------------------------------------------
    # BREATHING ACCESS
    # -----------------------------------------------------

    def get_breathing_params(self) -> tuple[float, float]:
        return self._antenna_amp_deg, self._antenna_freq_hz


    # -----------------------------------------------------
    # CONFIG + MANIFOLD
    # -----------------------------------------------------

    def _maybe_reload_config(self, now: float) -> None:
        if now - self._last_config_check < self.CONFIG_POLL_INTERVAL:
            return

        self._last_config_check = now

        if not os.path.exists(self.CONFIG_PATH):
            return

        mtime = os.path.getmtime(self.CONFIG_PATH)
        if mtime == self._last_mtime:
            return

        self._last_mtime = mtime

        try:
            with open(self.CONFIG_PATH, "r") as f:
                cfg = json.load(f)

            # ---------------------------
            # Read VADCC instead of target
            # ---------------------------
            vadcc = cfg.get("vadcc", [0.5, 0.5, 0.5, 0.5, 0.5])
            if len(vadcc) == 5:
                self._vadcc = tuple(float(v) for v in vadcc)

            # ---------------------------
            # Query manifold
            # ---------------------------
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

            # Breathing
            b = motion["breathing"]
            self._antenna_amp_deg = float(b["antenna_amplitude_deg"])
            self._antenna_freq_hz = float(b["antenna_frequency_hz"])

            # Core motion parameters
            self._strength = float(motion["strength"])
            self._base_gain = float(motion["base_gain"])
            self._attack_time = float(motion["attack_time"])
            self._release_time = float(motion["release_time"])
            self._axis_weights = tuple(float(a) for a in motion["axis_weights"])

            print("Affect manifold updated from VADCC")

        except Exception as e:
            print("Failed to load affect config:", e)

