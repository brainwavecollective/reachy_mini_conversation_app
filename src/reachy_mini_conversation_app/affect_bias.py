import time
import math
from typing import Tuple
import json
import os


Vec6 = Tuple[float, float, float, float, float, float]


class AffectBias:

    CONFIG_PATH = "affect_config.json"
    CONFIG_POLL_INTERVAL = 0.5  # seconds

    def __init__(self) -> None:
        self._target = (0, 0, 0, 0, 0, 0)
        self._strength = 0.0
        self._envelope = 0.0
        self._last_time = time.monotonic()

        self._base_gain = 0.6
        self._attack_time = 1.5
        self._release_time = 3.0
        self._axis_weights = (1, 1, 1, 1, 1, 1)

        self._last_config_check = 0.0
        self._last_mtime = 0.0

        self._antenna_amp_deg = 15.0
        self._antenna_freq_hz = 0.5


    def compute_bias(self, current):
        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now

        self._maybe_reload_config(now)
        self._update_envelope(dt)

        k = self._base_gain * self._strength * self._envelope

        if k <= 1e-5:
            return (0, 0, 0, 0, 0, 0)

        bias = []
        for i in range(6):
            delta = self._target[i] - current[i]
            weighted = delta * self._axis_weights[i]
            bias.append(k * weighted)

        return tuple(bias)

    def _update_envelope(self, dt):
        if self._strength > 0:
            rate = dt / max(self._attack_time, 1e-6)
            self._envelope = min(1.0, self._envelope + rate)
        else:
            rate = dt / max(self._release_time, 1e-6)
            self._envelope = max(0.0, self._envelope - rate)

    def get_breathing_params(self) -> tuple[float, float]:
        # degrees, Hz
        return self._antenna_amp_deg, self._antenna_freq_hz


    def _maybe_reload_config(self, now):
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

            self._strength = float(cfg.get("strength", 0.0))

            t = cfg.get("target", {})
            self._target = (
                float(t.get("x", 0.0)),
                float(t.get("y", 0.0)),
                float(t.get("z", 0.0)),
                float(t.get("roll", 0.0)),
                float(t.get("pitch", 0.0)),
                float(t.get("yaw", 0.0)),
            )

            breathing = cfg.get("breathing", {})
            self._antenna_amp_deg = float(breathing.get("antenna_amplitude_deg", 15.0))
            self._antenna_freq_hz = float(breathing.get("antenna_frequency_hz", 0.5))

            self._base_gain = float(cfg.get("base_gain", 0.6))
            self._attack_time = float(cfg.get("attack_time", 1.5))
            self._release_time = float(cfg.get("release_time", 3.0))

            axis = cfg.get("axis_weights", [1,1,1,1,1,1])
            if len(axis) == 6:
                self._axis_weights = tuple(float(a) for a in axis)

            print("Affect config reloaded")

        except Exception as e:
            print("Failed to load affect config:", e)
