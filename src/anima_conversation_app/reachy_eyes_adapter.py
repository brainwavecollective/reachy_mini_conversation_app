"""Eye color adapter for Anima → ReachyEyes.

Bridges the Anima emotional engine to the reachy-eyes hardware component
by forwarding VADCC tuples directly as VIBE commands. The eye firmware
interprets VADCC natively, so no manifold mapping is needed here.

Usage:
    eyes = robot.component["reachy-eyes"]
    adapter = ReachyEyesAdapter(eyes)
    anima.subscribe(adapter.update)
"""

import logging
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ReachyEyesAdapter:
    """Forward Anima VADCC to ReachyEyes as VIBE commands.

    The eye firmware accepts VADCC directly via the VIBE protocol command,
    so this adapter is intentionally thin — it owns only rate-limiting and
    graceful degradation when the eye component is absent.
    """

    # Minimum seconds between VIBE commands sent to the device.
    # The eye firmware smooths transitions itself (COLOR_TAU), so sending
    # faster than ~10 Hz would be wasteful without improving perceived quality.
    _MIN_SEND_INTERVAL: float = 0.1

    def __init__(
        self,
        eyes,  # ReachyEyes instance, or None when hardware is unavailable
        min_send_interval: float = _MIN_SEND_INTERVAL,
    ) -> None:
        """Initialise the adapter.

        Args:
            eyes: A ``ReachyEyes`` instance obtained from
                ``robot.component["reachy-eyes"]``. If ``None``, all
                ``update()`` calls are silently ignored so the rest of the
                app continues without eye hardware.
            min_send_interval: Minimum seconds between VIBE commands.
                Defaults to 0.1 s (10 Hz).
        """
        self._eyes = eyes
        self._min_send_interval = min_send_interval
        self._last_send_time: float = 0.0
        self._last_vadcc: Optional[Tuple[float, float, float, float, float]] = None
        self._last_log_time: float = 0.0

        if eyes is None:
            logger.warning("[EYE] No eye hardware available — ReachyEyesAdapter running in no-op mode")
        else:
            logger.info("[EYE] ReachyEyesAdapter initialised")

    # ------------------------------------------------------------------
    # Anima subscriber interface
    # ------------------------------------------------------------------

    def update(self, vadcc: Tuple[float, float, float, float, float]) -> None:
        """Receive a VADCC tuple from Anima and forward it to the eyes.

        This method is called by Anima on every emotional state update and
        must return quickly — no blocking I/O, no heavy computation.

        Args:
            vadcc: (valence, arousal, dominance, complexity, coherence),
                   each in [0.0, 1.0].
        """
        if self._eyes is None:
            return

        now = time.monotonic()
        if now - self._last_send_time < self._min_send_interval:
            return

        val, aro, dom, cplx, coh = vadcc

        try:
            self._eyes._device.send_command(
                f"VIBE {val:.4f} {aro:.4f} {dom:.4f} {cplx:.4f} {coh:.4f}"
            )
            self._last_send_time = now
        except Exception as e:
            # Never crash the calling thread (Anima's subscriber loop).
            logger.debug("[EYE] Failed to send VIBE command: %s", e)
            return

        # Sparse debug logging — at most once per second, only on change.
        rounded = tuple(round(v, 3) for v in vadcc)
        if rounded != self._last_vadcc and (now - self._last_log_time) >= 1.0:
            logger.debug(
                "[EYE] VIBE V:%.3f A:%.3f D:%.3f Cx:%.3f Ch:%.3f",
                val, aro, dom, cplx, coh,
            )
            self._last_vadcc = rounded
            self._last_log_time = now
