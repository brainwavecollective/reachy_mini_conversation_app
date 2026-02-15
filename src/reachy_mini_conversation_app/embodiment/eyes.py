"""
Eyes embodiment adapter.

Bridges AffectEngine VIBE output to Reachy Eyes serial device.
"""

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)


class EyesAdapter:
    """
    Emotional passthrough adapter.

    Receives VIBE (VADCC) state from AffectEngine and forwards it
    to the Reachy Eyes device.

    - If device is unavailable, runs in inactive mode.
    - Does not modify or interpret emotional values.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self._eyes: Any = None
        self._active: bool = False

    # --------------------------------------------------
    # CONNECTION
    # --------------------------------------------------

    async def connect(self) -> None:
        """
        Attempt to connect to Reachy Eyes device.

        If unavailable, adapter becomes inactive but does not raise.
        """
        if self.dry_run:
            logger.info("EyesAdapter running in dry-run mode.")
            self._active = False
            return

        try:
            import reachy_eyes

            self._eyes = reachy_eyes.create({"simulation": False})

            if self._eyes:
                logger.info("Connected to Reachy Eyes device.")
                self._active = True
            else:
                logger.warning("Reachy Eyes not found. Adapter inactive.")
                self._active = False

        except Exception as e:
            logger.warning("Failed to initialize Reachy Eyes: %s", e)
            self._active = False

    # --------------------------------------------------
    # AFFECT SUBSCRIPTION ENTRYPOINT
    # --------------------------------------------------

    def update(self, vibe: Sequence[float]) -> None:
        """
        Receive VIBE state from AffectEngine.

        Expects iterable of 5 floats: [V, A, D, Cx, Ch]
        """
        if not self._active:
            return

        try:
            message = (
                f"VIBE "
                f"{vibe[0]:.1f} "
                f"{vibe[1]:.1f} "
                f"{vibe[2]:.1f} "
                f"{vibe[3]:.1f} "
                f"{vibe[4]:.1f}"
            )

            # reachy_eyes exposes device transport internally
            self._eyes._device.send_command(message)

        except Exception as e:
            logger.error("Failed to send VIBE to eyes: %s", e)

    # --------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------

    async def close(self) -> None:
        """Cleanup device connection if active."""
        if self._eyes:
            try:
                self._eyes.cleanup()
                logger.info("EyesAdapter connection closed.")
            except Exception as e:
                logger.error("Error closing EyesAdapter: %s", e)
