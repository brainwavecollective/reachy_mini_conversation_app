"""
Movement embodiment adapter.

Bridges AffectEngine VIBE output to MovementManager.
"""

from typing import Sequence
from reachy_mini_conversation_app.moves import MovementManager


class MovementAdapter:
    """
    Thin adapter that forwards VIBE (VADCC) state
    into MovementManager's affect bias system.
    """

    def __init__(self, movement_manager: MovementManager) -> None:
        self._movement_manager = movement_manager

    def update(self, vibe: Sequence[float]) -> None:
        """
        Receive VIBE from AffectEngine and forward to MovementManager.
        """
        self._movement_manager.update_vadcc(tuple(vibe))
