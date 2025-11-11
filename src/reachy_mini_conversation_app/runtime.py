# src/reachy_mini_conversation_app/runtime.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from reachy_mini import ReachyMini
from .moves import MovementManager
from .audio.head_wobbler import HeadWobbler
from .utils import setup_logger, handle_vision_stuff


@dataclass
class AppContext:
    robot: ReachyMini
    movement_manager: MovementManager
    head_wobbler: HeadWobbler
    camera_worker: Optional[object]
    vision_manager: Optional[object]
    logger: object


def start_runtime(args) -> AppContext:
    """Create and start the same core services that main() uses, but for offline mode."""
    logger = setup_logger(args.debug)
    logger.info("Starting offline actions runner")

    robot = ReachyMini()

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)

    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )

    head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)

    # Start background services (match main.py ordering)
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    return AppContext(
        robot=robot,
        movement_manager=movement_manager,
        head_wobbler=head_wobbler,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        logger=logger,
    )


def stop_runtime(ctx: AppContext) -> None:
    """Stop services and disconnect, mirroring main.py teardown."""
    try:
        if ctx.vision_manager:
            ctx.vision_manager.stop()
        if ctx.camera_worker:
            ctx.camera_worker.stop()
        ctx.head_wobbler.stop()
        ctx.movement_manager.stop()
    finally:
        ctx.robot.client.disconnect()
        ctx.logger.info("Shutdown complete.")

