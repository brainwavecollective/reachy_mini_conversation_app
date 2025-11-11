# src/reachy_mini_conversation_app/runtime.py
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
"""Creates the same core services used by main().
This mirrors main.py logic so offline/online share the same stack.
"""
logger = setup_logger(args.debug)
logger.info("Bootstrapping Reachy Mini runtime …")


robot = ReachyMini()


# Camera + vision
camera_worker, _, vision_manager = handle_vision_stuff(args, robot)


movement_manager = MovementManager(
current_robot=robot,
camera_worker=camera_worker,
)


head_wobbler = HeadWobbler(set_speech_offsets=movement_manager.set_speech_offsets)


# Start background services (identical order to main.py)
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
"""Stops services in the reverse order and disconnects."""
try:
if ctx.vision_manager:
ctx.vision_manager.stop()
if ctx.camera_worker:
ctx.camera_worker.stop()
ctx.head_wobbler.stop()
ctx.movement_manager.stop()
finally:
# Prevent stray threads
ctx.robot.client.disconnect()
ctx.logger.info("Shutdown complete.")
