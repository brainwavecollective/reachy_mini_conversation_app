# bin/offline_actions.py
from __future__ import annotations
import argparse
import json
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, TypedDict

from reachy_mini_conversation_app.runtime import start_runtime, stop_runtime
from reachy_mini_conversation_app.utils import parse_args as parse_common_args


class MoveSpec(TypedDict, total=False):
    name: str
    repeat: int
    duration_sec: float


class ActionSpec(TypedDict, total=False):
    id: str
    moves: List[MoveSpec]
    reactive_sound: str | None
    head_tracking: bool | None  # optional flip per action


def load_actions(dir_path: Path) -> List[ActionSpec]:
    actions: List[ActionSpec] = []
    for f in sorted(dir_path.glob("*.json")):
        with f.open("r") as fh:
            data = json.load(fh)
        if "moves" not in data:
            raise SystemExit(f"Action file {f} missing 'moves'")
        data.setdefault("id", f.stem)
        data.setdefault("reactive_sound", None)
        data.setdefault("head_tracking", None)
        actions.append(data)  # type: ignore[arg-type]
    if not actions:
        raise SystemExit(f"No actions found under {dir_path}")
    return actions


def apply_action(ctx, action: ActionSpec) -> None:
    mm = ctx.movement_manager
    logger = ctx.logger

    logger.info("=== ACTION: %s ===", action["id"])

    # Optional head tracking toggle (if your camera worker supports it)
    if action.get("head_tracking") is not None and ctx.camera_worker:
        enabled = bool(action["head_tracking"])
        # If your API differs, update this call only:
        try:
            ctx.camera_worker.set_head_tracking(enabled)
        except AttributeError:
            logger.debug("camera_worker.set_head_tracking not available; skipping.")
        logger.info("Head tracking %s", "enabled" if enabled else "disabled")

    # Moves
    for m in action["moves"]:
        name = m.get("name", "")
        rep = int(m.get("repeat", 1))
        dur = m.get("duration_sec")
        if dur is not None:
            logger.info("Move: %s x%d (duration=%.1fs)", name, rep, float(dur))
        else:
            logger.info("Move: %s x%d", name, rep)
        for _ in range(max(1, rep)):
            # ADAPT HERE IF NEEDED:
            # If your MovementManager uses a different API, change just this line.
            mm.perform_move(name=name, duration=dur)  # <-- adjust if needed

    # Reactive sound
    snd = action.get("reactive_sound")
    if snd:
        started = time.monotonic()
        try:
            # ADAPT HERE IF NEEDED:
            mm.play_sound(snd)  # <-- adjust if needed
            lat = time.monotonic() - started
            logger.info("Reactive sound: %s (lat=%.3fs)", snd, lat)
        except Exception as e:
            logger.warning("Failed to play sound %s: %s", snd, e)


def cli(argv: List[str] | None = None) -> int:
    """
    Entry point for the offline actions runner.
    We parse offline-specific flags first, remove them from argv,
    then let the app's parse_args() handle the rest (debug, camera, vision...).
    """
    if argv is None:
        argv = sys.argv[1:]

    # 1) Parse OFFLINE flags first, leave the rest for main app parser
    offline_parser = argparse.ArgumentParser(add_help=False)
    offline_parser.add_argument("--actions-dir", default="actions", type=Path)
    offline_parser.add_argument(
        "--select",
        default="random",
        choices=["random", "roundrobin"],
        help="How to pick actions for the loop",
    )
    offline_parser.add_argument("--pause-sec", default=1.0, type=float)
    offline_parser.add_argument("--once", action="store_true", help="Run a single action then exit")
    offline_parser.add_argument("--seed", type=int, default=None, help="Seed RNG for reproducibility")

    offline_args, remaining = offline_parser.parse_known_args(argv)

    # 2) Now call the app's parse_args() on the remaining args by swapping sys.argv
    saved_argv = sys.argv[:]
    try:
        sys.argv = [saved_argv[0]] + remaining
        app_args = parse_common_args()  # pulls in --debug, --no-camera, --local-vision, --gradio, etc.
    finally:
        sys.argv = saved_argv

    # Merge offline args into app_args
    for k, v in vars(offline_args).items():
        setattr(app_args, k, v)

    # 3) Optional RNG seeding
    if app_args.seed is not None:
        random.seed(app_args.seed)

    # 4) Load actions and start runtime
    actions = load_actions(app_args.actions_dir)
    ctx = start_runtime(app_args)

    quit_requested = False

    def _sigint(_signum, _frame):
        nonlocal quit_requested
        quit_requested = True
        ctx.logger.info("Keyboard interrupt; shutting down.")

    signal.signal(signal.SIGINT, _sigint)

    try:
        idx = 0
        while not quit_requested:
            if app_args.select == "random":
                choice = random.choice(actions)
            else:  # round robin
                choice = actions[idx % len(actions)]
                idx += 1

            apply_action(ctx, choice)

            if app_args.once:
                break

            time.sleep(max(0.0, float(app_args.pause_sec)))

        return 0
    finally:
        stop_runtime(ctx)


if __name__ == "__main__":
    sys.exit(cli())

