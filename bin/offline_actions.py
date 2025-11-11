#!/usr/bin/env python3
import argparse, json, random, time, logging, os, glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional deps (graceful degradation)
try:
    import simpleaudio as sa  # playback (WAV)
except Exception:
    sa = None
try:
    import soundfile as sf    # decode many formats
    import numpy as np
except Exception:
    sf = None
    np = None

# Validation
from jsonschema import Draft202012Validator

# ---- import your stack (already in repo) ----
from reachy_mini import ReachyMini
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove
from reachy_mini_conversation_app.utils import handle_vision_stuff, setup_logger
from reachy_mini_conversation_app.audio.speech_tapper import SwayRollRT, HOP_MS

SCHEMA_PATH = Path(__file__).with_name("action_schema.json")

# ------------------------------------------------------------
# Round-robin helper
class RoundRobin:
    def __init__(self):
        self._idx: Dict[str, int] = {}
    def pick(self, key: str, items: List[Any]) -> Any:
        i = self._idx.get(key, 0)
        val = items[i % len(items)]
        self._idx[key] = i + 1
        return val

_rr = RoundRobin()

# ------------------------------------------------------------
# Picker utilities
def _normalize_items(raw_items: List[Any], kind: str) -> List[Tuple[str, float]]:
    """Return list of (value, weight). kind='move' uses 'name'; kind='sound' uses 'file' (fallback 'name')."""
    out: List[Tuple[str, float]] = []
    for it in raw_items:
        if isinstance(it, str):
            out.append((it, 1.0))
        elif isinstance(it, dict):
            val = it.get("name")
            if kind != "move":
                val = it.get("file", val)
            if isinstance(val, str):
                out.append((val, float(it.get("weight", 1.0))))
    return out

def resolve_choice(field: Any, kind: str, rr_key: str) -> str:
    """Resolve literals or {pick:{...}} to a concrete string."""
    if isinstance(field, str):
        return field
    if isinstance(field, dict) and "pick" in field:
        picker = field["pick"]
        items = _normalize_items(picker.get("items", []), kind)
        if not items:
            raise ValueError("empty picker items")
        strategy = picker.get("strategy", "random")
        values = [v for v, _ in items]
        if strategy == "random":
            return random.choice(values)
        if strategy == "weighted_random":
            weights = [w for _, w in items]
            if sum(weights) <= 0:
                weights = [1.0] * len(items)
            return random.choices(values, weights=weights, k=1)[0]
        if strategy == "round_robin":
            return _rr.pick(rr_key, values)
        return random.choice(values)
    raise ValueError(f"invalid field: {field}")

# ------------------------------------------------------------
# Hugging Face cache resolver for dataset audio
def resolve_sound_path(spec: str) -> str:
    # Absolute/relative path that exists?
    p = Path(spec)
    if p.exists():
        return str(p)

    # "hf:<basename>" lookup in HF dataset cache
    if spec.startswith("hf:"):
        basename = spec[3:]

        candidates: List[Path] = []
        # HF_HOME might hold cache roots
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            candidates.append(Path(hf_home))
        # Default hub cache
        candidates.append(Path.home() / ".cache" / "huggingface" / "hub")

        for root in candidates:
            pattern = str(root / "datasets--pollen-robotics--reachy-mini-emotions-library" / "**" / basename)
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return matches[0]

        logging.warning("HF sound not found: %s (searched %s)", spec, [str(c) for c in candidates])
        return basename

    # fallback: return as-is; playback will warn if missing
    return spec

# ------------------------------------------------------------
# Playback / reactive wobble (with optional truncation)
def play_sound_blocking(path: str, volume: float = 1.0, max_duration: float | None = None) -> float:
    p = Path(path)
    if not p.exists():
        logging.warning("Sound file missing: %s", path)
        return 0.0
    try:
        if sa:
            wave_obj = sa.WaveObject.from_wave_file(str(p))
            play_obj = wave_obj.play()
            # length
            dur = 0.0
            try:
                import wave
                with wave.open(str(p), "rb") as wf:
                    dur = wf.getnframes() / float(wf.getframerate())
            except Exception:
                pass
            if max_duration and max_duration > 0:
                time.sleep(min(max_duration, dur if dur > 0 else max_duration))
                try: play_obj.stop()
                except Exception: pass
                return min(max_duration, dur) if dur > 0 else max_duration
            else:
                play_obj.wait_done()
                return dur
        else:
            logging.info("simpleaudio not available; skipping audible playback.")
            return 0.0
    except Exception as e:
        logging.warning("Playback error for %s: %s", path, e)
        return 0.0

def reactive_wobble_from_file(path: str, set_offsets, latency_s: float = 0.08, max_duration: float | None = None) -> float:
    if sf is None or np is None:
        logging.warning("soundfile/numpy not available; reactive wobble will be skipped.")
        return 0.0

    p = Path(path)
    if not p.exists():
        logging.warning("Sound file missing: %s", path)
        return 0.0

    sway = SwayRollRT()
    hop_dt = HOP_MS / 1000.0

    try:
        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        hop_n = max(1, int(sr * hop_dt))
        base_t = time.monotonic()
        t_hops = 0
        spent = 0.0

        for i in range(0, len(mono), hop_n):
            if max_duration and spent >= max_duration:
                break
            chunk = mono[i : i + hop_n]
            frames = sway.feed(chunk, sr)
            for r in frames:
                if max_duration and spent >= max_duration:
                    break
                target = base_t + latency_s + t_hops * hop_dt
                sleep = target - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                set_offsets((
                    r["x_mm"]/1000.0, r["y_mm"]/1000.0, r["z_mm"]/1000.0,
                    r["roll_rad"], r["pitch_rad"], r["yaw_rad"]
                ))
                t_hops += 1
                spent += hop_dt

        # tail: damp to zero
        for _ in range(6):
            time.sleep(hop_dt)
            set_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        sway.reset()
        return spent if max_duration else len(mono) / float(sr)

    except Exception as e:
        logging.warning("Reactive wobble error (%s): %s", path, e)
        return 0.0

# ------------------------------------------------------------
# Load & validate actions
def load_schema() -> Dict[str, Any]:
    try:
        return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logging.error("Failed to load schema at %s: %s", SCHEMA_PATH, e)
        raise SystemExit(1)

def load_actions(dir_path: Path, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    if not dir_path.exists():
        logging.error("Actions dir not found: %s", dir_path)
        return actions
    files = sorted([*dir_path.glob("*.json")])
    if not files:
        logging.warning("No action JSON files in %s", dir_path)
        return actions
    validator = Draft202012Validator(schema)
    for fp in files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
            validator.validate(obj)
            actions.append(obj)
        except Exception as e:
            logging.error("Invalid action file %s: %s", fp.name, e)
    return actions

# ------------------------------------------------------------
# Action selection & execution
def select_action(actions: List[Dict[str, Any]], mode: str, by_name: Optional[str], seq_list: Optional[List[str]], i: int) -> Dict[str, Any]:
    if mode == "by-name" and by_name:
        for a in actions: 
            if a.get("name") == by_name: 
                return a
        raise RuntimeError(f"Action named '{by_name}' not found.")
    elif mode == "alphabetical":
        return actions[i % len(actions)]
    elif mode == "list" and seq_list:
        wanted = seq_list[i % len(seq_list)]
        for a in actions:
            if a.get("name") == wanted:
                return a
        raise RuntimeError(f"Action '{wanted}' not found.")
    else:
        return random.choice(actions)

def run_action(action: Dict[str, Any], movement: MovementManager, camera_worker: Any, pause_sec: float) -> None:
    name = action.get("name", "?")
    ht = (action.get("head_tracking") or {})
    ht_enabled = bool(ht.get("enabled", False))
    if camera_worker is not None:
        try: camera_worker.set_head_tracking_enabled(ht_enabled)
        except Exception: pass

    logging.info("=== ACTION: %s ===", name)
    for idx, step in enumerate(action.get("steps", []), 1):
        dur = step.get("duration", None)  # seconds or None

        if "move" in step:
            move_name = resolve_choice(step["move"], "move", rr_key=f"{name}.step{idx}.move")
            repeat = int(step.get("repeat", 1))
            logging.info("Move: %s x%d%s", move_name, repeat, f" (duration={dur}s)" if dur else "")
            for _ in range(repeat):
                movement.queue_move(DanceQueueMove(move_name))
            if isinstance(dur, (int, float)) and dur > 0:
                time.sleep(float(dur))

        elif "sound" in step:
            path_in = resolve_choice(step["sound"], "sound", rr_key=f"{name}.step{idx}.sound")
            path = resolve_sound_path(path_in)
            vol = float(step.get("volume", 1.0))
            logging.info("Sound: %s (vol=%.2f%s)", path, vol, f", duration={dur}s" if dur else "")
            play_sound_blocking(path, volume=vol, max_duration=dur)

        elif "reactive_sound" in step:
            path_in = resolve_choice(step["reactive_sound"], "sound", rr_key=f"{name}.step{idx}.rsound")
            path = resolve_sound_path(path_in)
            lat = float(step.get("latency", 0.08))
            logging.info("Reactive sound: %s (lat=%.3fs%s)", path, lat, f", duration={dur}s" if dur else "")
            reactive_wobble_from_file(path, movement.set_speech_offsets, latency_s=lat, max_duration=dur)

        else:
            logging.warning("Unknown step kind; skipping: %s", step)

        time.sleep(pause_sec)

# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("offline-actions runner (standalone)")
    ap.add_argument("--actions-dir", default="actions", help="Folder with action JSON files")
    ap.add_argument("--select", choices=["random", "alphabetical", "by-name", "list"], default="random")
    ap.add_argument("--action", help="Name for by-name mode")
    ap.add_argument("--actions", help="Comma-separated names for list mode")
    ap.add_argument("--pause-sec", type=float, default=1.0)
    ap.add_argument("--once", action="store_true", help="Run one action then exit")
    ap.add_argument("--head-tracker", choices=["yolo", "mediapipe"], default=None)
    ap.add_argument("--no-camera", action="store_true", default=False)
    ap.add_argument("--debug", action="store_true", default=False)
    args = ap.parse_args()

    logger = setup_logger(args.debug)
    logger.info("Starting offline actions runner")

    schema = load_schema()
    actions = load_actions(Path(args.actions_dir), schema)
    if not actions:
        raise SystemExit(1)

    # Robot & movement (reuse your stack)
    robot = ReachyMini()

    camera_worker = None
    vision_manager = None
    # handle_vision_stuff expects args with: head_tracker, no_camera, local_vision, gradio, debug
    if not args.no_camera:
        # fabricate missing fields for compatibility
        if not hasattr(args, "local_vision"): setattr(args, "local_vision", False)
        if not hasattr(args, "gradio"): setattr(args, "gradio", False)
        try:
            camera_worker, _, vision_manager = handle_vision_stuff(args, robot)
        except Exception as e:
            logger.warning("Head-tracker init failed: %s", e)

    movement = MovementManager(current_robot=robot, camera_worker=camera_worker)
    movement.start()

    try:
        i = 0
        sequence = args.actions.split(",") if args.actions else None
        while True:
            action = select_action(actions, args.select, args.action, sequence, i)
            run_action(action, movement, camera_worker, args.pause_sec)
            i += 1
            if args.once:
                break
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt; shutting down.")
    finally:
        movement.stop()
        if camera_worker: 
            try: camera_worker.stop()
            except Exception: pass
        if vision_manager: 
            try: vision_manager.stop()
            except Exception: pass
        robot.client.disconnect()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
