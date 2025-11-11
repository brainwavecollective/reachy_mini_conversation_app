#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import random
import shutil
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from reachy_mini_conversation_app.runtime import start_runtime, stop_runtime
from reachy_mini_conversation_app.utils import parse_args as parse_common_args
from reachy_mini_conversation_app.dance_emotion_moves import DanceQueueMove

# Optional JSON Schema validation (uses the schema in bin/action_schema.json)
try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None  # type: ignore

logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# Types aligned with bin/action_schema.json
# -----------------------------------------------------------------------------

class PickerItem(TypedDict, total=False):
    name: str
    file: str
    weight: float

class Picker(TypedDict, total=False):
    items: List[PickerItem] | List[str]
    strategy: str  # "random" | "weighted_random" | "round_robin"

class NameOrPicker(TypedDict, total=False):
    pick: Picker

class FileOrPicker(TypedDict, total=False):
    pick: Picker

class Step(TypedDict, total=False):
    move: str | NameOrPicker
    sound: str | FileOrPicker
    reactive_sound: str | FileOrPicker
    repeat: int
    volume: float
    latency: float
    duration: float

class ActionDoc(TypedDict, total=False):
    name: str
    tags: List[str]
    head_tracking: Dict[str, Any]
    steps: List[Step]


# -----------------------------------------------------------------------------
# Hugging Face hub snapshot audio index (no downloads)
# -----------------------------------------------------------------------------

HF_REPO = "pollen-robotics/reachy-mini-emotions-library"

def _hub_repo_root(explicit_root: Optional[str] = None) -> Optional[Path]:
    """Return the hub repo dir (.../hub/datasets--org--name) if it exists."""
    roots: List[Path] = []
    if explicit_root:
        roots.append(Path(explicit_root) / "hub")
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    org, name = HF_REPO.split("/", 1)
    hub_dirname = f"datasets--{org}--{name}"
    for base in roots:
        repo = (base / hub_dirname).resolve()
        if repo.exists():
            return repo
    return None

def _current_snapshot_dir(hub_repo: Path) -> Optional[Path]:
    """Read refs/* to get a commit id and return snapshots/<SHA> if present."""
    refs_dir = hub_repo / "refs"
    snaps_dir = hub_repo / "snapshots"
    if not refs_dir.exists() or not snaps_dir.exists():
        return None

    # Prefer 'main' if present, else any ref file
    ref_files: List[Path] = []
    if (refs_dir / "main").exists():
        ref_files.append(refs_dir / "main")
    ref_files += [p for p in refs_dir.iterdir() if p.is_file() and p.name != "main"]

    for rf in ref_files:
        try:
            sha = rf.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if sha:
            maybe = snaps_dir / sha
            if maybe.exists():
                return maybe
    return None

class HubAudioIndex:
    """Index audio files in the hub cache snapshots for a dataset (no downloads)."""

    def __init__(self, explicit_root: Optional[str] = None):
        self.repo_root = _hub_repo_root(explicit_root)
        self.snap_root = _current_snapshot_dir(self.repo_root) if self.repo_root else None
        self.by_basename: Dict[str, Path] = {}
        self.by_stem: Dict[str, List[Path]] = {}
        self.all_paths: List[Path] = []

    def build(self) -> None:
        if not self.snap_root:
            logging.warning("No hub snapshot root found for %s", HF_REPO)
            return
        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif", ".wma"}
        count = 0
        for p in self.snap_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in audio_exts:
                self.by_basename[p.name] = p
                self.by_stem.setdefault(p.stem, []).append(p)
                self.all_paths.append(p)
                count += 1
        logging.info("Indexed %d audio files under %s", count, self.snap_root)

    def list_basenames(self) -> List[str]:
        return sorted(self.by_basename.keys())

    def resolve_exact(self, rel: str) -> Optional[str]:
        """Try exact relative path, exact basename, or stem match (prefer .wav)."""
        if not self.snap_root:
            return None

        # 1) exact relative path
        candidate = self.snap_root / rel
        if candidate.exists():
            return str(candidate)

        # 2) exact basename
        if rel in self.by_basename:
            return str(self.by_basename[rel])

        # 3) stem match (prefer .wav)
        stem = Path(rel).stem
        if stem in self.by_stem:
            opts = sorted(self.by_stem[stem], key=lambda p: (p.suffix.lower() != ".wav", len(str(p))))
            return str(opts[0])

        return None

    def resolve_fuzzy(self, rel: str, cutoff: float = 0.7) -> Optional[str]:
        """Fuzzy match a name (basename or stem) with cutoff; prefer .wav on ties."""
        if not self.by_basename:
            return None
        names = list(self.by_basename.keys())
        # Try basename first
        best = difflib.get_close_matches(rel, names, n=1, cutoff=cutoff)
        if best:
            return str(self.by_basename[best[0]])

        # Try by stem list
        stem = Path(rel).stem
        stems = list({Path(n).stem for n in names})
        best_stem = difflib.get_close_matches(stem, stems, n=1, cutoff=cutoff)
        if best_stem:
            stem_name = best_stem[0]
            candidates = [self.by_basename[n] for n in names if Path(n).stem == stem_name]
            candidates.sort(key=lambda p: (p.suffix.lower() != ".wav", len(str(p))))
            return str(candidates[0]) if candidates else None

        return None


# Global hub index, initialized in cli()
_HUB_INDEX: Optional[HubAudioIndex] = None
_AUTO_FUZZY: bool = False
_FUZZY_CUTOFF: float = 0.7

def resolve_sound_path(spec: str) -> Tuple[str, bool]:
    """
    Resolve 'hf:*' via hub snapshot index; otherwise return filesystem path unchanged.
    Returns (resolved_path_or_spec, resolved_bool).
    """
    if not spec.startswith("hf:"):
        return spec, True

    if _HUB_INDEX is None:
        logging.error("HF hub index not initialized; cannot resolve %s", spec)
        return spec, False

    rel = spec[3:].lstrip("/")
    exact = _HUB_INDEX.resolve_exact(rel)
    if exact:
        return exact, True

    if _AUTO_FUZZY:
        fuzzy = _HUB_INDEX.resolve_fuzzy(rel, cutoff=_FUZZY_CUTOFF)
        if fuzzy:
            logging.warning("HF audio '%s' not found; using closest match: %s", rel, Path(fuzzy).name)
            return fuzzy, True

    # Log suggestions when not resolved
    close = difflib.get_close_matches(rel, _HUB_INDEX.list_basenames(), n=5, cutoff=0.6)
    if close:
        logging.warning("HF audio '%s' not found. Did you mean: %s", rel, ", ".join(close))
    else:
        logging.warning("HF audio '%s' not found in snapshot.", rel)

    return spec, False


# -----------------------------------------------------------------------------
# Picker utilities
# -----------------------------------------------------------------------------

class RoundRobinState:
    """Maintains per-action, per-key round-robin indices."""
    def __init__(self) -> None:
        self.indices: Dict[Tuple[str, int], int] = defaultdict(int)

    def next_idx(self, action_id: str, step_idx: int, n: int) -> int:
        key = (action_id, step_idx)
        idx = self.indices[key] % n
        self.indices[key] += 1
        return idx


def _flatten_picker_items(items: List[Any], want: str) -> List[Tuple[str, float]]:
    """
    Normalize items into (value, weight) tuples.
    `want` is "name" for moves or "file" for sounds.
    Items may be strings or dicts with {name|file, weight}.
    """
    out: List[Tuple[str, float]] = []
    for it in items:
        if isinstance(it, str):
            out.append((it, 1.0))
        elif isinstance(it, dict):
            val = it.get(want)
            if not isinstance(val, str):
                continue
            w = float(it.get("weight", 1.0))
            out.append((val, w if w >= 0 else 0.0))
    return out


def _pick_value(action_id: str, step_idx: int, pick: Picker, want: str, rr: RoundRobinState) -> Optional[str]:
    items_raw = pick.get("items", [])
    if not isinstance(items_raw, list) or not items_raw:
        return None

    items = _flatten_picker_items(items_raw, want)
    if not items:
        return None

    strategy = pick.get("strategy", "random")

    if strategy == "round_robin":
        idx = rr.next_idx(action_id, step_idx, len(items))
        return items[idx][0]

    if strategy == "weighted_random":
        weights = [max(0.0, w) for (_, w) in items]
        total = sum(weights)
        if total <= 0:
            return random.choice([v for (v, _) in items])
        r = random.random() * total
        acc = 0.0
        for v, w in items:
            acc += w
            if r <= acc:
                return v
        return items[-1][0]

    # default: random
    return random.choice([v for (v, _) in items])


def _resolve_name_or_picker(action_id: str, step_idx: int, val: Any, rr: RoundRobinState) -> Optional[str]:
    if isinstance(val, str):
        return val
    if isinstance(val, dict) and "pick" in val:
        return _pick_value(action_id, step_idx, val["pick"], want="name", rr=rr)
    return None


def _resolve_file_or_picker(action_id: str, step_idx: int, val: Any, rr: RoundRobinState) -> Optional[str]:
    if isinstance(val, str):
        return val
    if isinstance(val, dict) and "pick" in val:
        return _pick_value(action_id, step_idx, val["pick"], want="file", rr=rr)
    return None


# -----------------------------------------------------------------------------
# Loading & validation
# -----------------------------------------------------------------------------

def _validate_with_schema(doc: Dict[str, Any], schema_path: Path) -> None:
    if jsonschema is None:
        return  # schema validation optional; skip if lib unavailable
    with schema_path.open("r") as fh:
        schema = json.load(fh)
    jsonschema.validate(instance=doc, schema=schema)  # type: ignore


def load_actions(dir_path: Path, schema_path: Optional[Path]) -> List[ActionDoc]:
    actions: List[ActionDoc] = []
    for f in sorted(dir_path.glob("*.json")):
        with f.open("r") as fh:
            data = json.load(fh)
        if schema_path:
            _validate_with_schema(data, schema_path)
        if "name" not in data or "steps" not in data:
            raise SystemExit(f"Action file {f} missing 'name' or 'steps'")
        actions.append(data)  # modern schema only
    if not actions:
        raise SystemExit(f"No actions found under {dir_path}")
    return actions


# -----------------------------------------------------------------------------
# Execution helpers wired to your real APIs
# -----------------------------------------------------------------------------

def _apply_head_tracking_if_present(ctx, action: ActionDoc) -> None:
    ht = action.get("head_tracking")
    if not ht:
        return

    enabled = bool(ht.get("enabled", True))

    if ctx.camera_worker and hasattr(ctx.camera_worker, "set_head_tracking_enabled"):
        try:
            ctx.camera_worker.set_head_tracking_enabled(enabled)
        except Exception as e:
            ctx.logger.warning("Failed to set head tracking: %s", e)

    ctx.logger.info("Head tracking %s", "enabled" if enabled else "disabled")


def _assets_dir_from_robot(ctx) -> Path:
    """
    Locate the installed reachy_mini assets dir used by media.play_sound().
    We derive it from the error pattern, but more robustly via the package path.
    """
    import reachy_mini  # local import to avoid import-time side effects elsewhere
    pkg_root = Path(reachy_mini.__file__).resolve().parent
    return (pkg_root / "assets").resolve()


def _stage_into_assets_and_basename(ctx, source_path: Path) -> str:
    """
    Ensure the file exists under reachy_mini/assets, returns just the basename.
    If a file with the same basename already exists, we re-use it.
    Otherwise we copy the source into assets (symlink if possible on POSIX).
    """
    assets = _assets_dir_from_robot(ctx)
    assets.mkdir(parents=True, exist_ok=True)

    basename = source_path.name
    dest = assets / basename

    try:
        if dest.exists():
            # If it's already the same file (by size/mtime), skip; else overwrite.
            try:
                if dest.stat().st_size == source_path.stat().st_size:
                    return basename
            except Exception:
                pass
            # Overwrite to keep it fresh
            if dest.is_symlink() or dest.is_file():
                dest.unlink(missing_ok=True)

        # Prefer symlink to avoid copies if possible
        try:
            if dest.exists():
                dest.unlink(missing_ok=True)
            os.symlink(source_path, dest)
        except Exception:
            # Fallback to copy if symlink fails (e.g., on Windows or FS without perms)
            shutil.copy2(source_path, dest)
    except Exception as e:
        ctx.logger.warning("Failed staging sound into assets (%s): %s", dest, e)
        # As a last resort, just return basename; play_sound will still look in assets/
        # which may or may not have a match.
    return basename


def _play_sound(ctx, spec: str) -> None:
    """
    Resolve hf:* to a real path, stage it into reachy_mini/assets/, then call
    ReachyMini.media.play_sound(<basename>), which is how the main app plays 'wake_up.wav'.
    """
    resolved, ok = resolve_sound_path(spec)
    try:
        if ok and os.path.isabs(resolved) and os.path.exists(resolved):
            basename = _stage_into_assets_and_basename(ctx, Path(resolved))
            ctx.robot.media.play_sound(basename)
        else:
            # If it's already a bare name like 'wake_up.wav' (or unresolved hf:), pass through.
            # Media will look under reachy_mini/assets/.
            ctx.robot.media.play_sound(Path(resolved).name)
    except Exception as e:
        # Use the same logger & format you saw in logs for easy grepping
        logging.getLogger("reachy_mini_conversation_app.utils").warning(
            "play_sound failed for %s: %s", resolved, e
        )


def apply_action(ctx, action: ActionDoc, rr_state: RoundRobinState) -> None:
    mm = ctx.movement_manager
    logger = logging.getLogger(__name__)
    action_id = action.get("name", "unnamed")

    logger.info("=== START ACTION: %s ===", action_id)

    _apply_head_tracking_if_present(ctx, action)

    steps = action["steps"]
    for i, step in enumerate(steps):
        logger.info("---- STEP %d ----", i)

        duration = float(step.get("duration", 0.0))
        latency = float(step.get("latency", 0.08)) if "latency" in step else None

        did_move = False
        did_sound = False

        # --- Movement ---
        if "move" in step:
            move_name = _resolve_name_or_picker(action_id, i, step["move"], rr_state)
            if move_name:
                repeat = int(step.get("repeat", 1))
                logger.info("→ MOVE: '%s' x%d", move_name, repeat)
                for _ in range(max(1, repeat)):
                    mm.queue_move(DanceQueueMove(move_name))
                did_move = True
            else:
                logger.warning("!! Could not resolve move in step %d", i)

        # --- Sound ---
        if "sound" in step:
            snd = _resolve_file_or_picker(action_id, i, step["sound"], rr_state)
            if snd:
                logger.info("→ SOUND: '%s'%s", snd, f" (latency={latency:.2f}s)" if latency else "")
                if latency:
                    time.sleep(latency)
                _play_sound(ctx, snd)
                did_sound = True
            else:
                logger.warning("!! Could not resolve sound in step %d", i)

        # --- Reactive Sound ---
        if "reactive_sound" in step:
            snd = _resolve_file_or_picker(action_id, i, step["reactive_sound"], rr_state)
            if snd:
                logger.info("→ REACTIVE SOUND: '%s'%s", snd, f" (latency={latency:.2f}s)" if latency else "")
                if latency:
                    time.sleep(latency)
                _play_sound(ctx, snd)
                did_sound = True
            else:
                logger.warning("!! Could not resolve reactive_sound in step %d", i)

        # --- No actions? ---
        if not (did_move or did_sound):
            logger.info("→ No movement or sound in step %d", i)

        # --- Wait for step duration ---
        if duration > 0:
            logger.info("⏳ Waiting %.2fs to complete step duration", duration)
            time.sleep(duration)
        else:
            logger.info("⏭ No step duration specified; proceeding immediately")

    logger.info("=== END ACTION: %s ===", action_id)



# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def cli(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the offline actions runner.
    Parse offline-specific flags first, then pass remaining flags to the app's parse_args().
    """
    if argv is None:
        argv = sys.argv[1:]

    # 1) Parse OFFLINE flags first
    offline_parser = argparse.ArgumentParser(add_help=False)
    offline_parser.add_argument("--actions-dir", default="actions", type=Path)
    offline_parser.add_argument(
        "--select",
        default="random",
        choices=["random", "roundrobin"],
        help="How to pick which action file to run in the loop",
    )
    offline_parser.add_argument("--pause-sec", default=1.0, type=float)
    offline_parser.add_argument("--once", action="store_true", help="Run a single action then exit")
    offline_parser.add_argument("--seed", type=int, default=None, help="Seed RNG for reproducibility")
    offline_parser.add_argument("--schema", type=Path, default=Path("bin/action_schema.json"),
                                help="Path to JSON schema for actions (set to '' to disable)")

    # HF index options (optional)
    offline_parser.add_argument("--hf-home", type=str, default=None,
                                help="Override HF_HOME to find local cache (no downloads).")
    offline_parser.add_argument("--auto-fuzzy", action="store_true", help="Allow fuzzy match on hf: filenames")
    offline_parser.add_argument("--fuzzy-cutoff", type=float, default=0.7, help="Fuzzy cutoff (0-1)")

    offline_args, remaining = offline_parser.parse_known_args(argv)

    # 2) Call the app's parse_args() on the remaining args by swapping sys.argv
    saved_argv = sys.argv[:]
    try:
        sys.argv = [saved_argv[0]] + remaining
        app_args = parse_common_args()  # brings in --debug, --no-camera, --local-vision, --gradio, etc.
    finally:
        sys.argv = saved_argv

    # Merge: single Namespace carrying both sets
    for k, v in vars(offline_args).items():
        setattr(app_args, k, v)

    # 3) RNG seeding
    if app_args.seed is not None:
        random.seed(app_args.seed)

    # 4) Build HF audio index
    global _HUB_INDEX, _AUTO_FUZZY, _FUZZY_CUTOFF
    _HUB_INDEX = HubAudioIndex(explicit_root=app_args.hf_home)
    _HUB_INDEX.build()
    _AUTO_FUZZY = bool(getattr(app_args, "auto_fuzzy", False))
    _FUZZY_CUTOFF = float(getattr(app_args, "fuzzy_cutoff", 0.7))

    # 5) Load and validate actions
    schema_path: Optional[Path] = None
    if isinstance(app_args.schema, Path) and str(app_args.schema):
        schema_path = app_args.schema

    actions = load_actions(app_args.actions_dir, schema_path)

    # 6) Start runtime
    ctx = start_runtime(app_args)

    # 7) Handle Ctrl-C
    quit_requested = False

    def _sigint(_signum, _frame):
        nonlocal quit_requested
        quit_requested = True
        ctx.logger.info("Keyboard interrupt; shutting down.")

    signal.signal(signal.SIGINT, _sigint)

    # 8) Round-robin state for per-step pickers
    rr_state = RoundRobinState()

    try:
        idx = 0
        while not quit_requested:
            # Choose which action file to run this cycle
            if app_args.select == "random":
                action = random.choice(actions)
            else:  # round robin across files
                action = actions[idx % len(actions)]
                idx += 1

            apply_action(ctx, action, rr_state)

            if app_args.once:
                break

            time.sleep(max(0.0, float(app_args.pause_sec)))

        return 0
    finally:
        stop_runtime(ctx)


if __name__ == "__main__":
    sys.exit(cli())

