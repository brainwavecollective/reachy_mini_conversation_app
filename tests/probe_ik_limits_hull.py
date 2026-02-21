"""Reachy Mini IK Workspace Probe

Characterizes the true multi-axis operating envelope for head pose offsets
by Monte Carlo sampling across (z, pitch, roll) simultaneously — the three
axes driven by emotional expression anchors.

Run this script standalone — no robot connection required.
Outputs per-axis limits, a 2D conditional table (pitch given z), and a
3D achievable point cloud for convex hull fitting.

Usage:
    uv run probe_ik_limits.py
"""

import json
import time

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from reachy_mini.kinematics import AnalyticalKinematics


# ── Pose construction ────────────────────────────────────────────────────────

def make_pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    """Build a 4x4 pose matrix from translation (meters) and RPY (radians)."""
    pose = np.eye(4)
    pose[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    pose[:3, 3] = [x, y, z]
    return pose


# ── IK probe ─────────────────────────────────────────────────────────────────

def probe_ik(kin: AnalyticalKinematics, pose: np.ndarray, body_yaw: float = 0.0) -> bool:
    """Return True if pose is achievable."""
    try:
        joints = kin.ik(pose, body_yaw=body_yaw)
        return joints is not None and not np.any(np.isnan(joints))
    except Exception:
        return False


# ── 1D limit finder ──────────────────────────────────────────────────────────

def find_limit_1d(kin, axis: str, sign: int, other_defaults: dict,
                  coarse_step=0.005, fine_step=0.001, search_range=0.5) -> float:
    """Walk along one axis in one direction until IK fails.

    Returns the last achievable value before failure.
    """
    val = 0.0
    last_good = 0.0
    step = coarse_step * sign

    while abs(val) < search_range:
        val += step
        pose = make_pose(**{**other_defaults, axis: val})
        if not probe_ik(kin, pose):
            break
        last_good = val

    val = last_good
    step = fine_step * sign
    while abs(val) < search_range:
        val += step
        pose = make_pose(**{**other_defaults, axis: val})
        if not probe_ik(kin, pose):
            break
        last_good = val

    return last_good


# ── 3D Monte Carlo sampler ───────────────────────────────────────────────────

def sample_3d_envelope(kin, n_samples: int = 50_000,
                       z_range=(-0.06, 0.03),
                       pitch_range=(-1.6, 1.0),
                       roll_range=(-1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """Randomly sample (z, pitch, roll) space and test each point.

    Returns:
        achievable: (N, 3) array of achievable [z, pitch, roll] points
        failed:     (M, 3) array of failed points
    """
    samples = np.column_stack([
        np.random.uniform(z_range[0],     z_range[1],     n_samples),
        np.random.uniform(pitch_range[0], pitch_range[1], n_samples),
        np.random.uniform(roll_range[0],  roll_range[1],  n_samples),
    ])

    achievable = []
    failed = []

    for i, (z, pitch, roll) in enumerate(samples):
        if i % 5000 == 0:
            print(f"  [{i:>6}/{n_samples}] achievable so far: {len(achievable)}")
        pose = make_pose(z=z, pitch=pitch, roll=roll)
        if probe_ik(kin, pose):
            achievable.append([z, pitch, roll])
        else:
            failed.append([z, pitch, roll])

    return np.array(achievable), np.array(failed)


# ── Conditional pitch table ───────────────────────────────────────────────────

def pitch_limits_given_z_roll(kin, z_values: np.ndarray,
                               roll_values: np.ndarray) -> dict:
    """For each (z, roll) pair, find the achievable pitch range.

    Returns None for cells where z+roll alone is outside the workspace.
    """
    results = {}
    defaults = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}

    for z_val in z_values:
        for roll_val in roll_values:
            # Pre-check: is z+roll at pitch=0 even achievable?
            baseline_pose = make_pose(z=z_val, roll=roll_val)
            if not probe_ik(kin, baseline_pose):
                results[(round(z_val, 4), round(roll_val, 4))] = None
                continue

            other = {**defaults, 'z': z_val, 'roll': roll_val}
            p_pos = find_limit_1d(kin, 'pitch', +1, other,
                                  coarse_step=0.05, fine_step=0.002, search_range=1.6)
            p_neg = find_limit_1d(kin, 'pitch', -1, other,
                                  coarse_step=0.05, fine_step=0.002, search_range=1.6)
            results[(round(z_val, 4), round(roll_val, 4))] = (p_neg, p_pos)

    return results


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_angular(neg, pos, label):
    print(f"# {label}")
    print(f"#   neg: {neg:+.4f} rad  |  {np.rad2deg(neg):+.2f} deg")
    print(f"#   pos: {pos:+.4f} rad  |  {np.rad2deg(pos):+.2f} deg")


def fmt_linear(neg, pos, label):
    print(f"# {label}")
    print(f"#   neg: {neg:+.4f} m   |  {neg*1000:+.2f} mm")
    print(f"#   pos: {pos:+.4f} m   |  {pos*1000:+.2f} mm")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Initializing AnalyticalKinematics...")
    kin = AnalyticalKinematics()
    print("Ready.\n")

    defaults = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

    # ── Section 1: Per-axis limits ──────────────────────────────────────────
    print("=" * 60)
    print("1. PER-AXIS LIMITS (all other axes at neutral)")
    print("=" * 60)

    axes = {
        'z':     (-0.08, 0.08),
        'pitch': (-1.6,  1.6),
        'roll':  (-1.6,  1.6),
        'yaw':   (-1.6,  1.6),
        'x':     (-0.05, 0.05),
        'y':     (-0.05, 0.05),
    }

    limits = {}
    for axis, (neg_range, pos_range) in axes.items():
        other = {k: v for k, v in defaults.items() if k != axis}
        pos_lim = find_limit_1d(kin, axis, +1, other,
                                coarse_step=abs(pos_range) / 20,
                                fine_step=abs(pos_range) / 200,
                                search_range=abs(pos_range))
        neg_lim = find_limit_1d(kin, axis, -1, other,
                                coarse_step=abs(neg_range) / 20,
                                fine_step=abs(neg_range) / 200,
                                search_range=abs(neg_range))
        limits[axis] = (neg_lim, pos_lim)

        if axis in ('z', 'x', 'y'):
            print(f"  {axis:6s}: [{neg_lim:+.4f}, {pos_lim:+.4f}] m"
                  f"  ({neg_lim*1000:+.1f}mm to {pos_lim*1000:+.1f}mm)")
        else:
            print(f"  {axis:6s}: [{neg_lim:+.4f}, {pos_lim:+.4f}] rad"
                  f"  ({np.rad2deg(neg_lim):+.1f}° to {np.rad2deg(pos_lim):+.1f}°)")

    # ── Section 2: Conditional pitch table (z × roll) ──────────────────────
    print()
    print("=" * 60)
    print("2. PITCH LIMITS GIVEN (Z, ROLL) — 5x5 grid")
    print("=" * 60)

    # Use 60% of z range — 80% still clips the workspace at extremes
    # Roll samples use 70% to stay inside the single-axis roll limit
    z_samples    = np.linspace(limits['z'][0]    * 0.6, limits['z'][1]    * 0.6, 5)
    roll_samples = np.linspace(limits['roll'][0] * 0.7, limits['roll'][1] * 0.7, 5)

    cond_table = pitch_limits_given_z_roll(kin, z_samples, roll_samples)

    print(f"\n  {'z \\ roll':>10}", end="")
    for roll_val in roll_samples:
        print(f"  {np.rad2deg(roll_val):>+8.1f}°", end="")
    print()

    for z_val in z_samples:
        print(f"  {z_val*1000:>+8.1f}mm", end="")
        for roll_val in roll_samples:
            key = (round(z_val, 4), round(roll_val, 4))
            val = cond_table[key]
            if val is None:
                print(f"  {'[unreachable]':>18}", end="")
            else:
                p_neg, p_pos = val
                print(f"  [{np.rad2deg(p_neg):>+5.1f}°,{np.rad2deg(p_pos):>+5.1f}°]", end="")
        print()

    # ── Section 3: Monte Carlo 3D envelope ─────────────────────────────────
    print()
    print("=" * 60)
    print("3. MONTE CARLO 3D ENVELOPE (z, pitch, roll) — 50k samples")
    print("=" * 60)
    print()

    t0 = time.time()
    achievable, failed = sample_3d_envelope(
        kin,
        n_samples=50_000,
        z_range=(limits['z'][0] * 1.1, limits['z'][1] * 1.1),
        pitch_range=(limits['pitch'][0] * 1.1, limits['pitch'][1] * 1.1),
        roll_range=(limits['roll'][0] * 1.1, limits['roll'][1] * 1.1),
    )
    elapsed = time.time() - t0

    print(f"\n  Sampled {len(achievable) + len(failed):,} points in {elapsed:.1f}s")
    print(f"  Achievable: {len(achievable):,}  ({100*len(achievable)/(len(achievable)+len(failed)):.1f}%)")
    print(f"  Failed:     {len(failed):,}")

    hull = ConvexHull(achievable)
    print(f"  Convex hull: {len(hull.vertices)} vertices, {len(hull.simplices)} simplices")

    print(f"\n  Achievable cloud bounds:")
    print(f"    z:     [{achievable[:,0].min():+.4f}, {achievable[:,0].max():+.4f}] m"
          f"  ({achievable[:,0].min()*1000:+.1f}mm to {achievable[:,0].max()*1000:+.1f}mm)")
    print(f"    pitch: [{achievable[:,1].min():+.4f}, {achievable[:,1].max():+.4f}] rad"
          f"  ({np.rad2deg(achievable[:,1].min()):+.1f}° to {np.rad2deg(achievable[:,1].max()):+.1f}°)")
    print(f"    roll:  [{achievable[:,2].min():+.4f}, {achievable[:,2].max():+.4f}] rad"
          f"  ({np.rad2deg(achievable[:,2].min()):+.1f}° to {np.rad2deg(achievable[:,2].max()):+.1f}°)")

    hull_data = {
        "equations":    hull.equations.tolist(),
        "vertices":     achievable[hull.vertices].tolist(),
        "axes":         ["z_m", "pitch_rad", "roll_rad"],
        "n_samples":    len(achievable) + len(failed),
        "n_achievable": len(achievable),
    }
    hull_path = "ik_workspace_hull.json"
    with open(hull_path, "w") as f:
        json.dump(hull_data, f, indent=2)
    print(f"\n  Hull saved to: {hull_path}")

    # ── Section 4: Final output ─────────────────────────────────────────────
    print()
    print("=" * 60)
    print("4. DISCOVERED LIMITS")
    print("=" * 60)
    print()
    print("# IK workspace limits — generated by probe_ik_limits.py")
    print("# Exact per-axis boundaries (all other axes at neutral).")
    print("# For multi-axis use, load ik_workspace_hull.json.")
    print()
    fmt_linear (limits['z'][0],     limits['z'][1],     "Z (translation)")
    print(f"IK_HEAD_Z_LIMIT     = ({limits['z'][0]:.4f}, {limits['z'][1]:.4f})  # meters")
    print()
    fmt_angular(limits['pitch'][0], limits['pitch'][1], "Pitch")
    print(f"IK_HEAD_PITCH_LIMIT = ({limits['pitch'][0]:.4f}, {limits['pitch'][1]:.4f})  # radians")
    print()
    fmt_angular(limits['roll'][0],  limits['roll'][1],  "Roll")
    print(f"IK_HEAD_ROLL_LIMIT  = ({limits['roll'][0]:.4f}, {limits['roll'][1]:.4f})  # radians")
    print()
    fmt_angular(limits['yaw'][0],   limits['yaw'][1],   "Yaw")
    print(f"IK_HEAD_YAW_LIMIT   = ({limits['yaw'][0]:.4f}, {limits['yaw'][1]:.4f})  # radians")
    print()


if __name__ == "__main__":
    main()
