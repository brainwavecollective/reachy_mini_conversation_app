"""Reachy Mini IK Workspace Probe
 
Characterizes the safe operating envelope for head pose offsets
by probing the AnalyticalKinematics IK solver across the pose space
relevant to emotional expression (pitch and z primarily).

Run this script standalone — no robot connection required.
Outputs safe range constants ready to paste into moves.py.

Usage:
    python probe_workspace.py

Requirements:
    pip install reachy-mini  (or install from local SDK)
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from reachy_mini.kinematics import AnalyticalKinematics


def make_pose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    """Build a 4x4 pose matrix from translation (meters) and RPY (radians)."""
    pose = np.eye(4)
    pose[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    pose[:3, 3] = [x, y, z]
    return pose


def probe_ik(kin: AnalyticalKinematics, pose, body_yaw: float = 0.0) -> bool:
    """Return True if pose is achievable."""
    try:
        joints = kin.ik(pose, body_yaw=body_yaw)
        return joints is not None and not np.any(np.isnan(joints))
    except (ValueError, Exception):
        return False


def find_limit_1d(kin, axis: str, sign: int, other_defaults: dict,
                  coarse_step=0.005, fine_step=0.001, search_range=0.5) -> float:
    """Binary-search the limit along one axis in one direction.
    
    Returns the last achievable value before IK fails.
    """
    val = 0.0
    last_good = 0.0
    step = coarse_step * sign

    # Coarse pass
    while abs(val) < search_range:
        val += step
        kwargs = {**other_defaults, axis: val}
        pose = make_pose(**kwargs)
        if not probe_ik(kin, pose):
            break
        last_good = val

    # Fine pass: back off and step finely
    val = last_good
    step = fine_step * sign
    while abs(val) < search_range:
        val += step
        kwargs = {**other_defaults, axis: val}
        pose = make_pose(**kwargs)
        if not probe_ik(kin, pose):
            break
        last_good = val

    return last_good


def probe_2d_envelope(kin, axis_a: str, axis_b: str, steps=20,
                      range_a=(-0.3, 0.3), range_b=(-0.05, 0.05)) -> dict:
    """Sweep a 2D grid of axis_a vs axis_b and record which cells are achievable.
    
    Returns a dict with the achievable range for axis_a at each axis_b sample.
    """
    results = {}
    b_values = np.linspace(range_b[0], range_b[1], steps)
    
    for b_val in b_values:
        achievable_a = []
        for a_val in np.linspace(range_a[0], range_a[1], steps * 2):
            kwargs = {axis_a: a_val, axis_b: b_val, 'x': 0, 'y': 0}
            # fill remaining axes with 0
            for ax in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                if ax not in kwargs:
                    kwargs[ax] = 0.0
            pose = make_pose(**kwargs)
            if probe_ik(kin, pose):
                achievable_a.append(a_val)
        
        if achievable_a:
            results[round(b_val, 4)] = (min(achievable_a), max(achievable_a))
        else:
            results[round(b_val, 4)] = None

    return results


def main():
    print("Initializing AnalyticalKinematics...")
    kin = AnalyticalKinematics()
    print("Ready.\n")

    defaults = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}

    print("=" * 60)
    print("PROBING INDIVIDUAL AXIS LIMITS (all others at neutral)")
    print("=" * 60)

    axes = {
        'z':     (-0.08, 0.08),
        'pitch': (-1.5,  1.5),
        'roll':  (-1.5,  1.5),
        'yaw':   (-1.5,  1.5),
        'x':     (-0.05, 0.05),
        'y':     (-0.05, 0.05),
    }

    limits = {}
    for axis, (neg_range, pos_range) in axes.items():
        other = {k: v for k, v in defaults.items() if k != axis}
        
        pos_limit = find_limit_1d(kin, axis, +1, other,
                                   coarse_step=abs(pos_range) / 20,
                                   fine_step=abs(pos_range) / 200,
                                   search_range=abs(pos_range))
        neg_limit = find_limit_1d(kin, axis, -1, other,
                                   coarse_step=abs(neg_range) / 20,
                                   fine_step=abs(neg_range) / 200,
                                   search_range=abs(neg_range))
        
        limits[axis] = (neg_limit, pos_limit)
        print(f"  {axis:6s}: [{neg_limit:+.4f}, {pos_limit:+.4f}] rad/m"
              f"  ({np.rad2deg(neg_limit):+.1f}° to {np.rad2deg(pos_limit):+.1f}°)")

    print()
    print("=" * 60)
    print("PROBING PITCH+Z COMBINED ENVELOPE (most relevant for emotions)")
    print("=" * 60)
    
    # The emotional offsets drive primarily pitch and z together
    # Find safe pitch range at several z offsets
    # Use inner 70% of z range for combined sweep — extremes are not useful for expression
    # and drag the combined pitch limit to near zero
    z_inner_neg = limits['z'][0] * 0.7
    z_inner_pos = limits['z'][1] * 0.7
    z_samples = np.linspace(z_inner_neg, z_inner_pos, 7)
    print(f"\n  Pitch range at each z offset (inner 70% of z envelope):")
    
    combined_limits = {}
    for z_val in z_samples:
        other = {**defaults, 'z': z_val}
        other.pop('pitch')
        p_pos = find_limit_1d(kin, 'pitch', +1, other, coarse_step=0.05, fine_step=0.001, search_range=1.5)
        p_neg = find_limit_1d(kin, 'pitch', -1, other, coarse_step=0.05, fine_step=0.001, search_range=1.5)
        combined_limits[round(z_val, 4)] = (p_neg, p_pos)
        print(f"    z={z_val:+.4f}m  pitch: [{p_neg:+.4f}, {p_pos:+.4f}] rad"
              f"  ({np.rad2deg(p_neg):+.1f}° to {np.rad2deg(p_pos):+.1f}°)")

    # Tightest pitch across all z samples (worst case for combined motion)
    all_pitch_neg = min(v[0] for v in combined_limits.values())
    all_pitch_pos = min(v[1] for v in combined_limits.values())

    print()
    print("=" * 60)
    print("DISCOVERED LIMITS — paste into your app and apply margin there")
    print("=" * 60)
    print()
    print("# IK workspace limits — generated by probe_workspace.py")
    print("# Exact boundaries where IK starts failing.")
    print("# Apply your own margin when using these in moves.py.")
    print()

    def fmt_angular(neg, pos, label):
        print(f"# {label}")
        print(f"#   neg: {neg:.4f} rad  |  {np.rad2deg(neg):+.2f} deg")
        print(f"#   pos: {pos:.4f} rad  |  {np.rad2deg(pos):+.2f} deg")

    def fmt_linear(neg, pos, label):
        print(f"# {label}")
        print(f"#   neg: {neg:.4f} m  |  {neg*1000:+.2f} mm")
        print(f"#   pos: {pos:.4f} m  |  {pos*1000:+.2f} mm")

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
    fmt_linear (limits['x'][0],     limits['x'][1],     "X (translation)")
    print(f"IK_HEAD_X_LIMIT     = ({limits['x'][0]:.4f}, {limits['x'][1]:.4f})  # meters")
    print()
    fmt_linear (limits['y'][0],     limits['y'][1],     "Y (translation)")
    print(f"IK_HEAD_Y_LIMIT     = ({limits['y'][0]:.4f}, {limits['y'][1]:.4f})  # meters")
    print()
    print("# Pitch limit at each z offset (worst case for combined pitch+z motion):")
    for z_val, (p_neg, p_pos) in combined_limits.items():
        print(f"#   z={z_val:+.4f}m ({z_val*1000:+.1f}mm)  pitch: [{p_neg:.4f}, {p_pos:.4f}] rad  [{np.rad2deg(p_neg):+.1f}°, {np.rad2deg(p_pos):+.1f}°]")
    print()
    fmt_angular(all_pitch_neg, all_pitch_pos, "Pitch — worst case across z sweep")
    print(f"IK_HEAD_PITCH_WITH_Z_LIMIT = ({all_pitch_neg:.4f}, {all_pitch_pos:.4f})  # radians")
    print()


if __name__ == "__main__":
    main()
