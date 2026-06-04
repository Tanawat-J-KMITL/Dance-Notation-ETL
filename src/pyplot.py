import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from kinematics import Joint
from matplotlib.animation import FuncAnimation

# Configuration
matplotlib.use('QtAgg')


# Graph plotting helpers
def extract_joints_and_bones(
    root: Joint
) -> tuple[list[Joint], list[tuple[Joint, Joint]]]:
    """Flatten the Joint tree -> list of joints & parent-child bone pairs."""
    joints = []
    bones  = []

    def _walk(joint: Joint):
        joints.append(joint)
        for child in joint.children:
            bones.append((joint, child))
            _walk(child)

    _walk(root)
    return joints, bones


def extract_positions_optimized(
    frames: list[Joint], joints: list[Joint]
) -> np.ndarray:
    """
    Computes world positions using a single top-down tree walk per frame.
    Bypasses the expensive recursive HTM .root property.
    """
    joint_count = len(joints)
    frame_count = len(frames)
    positions   = np.zeros((frame_count, joint_count, 3))

    # Map joint IDs to flat array indices for O(1) filling
    idx_of = {j.id: i for i, j in enumerate(joints)}

    for f, frame_root in enumerate(frames):
        def _walk_and_compute(joint: Joint, parent_world: np.ndarray):
            # Compute world transform downstream linearly
            local_T = joint.transform.local
            world_T = local_T if parent_world \
                is None else parent_world @ local_T

            # Save world position directly
            if joint.id in idx_of:
                positions[f, idx_of[joint.id]] = world_T[:3, 3]

            for child in joint.children:
                _walk_and_compute(child, world_T)

        _walk_and_compute(frame_root, None)

    return positions


def plot_skeleton(frames: list[Joint]):
    """Animate a list of per-frame Joint trees with optimized lookups."""
    if not frames:
        return

    # ── 1. Precalculate Topology & Positions Once ─────────────────────
    joints, bones = extract_joints_and_bones(frames[0])
    positions     = extract_positions_optimized(frames, joints)
    frame_count   = len(frames)

    # Precalculate flat array indices for bones to keep the animation loop fast
    idx_of       = {j.id: i for i, j in enumerate(joints)}
    bone_indices = [(idx_of[p.id], idx_of[c.id]) for p, c in bones]

    # ── 2. Figure Setup ───────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='3d')

    mn  = positions.reshape(-1, 3).min(axis=0)
    mx  = positions.reshape(-1, 3).max(axis=0)
    pad = (mx - mn).max() * 0.1

    mid_x = (mx[0] + mn[0]) / 2
    mid_y = (mx[1] + mn[1]) / 2
    mid_z = (mx[2] + mn[2]) / 2

    # Find the maximum span across any axis and add padding
    max_range = (mx - mn).max()
    half_extent = (max_range / 2) + pad

    ax.set_xlim(mid_x - half_extent, mid_x + half_extent)
    ax.set_ylim(mid_z - half_extent, mid_z + half_extent)  # Z → matplotlib Y
    ax.set_zlim(mid_y - half_extent, mid_y + half_extent)  # Y → matplotlib Z

    ax.set_box_aspect((1, 1, 1))
    ax.invert_yaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    scatter    = ax.scatter([], [], [], c='red', s=20)
    bone_lines = [ax.plot([], [], [], c='black', lw=1)[0] for _ in bones]
    title      = ax.set_title("")

    # ── 3. Optimized Animation Callback ───────────────────────────────
    def update(frame):
        pts = positions[frame]

        # Update joint dots (Swapping Y and Z)
        scatter._offsets3d = (pts[:, 0], pts[:, 2], pts[:, 1])

        # Update bone lines using precalculated integer indices
        for line, (pi, ci) in zip(bone_lines, bone_indices):
            line.set_data([pts[pi, 0], pts[ci, 0]], [pts[pi, 2], pts[ci, 2]])
            line.set_3d_properties([pts[pi, 1], pts[ci, 1]])

        title.set_text(f"Frame {frame}/{frame_count - 1}")
        return [scatter, *bone_lines, title]

    ani = FuncAnimation(
        fig, update, frames=frame_count, interval=16, blit=False
    )
    plt.show()
    return ani
