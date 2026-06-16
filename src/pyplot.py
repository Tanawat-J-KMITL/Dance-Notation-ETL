"""pyplot.py — 3-D skeleton animation using Plotly.

Single public function:

    plot_skeleton(frames) → go.Figure

Coordinate remapping
--------------------
BVH/kinematics uses the convention  X = right, Y = up, Z = forward.
The plotter maps  bvh-Y → plot-Z  and  bvh-Z → plot-Y (inverted) so
that "forward" points into the screen, matching the old matplotlib view.
"""

import numpy as np
import plotly.graph_objects as go
from kinematics import Joint


def extract_joints_and_bones(
    root: Joint,
) -> tuple[list[Joint], list[tuple[Joint, Joint]]]:
    joints, bones = [], []

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
    joint_count = len(joints)
    frame_count = len(frames)
    positions   = np.zeros((frame_count, joint_count, 3))
    idx_of      = {j.id: i for i, j in enumerate(joints)}

    for f, frame_root in enumerate(frames):
        def _walk_and_compute(joint: Joint, parent_world):
            local_T = joint.transform.local
            world_T = local_T if parent_world is None else parent_world @ local_T
            if joint.id in idx_of:
                positions[f, idx_of[joint.id]] = world_T[:3, 3]
            for child in joint.children:
                _walk_and_compute(child, world_T)

        _walk_and_compute(frame_root, None)

    return positions


def _bone_segments(pts_x, pts_y, pts_z, bone_indices):
    """Interleave None separators so a single Scatter3d draws all bones."""
    xs, ys, zs = [], [], []
    for pi, ci in bone_indices:
        xs += [float(pts_x[pi]), float(pts_x[ci]), None]
        ys += [float(pts_y[pi]), float(pts_y[ci]), None]
        zs += [float(pts_z[pi]), float(pts_z[ci]), None]
    return xs, ys, zs


def plot_skeleton(frames: list[Joint]):
    """Animate a list of per-frame Joint trees using a Plotly figure."""
    if not frames:
        return

    joints, bones = extract_joints_and_bones(frames[0])
    positions     = extract_positions_optimized(frames, joints)
    frame_count   = len(frames)
    idx_of        = {j.id: i for i, j in enumerate(joints)}
    bone_indices  = [(idx_of[p.id], idx_of[c.id]) for p, c in bones]

    # Coordinate remap: bvh-Y → plot-Z,  bvh-Z → plot-Y (inverted)
    px_ = positions[:, :, 0]
    py_ = -positions[:, :, 2]
    pz_ = positions[:, :, 1]

    def frame_traces(f):
        bx, by, bz = _bone_segments(px_[f], py_[f], pz_[f], bone_indices)
        return [
            go.Scatter3d(x=px_[f], y=py_[f], z=pz_[f],
                         mode="markers", marker=dict(size=4, color="red"),
                         name="joints", showlegend=False),
            go.Scatter3d(x=bx, y=by, z=bz,
                         mode="lines", line=dict(color="black", width=2),
                         name="bones", showlegend=False),
        ]

    anim_frames = [
        go.Frame(data=frame_traces(f), name=str(f))
        for f in range(frame_count)
    ]

    flat   = positions.reshape(-1, 3)
    mn, mx = flat.min(axis=0), flat.max(axis=0)
    mid    = (mn + mx) / 2
    half   = ((mx - mn).max() / 2) * 1.1

    fig = go.Figure(
        data=frame_traces(0),
        frames=anim_frames,
        layout=go.Layout(
            scene=dict(
                xaxis=dict(range=[mid[0] - half, mid[0] + half], title="X"),
                yaxis=dict(range=[-mid[2] - half, -mid[2] + half], title="Z (inv)"),
                zaxis=dict(range=[mid[1] - half, mid[1] + half], title="Y"),
                aspectmode="cube",
            ),
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=0.05, redraw=True),
                                          fromcurrent=True)]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                             mode="immediate")]),
                ],
            )],
            sliders=[dict(
                steps=[
                    dict(method="animate",
                         args=[[str(f)], dict(mode="immediate",
                                              frame=dict(duration=0, redraw=True))],
                         label=str(f))
                    for f in range(frame_count)
                ],
                transition=dict(duration=0),
                x=0, y=0, len=1.0,
            )],
        ),
    )
    fig.show()
    return fig
