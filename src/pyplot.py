"""pyplot.py — 3-D skeleton animation using Plotly.

Single public function:

    plot_skeleton(frames) → go.Figure

Coordinate remapping
--------------------
BVH/kinematics uses the convention  X = right, Y = up, Z = forward.
The plotter maps  bvh-Y → plot-Z  and  bvh-X → plot-Y (inverted) so
that "right" points into the screen, matching the old matplotlib view.
"""

import numpy as np
import plotly.graph_objects as go
from kinematics import Joint

PLAYER_SIZE = 600


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
            world_T = local_T if parent_world is None \
                else parent_world @ local_T
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

    # Coordinate remap: bvh-Y → plot-Z,  bvh-X → plot-Y (inverted)
    px_ = positions[:, :, 2]
    py_ = -positions[:, :, 0]
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
            width=PLAYER_SIZE,
            height=PLAYER_SIZE,
            scene=dict(
                xaxis=dict(range=[mid[2] - half, mid[2] + half], title="Z"),
                yaxis=dict(range=[-mid[0] - half, -mid[0] + half], title="X"),
                zaxis=dict(range=[mid[1] - half, mid[1] + half], title="Y"),
                aspectmode="cube",
            ),
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, dict(
                        frame=dict(duration=12, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True
                    )]),
                    dict(label="Pause", method="animate", args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate"
                    )]),
                ],
            )],
            sliders=[dict(
                steps=[
                    dict(method="animate", args=[[str(f)], dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True)
                    )], label=str(f))
                    for f in range(frame_count)
                ],
                transition=dict(duration=0),
                x=0, y=0, len=1.0,
            )],
        ),
    )
    fig.show()
    return fig


def save_video(fig: go.Figure, path: str, fps: int = 30, step: int = 1):
    """Render frames with matplotlib and encode via ffmpeg.

    path: output filename — "out.mp4" or "out.gif"
    step: render every Nth frame to trade quality for speed.
    Requires ffmpeg on PATH (pkgs.ffmpeg in flake.nix).
    """
    import subprocess
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene  = fig.layout.scene
    size   = PLAYER_SIZE / 100

    if path.endswith(".gif"):
        cmd = [
            "ffmpeg", "-y",
            "-f", "image2pipe", "-vcodec", "png", "-r", str(fps), "-i", "-",
            "-vf", "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop", "0",
            path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "image2pipe", "-vcodec", "png", "-r", str(fps), "-i", "-",
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            path,
        ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    for frame in fig.frames[::step]:
        mfig = plt.figure(figsize=(size, size))
        ax   = mfig.add_subplot(111, projection="3d")

        for trace in frame.data:
            mode = getattr(trace, "mode", "") or ""
            if "markers" in mode:
                ax.scatter(trace.x, trace.y, trace.z, c="red", s=16)
            elif "lines" in mode:
                xs, ys, zs = list(trace.x), list(trace.y), list(trace.z)
                i = 0
                while i < len(xs):
                    if xs[i] is None:
                        i += 1
                        continue
                    j = xs.index(None, i) if None in xs[i:] else len(xs)
                    ax.plot(xs[i:j], ys[i:j], zs[i:j], "k-", linewidth=1)
                    i = j + 1

        if scene.xaxis.range:
            ax.set_xlim(scene.xaxis.range)
        if scene.yaxis.range:
            ax.set_ylim(scene.yaxis.range)
        if scene.zaxis.range:
            ax.set_zlim(scene.zaxis.range)

        ax.set_xlabel("Z")
        ax.set_ylabel("X")
        ax.set_zlabel("Y")

        buf = io.BytesIO()
        mfig.savefig(buf, format="png", dpi=100)
        plt.close(mfig)
        proc.stdin.write(buf.getvalue())

    proc.stdin.close()
    proc.wait()
