"""motion.py — Normalised, velocity-annotated BVH motion stream.

The central class is MotionStream, which wraps a BVH file and exposes its
frames as a resettable Python iterator.  Internally it:

1. Parses the BVH file via a module-level LRU cache so the same file is only
   parsed once regardless of how many MotionStream instances point to it.
2. Normalises every frame: the root joint is translated to the origin and
   rotated to the identity pose, and the entire skeleton is scaled so the
   spine length equals target_spine_length (default 1.0).
3. Computes per-frame velocity using batch numpy operations (no scipy in the
   hot path), storing results as (N, J, 3) linear and (N, J, 4) angular arrays.

Velocity convention
-------------------
velocity[i] = motion from frame i to frame i+1 (forward difference).
  linear  : world-space position delta, Δpos = pos[i+1] - pos[i]  shape (3,)
  angular : quaternion delta, q_prev⁻¹ * q_curr  shape (4,) xyzw

The iterator yields (frame: Joint, velocity: dict | None); velocity is None
for the first frame (no predecessor).

Quick start
-----------
    stream = MotionStream("recording.bvh")
    for frame, vel in stream:
        if vel is not None:
            print(vel["torso_1"]["linear"])   # (3,) position delta

    # For DTW — flat numpy array, shape (N_frames-1, N_joints * 7)
    arr = stream.as_array()
"""

import functools
import numpy as np
import bvhio
from kinematics import Joint
from model import convertBvhToHierarchy


_SPINE_CHAIN = [
    "root",
    "torso_1", "torso_2", "torso_3", "torso_4",
    "torso_5", "torso_6", "torso_7",
    "neck_1",
]


# ── BVH cache ────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=8)
def _parse_bvh(path: str) -> bvhio.BvhContainer:
    # bvhio converts all Euler keyframes to quaternions upfront — the dominant
    # cost.  Caching means multiple MotionStream instances on the same file pay
    # that cost only once.
    return bvhio.readAsBvh(path)


# ── Batch quaternion helpers (pure numpy, no scipy) ──────────────────────────

def _bquat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(N, 4) xyzw → (N, 3, 3) rotation matrices."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = len(q)
    R = np.empty((N, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _bquat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, (N, 4) x (N, 4) → (N, 4) in xyzw convention."""
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], axis=1)


def _bquat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (= inverse for unit quats), (N, 4) → (N, 4)."""
    c = q.copy()
    c[:, :3] *= -1
    return c


# ── MotionStream ─────────────────────────────────────────────────────────────

class MotionStream:
    """Preloaded BVH motion exposed as a resettable iterator.

    Each iteration yields (frame: Joint, velocity: dict | None).
    The first frame yields velocity=None (no predecessor).
    Subsequent frames yield per-joint deltas from the previous frame:
        {"linear": np.ndarray(3,), "angular": np.ndarray(4,)}

    Use .as_array() to get a (N_frames-1, N_joints * 7) matrix for DTW.
    """

    def __init__(self, bvh_path: str, target_spine_length: float = 1.0):
        self._target_spine_length = target_spine_length
        self._spine_len: float | None = None
        self._index = 0

        bvh = _parse_bvh(bvh_path)
        self._frames: list[Joint] = []

        for i in range(bvh.FrameCount):
            frame = convertBvhToHierarchy(bvh, i)
            self._normalize(frame)
            self._frames.append(frame)

        self._joint_ids, self._lin, self._ang, self._velocities = \
            self._batch_velocities()

    def _batch_velocities(self):
        """Compute all per-joint velocities in batch numpy ops.

        World transforms are computed by walking joints in DFS (topological)
        order and accumulating parent state downward — no recursive property
        accesses, no scipy calls.
        """
        frames = self._frames
        N = len(frames)

        # Topology: DFS order so every parent index < child index.
        joint_ids: list[str] = []
        parent_idx: list[int] = []

        def _topo(j: Joint, p: int) -> None:
            i = len(joint_ids)
            joint_ids.append(j.id)
            parent_idx.append(p)
            for child in j.children:
                _topo(child, i)

        _topo(frames[0], -1)
        J = len(joint_ids)

        # Extract local offsets + quats for every frame into (N, J, 3/4).
        offsets = np.empty((N, J, 3))
        quats   = np.empty((N, J, 4))
        for f, frame in enumerate(frames):
            for j, jid in enumerate(joint_ids):
                joint = frame[jid]
                offsets[f, j] = joint._offset
                quats[f, j]   = joint._quat

        # Propagate world transforms top-down (parent precedes child in DFS).
        world_pos  = np.zeros((N, J, 3))
        world_quat = np.zeros((N, J, 4))
        world_quat[:, :, 3] = 1.0  # identity w

        for j in range(J):
            p = parent_idx[j]
            if p == -1:
                # Root: offset and quat are zeroed/identity by _normalize.
                world_pos[:, j]  = offsets[:, j]
                world_quat[:, j] = quats[:, j]
            else:
                pR = _bquat_to_matrix(world_quat[:, p])  # (N, 3, 3)
                rot_offset = np.einsum('nij,nj->ni', pR, offsets[:, j])
                world_pos[:, j] = rot_offset + world_pos[:, p]
                world_quat[:, j] = _bquat_mul(world_quat[:, p], quats[:, j])

        # Velocity between consecutive frames.
        lin = world_pos[1:] - world_pos[:-1]  # (N-1, J, 3)
        M   = N - 1
        ang = _bquat_mul(
            _bquat_conj(world_quat[:-1].reshape(M * J, 4)),
            world_quat[1:].reshape(M * J, 4),
        ).reshape(M, J, 4)  # (N-1, J, 4)

        # Convert to list-of-dicts for the streaming API.
        velocities = [
            {jid: {"linear": lin[n, j], "angular": ang[n, j]}
             for j, jid in enumerate(joint_ids)}
            for n in range(M)
        ]
        return joint_ids, lin, ang, velocities

    # ── iterator ─────────────────────────────────────────────────────────────

    def __iter__(self) -> "MotionStream":
        self._index = 0
        return self

    def __next__(self) -> tuple[Joint, dict | None]:
        if self._index >= len(self._frames):
            raise StopIteration
        i = self._index
        self._index += 1
        vel = self._velocities[i - 1] if i > 0 else None
        return self._frames[i], vel

    # ── direct access ────────────────────────────────────────────────────────

    @property
    def frames(self) -> list[Joint]:
        return self._frames

    @property
    def velocities(self) -> list[dict]:
        """Per-frame velocity dicts; len = len(frames) - 1."""
        return self._velocities

    def as_array(self, joint_ids: list[str] | None = None) -> np.ndarray:
        """Return velocities as a (N_frames-1, N_joints * 7) numpy array.

        Feature order per joint:
        [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z, ang_w].
        Defaults to DFS joint order; pass joint_ids to filter or reorder.
        """
        if not self._velocities:
            return np.empty((0, 0))
        if joint_ids is None:
            lin, ang = self._lin, self._ang
        else:
            id_to_j  = {jid: j for j, jid in enumerate(self._joint_ids)}
            idx      = [id_to_j[jid] for jid in joint_ids]
            lin, ang = self._lin[:, idx], self._ang[:, idx]
        N, J, _ = lin.shape
        return np.concatenate([lin, ang], axis=2).reshape(N, J * 7)

    # ── normalization ────────────────────────────────────────────────────────

    def _get_spine_length(self, root: Joint) -> float:
        if self._spine_len is not None:
            return self._spine_len
        total = 0.0
        for i in range(1, len(_SPINE_CHAIN)):
            parent = root[_SPINE_CHAIN[i - 1]]
            child  = root[_SPINE_CHAIN[i]]
            if parent is None or child is None:
                continue
            total += np.linalg.norm(child.offset)
        self._spine_len = total
        return total

    def _normalize(self, root: Joint) -> None:
        root.offset = np.array([0.0, 0.0, 0.0])
        root.quat   = np.array([0.0, 0.0, 0.0, 1.0])
        current = self._get_spine_length(root)
        if current < 1e-6:
            return
        scale = self._target_spine_length / current

        def _walk(j: Joint) -> None:
            j.offset = j.offset * scale
            for child in j.children:
                _walk(child)

        _walk(root)
