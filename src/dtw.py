"""dtw.py — Quaternion-aware Dynamic Time Warping for motion streams.

Provides two modes of use:

Offline (full sequence)
-----------------------
    distance, path = dtw(stream_a, stream_b)

    cost_matrix(stream_a, stream_b) → (N, M) ndarray  # inspect before DP

Online (live / streaming)
-------------------------
    live = LiveDTW(reference_stream)

    for frame, vel in live_source:
        if vel is not None:
            dist = live.update(vel)   # O(N) per frame

    live.reset()   # start a new segment against the same reference

Distance metric
---------------
Each frame pair (i, j) accumulates a cost summed over all joints:

    cost[i, j] = lin_weight * Σ_joints  ‖Δpos_a[i] - Δpos_b[j]‖₂
               + ang_weight * Σ_joints  arccos(|q_a[i] · q_b[j]|)

The angular term is the geodesic distance on S³.  The absolute-value dot
product handles the quaternion double-cover (q and -q represent the same
rotation but have Euclidean distance 2).

DTW DP
------
The offline DP is implemented via an anti-diagonal wavefront: all cells on
anti-diagonal d = i+j are independent and can be filled with a single numpy
operation, replacing O(N·M) Python iterations with O(N+M) numpy calls.

The online DP maintains a single column of shape (N+1,) and updates it in
O(N) per incoming frame.  The column-wise recurrence is inherently sequential
(curr[i] depends on curr[i-1]), so it cannot be further vectorised.
"""

import numpy as np
from motion import MotionStream


# ── Distance functions ───────────────────────────────────────────────────────

def _lin_cost_row(lin_ai: np.ndarray, lin_b: np.ndarray) -> np.ndarray:
    """Euclidean distance on linear velocity, summed over joints.

    lin_ai : (J, 3) — one frame from sequence A
    lin_b  : (M, J, 3) — all frames from sequence B
    returns: (M,)
    """
    return np.linalg.norm(lin_ai - lin_b, axis=-1).sum(axis=-1)


def _ang_cost_row(ang_ai: np.ndarray, ang_b: np.ndarray) -> np.ndarray:
    """Geodesic distance on S³, summed over joints.

    Metric: arccos(|q_a · q_b|)
      - The absolute value handles the quaternion double-cover (q ≡ -q).
      - For unit quaternions this equals the angle of the relative rotation,
        i.e. the shortest arc on the 3-sphere.

    ang_ai : (J, 4) — one frame from sequence A
    ang_b  : (M, J, 4) — all frames from sequence B
    returns: (M,)
    """
    dot = np.einsum('jk,mjk->mj', ang_ai, ang_b)   # (M, J)
    return np.arccos(np.clip(np.abs(dot), 0.0, 1.0)).sum(axis=-1)


# ── Cost matrix ──────────────────────────────────────────────────────────────

def cost_matrix(
    stream_a: MotionStream,
    stream_b: MotionStream,
    joint_ids: list[str] | None = None,
    lin_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> np.ndarray:
    """Return the (N, M) pairwise frame-distance matrix between two streams.

    Each cell [i, j] is:
        lin_weight * Σ_joints ||Δpos_a[i] - Δpos_b[j]||₂
      + ang_weight * Σ_joints arccos(|q_a[i] · q_b[j]|)
    """
    lin_a, ang_a, lin_b, ang_b = _resolve(stream_a, stream_b, joint_ids)
    N, M = len(lin_a), len(lin_b)
    C = np.empty((N, M))
    for i in range(N):
        C[i] = (
            lin_weight * _lin_cost_row(
                lin_a[i], lin_b
            ) + ang_weight * _ang_cost_row(
                ang_a[i], ang_b
            )
        )
    return C


# ── DTW ──────────────────────────────────────────────────────────────────────

def dtw(
    stream_a: MotionStream,
    stream_b: MotionStream,
    joint_ids: list[str] | None = None,
    lin_weight: float = 1.0,
    ang_weight: float = 1.0,
) -> tuple[float, list[tuple[int, int]]]:
    """Quaternion-aware DTW between two motion streams.

    Args:
        stream_a, stream_b: loaded MotionStream objects
        joint_ids:          joints to include; defaults to all joints in A ∩ B,
                            in A's DFS order
        lin_weight:         weight on linear (position delta) distance
        ang_weight:         weight on angular (quaternion delta, geodesic)
                            distance

    Returns:
        distance:  scalar DTW distance (normalised by path length)
        path:      optimal warping path as list of (i, j) frame-index pairs
    """
    C           = cost_matrix(
        stream_a, stream_b, joint_ids, lin_weight, ang_weight
    )
    dist, D     = _dp(C)
    path        = _backtrack(D)
    return dist / len(path), path


# ── DP and backtrack ──────────────────────────────────────────────────────

def _dp(C: np.ndarray) -> tuple[float, np.ndarray]:
    """DTW dynamic programming via anti-diagonal wavefront (vectorised).

    Each anti-diagonal d = i+j shares no data dependencies within the diagonal,
    so all its cells can be filled with a single numpy operation instead of a
    Python loop — replacing O(N*M) Python iterations with O(N+M) numpy calls.
    """
    N, M = C.shape
    D = np.full((N + 1, M + 1), np.inf)
    D[0, 0] = 0.0

    for d in range(2, N + M + 1):          # d = i+j (1-indexed in D)
        i = np.arange(max(1, d - M), min(N, d - 1) + 1)
        j = d - i
        D[i, j] = C[i - 1, j - 1] + np.minimum(
            np.minimum(D[i - 1, j], D[i, j - 1]),
            D[i - 1, j - 1],
        )

    return float(D[N, M]), D[1:, 1:]


def _backtrack(D: np.ndarray) -> list[tuple[int, int]]:
    """
    Trace the optimal warping path back through the accumulated cost matrix.
    """
    i, j = np.array(D.shape) - 1
    path = [(int(i), int(j))]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
            if step == 0:
                i -= 1
                j -= 1
            elif step == 1:
                i -= 1
            else:
                j -= 1
        path.append((int(i), int(j)))
    return path[::-1]


# ── Live / streaming DTW ─────────────────────────────────────────────────────

class LiveDTW:
    """Online DTW against a fixed reference, updated one live frame at a time.

    Keeps one DP column of shape (N+1,) where N = len(reference.velocities).
    Each call to update() costs O(N) — a tight loop over the reference length —
    so it is suitable for real-time use at typical motion-capture frame rates.

    The returned value after j live frames is D[N, j]: the cumulative cost of
    warping all N reference frames to the j live frames received so far.
    Normalise by .frames_seen if you want a per-frame average.

    Usage::

        ref  = motion.MotionStream("reference.bvh")
        live = LiveDTW(ref)

        prev_frame = None
        for raw_frame in sensor:          # whatever produces live Joint trees
            normalize(raw_frame)
            if prev_frame is not None:
                vel = motion._compute_velocity(prev_frame, raw_frame)
                dist = live.update(vel)
                print(dist)
            prev_frame = raw_frame

    Call reset() to start a new segment without reloading the reference.
    """

    def __init__(
        self,
        reference: MotionStream,
        joint_ids: list[str] | None = None,
        lin_weight: float = 1.0,
        ang_weight: float = 1.0,
        max_frames: int | None = None,
        band: int | None = None,
    ):
        ids    = joint_ids or reference._joint_ids
        id_map = {jid: j for j, jid in enumerate(reference._joint_ids)}
        idx    = [id_map[jid] for jid in ids]

        self._ref_lin   = reference._lin[:max_frames, idx]   # (N, J, 3)
        self._ref_ang   = reference._ang[:max_frames, idx]   # (N, J, 4)
        self._joint_ids = ids
        self._lin_w     = lin_weight
        self._ang_w     = ang_weight
        self._N         = len(self._ref_lin)
        self._band      = band if band is not None else self._N
        self._j         = 0
        self._incr      = float("inf")
        self._col       = self._init_col()

    # ── public interface ─────────────────────────────────────────────────────

    def update(self, velocity: dict) -> float:
        """
        Feed one live velocity frame; return the cumulative DTW cost D[N, j].

        velocity: dict mapping joint_id → {"linear": (3,), "angular": (4,)}
                  Same format yielded by MotionStream.__next__; do not call
                  with the first frame's None velocity.
        """
        lin = np.stack(
            [velocity[jid]["linear"] for jid in self._joint_ids]
        )  # (J, 3)
        ang = np.stack(
            [velocity[jid]["angular"] for jid in self._joint_ids]
        )  # (J, 4)

        # Cost of this live frame against every reference frame: (N,)
        lin_c = np.linalg.norm(
            lin - self._ref_lin, axis=-1
        ).sum(axis=-1)
        dot = np.einsum('jk,njk->nj', ang, self._ref_ang)  # (N, J)
        ang_c = np.arccos(np.clip(np.abs(dot), 0.0, 1.0)).sum(axis=-1)
        cost = self._lin_w * lin_c + self._ang_w * ang_c   # (N,)

        # Sequential column update with optional Sakoe-Chiba band.
        # Cells outside |i - j| > band are set to inf, preventing the path
        # from compressing the match window to fewer than N - band frames.
        j    = self._j + 1  # 1-indexed live frame about to be processed
        prev = self._col
        curr = np.full(self._N + 1, np.inf)
        curr[0] = 0.0   # subsequence DTW: D[0,j]=0 for all j
        lo = max(1, j - self._band)
        hi = min(self._N, j + self._band)
        for i in range(lo, hi + 1):
            curr[i] = cost[i - 1] + min(prev[i], curr[i - 1], prev[i - 1])

        prev_cost    = float(self._col[self._N])
        self._col    = curr
        self._j     += 1
        self._incr   = float(curr[self._N]) - prev_cost
        return float(curr[self._N])

    def reset(self) -> None:
        """
        Discard live-stream history; keep the reference for a new segment.
        """
        self._col  = self._init_col()
        self._j    = 0
        self._incr = float("inf")

    @property
    def increment(self) -> float:
        """Marginal cost of the last live frame, clipped to [0, ∞).

        D[N, j] - D[N, j-1] can be negative when a new frame allows the DTW
        to find a better path (i.e. an even better fit than before).  We clip
        to zero so the distribution stays non-negative and bimodal: near-zero
        for motion frames, large positive for idle frames.
        """
        return max(0.0, self._incr)

    @property
    def cost(self) -> float:
        """Raw accumulated DTW cost D[N, j] after j live frames."""
        return float(self._col[self._N])

    @property
    def subseq_distance(self) -> float:
        """Best-match cost normalised by reference length N.

        With subsequence-DTW initialisation (curr[0]=0), D[N,j] is the
        minimum cost of aligning the full N-frame reference to *any*
        contiguous window of live frames ending at j.  Dividing by N
        gives a per-reference-frame average that is low when the current
        live motion matches the reference and high otherwise.
        """
        return float(self._col[self._N]) / self._N if self._N else 0.0

    @property
    def distance(self) -> float:
        """DTW cost normalised by frames seen (comparable across lengths)."""
        return float(self._col[self._N]) / self._j if self._j else 0.0

    @property
    def frames_seen(self) -> int:
        """Number of live velocity frames fed so far."""
        return self._j

    # ── internal ─────────────────────────────────────────────────────────────

    def _init_col(self) -> np.ndarray:
        col    = np.full(self._N + 1, np.inf)
        col[0] = 0.0
        return col


# ── Helpers ────────────────────────────────────────────────────────────────

def _resolve(
    stream_a: MotionStream,
    stream_b: MotionStream,
    joint_ids: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract aligned (lin, ang) arrays for the chosen joints."""
    ids = joint_ids or [
        jid for jid in stream_a._joint_ids
        if jid in set(stream_b._joint_ids)
    ]
    a_map = {jid: j for j, jid in enumerate(stream_a._joint_ids)}
    b_map = {jid: j for j, jid in enumerate(stream_b._joint_ids)}
    ia    = [a_map[jid] for jid in ids]
    ib    = [b_map[jid] for jid in ids]
    return stream_a._lin[:, ia], \
        stream_a._ang[:, ia], \
        stream_b._lin[:, ib], \
        stream_b._ang[:, ib]
