# Dance Notation ETL — Technical Documentation

## Overview

This project is an ETL (Extract–Transform–Load) pipeline that converts raw motion-capture recordings of traditional Japanese folk dances into labelled, machine-readable notation. The pipeline reads BVH files produced by a Sony mocopi suit, normalises the skeleton, computes velocity features, and detects named movement segments using streaming Dynamic Time Warping.

```
BVH file
   │
   ▼  [Extract]
Joint tree (per-frame pose)
   │
   ▼  [Transform: Normalise]
Origin-anchored, scale-normalised Joint tree
   │
   ▼  [Transform: Feature extraction]
Per-frame velocity vectors  (N_frames − 1) × (J joints × 7 features)
   │
   ▼  [Transform: Segment + classify]
Labelled segments  [(begin, end, label), …]
   │
   ▼  [Load — planned]
CSV / socket / IPC stream
```

---

## Module Reference

### [src/kinematics.py](src/kinematics.py)

**Purpose:** Core data structures for a single-frame skeleton pose.

#### `Joint`

A node in an n-ary tree. Every joint stores:

| Attribute | Type | Meaning |
|-----------|------|---------|
| `_offset` | `(3,) ndarray` | Local translation from parent in parent space |
| `_quat` | `(4,) ndarray` | Local rotation as `(x, y, z, w)` quaternion |
| `_parent` | `Joint \| None` | Parent node; `None` for the root |
| `_children` | `deque[Joint]` | Ordered list of child joints |
| `_id` | `str` | Unique string identifier (`"torso_1"`, `"head"`, …) |
| `_id_hash` | `dict[str, Joint]` | Root-owned map for O(1) lookup by id |

Key methods:

- `append(id)` — create and wire a child joint, returns it
- `__getitem__(key)` — find any joint in the tree by id in O(1) via the root hash

Writing `offset` or `quat` automatically invalidates the cached transforms for the joint **and all its descendants** via `HomoTransform._invalidate()`.

#### `HomoTransform`

Owned 1-to-1 by a `Joint`. Exposes two lazily-computed, cached 4×4 homogeneous transformation matrices:

| Property | Meaning |
|----------|---------|
| `local` | Parent-space → joint-space: `[R(quat) \| offset]` |
| `root` | World-space → joint-space: accumulated product from root down |

Derived accessors:
- `position` — world-space `(3,)` position (`root[:3, 3]`)
- `rotation` — world-space `(4,)` quaternion

Cache invalidation propagates downward through the tree so that no stale transforms are ever used.

---

### [src/model.py](src/model.py)

**Purpose:** Parse a BVH file (via `bvhio`) and convert each frame into a `Joint` tree.

#### `convertBvhToHierarchy(bvh, frame)`

Walks the `bvhio` node tree and recursively builds a `Joint` tree for one frame:

1. Creates the root `Joint`
2. Sets its offset from `bvh.Root.Offset` **plus** the keyframe `Position` channel (the root can translate in BVH)
3. Sets its quaternion from the keyframe `Rotation` channel
4. Recurses into `bvh.Root.Children`, appending child joints using their `Offset` (rest-pose bone vector) and keyframe `Rotation`

This gives one fully-populated `Joint` tree per frame.

#### `load_model(path)`

Returns a closure that yields successive frames as `Joint` trees. Parses the BVH file once up-front (the expensive step: `bvhio` converts all Euler keyframes to quaternions). Prefer `MotionStream` for any use that needs velocity or multiple replays.

---

### [src/motion.py](src/motion.py)

**Purpose:** Normalised, velocity-annotated, resettable motion stream.

#### BVH cache

```python
@functools.lru_cache(maxsize=8)
def _parse_bvh(path): ...
```

`bvhio.readAsBvh()` is the dominant cost (~4 s for 1800 frames). The LRU cache ensures that multiple `MotionStream` instances pointing to the same file share one parse.

#### `MotionStream`

Constructor pipeline for each BVH file:

1. Parse BVH (cached)
2. For every frame: call `convertBvhToHierarchy` → `_normalize`
3. Call `_batch_velocities()` once to pre-compute all velocity frames

**`_normalize(root)`**

Applies three normalisations in-place on the `Joint` tree:

| Step | Operation |
|------|-----------|
| Translation | `root.offset = [0, 0, 0]` (anchor to world origin) |
| Rotation | `root.quat = [0, 0, 0, 1]` (identity, removes facing direction) |
| Scale | Multiply every joint's offset by `target_spine_length / spine_length` |

The spine length is measured once per stream (cached in `_spine_len`) along the chain `root → torso_1 → … → torso_7 → neck_1`. This makes all recordings body-size–invariant.

**`_batch_velocities()`**

Computes world-space positions and quaternions for all frames without any recursive Python calls:

1. Walk joints in **DFS topological order** so every parent index is strictly less than every child index
2. Extract `(N, J, 3)` offsets and `(N, J, 4)` quats into dense numpy arrays
3. Propagate world transforms top-down using a vectorised batch quaternion-rotation (`_bquat_to_matrix`, `np.einsum`)
4. Linear velocity: `lin[n] = world_pos[n+1] - world_pos[n]`  → `(N-1, J, 3)`
5. Angular velocity: `ang[n] = q_prev⁻¹ ⊗ q_curr`  → `(N-1, J, 4)` (relative quaternion, geodesic delta)

Helper functions (`_bquat_to_matrix`, `_bquat_mul`, `_bquat_conj`) are pure numpy — no scipy in the hot path.

**Iterator interface**

```python
for frame, vel in stream:
    # frame : Joint tree for this frame
    # vel   : dict[joint_id → {"linear": (3,), "angular": (4,)}]
    #         None for the first frame (no predecessor)
```

**`as_array(joint_ids=None)`**

Flattens velocities to a `(N-1, J*7)` numpy array: `[lin_x, lin_y, lin_z, ang_x, ang_y, ang_z, ang_w]` per joint. Used by the offline DTW.

---

### [src/dtw.py](src/dtw.py)

**Purpose:** Quaternion-aware Dynamic Time Warping for motion streams.

#### Distance metric

For every frame pair `(i, j)`, the per-joint costs are summed:

```
cost[i, j] = lin_weight × Σ_joints  ‖Δpos_a[i] − Δpos_b[j]‖₂
           + ang_weight × Σ_joints  arccos(|q_a[i] · q_b[j]|)
```

The angular term is the **geodesic distance on S³** — the shortest arc on the unit hypersphere of quaternions. `|·|` handles the double-cover (q and −q represent the same rotation but differ in Euclidean distance).

#### `cost_matrix(stream_a, stream_b)`

Returns the `(N, M)` pairwise cost matrix. Each row `i` is computed via `_lin_cost_row` and `_ang_cost_row`, which are vectorised over all M reference frames with `np.linalg.norm` and `np.einsum`.

#### `dtw(stream_a, stream_b)` — offline mode

1. Build the cost matrix
2. Run `_dp(C)` — anti-diagonal wavefront DP (see below)
3. `_backtrack(D)` — trace the optimal warping path
4. Return `total_cost / path_length` (normalised distance) and the path

**Anti-diagonal DP (`_dp`)**

All cells on diagonal `d = i + j` are independent and are filled in one numpy operation instead of a Python loop, reducing `O(N × M)` Python iterations to `O(N + M)` numpy calls:

```python
for d in range(2, N + M + 1):
    i = np.arange(max(1, d - M), min(N, d - 1) + 1)
    j = d - i
    D[i, j] = C[i-1, j-1] + np.minimum(np.minimum(D[i-1, j], D[i, j-1]), D[i-1, j-1])
```

#### `LiveDTW` — online/streaming mode

Maintains a single DP column `(N+1,)` and updates it in **O(N)** per incoming live frame. This is the subsequence-DTW initialisation: `col[0] = 0` for every new live frame `j`, which means `D[N, j]` tracks the best match of the entire reference against **any contiguous window of live frames ending at j**.

Key properties:

| Property | Meaning |
|----------|---------|
| `subseq_distance` | `D[N, j] / N` — match cost per reference frame; low when matched |
| `distance` | `D[N, j] / j` — normalised by live frames seen |
| `increment` | Marginal cost of the last live frame, clipped to `[0, ∞)` |
| `cost` | Raw accumulated `D[N, j]` |

Optional **Sakoe-Chiba band** (`band` parameter): cells with `|i − j| > band` are kept at `inf`, preventing the path from compressing the match into fewer than `N − band` frames.

`reset()` clears the DP column and frame counter without reloading the reference — supports repeated detection on the same stream.

---

### [src/segmentation.py](src/segmentation.py)

**Purpose:** Segment a motion stream into labelled gait events.

#### `GaitDetector`

**Calibration (`calibrate(stream)`)**

Runs the stream once to collect raw `subseq_distance` values from each `LiveDTW` probe, discarding the first `warmup_frames` values (default: `max(reference_N) + 30`) to let the DP accumulator settle.

Then for each gait:

1. **Auto-detect EMA alpha**: sweep 40 candidate values `α ∈ [0.01, 0.5]`, apply EMA to the calibration signal, compute Otsu's between-class variance, keep the α that maximises it.
2. **Otsu threshold**: apply the chosen EMA to the calibration signal, find the threshold that maximises between-class variance on the smoothed distribution.

**Detection (`detect(stream)`)**

Real-time two-state machine: **IDLE** and **ACTIVE**.

For every incoming frame:

1. Feed velocity to all `LiveDTW` probes
2. After warmup, apply per-gait EMA to `subseq_distance`
3. Normalise by calibrated threshold → score per gait
4. Winner = gait with minimum score; check if `score ≥ 1.0` (when `detect_high=True`)
5. State transitions:
   - `IDLE → ACTIVE`: triggered, record `detection_begin`
   - `ACTIVE → IDLE`: un-triggered, emit `(begin, end, label)`, reset all DTW probes and EMA states
   - Label change while `ACTIVE`: close current segment, open new one

**Otsu helpers**

`_otsu_threshold(values)` and `_otsu_between_class_var(values)` operate on a 256-bin histogram of the normalised distance signal. They find the threshold that maximises the weighted between-class variance `w₀ × w₁ × (μ₀ − μ₁)²`.

---

### [src/pyplot.py](src/pyplot.py)

**Purpose:** 3D skeleton animation using Plotly.

#### Coordinate remapping

BVH/kinematics convention: `X = right, Y = up, Z = forward`.
Plotly display mapping:

| BVH axis | Plot axis | Note |
|----------|-----------|------|
| `Y` | `Z` (vertical) | |
| `X` | `Y` (inverted) | "right" goes into the screen |
| `Z` | `X` | |

#### `plot_skeleton(frames)`

1. `extract_joints_and_bones` — DFS walk on frame 0 to get joint list and parent–child bone pairs
2. `extract_positions_optimized` — walks each frame's joint tree, accumulates HTM products, stores world positions in `(N_frames, J, 3)`
3. Builds one `go.Frame` per animation frame with:
   - `go.Scatter3d` markers for joints (red dots)
   - `go.Scatter3d` lines for bones (black lines, `None`-separated to draw all in one trace)
4. Adds Play/Pause buttons and a frame slider to the Plotly layout

#### `save_video(fig, path, fps, step)`

Renders each Plotly frame via matplotlib (Agg backend) and pipes PNG bytes to `ffmpeg` stdin, producing an MP4 or GIF. Requires `ffmpeg` on `PATH`.

---

### [src/etl.ipynb](src/etl.ipynb)

The main entry point. Sequential cells:

| Cell | What it does |
|------|--------------|
| Imports | Loads all modules plus Plotly |
| Load streams | Creates `MotionStream` objects for 5 reference recordings (`ref1`–`ref5`) and one target `stream` |
| Plot skeleton | Calls `plot.plot_skeleton(frames)` to display a 3D animated skeleton of the target stream |
| Calibrate & detect | Creates a `GaitDetector` with one gait (`"Picking up object"` → `ref1`), calibrates on `stream`, runs detection; prints threshold and found segments |
| DTW distance plot | Re-runs both probes frame-by-frame, collects raw and EMA-smoothed distances, plots them as a two-panel Plotly figure with threshold lines and green detection bands |
| Profiling (commented) | `cProfile` block for benchmarking the pipeline |

---

## Data Flow — End to End

```
MCPM_*.bvh
    │
    │  bvhio.readAsBvh()          [cached by path]
    ▼
bvhio.BvhContainer
    │
    │  convertBvhToHierarchy()    [per frame]
    ▼
Joint tree (one per frame, raw pose)
    │
    │  MotionStream._normalize()  [per frame, in-place]
    ▼
Joint tree (origin-anchored, spine-normalised)
    │
    │  _batch_velocities()        [once, full numpy batch]
    ▼
lin: (N-1, J, 3)   ang: (N-1, J, 4)   velocities: list[dict]
    │
    │  LiveDTW.update()           [O(N_ref) per live frame]
    ▼
subseq_distance: float   [per frame, per gait]
    │
    │  EMA smoothing + Otsu threshold
    ▼
IDLE / ACTIVE state machine
    │
    ▼
[(begin, end, label), …]
```