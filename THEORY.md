# Dance Notation ETL — Theory

This document explains the mathematical and conceptual foundations behind every design choice in the pipeline. References to specific code files are included where relevant.

---

## 1. Skeleton Representation: Joint Trees and HTMs

### 1.1 Why a tree?

The human body is a **kinematic chain**: the position of the hand depends on the elbow, which depends on the shoulder, which depends on the spine. A tree structure captures this dependency exactly — each joint's world-space pose is determined by chaining transformations from the root outward.

### 1.2 Homogeneous Transformation Matrices (HTM)

A single 4×4 matrix encodes both rotation **R** (3×3) and translation **t** (3×1) in one algebraic object:

```
T = [ R | t ]
    [ 0 | 1 ]
```

Composing two transforms (parent → child) is a matrix multiplication:

```
T_world = T_parent × T_local
```

This is exactly what `HomoTransform.root` computes by walking up the ancestor chain. The key advantage: world position of any joint is simply `T_world[:3, 3]`, and world rotation is `R.from_matrix(T_world[:3, :3]).as_quat()`.

### 1.3 Lazy caching and cache invalidation

Computing the full chain product on every access would be O(depth) per joint per frame. The `HomoTransform` class caches both `local` and `root` matrices and marks them dirty whenever `offset` or `quat` is written. Dirtying propagates **downward** through descendants (not upward) because a change to a parent invalidates all of its subtree, but never its ancestors. This makes bulk updates (e.g., scaling every joint during normalisation) correct without manual cache management.

---

## 2. Quaternion Rotation

### 2.1 Why not Euler angles?

Euler angles (roll, pitch, yaw) decompose rotation into three sequential rotations around fixed axes. They suffer from two problems:

1. **Gimbal lock**: when two rotation axes align, the representation loses a degree of freedom, making interpolation undefined.
2. **Order-dependence**: the result depends on which axis is rotated first (ZYX, XYZ, etc.).

Quaternions are a 4-dimensional number system `q = (x, y, z, w)` where the rotation axis is `n = (x, y, z)/sin(θ/2)` and the rotation angle is encoded in `w = cos(θ/2)`. They avoid gimbal lock entirely and compose cleanly by multiplication.

### 2.2 Quaternion operations used in the pipeline

**Rotation matrix from quaternion** (`_bquat_to_matrix`):

```
R = [ 1−2(y²+z²)    2(xy−zw)     2(xz+yw)  ]
    [ 2(xy+zw)      1−2(x²+z²)   2(yz−xw)  ]
    [ 2(xz−yw)      2(yz+xw)     1−2(x²+y²) ]
```

This is used in the world-propagation step of `_batch_velocities` to rotate child offsets into parent space: `child_world_pos = R_parent @ child_offset + parent_world_pos`.

**Hamilton product** (`_bquat_mul`): composition of two rotations `q₁ ⊗ q₂`. Used to accumulate world quaternions and to compute angular velocity (relative rotation between frames).

**Conjugate / inverse** (`_bquat_conj`): for a unit quaternion `q = (x,y,z,w)`, the conjugate `q* = (−x,−y,−z,w)` is also its inverse. Angular velocity is `Δq = q_prev⁻¹ ⊗ q_curr`.

### 2.3 The double-cover problem

Every 3D rotation is represented by **two** unit quaternions: `q` and `−q`. This means `q₁ · q₂` (the 4D dot product) can be either positive or negative even when the rotations are close. The geodesic distance on S³ must account for this:

```
d_geodesic(q₁, q₂) = arccos(|q₁ · q₂|)
```

The absolute value `|·|` selects the shorter arc on the 3-sphere, ensuring the minimum of the two equivalent angles. This is the formula used in `_ang_cost_row` and in `LiveDTW.update`.

---

## 3. Skeleton Normalisation

Normalisation makes motion features **speaker-independent** (invariant to body size and starting pose).

### 3.1 Translation normalisation

The root joint is anchored to the world origin at every frame:

```python
root.offset = [0, 0, 0]
```

Without this, a performer who walks across the room would generate large linear velocity signals unrelated to the gesture being performed. By anchoring to the origin, only motion **relative to the body's own reference frame** contributes to the features.

### 3.2 Rotation normalisation

```python
root.quat = [0, 0, 0, 1]   # identity quaternion
```

This strips the facing direction of the performer. A dancer facing north and a dancer facing east performing the same gesture would otherwise produce different feature vectors.

### 3.3 Bone-length (scale) normalisation

All bone offsets are scaled by `target_spine_length / measured_spine_length` (default target = 1.0). The measured spine length is the sum of bone lengths along:

```
root → torso_1 → torso_2 → torso_3 → torso_4 → torso_5 → torso_6 → torso_7 → neck_1
```

**Anatomical basis.** The spine is chosen as the normalisation landmark because it is the most stable segment of the skeletal hierarchy. The human lumbar and thoracic spine has a consistent proportional relationship with total body height across adults; its length is not affected by limb position or joint angle, making it a reliable metric for inter-subject scale normalisation. (Refer: Gilad & Nissan, *The Anatomical Record*, 10.1002/ar.21426, for quantitative data on spinal segment lengths across subjects.)

The `_SPINE_CHAIN` list in `motion.py` maps directly to the mocopi suit's joint naming convention for the torso vertebral segments.

---

## 4. Velocity as a Motion Feature

Rather than using absolute joint positions as the classification feature, the pipeline uses **frame-to-frame velocity**:

```
linear[n]  = world_pos[n+1] − world_pos[n]      # (J, 3)
angular[n] = q_world[n]⁻¹ ⊗ q_world[n+1]       # (J, 4)
```

**Why velocity and not position?** Velocity is translation-invariant at the signal level — once the root is anchored to the origin each frame, linear velocity captures joint motion but is insensitive to where in space the motion occurs. More importantly, velocity captures the **dynamics** (speed and direction of movement), not just a snapshot of configuration. Two poses that look similar but are reached via different trajectories produce different velocity signals, giving the DTW more discriminative power.

The feature vector per frame is the concatenation `[lin_x, lin_y, lin_z, ang_x, ang_y, ang_z, ang_w]` for each joint, giving a `(J × 7)` dimensional feature per frame.

---

## 5. Dynamic Time Warping (DTW)

Reference: Świtoński, Josiński & Wojciechowski, *Multidimensional Systems and Signal Processing* 30:1437–1468 (2019), DOI 10.1007/s11045-018-0611-3.

### 5.1 The problem DTW solves

Two performances of the same movement rarely take exactly the same amount of time. A simple Euclidean distance between frame-aligned sequences will measure the timing difference as dissimilarity even when the choreography is identical. DTW finds the **optimal time-warping** — a monotonic mapping from one sequence's time axis to the other's — that minimises total alignment cost.

### 5.2 The cost matrix

For sequences A (length N) and B (length M), the cost matrix C is N×M:

```
C[i, j] = lin_weight × Σ_joints ‖Δpos_A[i] − Δpos_B[j]‖₂
         + ang_weight × Σ_joints arccos(|q_A[i] · q_B[j]|)
```

Each cell measures how similar one frame of A is to one frame of B, summed across all joints.

### 5.3 The DP recurrence

The accumulated cost matrix D is filled by:

```
D[0, 0] = 0
D[i, j] = C[i, j] + min(D[i−1, j],   # insertion
                         D[i, j−1],   # deletion
                         D[i−1, j−1]) # match
```

The optimal DTW distance is `D[N, M]`. The optimal alignment path is recovered by backtracking from `D[N, M]` to `D[0, 0]`, always stepping to the smallest predecessor.

### 5.4 Anti-diagonal wavefront optimisation

Cells on the same anti-diagonal `d = i + j` have no data dependency on each other and can be computed simultaneously. This replaces O(N × M) Python iterations with O(N + M) numpy vectorised operations — a large speedup for long sequences.

### 5.5 Subsequence DTW (LiveDTW)

For action detection in a continuous stream, the goal is not to align two complete sequences but to find **where** in the live stream a reference sequence best matches. Subsequence DTW modifies the initialisation:

```
D[0, j] = 0  for all j         (allow matching to start anywhere)
```

This means `D[N, j]` equals the minimum cost of aligning all N reference frames to any contiguous window of live frames ending at frame `j`. When `D[N, j] / N` is small, the recent live motion resembles the reference.

The online (streaming) algorithm keeps only one DP column of shape `(N+1,)` in memory and updates it in O(N) per incoming live frame, making it suitable for real-time use.

### 5.6 Sakoe-Chiba band

An optional constraint `|i − j| ≤ band` restricts the warping path to a band around the diagonal. This prevents pathological warpings where the entire reference is matched to a single live frame. Cells outside the band are kept at `inf`. The default in `LiveDTW` is `band = N` (unconstrained).

### 5.7 Distance normalisation

The raw DTW cost `D[N, M]` grows with both sequence length and number of joints. Two normalised variants are exposed:

- `subseq_distance = D[N, j] / N` — per reference frame; directly comparable across gaits with different reference lengths
- `distance = D[N, j] / j` — per live frame seen; comparable across different detection window sizes

The segmentation pipeline uses `subseq_distance` as the primary signal.

---

## 6. Signal Smoothing: Exponential Moving Average (EMA)

The raw `subseq_distance` signal is noisy because each individual frame comparison has high variance. An EMA with smoothing factor `α`:

```
EMA[t] = α × raw[t]  +  (1 − α) × EMA[t−1]
```

- **Small α** (e.g., 0.05): heavily smoothed, slow to react, loses sharp boundaries
- **Large α** (e.g., 0.5): closer to raw signal, more responsive, noisier

The pipeline **auto-detects** the optimal α for each gait during calibration by sweeping 40 candidate values and choosing the one that maximises the Otsu between-class variance on the smoothed calibration signal.

---

## 7. Automatic Threshold Selection: Otsu's Method

Otsu's method (Nobuyuki Otsu, 1979) finds the threshold that best separates a 1D signal into two classes by maximising the **between-class variance**:

```
σ²_B(t) = w₀(t) × w₁(t) × (μ₀(t) − μ₁(t))²
```

where `w₀`, `w₁` are the class probabilities (histogram weights) and `μ₀`, `μ₁` are the class means below and above threshold `t`. The optimal threshold maximises `σ²_B`.

In the context of this pipeline, the EMA-smoothed `subseq_distance` distribution is bimodal: **low values** when the performer is executing the target gesture, **high values** during other motion or rest. Otsu's method automatically finds the boundary between these two modes without requiring a hand-tuned threshold. The threshold is derived from the calibration recording rather than from a held-out distribution, so it adapts to recording conditions, performer, and environment.

---

## 8. Segmentation State Machine

The detector runs a two-state machine per incoming frame:

```
         triggered (score ≥ θ)
  ┌──────────────────────────────────────┐
  │                                      │
IDLE ──────────────────────────────► ACTIVE
  ▲                                      │
  │    un-triggered (score < θ)          │
  └──────────────────────────────────────┘
      emit (begin, end, label), reset
```

When `detect_high=True` (used in the notebook), the detector fires when the EMA-smoothed distance is **above** the threshold — meaning the reference motion is being observed. On every state exit, all `LiveDTW` probes and EMA states are reset so that the same gesture can be re-detected immediately after with a clean DP column.

---

## 9. Joint Selection and Discriminativeness

The pipeline currently uses all joints in the skeleton for DTW comparison. The referenced paper (Świtoński et al.) demonstrates that many joints carry redundant or uninformative signals: discriminative joints can be identified via **hill climbing** or **genetic algorithm** search over subsets. Restricting DTW to a selected subset reduces computational cost and can improve classification accuracy by removing noise.

The `joint_ids` parameter in `cost_matrix`, `dtw`, and `LiveDTW` is the hook for this: passing a curated list of joint ids selects only those joints' features. This is currently left to future work in the pipeline's roadmap.

---

## 10. Coordinate System and BVH Convention

The BVH (BioVision Hierarchy) file format stores rotations as Euler angles per keyframe. The `bvhio` library converts all keyframes to quaternions at load time. BVH uses the convention:

| Axis | Meaning |
|------|---------|
| X | right |
| Y | up |
| Z | forward (into screen) |

The visualiser (`pyplot.py`) remaps to Plotly's default display:
- BVH Y → plot Z (vertical)
- BVH X → plot Y (negated, so "right" faces into screen)
- BVH Z → plot X

This remapping is purely cosmetic and does not affect the feature vectors used by DTW.

---

## Summary of Design Choices

| Choice | Alternative | Reason |
|--------|-------------|--------|
| Quaternions for rotation | Euler angles | No gimbal lock; smooth geodesic interpolation |
| Velocity features | Raw positions | Translation-invariant; captures dynamics |
| Spine-length normalisation | Height or limb length | Spine is anatomically stable; unaffected by joint angle |
| Subsequence DTW | Full-sequence DTW | Finds match within continuous stream; no prior segmentation needed |
| Otsu threshold | Hand-tuned value | Adapts to recording conditions without labelled training data |
| EMA smoothing + auto α | Fixed low-pass filter | Adapts smoothing to signal's own bimodal structure |
| Anti-diagonal DP | Row-by-row Python loop | O(N+M) numpy calls vs O(N×M) Python iterations |
