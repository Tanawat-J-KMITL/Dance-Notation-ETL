"""model.py — BVH file loading and conversion to the Joint tree format.

Provides two public functions:

- convertBvhToHierarchy(bvh, frame) — converts one frame of a parsed BVH
  container into a kinematics.Joint tree.
- load_model(path) — parses a BVH file and returns a closure that yields one
  Joint tree per call, stepping through frames in order.

Note: bvhio.readAsBvh() converts all Euler keyframes to quaternions up-front,
which is the dominant cost (~4 s for a 1800-frame file).  The per-frame Joint
tree construction done by the closure is comparatively cheap.  If the same file
is needed by multiple consumers, prefer motion.MotionStream which caches the
parsed BVH object.
"""

import bvhio
import numpy as np
import kinematics as kn
from collections.abc import Callable


def _glm_vec3(v) -> np.ndarray:
    """Convert a pyglm vec3 (with .x/.y/.z) to a numpy (3,) array."""
    return np.array([v.x, v.y, v.z])


def _glm_quat(q) -> np.ndarray:
    """Convert a pyglm quat (w, x, y, z) to scipy/numpy (x, y, z, w) order."""
    return np.array([q.x, q.y, q.z, q.w])


def _build_tree(bvh_joint: bvhio.BvhJoint, parent: kn.Joint, frame: int):
    """
    Recursively append bvh_joint and its descendants to parent for the
    given frame.
    """
    j = parent.append(bvh_joint.Name)
    j.offset = _glm_vec3(bvh_joint.Offset)
    j.quat   = _glm_quat(bvh_joint.Keyframes[frame].Rotation)
    for child in bvh_joint.Children:
        _build_tree(child, j, frame)


def convertBvhToHierarchy(bvh: bvhio.BvhContainer, frame: int = 0) -> kn.Joint:
    """Build a Joint tree from one frame of a parsed BVH container.

    The root joint's world position is taken from the keyframe Position channel
    (not from Offset, which is the rest-pose offset in BVH convention).
    All other joints use their Offset as the local bone vector and the keyframe
    Rotation for the local quaternion.

    Args:
        bvh:   parsed BVH container from bvhio.readAsBvh()
        frame: zero-based frame index (default 0)

    Returns:
        Root Joint with the full skeleton tree attached.
    """
    root        = kn.Joint()
    root.offset = _glm_vec3(bvh.Root.Offset)
    root.quat   = _glm_quat(bvh.Root.Keyframes[frame].Rotation)
    root.offset += _glm_vec3(bvh.Root.Keyframes[frame].Position)
    for child in bvh.Root.Children:
        _build_tree(child, root, frame)
    return root


def load_model(bvh_model: str) -> Callable:
    """Parse a BVH file and return a frame-iterator closure.

    Each call to the returned closure yields the next frame as a Joint tree.
    Returns 0 (falsy) once all frames have been consumed.

    Args:
        bvh_model: path to the .bvh file

    Returns:
        A zero-argument callable; call repeatedly to get successive frames.

    Example::

        stream = load_model("recording.bvh")
        frame = stream()
        while frame:
            process(frame)
            frame = stream()
    """
    bvh         = bvhio.readAsBvh(bvh_model)
    frame_count = bvh.FrameCount
    index       = 0

    def closure():
        nonlocal index
        if index >= frame_count:
            return 0
        frame  = convertBvhToHierarchy(bvh, index)
        index += 1
        return frame

    return closure
