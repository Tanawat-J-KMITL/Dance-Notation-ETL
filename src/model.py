import bvhio
import numpy as np
import kinematics as kn
from collections.abc import Callable


def _glm_vec3(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z])


def _glm_quat(q) -> np.ndarray:
    # glm is (w,x,y,z) → scipy needs (x,y,z,w)
    return np.array([q.x, q.y, q.z, q.w])


def _build_tree(bvh_joint: bvhio.BvhJoint, parent: kn.Joint, frame: int):
    j = parent.append(bvh_joint.Name)
    j.offset = _glm_vec3(bvh_joint.Offset)
    j.quat   = _glm_quat(bvh_joint.Keyframes[frame].Rotation)
    for child in bvh_joint.Children:
        _build_tree(child, j, frame)


def convertBvhToHierarchy(bvh: bvhio.BvhContainer, frame: int = 0) -> kn.Joint:
    root        = kn.Joint()
    root.offset = _glm_vec3(bvh.Root.Offset)
    root.quat   = _glm_quat(bvh.Root.Keyframes[frame].Rotation)
    # root position lives in the keyframe, not the offset
    root.offset += _glm_vec3(bvh.Root.Keyframes[frame].Position)
    for child in bvh.Root.Children:
        _build_tree(child, root, frame)
    return root


# New idea: Node(N) = Transform(0) + Transform(N),
#           Diff = Node(1) - Node(2)
#                = (Transform(0) + Transform(1))
#                   - (Transform(0) + Transform(1))
#                = Transform(1) - Transform(2)
# Note: (-) here actually means: delta of HTM, which is T1^-1 * T2
# So, frames = ( Transform(0), [Node(1), Node(2), ..., Node(N)] )
# We can then get velocity as a vector from previos frame, and current


def load_model(
    bvh_model: str,
    frameMap: Callable[[kn.Joint], kn.Joint] = None
) -> tuple[list[kn.Joint], int]:
    bvh         = bvhio.readAsBvh(bvh_model)
    frame_count = bvh.FrameCount
    frames      = [
        convertBvhToHierarchy(bvh, f) for f in range(frame_count)
    ]
    if frameMap is not None:
        frames = [
            frameMap(frame) for frame in frames
        ]
    return frames
