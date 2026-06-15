"""kinematics.py — Skeleton joint tree and homogeneous-transform helpers.

Core data structures for representing a single-frame pose as a tree of Joint
nodes.  Each Joint stores its local offset and quaternion rotation; world-space
transforms are computed on demand via the HomoTransform helper and cached until
any ancestor changes.

Typical usage
-------------
    root = Joint()                # root node (id = "root")
    hip  = root.append("hip")
    hip.offset = np.array([0, 1, 0])
    hip.quat   = np.array([0, 0, 0, 1])   # identity

    world_pos = hip.transform.position    # (3,) in world space
    world_rot = hip.transform.rotation    # (4,) quaternion in world space
"""

import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R


class HomoTransform:
    """4x4 homogeneous-transform helper owned by a single Joint.

    Lazily computes and caches two transforms:
    - local  (4x4): parent-space → joint-space  [R | offset]
    - root   (4x4): world-space  → joint-space  (chain from root to this joint)

    Both caches are invalidated recursively whenever the joint's offset or
    quaternion is written, so callers never need to manage the cache manually.
    """

    def __init__(self, joint: "Joint"):
        self._joint        = joint
        self._matrix       = None  # cache: local transform
        self._world_matrix = None  # cache: world (root→joint) transform

    def _invalidate(self):
        """
        Recursively clear local and world caches for this joint and
        all descendants.
        """
        self._matrix       = None
        self._world_matrix = None
        for child in self._joint._children:
            child.transform._invalidate()

    @property
    def local(self) -> np.ndarray:
        """
        4x4 HTM from parent space to this joint's space: [R(quat) | offset].
        """
        if self._matrix is not None:
            return self._matrix
        j = self._joint
        T = np.eye(4)
        T[:3, :3] = R.from_quat(j._quat).as_matrix()
        T[:3,  3] = j._offset
        self._matrix = T
        return T

    @property
    def root(self) -> np.ndarray:
        """
        4x4 HTM from world space to this joint's space (full chain from root).

        Computed as parent.root @ self.local and cached; invalidated whenever
        any ancestor's offset or quat changes.
        """
        if self._world_matrix is not None:
            return self._world_matrix
        j = self._joint
        if j._parent is None:
            T = self.local
        else:
            T = j._parent.transform.root @ self.local
        self._world_matrix = T
        return T

    @property
    def position(self) -> np.ndarray:
        """World-space position of the joint as (3,) array."""
        return self.root[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        """
        World-space rotation of the joint as (4,) quaternion (x, y, z, w).
        """
        return R.from_matrix(self.root[:3, :3]).as_quat()

    def translate(self, offset: np.ndarray) -> np.ndarray:
        """
        Return the world position of a point at local offset from this joint.
        """
        T = self.local
        return T[:3, :3] @ offset + T[:3, 3]

    def rotate(self, quat: np.ndarray) -> np.ndarray:
        """
        Return the composed quaternion of quat applied after this joint's
        local rotation.
        """
        q1 = R.from_matrix(self.local[:3, :3])
        q2 = R.from_quat(quat)
        return (q2 * q1).as_quat()


class Joint:
    """
    A node in a skeleton tree, storing a local offset and quaternion rotation.

    Each Joint is identified by a string id, knows its parent and children, and
    owns a HomoTransform that computes world-space position and rotation on
    demand.

    The root joint (parent=None) always has id="root" and acts as the world
    origin. All other joints are created via parent.append(id) which wires the
    parent/child relationship and registers the joint in the root's id hash
    map for O(1) lookup.

    Coordinates
    -----------
    offset : (3,) ndarray — local translation from parent
    quat   : (4,) ndarray — local rotation as (x, y, z, w) quaternion

    Writing either property invalidates the transform cache of this joint and
    all of its descendants.
    """

    HomoTransform = HomoTransform

    def __init__(
        self,
        parent: "Joint" = None,
        id: str = None
    ):
        # Coordinates
        self._offset    = np.array([.0, .0, .0])        # (x, y, z)
        self._quat      = np.array([.0, .0, .0, 1.0])   # (x, y, z, w)
        # Tree
        self._root      = None
        self._parent    = parent
        self._children  = deque([])
        self._indx      = 0
        self._id        = id
        self._id_hash   = {"root": self}
        # Transform helper (member, owns its own cache)
        self._transform = Joint.HomoTransform(self)
        # init
        if self._parent is None:
            self._root = self
            self._id = "root"
        else:
            self._root = parent.root
            if self._id is None:
                # parent -> parent_1
                self._id = f"{
                    self._parent.id
                }_{
                    str(len(self._parent._children))
                }"

    def __getitem__(self, key: str) -> "Joint | None":
        """ Return Joint with same id as key, None if not found """
        return self._root._id_hash.get(key)

    # Read-only
    @property
    def root(self) -> "Joint":
        """ Root Joint (aka. world) """
        return self._root

    @property
    def parent(self) -> "Joint":
        """ Parent Joint (None if root) """
        return self._parent

    @property
    def children(self) -> deque:
        """ Parent Joint (None if root) """
        return self._children

    @property
    def id(self) -> str:
        """ ID of Joint """
        return self._id

    @property
    def transform(self) -> "Joint.HomoTransform":
        """ The joint's HomoTransform helper """
        return self._transform

    @transform.setter
    def transform(self, T: np.ndarray) -> None:
        """ The joint's transformation matrix """
        self._transform._matrix = T.copy()
        self._offset = T[:3, 3].copy()
        self._quat = R.from_matrix(T[:3, :3]).as_quat()

    # Coordinates (invalidate the transform cache on write)
    @property
    def offset(self) -> np.ndarray:
        """ Offset of Joint from parent """
        return self._offset

    @offset.setter
    def offset(self, new_value: np.ndarray) -> None:
        self._transform._invalidate()
        self._offset = new_value

    @property
    def quat(self) -> np.ndarray:
        """ Rotation of Joint """
        return self._quat

    @quat.setter
    def quat(self, new_value: np.ndarray) -> None:
        self._transform._invalidate()
        self._quat = new_value

    # Methods
    def append(self, id: str = None) -> "Joint":
        """ Appends a Joint node to the parent """
        new_joint = Joint(parent=self, id=id)
        self._root._id_hash[new_joint._id] = new_joint
        self._children.append(new_joint)
        return new_joint
