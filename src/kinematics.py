# Imports
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R


class HomoTransform:
    """ Computes homogeneous transforms for the joint it belongs to. """

    def __init__(self, joint: "Joint"):
        self._joint  = joint
        self._matrix = None  # cache of the local transform

    def _invalidate(self):
        """ Recursively invalidate cache """
        self._matrix = None
        for child in self._joint._children:
            child.transform._invalidate()

    @property
    def local(self) -> np.ndarray:
        """ Local transform (parent -> joint) """
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
        """ Transform from root to joint """
        j = self._joint
        if j._parent is None:
            return self.local
        return j._parent.transform.root @ self.local

    @property
    def position(self) -> np.ndarray:
        """ Absolute position of the joint """
        return self.root[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        """ Absolute quaternion rotation of the joint """
        return R.from_matrix(self.root[:3, :3]).as_quat()

    # Methods
    def translate(self, offset: np.ndarray) -> np.ndarray:
        """ Applies a local offset on HTM """
        T = self.local
        R_matrix = T[:3, :3]
        t_vector = T[:3, 3]

        # Rotate the offset vector into the joint's frame,
        # then add the current translation
        return R_matrix @ offset + t_vector

    def rotate(self, quat: np.ndarray) -> np.ndarray:
        """ Rotate from HTM """
        # Extract current rotation directly from the HTM matrix
        q1 = R.from_matrix(self.local[:3, :3])
        q2 = R.from_quat(quat)
        return (q2 * q1).as_quat()


class Joint:
    """ Stores a joint's coordinates and its place in the skeleton tree """

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
