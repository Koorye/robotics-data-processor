import numpy as np
from scipy.spatial.transform import Rotation as R


def quaternion_to_euler(quat, order='xyz', scalar_first=False):
    r = R.from_quat(quat, scalar=scalar_first)
    return r.as_euler(order, degrees=True).tolist()


def euler_to_quaternion(euler, order='xyz', scalar_first=False):
    r = R.from_euler(order, euler, degrees=True)
    return r.as_quat(scalar_first=scalar_first).tolist()


def matrix_to_euler(rot_matrix, order='xyz'):
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    r = R.from_matrix(rot_matrix)
    return r.as_euler(order, degrees=True).tolist()

def euler_to_matrix(euler, order='xyz'):
    r = R.from_euler(order, euler, degrees=True)
    return r.as_matrix().tolist()


def matrix6d_to_euler(matrix6d, order='xyz'):
    matrix6d = np.array(matrix6d).reshape(3, 2)
    z_axis = np.cross(matrix6d[:, 0], matrix6d[:, 1])
    rot_matrix = np.stack([matrix6d[:, 0], matrix6d[:, 1], z_axis], axis=-1)
    r = R.from_matrix(rot_matrix)
    return r.as_euler(order, degrees=True).tolist()


def euler_to_matrix6d(euler, order='xyz'):
    r = R.from_euler(order, euler, degrees=True)
    rot_matrix = r.as_matrix()
    return rot_matrix[:, :2].flatten().tolist()


class BaseOperator:
    def __init__(self):
        pass

    def __call__(self):
        pass


class AngleOperator:
    def __init__(
        self, 
        from_type='quaternion', 
        to_type='euler', 
        order='xyz', 
        scalar_first=False
    ):
        self.from_type = from_type
        self.to_type = to_type
        self.order = order
        self.scalar_first = scalar_first
    
    def __call__(self, angle):
        pass