# position in robot frame: (0, 0, 1.0)
# position in world frame: (0, 0.71, 0.71)

import math
from scipy.spatial.transform import Rotation as R

euler = (0, 45 / 180 * math.pi, 90 / 180 * math.pi)
rot = R.from_euler('xyz', euler)
print('euler:', euler)
print('as_matrix:\n', rot.as_matrix())

pos = (0, 0, 1)
pos_rotated = rot.apply(pos)
print('pos:', pos)
print('pos rotated:', pos_rotated)
