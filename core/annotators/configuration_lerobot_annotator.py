import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LerobotAnnotatorConfig:
    repo_id: str
    video_backend: Optional[str] = None

    operators: List[dict] = field(default_factory=lambda: [
        {
            'type': 'position',
            'name': 'position_left',
            'window_size': 1,
            'state_key': 'observation.state',
            'xyz_range': (0, 3),
        }, {
            'type': 'position',
            'name': 'position_right',
            'window_size': 1,
            'state_key': 'observation.state',
            'xyz_range': (10, 13),
        }, {
            'type': 'angle',
            'name': 'angle_left',
            'window_size': 1,
            'state_key': 'observation.state',
            'rpy_range': (3, 9),
        }, {
            'type': 'angle',
            'name': 'angle_right',
            'window_size': 1,
            'state_key': 'observation.state',
            'rpy_range': (13, 19),
        }, {
            'type': 'gripper',
            'name': 'gripper_left',
            'window_size': 1,
            'state_key': 'observation.state',
            'gripper_indice': 9,
        }, {
            'type': 'gripper',
            'name': 'gripper_right',
            'window_size': 1,
            'state_key': 'observation.state',
            'gripper_indice': 19,
        }, {
            'type': 'position_rotation',
            'name': 'position_aligned_left',
            'window_size': 1,
            'position_key': 'position_left',
            'rotation_euler': (0, 0, 0.5 * math.pi),
        }, {
            'type': 'position_rotation',
            'name': 'position_aligned_right',
            'window_size': 1,
            'position_key': 'position_right',
            'rotation_euler': (0, 0, 0.5 * math.pi),
        }, {
            'type': 'movement',
            'name': 'movement_left',
            'window_size': 3,
            'position_key': 'position_aligned_left',
        }, {
            'type': 'movement',
            'name': 'movement_right',
            'window_size': 3,
            'position_key': 'position_aligned_right',
        }, {
            'type': 'velocity', 
            'name': 'velocity_left',
            'window_size': 1, 
            'movement_key': 'movement_left',
        }, {
            'type': 'velocity', 
            'name': 'velocity_right',
            'window_size': 1, 
            'movement_key': 'movement_right',
        }, {
            'type': 'acceleration',
            'name': 'acceleration_left',
            'window_size': 2,
            'vel_key': 'velocity_left',
        }, {
            'type': 'acceleration',
            'name': 'acceleration_right',
            'window_size': 2,
            'vel_key': 'velocity_right',
        }, {
            'type': 'gripper_movement',
            'name': 'gripper_movement_left',
            'window_size': 3,
            'gripper_key': 'gripper_left',
        }, {
            'type': 'gripper_movement',
            'name': 'gripper_movement_right',
            'window_size': 3,
            'gripper_key': 'gripper_right',
        }, {
            'type': 'gripper_summary',
            'name': 'gripper_summary_left',
            'gripper_key': 'gripper_left',
            'threshold': 500,
        }, {
            'type': 'gripper_summary',
            'name': 'gripper_summary_right',
            'gripper_key': 'gripper_right',
            'threshold': 500,
        }, {
            'type': 'movement_summary',
            'name': 'movement_summary_left',
            'movement_key': 'movement_left',
            'threshold': 1e-3,
        }, {
            'type': 'movement_summary',
            'name': 'movement_summary_right',
            'movement_key': 'movement_right',
            'threshold': 1e-3,
        }, {
            'type': 'gripper_movement_summary',
            'name': 'gripper_movement_summary_left',
            'gripper_movement_key': 'gripper_movement_left',
            'threshold': 10,
        }, {
            'type': 'gripper_movement_summary',
            'name': 'gripper_movement_summary_right',
            'gripper_movement_key': 'gripper_movement_right',
            'threshold': 10,
        }, {
            'type': 'velocity_summary',
            'name': 'velocity_summary_left',
            'velocity_key': 'velocity_left',
        }, {
            'type': 'velocity_summary',
            'name': 'velocity_summary_right',
            'velocity_key': 'velocity_right',
        }, {
            'type': 'acceleration_summary',
            'name': 'acceleration_summary_left',
            'acceleration_key': 'acceleration_left',
        }, {
            'type': 'acceleration_summary',
            'name': 'acceleration_summary_right',
            'acceleration_key': 'acceleration_right',
        }
    ])