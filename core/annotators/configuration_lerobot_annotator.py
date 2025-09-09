from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LerobotAnnotatorConfig:
    repo_id: str
    video_backend: Optional[str] = None

    operators: List[dict] = field(default_factory=lambda: [
        {
            'type': 'position',
            'name': 'pos_left',
            'window_size': 1,
            'state_key': 'observation.state',
            'xyz_range': (0, 3),
        },
        {
            'type': 'position',
            'name': 'pos_right',
            'window_size': 1,
            'state_key': 'observation.state',
            'xyz_range': (10, 13),
        },
        {
            'type': 'direction',
            'name': 'dir_left',
            'window_size': 1,
            'state_key': 'observation.state',
            'rpy_range': (3, 6),
        },
        {
            'type': 'direction',
            'name': 'dir_right',
            'window_size': 1,
            'state_key': 'observation.state',
            'rpy_range': (13, 16),
        },
        {
            'type': 'movement',
            'name': 'mov_left',
            'window_size': 10,
            'position_key': 'pos_left',
        }, {
            'type': 'movement',
            'name': 'mov_right',
            'window_size': 10,
            'position_key': 'pos_right',
        }, {
            'type': 'velocity', 
            'name': 'vel_left',
            'window_size': 1, 
            'movement_key': 'mov_left',
        }, {
            'type': 'velocity', 
            'name': 'vel_right',
            'window_size': 1, 
            'movement_key': 'mov_right',
        }, {
            'type': 'acceleration',
            'name': 'accel_left',
            'window_size': 5,
            'vel_key': 'vel_left',
        }, {
            'type': 'acceleration',
            'name': 'accel_right',
            'window_size': 5,
            'vel_key': 'vel_right',
        }, {
            'type': 'movement_summary',
            'name': 'mov_summary_left',
            'movement_key': 'mov_left',
        }, {
            'type': 'movement_summary',
            'name': 'mov_summary_right',
            'movement_key': 'mov_right',
        }, {
            'type': 'velocity_summary',
            'name': 'vel_summary_left',
            'vel_key': 'vel_left',
        }, {
            'type': 'velocity_summary',
            'name': 'vel_summary_right',
            'vel_key': 'vel_right',
        }, {
            'type': 'acceleration_summary',
            'name': 'accel_summary_left',
            'accel_key': 'accel_left',
        }, {
            'type': 'acceleration_summary',
            'name': 'accel_summary_right',
            'accel_key': 'accel_right',
        }
    ])