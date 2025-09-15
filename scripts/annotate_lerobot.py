import math

import sys
sys.path.append('.')

from core.annotators.configuration_lerobot_annotator import LerobotAnnotatorConfig
from core.annotators.lerobot_annotator import LerobotAnnotator


def main(config: LerobotAnnotatorConfig):
    annotator = LerobotAnnotator(config)
    annotator.annotate()


if __name__ == '__main__':
    config = LerobotAnnotatorConfig(
        repo_id='realman/eval_v1_anno',
        operators=[
            {
                'type': 'position',
                'name': 'position_left',
                'window_size': 1,
                'state_key': 'observation.state',
                'xyz_range': (8, 11),
            }, {
                'type': 'position',
                'name': 'position_right',
                'window_size': 1,
                'state_key': 'observation.state',
                'xyz_range': (25, 28),
            }, {
                'type': 'angle',
                'name': 'angle_left',
                'window_size': 1,
                'state_key': 'observation.state',
                'rpy_range': (11, 17),
            }, {
                'type': 'angle',
                'name': 'angle_right',
                'window_size': 1,
                'state_key': 'observation.state',
                'rpy_range': (28, 34),
            }, {
                'type': 'gripper',
                'name': 'gripper_left',
                'window_size': 1,
                'state_key': 'observation.state',
                'gripper_indice': 7,
            }, {
                'type': 'gripper',
                'name': 'gripper_right',
                'window_size': 1,
                'state_key': 'observation.state',
                'gripper_indice': 24,
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
                'threshold': 600,
            }, {
                'type': 'gripper_summary',
                'name': 'gripper_summary_right',
                'gripper_key': 'gripper_right',
                'threshold': 600,
            }, {
                'type': 'movement_summary',
                'name': 'movement_summary_left',
                'movement_key': 'movement_left',
                'threshold': 2e-3,
            }, {
                'type': 'movement_summary',
                'name': 'movement_summary_right',
                'movement_key': 'movement_right',
                'threshold': 2e-3,
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
                'slow_threshold': 2e-3,
                'fast_threshold': 5e-3,
            }, {
                'type': 'velocity_summary',
                'name': 'velocity_summary_right',
                'velocity_key': 'velocity_right',
                'slow_threshold': 2e-3,
                'fast_threshold': 5e-3,
            }, {
                'type': 'acceleration_summary',
                'name': 'acceleration_summary_left',
                'acceleration_key': 'acceleration_left',
                'threshold': 2e-3,
            }, {
                'type': 'acceleration_summary',
                'name': 'acceleration_summary_right',
                'acceleration_key': 'acceleration_right',
                'threshold': 2e-3,
            }, {
                'type': 'scene_description',
                'name': 'scene_description',
                'scene_description_dir': 'data/annotations/scenes',
                'task_meta_path': 'lerobot/realman/eval_v1_anno/meta/tasks.jsonl',

            }, {
                'type': 'subtask',
                'name': 'subtask',
                'subtask_annotation_path': 'data/annotations/project-563-at-2025-09-11-12-06-18445096.json',
            }
        ]
    )
    main(config)