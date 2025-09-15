import json
import numpy as np
import os
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .base_data_convertor import BaseDataConvertor
from .configuration_data_convertor import LeRobotDataConvertorConfig


def _get_default_lerobot_root():
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'lerobot')


def _extract_joint(state):
    return np.concatenate([
        state[..., 0:8],   # left joint + gripper
        state[..., 17:25], # right joint + gripper
    ])


def _generate_task(frame, annotation):
    task = 'task: {}\n'.format(frame['task'])
    task += 'description: {}\n'.format(annotation['scene_description'])
    task += 'subtask: {}\n'.format(annotation['subtask'])
    task += 'movement left: {} right: {}\n'.format(
        annotation['movement_summary_left'],
        annotation['movement_summary_right']
    )
    # task += 'velocity left: {}, velocity right: {}\n'.format(
    #     annotation['velocity_summary_left'],
    #     annotation['velocity_summary_right']
    # )
    return task.strip().replace('the ', '').replace('.', '').replace('a ', '').replace('is ', '').replace('are ', '')


def _parse_episode(repo_id, episode):
    episode_index = episode[0]['episode_index']
    annotation_path = os.path.join(_get_default_lerobot_root(), repo_id, 'annotations', f'episode_{episode_index:06d}.json')

    with open(annotation_path, 'r') as f:
        annotations = json.load(f)

    assert len(annotations) == len(episode)
    new_episodes = []
    new_episode = []
    prev_task = None

    for frame, annotation in zip(episode, annotations):
        task = _generate_task(frame, annotation)
        new_frame = {
            'observation.images.cam_high': frame['observation.images.cam_high'].numpy(),
            'observation.images.cam_left_wrist': frame['observation.images.cam_left_wrist'].numpy(),
            'observation.images.cam_right_wrist': frame['observation.images.cam_right_wrist'].numpy(),
            'observation.state': _extract_joint(frame['observation.state'].numpy()),
            'action': _extract_joint(frame['action'].numpy()),
            'task': task,
        }
        if prev_task is None:
            prev_task = task
        elif prev_task != task:
            new_episodes.append(new_episode)
            new_episode = []
            prev_task = task
        
        new_episode.append(new_frame)

    if len(new_episode) > 0:
        new_episodes.append(new_episode)
    
    return new_episodes


class LeRobotDataConvertor(BaseDataConvertor):
    def __init__(self, config: LeRobotDataConvertorConfig):
        super().__init__(config)
        self.config = config
    
    def _yield_episodes(self):
        episode = []

        dataset = LeRobotDataset(self.config.source_repo_id, video_backend=self.config.source_video_backend)
        prev_episode_index = None
        for sample in dataset:
            episode_index = sample['episode_index']
            if prev_episode_index is None:
                prev_episode_index = episode_index

            if episode_index != prev_episode_index:
                new_episodes = _parse_episode(self.config.source_repo_id, episode)
                for new_episode in new_episodes:
                    yield new_episode
                episode = []
                prev_episode_index = episode_index
            
            episode.append(sample)
        
        if len(episode) > 0:
            new_episodes = _parse_episode(self.config.source_repo_id, episode)
            for new_episode in new_episodes:
                yield new_episode