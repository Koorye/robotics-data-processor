import imageio
import os
import numpy as np
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .configuration_data_convertor import DataConvertorConfig


def load_image(path):
    if isinstance(path, np.ndarray):
        return path
    return imageio.v3.imread(path)


def get_lerobot_default_root():
    return os.path.expanduser('~/.cache/huggingface/lerobot')


class BaseDataConvertor(ABC):
    def __init__(self, config: DataConvertorConfig):
        self.config = config

        if self.config.overwrite:
            self._check_overwrite()
        
        self.create_dataset()
    
    @abstractmethod
    def _yield_episodes(self) -> List[Dict[str, Any]]:
        """
        Return a list of frames for the given episode.
        Each frame is a dictionary with keys corresponding to data fields.
        """
        pass

    def _check_overwrite(self):
        if self.config.data_root is not None:
            data_root = self.config.data_root
            if os.path.exists(data_root):
                print(f'Overwriting data root: {data_root}? (y/n)', end=' ')
                if input().strip().lower() != 'y':
                    print('Exiting without overwriting.')
                    return
                shutil.rmtree(data_root, ignore_errors=True)
        else:
            data_root = get_lerobot_default_root()
            data_root = os.path.join(data_root, self.config.repo_id)
            if os.path.exists(data_root):
                print(f'Overwriting data root: {data_root}? (y/n)', end=' ')
                if input().strip().lower() != 'y':
                    print('Exiting without overwriting.')
                    return
                shutil.rmtree(data_root, ignore_errors=True)

    def create_dataset(self):
        if self.config.check_only:
            print('Check only mode, skipping dataset creation.')
            return
        
        image_config = {
            'dtype': self.config.image_dtype,
            'shape': (self.config.image_height, self.config.image_width, 3),
            'names': ['height', 'width', 'channel'],
        }
        features = {image_name: image_config for image_name in self.config.image_names}

        depth_config = {
            'dtype': self.config.depth_dtype,
            'shape': (self.config.depth_height, self.config.depth_width),
            'name': ['height', 'width'],
        }
        for depth_name in self.config.depth_names:
            features[depth_name] = depth_config

        features[self.config.action_key] = {
            'dtype': self.config.action_dtype,
            'shape': (self.config.action_len,),
            'names': self.action_names,
        }

        features[self.config.state_key] = {
            'dtype': self.config.state_dtype,
            'shape': (self.config.action_len,),
            'names': self.state_names,
        }
        
        if self.config.data_root is not None:
            self.config.data_root = os.path.join(self.config.data_root, self.config.repo_id)
        
        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            root=self.config.data_root,
            fps=self.config.fps,
            video_backend=self.config.video_backend,
            features=features,
        )
    
    def process_data(self):
        for episode in self._yield_episodes():
            if self.config.check_only:
                print('Check only mode, skipping data processing.')
                continue
            for frame in episode:
                task = frame.get('task', self.config.default_task)
                self.dataset.add_frame(frame, task=task)
            self.dataset.save_episode()