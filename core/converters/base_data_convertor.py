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
        self.dataset = None

        if self.config.overwrite:
            self._check_overwrite()
        
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

    def create_dataset(self, example_data: dict[str, np.ndarray]):
        image_dtype = 'video' if self.config.video_backend != 'none' else 'image'
        features = {}
        for key, value in example_data.items():
            if key.startswith(self.config.image_prefix):
                features[key] = {
                    'dtype': image_dtype,
                    'shape': value.shape,
                    'names': ['height', 'width', 'channel'],
                }
            elif key != 'task':
                features[key] = {
                    'dtype': str(value.dtype),
                    'shape': value.shape,
                    'names': [key],
                }

        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            root=self.config.data_root,
            fps=self.config.fps,
            use_videos=True if self.config.video_backend != 'none' else False,
            video_backend=self.config.video_backend,
            image_writer_processes=self.config.image_writer_processes,
            image_writer_threads=self.config.image_writer_threads,
            features=features,
        )
    
    def convert(self):
        for episode in self._yield_episodes():
            if self.config.check_only:
                print('Check only mode, skipping data processing.')
                continue
        
            if self.dataset is None:
                self.create_dataset(episode[0])
            
            for frame in episode:
                if 'task' in frame:
                    task = frame['task']
                    del frame['task']
                else:
                    task = self.config.default_task

                self.dataset.add_frame(frame, task=task)
            self.dataset.save_episode()