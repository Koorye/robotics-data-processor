import io
import h5py
import numpy as np
import os
from PIL import Image

from .base_data_convertor import BaseDataConvertor
from .configuration_data_convertor import HDF5DataConvertorConfig

_TASK_MAPPING = {
    'basket_towel': 'put the towel into in basket.',
    'box_put_peach': 'open the box and put the peach in it.',
    'close_drawer': 'put the peach in the drawer and close it.',
    'fold_towel': 'fold the towel.',
    'pass_bowl': 'pass the bowl from the left to the right.',
}


def decode_image(image_buffer):
    img = Image.open(io.BytesIO(image_buffer))
    return np.array(img)


def extract_joint(state):
    return np.concatenate([
        state[..., 50:57], # left joint
        state[..., 60:61], # left gripper
        state[..., 0:7],   # right joint
        state[..., 10:11], # right gripper
    ], axis=-1)


def extract_pose(state):
    return np.concatenate([
        state[..., 80:83], # left xyz
        state[..., 83:89], # left rpy
        state[..., 60:61], # left gripper
        state[..., 30:33], # right xyz
        state[..., 33:39], # right rpy
        state[..., 10:11], # right gripper
    ], axis=-1)


def extract_joint_and_pose(state):
    return np.concatenate([
        state[..., 50:57], # left joint [0, 7)
        state[..., 60:61], # left gripper [7, 8)
        state[..., 80:83], # left xyz [8, 11)
        state[..., 83:89], # left rpy [11, 17)
        state[..., 0:7],   # right joint [17, 24)
        state[..., 10:11], # right gripper [24, 25)
        state[..., 30:33], # right xyz [25, 28)
        state[..., 33:39], # right rpy [28, 34)
    ], axis=-1)


def parse_hdf5(f, hdf5_path):
    def as_frames(output):
        # {'a': (N, ...), 'b': (N, ...)} -> [{'a': (...), 'b': (...)}, ...]
        return [dict(zip(output.keys(), t)) for t in zip(*output.values())]

    images = dict()
    for key in f['observations']['images'].keys():
        images[key] = [decode_image(img) for img in f['observations']['images'][key][:]]
    state = f['observations']['qpos'][:]
    action = f['action'][:]

    output = {
        'observation.state': extract_joint_and_pose(state),
        'action': extract_joint_and_pose(action),
    }
    for key, value in images.items():
        output[f'observation.images.{key}'] = np.array(value)
    
    task = hdf5_path.replace('\\', '/').split('/')[-2]
    task = _TASK_MAPPING[task]
    output['task'] = [task for _ in range(len(state))]

    return as_frames(output)


class HDF5DataConvertor(BaseDataConvertor):
    def __init__(self, config: HDF5DataConvertorConfig):
        super().__init__(config)
    
    def _yield_episodes(self):
        hdf5_paths = []
        for root, dirs, files in os.walk(self.config.root):
            for file in files:
                if file.endswith('.hdf5') or file.endswith('.h5'):
                    hdf5_paths.append(os.path.join(root, file))
        
        hdf5_paths.sort()

        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as f:
                result = parse_hdf5(f, hdf5_path)
            
            yield result