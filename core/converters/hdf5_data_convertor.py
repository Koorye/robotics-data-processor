import io
import h5py
import numpy as np
import os
from PIL import Image

from .base_data_convertor import BaseDataConvertor
from .configuration_data_convertor import HDF5DataConvertorConfig


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


def parse_hdf5(f):
    def as_frames(output):
        # {'a': (N, ...), 'b': (N, ...)} -> [{'a': (...), 'b': (...)}, ...]
        return [dict(zip(output.keys(), t)) for t in zip(*output.values())]

    images = dict()
    for key in f['observations']['images'].keys():
        images[key] = [decode_image(img) for img in f['observations']['images'][key][:]]
    state = f['observations']['qpos'][:]
    action = f['action'][:]

    output = {
        # 'observation.state': extract_joint(state),
        # 'action': extract_joint(action),
        'observation.state': extract_pose(state),
        'action': extract_pose(action),
    }
    for key, value in images.items():
        output[f'observation.images.{key}'] = np.array(value)
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
                result = parse_hdf5(f)
            
            yield result