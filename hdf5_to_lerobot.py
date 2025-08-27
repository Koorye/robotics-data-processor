"""
Convert hdf5 dataset to lerobot format.

Examples:

1. Basic usage:
```python
python hdf5_to_lerobot.py \
    --root data/of/your/hdf5/files \
    --repo_id name/task \
    --fps 30 \
    --video_backend pyav \
    --image_writer_threads 10 \
    --image_writer_processes 5
```

2. Limit the number of episodes:
```python
python hdf5_to_lerobot.py \
    --root data/of/your/hdf5/files \
    --repo_id name/task \
    --fps 30 \
    --video_backend pyav \
    --image_writer_threads 10 \
    --image_writer_processes 5 \
    --max_episodes 3
```

"""

import argparse
import h5py
import imageio
import io
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def decode_image(image_buffer):
    img = Image.open(io.BytesIO(image_buffer))
    return np.array(img)


def save_images(images, filename):
    imageio.mimsave(filename, images, fps=30)


def create_lerobot_dataset(repo_id, example_data, **kwargs) -> LeRobotDataset:
    features = dict()
    for key, value in example_data.items():
        if key.startswith('observation.images.'):
            features[key] = {
                'dtype': 'video',
                'shape': value.shape,
                'name': ['height', 'width', 'channel'],
            }
        else:
            features[key] = {
                'dtype': 'float32',
                'shape': value.shape,
                'name': [key.split('.')[-1]],
            }
    return LeRobotDataset.create(repo_id, features=features, **kwargs)


def extract_action(action):
    return np.concatenate([
        action[..., 50:57], # left joint
        action[..., 60:61], # left gripper
        action[..., 0:7],   # right joint
        action[..., 10:11], # right gripper
    ], axis=-1)


def load_hdf5(infile):
    def as_frames(output):
        # {'a': (N, ...), 'b': (N, ...)} -> [{'a': (...), 'b': (...)}, ...]
        return [dict(zip(output.keys(), t)) for t in zip(*output.values())]

    with h5py.File(infile, 'r') as f:
        images = dict()
        for key in f['observations']['images'].keys():
            images[key] = [decode_image(img) for img in f['observations']['images'][key][:]]
        state = f['observations']['qpos'][:]
        action = f['action'][:]

        output = {
            'observation.state': extract_action(state),
            'action': extract_action(action),
        }
        for key, value in images.items():
            output[f'observation.images.{key}'] = np.array(value)
        return as_frames(output)


def find_hdf5_paths(root):
    hdf5_paths = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith('.hdf5'):
                hdf5_paths.append(os.path.join(dirpath, filename))
    hdf5_paths.sort()
    return hdf5_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Input directory containing hdf5 files')
    parser.add_argument('--repo_id', type=str, required=True, help='Output lerobot repository id')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for videos')
    parser.add_argument('--video_backend', type=str, default='pyav', help='Video backend for lerobot (pyav or torchcodec)')
    parser.add_argument('--image_writer_threads', type=int, default=10, help='Number of threads for image writing')
    parser.add_argument('--image_writer_processes', type=int, default=5, help='Number of processes for image writing')
    parser.add_argument('--max_episodes', type=int, default=-1, help='Maximum number of episodes to process (-1 for all)')
    args = parser.parse_args()

    hdf5_paths = find_hdf5_paths(args.root)
    if not hdf5_paths:
        print(f'No hdf5 files found in {args.root}')
        exit(1)
    if args.max_episodes > 0:
        hdf5_paths = hdf5_paths[:args.max_episodes]
    
    print('HDF5 files to process:')
    for path in hdf5_paths:
        print(' -', path)

    example_data = load_hdf5(hdf5_paths[0])[0]
    dataset = create_lerobot_dataset(
        repo_id=args.repo_id, 
        example_data=example_data,
        fps=args.fps,
        video_backend=args.video_backend,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    for ep, path in enumerate(hdf5_paths):
        frames = load_hdf5(path)
        for frame in tqdm(frames, desc=f'Processing episode {ep} {path}'):
            dataset.add_frame(frame, task='catch banana into basket')
        dataset.save_episode()
