"""
Convert hdf5 dataset to lerobot format.

Examples:

1. Basic usage:
```python
python hdf5_to_lerobot.py \
    --root data/realman/pass_bowl \
    --repo_id realman/pass_bowl \
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
import shutil
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def get_lerobot_root():
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'lerobot')


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
            'observation.state': extract_joint(state),
            'action': extract_joint(action),
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


def hdf5_to_lerobot(hdf5_paths, dataset):
    for ep, path in enumerate(hdf5_paths):
        frames = load_hdf5(path)
        for frame in tqdm(frames, desc=f'Processing episode {ep} {path}'):
            dataset.add_frame(frame, task='catch banana into basket')
        dataset.save_episode()


def main(args):
    save_root = os.path.join(get_lerobot_root(), args.repo_id)
    if os.path.exists(save_root):
        if not args.overwrite:
            print(f'Dataset {args.repo_id} already exists. Use --overwrite to overwrite.')
            return
        else:
            print(f'Overwriting existing dataset {args.repo_id}.')
            shutil.rmtree(save_root)

    hdf5_paths = find_hdf5_paths(args.root)

    if not hdf5_paths:
        print(f'No hdf5 files found in {args.root}')
        return
    
    if args.max_episodes > 0:
        hdf5_paths = hdf5_paths[:args.max_episodes]
    
    print('HDF5 files to process:')
    for path in hdf5_paths:
        print(' -', path)
    
    dataset = create_lerobot_dataset(
        repo_id=args.repo_id,
        example_data=load_hdf5(hdf5_paths[0])[0],
        fps=args.fps,
        video_backend=args.video_backend,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )
    hdf5_to_lerobot(hdf5_paths, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Input directory containing hdf5 files')
    parser.add_argument('--repo_id', type=str, required=True, help='Output lerobot repository id')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for videos')
    parser.add_argument('--video_backend', type=str, default='pyav', help='Video backend for lerobot (pyav or torchcodec)')
    parser.add_argument('--image_writer_threads', type=int, default=10, help='Number of threads for image writing')
    parser.add_argument('--image_writer_processes', type=int, default=5, help='Number of processes for image writing')
    parser.add_argument('--max_episodes', type=int, default=-1, help='Maximum number of episodes to process (-1 for all)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing dataset if it exists')
    args = parser.parse_args()
    main(args)
