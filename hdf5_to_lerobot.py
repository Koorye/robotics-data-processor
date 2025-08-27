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
            'observation.state': state,
            'action': action
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
    return hdf5_paths


hdf5_paths = find_hdf5_paths('data/examples/realman/task_catch_banana_200_3.30')
example_data = load_hdf5(hdf5_paths[0])[0]
dataset = create_lerobot_dataset(
    'realman/catch_banana', 
    example_data=example_data,
    fps=30, video_backend='pyav',
    image_writer_threads=10,
    image_writer_processes=5,
)
for ep, path in enumerate(hdf5_paths):
    frames = load_hdf5(path)
    for frame in tqdm(frames, desc=f'Processing episode {ep}'):
        dataset.add_frame(frame, task='catch banana into basket')
    dataset.save_episode()
