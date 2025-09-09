import numpy as np
import os
from pyarrow import parquet
from PIL import Image
import matplotlib.pyplot as plt


parquet_root = 'lerobot/realman/peach_basket/data/chunk-000'
image_root = 'lerobot/realman/peach_basket/images'
image_keys = [
    'observation.images.image_top',
    'observation.images.image_left',
    'observation.images.image_right',
]

for episode in range(0, 100):
    parquet_path = os.path.join(parquet_root, f'episode_{episode:06d}.parquet')
    table = parquet.read_table(parquet_path)
    data = table.to_pydict()
    del data['timestamp']
    del data['frame_index']
    del data['episode_index']
    del data['index']
    del data['task_index']

    for image_key in image_keys:
        image_dir = os.path.join(image_root, image_key, f'episode_{episode:06d}')
        image_names = os.listdir(image_dir)
        image_paths = [os.path.join(image_dir, name) for name in image_names]
        images = [np.array(Image.open(path).convert('RGB')) for path in image_paths]
        data[image_key] = images
    
    print(data.keys())
    for i in range(len(data['action'])):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for j, image_key in enumerate(image_keys):
            axs[j].imshow(data[image_key][i])
            axs[j].axis('off')
        plt.suptitle(f"Episode {episode}, Step {i}, \nAction: {data['action'][i]}")
        print(f"Episode {episode}, Step {i}, Action: {data['action'][i]}")
        plt.show()