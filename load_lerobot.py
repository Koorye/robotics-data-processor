import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


dataset = LeRobotDataset("realman/catch_banana", video_backend='pyav')

prev_episode = None


def image_torch_to_numpy(img):
    return img.permute(1, 2, 0).cpu().numpy()


def extract_action(action_tensor):
    action = action_tensor.cpu().numpy()
    return np.concatenate([
        action[50:57], # left joint
        action[60:61], # left gripper
        action[0:7],   # right joint
        action[10:11], # right gripper
    ]).tolist()


def extract_sample(sample):
    episode = sample["episode_index"]
    step = sample["frame_index"]
    image_keys = [key for key in sample.keys() if key.startswith("observation.image")]
    frames = {key: image_torch_to_numpy(sample[key]) for key in image_keys}
    state = extract_action(sample["observation.state"])
    action = extract_action(sample["action"])
    return {
        "episode": episode,
        "step": step,
        "frames": frames,
        "state": state,
        "action": action,
    }


example = extract_sample(dataset[0])


plt.ion()
plt.subplots(1, len(example["frames"]), figsize=(15, 7))

for sample in dataset:
    data = extract_sample(sample)
    episode = data["episode"]
    step = data["step"]
    frames = data["frames"]
    state = data["state"]
    action = data["action"]

    print(f'Episode: {episode}, Step: {step}')
    print('State:', ' '.join([f'{s:.2f}' for s in state]))
    print('Action:', ' '.join([f'{a:.2f}' for a in action]))

    plt.clf()
    for i, (key, frame) in enumerate(frames.items()):
        plt.subplot(1, len(frames), i + 1)
        plt.imshow(frame)
        plt.title(key)
        plt.axis('off')
    
    plt.suptitle(f'Episode: {episode}, Step: {step}\nState: {" ".join([f"{s:.2f}" for s in state])}\nAction: {" ".join([f"{a:.2f}" for a in action])}')
    plt.pause(0.1)
