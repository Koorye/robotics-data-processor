import argparse
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


def get_default_lerobot_root():
    return os.path.expanduser('~/.cache/huggingface/lerobot')


def find_episodes(repo_id):
    with open(os.path.join(get_default_lerobot_root(), repo_id, 'meta', 'episodes.jsonl'), 'r') as f:
        lines = f.readlines()
    episodes = [json.loads(line) for line in lines]
    return episodes


def load_annotation(repo_id, episode_index):
    annotation_path = os.path.join(get_default_lerobot_root(), repo_id, 'annotations', f'episode_{episode_index:06d}.json')
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    return annotation


def stat_annotation(annotation, keys):
    frames = [frame['frame_index'] for frame in annotation]
    plt.figure(figsize=(12, 6))
    for key in keys:
        values = [frame[key] for frame in annotation if key in frame]
        df = pd.DataFrame({'frame': frames, key: values})
        plt.subplot(len(keys), 1, keys.index(key) + 1)
        sns.lineplot(data=df, x='frame', y=key)
    plt.show()


def main(args):
    episodes = find_episodes(args.repo_id)
    if not episodes:
        print(f'No episodes found in {args.repo_id}')
        return

    for ep in episodes:
        annotation = load_annotation(args.repo_id, ep['episode_index'])
        stat_annotation(annotation, args.keys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str, required=True, help='Lerobot repository ID')
    parser.add_argument('--keys', type=str, nargs='+', required=True, help='Keys to plot from annotation')
    args = parser.parse_args()

    main(args)