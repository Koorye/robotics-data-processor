import json
import os
from pyarrow import parquet

from .configuration_lerobot_annotator import LerobotAnnotatorConfig
from .operators import make_operator_from_config


def get_default_lerobot_root():
    return os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'lerobot')


class LerobotAnnotator:
    def __init__(self, config: LerobotAnnotatorConfig):
        self.config = config
        self.operators = [make_operator_from_config(op_cfg) for op_cfg in config.operators]

    def annotate(self):
        os.makedirs(os.path.join(get_default_lerobot_root(), self.config.repo_id, 'annotations'), exist_ok=True)

        parquet_paths = []
        for root, dirs, files in os.walk(os.path.join(get_default_lerobot_root(), self.config.repo_id, 'data')):
            for file in files:
                if file.endswith('.parquet'):
                    parquet_path = os.path.join(root, file)
                    parquet_paths.append(parquet_path)
        
        parquet_paths.sort()

        for path in parquet_paths:
            episode = parquet.read_table(path).to_pydict()
            # {'a': [...], 'b': [...]} -> [{'a': ..., 'b': ...}, ...]
            episode = [dict(zip(episode.keys(), values)) for values in zip(*episode.values())]
            self._annotate_episode(episode)
    
    def _annotate_episode(self, episode):
        annotation_path = os.path.join(get_default_lerobot_root(), self.config.repo_id, 'annotations', f'episode_{episode[0]["episode_index"]:06d}.json')
        if os.path.exists(annotation_path):
            annotations = json.load(open(annotation_path, 'r'))
        else:
            annotations = [{"frame_index": i} for i in range(len(episode))]
        
        for operator in self.operators:
            annotations = operator.operate(episode, annotations)

        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=4)
        
        print(f'Annotated episode {episode[0]["episode_index"]} with {len(episode)} frames.')


if __name__ == '__main__':
    config = LerobotAnnotatorConfig(
        repo_id='realman/catch_banana',
        video_backend='pyav',
    )
    annotator = LerobotAnnotator(config)
    annotator.annotate()