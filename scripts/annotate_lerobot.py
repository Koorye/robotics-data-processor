import draccus

import sys
sys.path.append('.')
from core.annotators.configuration_lerobot_annotator import LerobotAnnotatorConfig
from core.annotators.lerobot_annotator import LerobotAnnotator


@draccus.wrap()
def main(config: LerobotAnnotatorConfig):
    annotator = LerobotAnnotator(config)
    annotator.annotate()


if __name__ == '__main__':
    main()