import argparse
import sys
sys.path.append('.')

from core.converters.configuration_data_convertor import LeRobotDataConvertorConfig
from core.converters.lerobot_data_convertor import LeRobotDataConvertor


def main(args):
    config = LeRobotDataConvertorConfig(
        source_repo_id=args.source_repo_id,
        source_video_backend=args.source_video_backend,
        repo_id=args.repo_id,
        fps=args.fps,
        video_backend=args.video_backend,
        overwrite=args.overwrite,
        check_only=args.check_only,
        image_prefix=args.image_prefix,
        default_task=args.default_task,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )
    convertor = LeRobotDataConvertor(config)
    convertor.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_repo_id', type=str, required=True, help='Source Lerobot repository ID to read the dataset.')
    parser.add_argument('--source_video_backend', type=str, default='pyav', help='Video backend for source lerobot (pyav or torchcodec).')
    parser.add_argument('--repo_id', type=str, required=True, help='Lerobot repository ID to save the dataset.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output videos.')
    parser.add_argument('--video_backend', type=str, default='pyav', help='Video backend for lerobot (pyav or torchcodec).')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing dataset.')
    parser.add_argument('--check_only', action='store_true', help='If set, only check the data without converting.')
    parser.add_argument('--image_prefix', type=str, default='observation.images', help='Prefix for image keys in the data.')
    parser.add_argument('--default_task', type=str, default='do something', help='Default task for lerobot.')
    parser.add_argument('--image_writer_processes', type=int, default=1, help='Number of processes for image writing.')
    parser.add_argument('--image_writer_threads', type=int, default=1, help='Number of threads for image writing.')
    args = parser.parse_args()
    main(args)