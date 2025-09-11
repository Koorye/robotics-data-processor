import argparse
import sys
sys.path.append('.')

from core.converters.hdf5_data_convertor import HDF5DataConvertor
from core.converters.configuration_data_convertor import HDF5DataConvertorConfig


def main(args):
    config = HDF5DataConvertorConfig(
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        video_backend=args.video_backend,
        overwrite=args.overwrite,
        check_only=args.check_only,
        image_prefix=args.image_prefix,
        default_task=args.default_task,
    )
    convertor = HDF5DataConvertor(config)
    convertor.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Root directory containing HDF5 files.')
    parser.add_argument('--repo_id', type=str, required=True, help='Lerobot repository ID to save the dataset.')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output videos.')
    parser.add_argument('--video_backend', type=str, default='pyav', help='Video backend for lerobot (pyav or torchcodec).')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing dataset.')
    parser.add_argument('--check_only', action='store_true', help='If set, only check the data without converting.')
    parser.add_argument('--image_prefix', type=str, default='observation.images', help='Prefix for image keys in the data.')
    parser.add_argument('--default_task', type=str, default='do something', help='Default task for lerobot.')
    args = parser.parse_args()
    main(args)