"""
Multi-process conversion of HDF5 files to lerobot dataset format.

Examples:

1. Basic usage:
```python
python multi_process_hdf5_to_lerobot.py \
    --root data/of/your/hdf5/files \
    --repo_id name/task \
    --fps 30 \
    --video_backend pyav \
    --image_writer_threads 10 \
    --image_writer_processes 5 \
    --num_processes -1
```

2. Limit the number of episodes:
```python
python multi_process_hdf5_to_lerobot.py \
    --root data/of/your/hdf5/files \
    --repo_id name/task \
    --fps 30 \
    --video_backend pyav \
    --image_writer_threads 10 \
    --image_writer_processes 5 \
    --max_episodes 3 \
    --num_processes -1
```

"""

import argparse
import multiprocessing
import os
import shutil
from tqdm import tqdm

from hdf5_to_lerobot import (
    create_lerobot_dataset, 
    find_hdf5_paths, 
    get_lerobot_root,
    hdf5_to_lerobot, 
    load_hdf5,
)
from merge import merge_datasets


def split_list(lst, n):
    """Split list lst into n approximately equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def process(hdf5_paths, args):
    dataset = create_lerobot_dataset(
        repo_id=args.repo_id,
        example_data=load_hdf5(hdf5_paths[0])[0],
        fps=args.fps,
        video_backend=args.video_backend,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )
    hdf5_to_lerobot(hdf5_paths, dataset)
    return dataset.repo_id


def main(args):
    save_root = os.path.join(get_lerobot_root(), args.repo_id)
    if os.path.exists(save_root):
        if not args.overwrite:
            print(f'Dataset {args.repo_id} already exists. Use --overwrite to overwrite.')
            return
        else:
            print(f'Overwriting existing dataset {args.repo_id}.')
            # remove path startswith save_root
            for root, dirs, files in os.walk(get_lerobot_root()):
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    if dir_path.startswith(save_root):
                        shutil.rmtree(dir_path)
    
    hdf5_paths = find_hdf5_paths(args.root)

    if not hdf5_paths:
        print(f'No hdf5 files found in {args.root}')
        exit(1)
    
    if args.max_episodes > 0:
        hdf5_paths = hdf5_paths[:args.max_episodes]
    
    num_cpus = multiprocessing.cpu_count()
    if args.num_processes < 1:
        args.num_processes = num_cpus
        print(f'Setting num_processes to number of cpus: {num_cpus}')
    
    hdf5_paths_chunks = split_list(hdf5_paths, args.num_processes)

    print('HDF5 files to process:')
    for i, hdf5_paths in enumerate(hdf5_paths_chunks):
        print(f'- Process {i}:')
        for path in hdf5_paths:
            print(f'  - {path}')

    repo_ids = [args.repo_id + f"_part{i}" for i in range(len(hdf5_paths_chunks))]
    processes = []
    for repo_id, hdf5_paths in zip(repo_ids, hdf5_paths_chunks):
        print(f'Process will write to repo_id: {repo_id} with {len(hdf5_paths)} episodes')
        args = argparse.Namespace(**vars(args))
        args.repo_id = repo_id
        p = multiprocessing.Process(target=process, args=(hdf5_paths, args))
        processes.append(p)
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print('Merging datasets...')
    source_roots = [os.path.join(get_lerobot_root(), args.repo_id + f"_part{i}") 
                    for i in range(len(hdf5_paths_chunks))]
    merge_datasets(source_roots, save_root, default_fps=args.fps)
    print(f'Merged dataset saved to {args.repo_id}')

    for source_root in source_roots:
        shutil.rmtree(source_root)
        print(f'Removed temporary dataset {source_root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Input directory containing hdf5 files')
    parser.add_argument('--repo_id', type=str, required=True, help='Output lerobot repository id')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for videos')
    parser.add_argument('--video_backend', type=str, default='pyav', help='Video backend for lerobot (pyav or torchcodec)')
    parser.add_argument('--image_writer_threads', type=int, default=10, help='Number of threads for image writing')
    parser.add_argument('--image_writer_processes', type=int, default=5, help='Number of processes for image writing')
    parser.add_argument('--max_episodes', type=int, default=-1, help='Maximum number of episodes to process (-1 for all)')
    parser.add_argument('--num_processes', type=int, default=-1, help='Number of processes for parallel processing')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing dataset if it exists')
    args = parser.parse_args()
    main(args)
