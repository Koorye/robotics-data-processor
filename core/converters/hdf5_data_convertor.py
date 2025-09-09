import os

from .base_data_convertor import BaseDataConvertor
from .configuration_data_convertor import HDF5DataConvertorConfig


class HDF5DataConvertor(BaseDataConvertor):
    def __init__(self, config: HDF5DataConvertorConfig):
        super().__init__(config)
    
    def _yield_episodes(self):
        hdf5_paths = []
        for root, dirs, files in os.walk(self.config.root):
            for file in files:
                if file.endswith('.hdf5') or file.endswith('.h5'):
                    hdf5_paths.append(os.path.join(root, file))
        
        hdf5_paths.sort()
        for path in hdf5_paths:
            print(f'Processing {path}...')
            dataset = LeRobotDataset(path)
            for episode in dataset.episodes:
                yield episode.frames
