
from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class DataConvertorConfig:
    overwrite: bool = True
    check_only: bool = False

    repo_id: str = 'realman/test'
    data_root: Optional[str] = None
    fps: int = 30
    video_backend: str = 'pyav'
    image_writer_processes: int = 1
    image_writer_threads: int = 1

    image_prefix: str = 'observation.images'
    default_task: str = 'do something'


@dataclass
class HDF5DataConvertorConfig(DataConvertorConfig):
    root: str = ''