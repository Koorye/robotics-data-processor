import math
from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class BaseDataConvetorConfig:
    overwrite: bool = True
    check_only: bool = False

    source_data_roots: List[str] = field(default_factory=lambda: [])

    image_height: int = 480
    image_width: int = 640
    image_dtype: str = 'video'
    image_names: List[str] = field(default_factory=lambda: [])

    depth_height: int = 480
    depth_width: int = 640
    depth_dtype: str = 'float32'
    depth_names: List[str] = field(default_factory=lambda: [])

    state_key: str = 'observation.state'
    state_dtype: str = 'float32'
    state_names: List[str] = field(default_factory=lambda: [])

    action_key: str = 'action'
    action_dtype: str = 'float32'
    action_names: List[str] = field(default_factory=lambda: [])

    position_nonoop_threshold: float = 1e-4
    rotation_nonoop_threshold: float = math.radians(0.1)
    gripper_nonoop_threshold: float = 1e-4

    default_task: str = 'do something'

    repo_id: str = 'realman/test'
    data_root: Optional[str] = None
    fps: int = 30
    video_backend: str = 'pyav'

    def __post_init__(self):
        self.action_len = len(self.action_names)


@dataclass
class HDF5DataConvertorConfig(BaseDataConvetorConfig):
    root: str
    image_froms: List[Any] = field(default_factory=lambda: [])
    depth_froms: List[str] = field(default_factory=lambda: [])
    state_from: str = ''
    action_from: str = ''