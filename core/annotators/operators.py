import copy
import numpy as np
from abc import ABC, abstractmethod


class BaseOperator(ABC):
    def __init__(
        self, 
        name,
        window_size=1, 
    ):
        self.name = name
        self.window_size = window_size
    
    @abstractmethod
    def _operate(self, frame_window, annotation_window):
        pass

    def operate(self, episode, annotations):
        frame_windows, annotation_windows = self._split_windows(episode, annotations)

        results = []
        for frame_window, annotation_window in zip(frame_windows, annotation_windows):
            result = self._operate(frame_window, annotation_window)
            results.append(result)

        for result, annotation in zip(results, annotations):
            annotation[self.name] = result

        return annotations

    def _split_windows(self, episode, annotations):
        # e.g. [a, b, c, d, e], window_size=3 
        # -> [[a,a,a], [a,a,b], [a,b,c], [b,c,d], [c,d,e]]
        if self.window_size <= 1:
            return [[frame] for frame in episode], [[annotation] for annotation in annotations]
        
        frame_windows, annotation_windows = [], []
        for i in range(len(episode)):
            start = max(0, i - self.window_size + 1)
            frame_window = episode[start:i + 1]
            annotation_window = annotations[start:i + 1]

            if len(frame_window) < self.window_size:
                frame_window = [copy.deepcopy(frame_window[0])] * (self.window_size - len(frame_window)) + frame_window
                annotation_window = [copy.deepcopy(annotation_window[0])] * (self.window_size - len(annotation_window)) + annotation_window

            frame_windows.append(frame_window)
            annotation_windows.append(annotation_window)

        return frame_windows, annotation_windows


class PositionOperator(BaseOperator):
    def __init__(
        self,
        state_key,
        xyz_range=(0, 2),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.state_key = state_key
        self.xyz_range = xyz_range
    
    def _operate(self, frame_window, annotation_window):
        curr_xyz = frame_window[-1][self.state_key][self.xyz_range[0]:self.xyz_range[1]]
        return list(curr_xyz)


class DirectionOperator(BaseOperator):
    def __init__(
        self,
        state_key,
        rpy_range=(0, 2),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.state_key = state_key
        self.rpy_range = rpy_range
    
    def _operate(self, frame_window, annotation_window):
        curr_rpy = frame_window[-1][self.state_key][self.rpy_range[0]:self.rpy_range[1]]
        return list(curr_rpy)


class MovementOperator(BaseOperator):
    def __init__(
        self,
        position_key,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.position_key = position_key
    
    def _operate(self, frame_window, annotation_window):
        if self.window_size < 2:
            return 0.0
        
        prev_xyz = annotation_window[0][self.position_key]
        curr_xyz = annotation_window[-1][self.position_key]
        return (np.array(curr_xyz) - np.array(prev_xyz)).tolist()


class VelocityOperator(BaseOperator):
    def __init__(
        self,
        movement_key,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.movement_key = movement_key

    def _operate(self, frame_window, annotation_window):
        return np.linalg.norm(annotation_window[-1][self.movement_key]).item()


class AccelerationOperator(BaseOperator):
    def __init__(
        self,
        vel_key,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vel_key = vel_key

    def _operate(self, frame_window, annotation_window):
        if self.window_size < 2:
            return 0.0
        
        prev_vel = annotation_window[0][self.vel_key]
        curr_vel = annotation_window[-1][self.vel_key]
        accel = (curr_vel - prev_vel)
        return accel


class MovementSummaryOperator(BaseOperator):
    def __init__(
        self,
        movement_key,
        threshold=1e-3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.movement_key = movement_key
        self.threshold = threshold

    def _operate(self, frame_window, annotation_window):
        dx, dy, dz = annotation_window[-1][self.movement_key]
        if max(abs(dx), abs(dy), abs(dz)) < self.threshold:
            return 'stationary'
        
        if abs(dx) > abs(dy) and abs(dx) > abs(dz):
            if dx > 0:
                return 'right'
            else:
                return 'left'
        elif abs(dy) > abs(dx) and abs(dy) > abs(dz):
            if dy > 0:
                return 'forward'
            else:
                return 'backward'
        elif abs(dz) > abs(dx) and abs(dz) > abs(dy):
            if dz > 0:
                return 'up'
            else:
                return 'down'
        else:
            return 'stationary'


class VelocitySummaryOperator(BaseOperator):
    def __init__(
        self,
        vel_key,
        slow_threshold=1e-3,
        fast_threshold=5e-3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vel_key = vel_key
        self.slow_threshold = slow_threshold
        self.fast_threshold = fast_threshold

    def _operate(self, frame_window, annotation_window):
        vel = annotation_window[-1][self.vel_key]
        if vel < self.slow_threshold:
            return 'stationary'
        elif vel < self.fast_threshold:
            return 'slow'
        else:
            return 'fast'


class AccelerationSummaryOperator(BaseOperator):
    def __init__(
        self,
        accel_key,
        decel_threshold=-1e-4,
        accel_threshold=1e-4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.accel_key = accel_key
        self.decel_threshold = decel_threshold
        self.accel_threshold = accel_threshold

    def _operate(self, frame_window, annotation_window):
        accel = annotation_window[-1][self.accel_key]
        if accel < self.decel_threshold:
            return 'decelerating'
        elif accel > self.accel_threshold:
            return 'accelerating'
        else:
            return 'constant'


def make_operator_from_config(config):
    op_type = config['type']
    del config['type']

    if op_type == 'position':
        return PositionOperator(**config)
    elif op_type == 'direction':
        return DirectionOperator(**config)
    elif op_type == 'movement':
        return MovementOperator(**config)
    elif op_type == 'velocity':
        return VelocityOperator(**config)
    elif op_type == 'acceleration':
        return AccelerationOperator(**config)
    elif op_type == 'movement_summary':
        return MovementSummaryOperator(**config)
    elif op_type == 'velocity_summary':
        return VelocitySummaryOperator(**config)
    elif op_type == 'acceleration_summary':
        return AccelerationSummaryOperator(**config)
    else:
        raise ValueError(f"Unknown operator type: {op_type}")