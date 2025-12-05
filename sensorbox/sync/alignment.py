"""
Frame alignment for multi-sensor data.

Aligns frames from different sensors based on timestamps.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from collections import deque
import threading

from ..core.frame import SensorFrame


@dataclass
class AlignedFrame:
    """
    A set of frames aligned to a common timestamp.
    
    Contains frames from multiple sensors that were captured
    at approximately the same time.
    """
    timestamp: float  # Reference timestamp
    wall_time: datetime
    frames: Dict[str, SensorFrame] = field(default_factory=dict)
    alignment_errors: Dict[str, float] = field(default_factory=dict)  # Time delta per sensor
    
    def get(self, sensor_id: str) -> Optional[SensorFrame]:
        """Get frame for a specific sensor."""
        return self.frames.get(sensor_id)
    
    def __getitem__(self, sensor_id: str) -> Optional[SensorFrame]:
        return self.get(sensor_id)
    
    @property
    def sensor_ids(self) -> List[str]:
        """List of sensor IDs in this aligned frame."""
        return list(self.frames.keys())
    
    @property
    def max_alignment_error(self) -> float:
        """Maximum alignment error across all sensors."""
        if not self.alignment_errors:
            return 0.0
        return max(abs(e) for e in self.alignment_errors.values())
    
    def is_complete(self, expected_sensors: List[str]) -> bool:
        """Check if all expected sensors have frames."""
        return all(sid in self.frames for sid in expected_sensors)


class FrameBuffer:
    """
    Thread-safe buffer for sensor frames with timestamp indexing.
    """
    
    def __init__(self, max_size: int = 100, max_age: float = 5.0):
        """
        Args:
            max_size: Maximum number of frames to buffer
            max_age: Maximum age of frames in seconds
        """
        self._max_size = max_size
        self._max_age = max_age
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def add(self, frame: SensorFrame) -> None:
        """Add a frame to the buffer."""
        with self._lock:
            self._buffer.append(frame)
            self._cleanup_old_frames()
    
    def _cleanup_old_frames(self) -> None:
        """Remove frames older than max_age."""
        if not self._buffer:
            return
        
        newest_ts = self._buffer[-1].timestamp
        cutoff = newest_ts - self._max_age
        
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()
    
    def find_nearest(self, target_ts: float) -> Optional[Tuple[SensorFrame, float]]:
        """
        Find the frame nearest to the target timestamp.
        
        Returns:
            Tuple of (frame, time_delta) or None if buffer is empty
        """
        with self._lock:
            if not self._buffer:
                return None
            
            best_frame = None
            best_delta = float('inf')
            
            for frame in self._buffer:
                delta = abs(frame.timestamp - target_ts)
                if delta < best_delta:
                    best_delta = delta
                    best_frame = frame
            
            return (best_frame, target_ts - best_frame.timestamp) if best_frame else None
    
    def get_latest(self) -> Optional[SensorFrame]:
        """Get the most recent frame."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


class FrameAligner:
    """
    Aligns frames from multiple sensors based on timestamps.
    
    Uses a primary sensor as the time reference and finds matching
    frames from other sensors within a tolerance window.
    
    Example:
        aligner = FrameAligner(
            primary_sensor="csi_camera_0",
            tolerance=0.050,  # 50ms
        )
        
        # Add frames as they arrive
        aligner.add_frame(camera_frame)
        aligner.add_frame(lidar_frame)
        
        # Get aligned frames
        for aligned in aligner.get_aligned_frames():
            cam = aligned["csi_camera_0"]
            lidar = aligned["rplidar"]
    """
    
    def __init__(
        self,
        primary_sensor: str,
        tolerance: float = 0.050,
        buffer_size: int = 100,
    ):
        """
        Initialize frame aligner.
        
        Args:
            primary_sensor: Sensor ID to use as time reference
            tolerance: Maximum time difference for alignment (seconds)
            buffer_size: Size of frame buffers
        """
        self._primary_sensor = primary_sensor
        self._tolerance = tolerance
        self._buffer_size = buffer_size
        
        self._buffers: Dict[str, FrameBuffer] = {}
        self._lock = threading.Lock()
        
        # Stats
        self._total_alignments = 0
        self._successful_alignments = 0
    
    def _get_buffer(self, sensor_id: str) -> FrameBuffer:
        """Get or create buffer for a sensor."""
        if sensor_id not in self._buffers:
            self._buffers[sensor_id] = FrameBuffer(max_size=self._buffer_size)
        return self._buffers[sensor_id]
    
    def add_frame(self, frame: SensorFrame) -> None:
        """Add a frame to the appropriate buffer."""
        with self._lock:
            buffer = self._get_buffer(frame.sensor_id)
            buffer.add(frame)
    
    def align_to_timestamp(
        self,
        target_ts: float,
        sensor_ids: Optional[List[str]] = None,
    ) -> AlignedFrame:
        """
        Align frames from all sensors to a target timestamp.
        
        Args:
            target_ts: Target timestamp to align to
            sensor_ids: Specific sensors to include (None = all)
        
        Returns:
            AlignedFrame with matched frames
        """
        with self._lock:
            frames = {}
            errors = {}
            
            sensors = sensor_ids or list(self._buffers.keys())
            
            for sensor_id in sensors:
                if sensor_id not in self._buffers:
                    continue
                
                result = self._buffers[sensor_id].find_nearest(target_ts)
                if result:
                    frame, delta = result
                    if abs(delta) <= self._tolerance:
                        frames[sensor_id] = frame
                        errors[sensor_id] = delta
            
            self._total_alignments += 1
            if frames:
                self._successful_alignments += 1
            
            return AlignedFrame(
                timestamp=target_ts,
                wall_time=datetime.now(),
                frames=frames,
                alignment_errors=errors,
            )
    
    def align_to_primary(self) -> Optional[AlignedFrame]:
        """
        Align all sensors to the latest frame from the primary sensor.
        
        Returns:
            AlignedFrame or None if no primary frame available
        """
        with self._lock:
            if self._primary_sensor not in self._buffers:
                return None
            
            primary_frame = self._buffers[self._primary_sensor].get_latest()
            if primary_frame is None:
                return None
            
            return self.align_to_timestamp(primary_frame.timestamp)
    
    def get_stats(self) -> dict:
        """Get alignment statistics."""
        with self._lock:
            success_rate = (
                self._successful_alignments / self._total_alignments
                if self._total_alignments > 0 else 0.0
            )
            return {
                "total_alignments": self._total_alignments,
                "successful_alignments": self._successful_alignments,
                "success_rate": success_rate,
                "buffer_sizes": {
                    sid: len(buf) for sid, buf in self._buffers.items()
                },
            }
    
    def clear(self) -> None:
        """Clear all buffers."""
        with self._lock:
            for buffer in self._buffers.values():
                buffer.clear()
            self._total_alignments = 0
            self._successful_alignments = 0
