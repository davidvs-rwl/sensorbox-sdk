"""
SensorFrame: Unified data structure for all sensor readings.

Every sensor reading in the system is wrapped in a SensorFrame,
providing consistent timestamps, metadata, and type information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import numpy as np


class SensorType(Enum):
    """Supported sensor types."""
    CAMERA = "camera"
    LIDAR = "lidar"


class FrameType(Enum):
    """Type of data payload in the frame."""
    IMAGE = "image"
    POINT_CLOUD = "point_cloud"
    SCAN = "scan"


@dataclass
class SensorMetadata:
    """Static metadata about a sensor (set once at connection)."""
    sensor_id: str
    sensor_type: SensorType
    manufacturer: str = ""
    model: str = ""
    serial_number: str = ""
    firmware_version: str = ""
    config: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)


@dataclass
class SensorFrame:
    """
    A single reading from a sensor.
    
    Attributes:
        sensor_id: Unique identifier for the sensor that produced this frame
        sensor_type: Type of sensor (camera, lidar)
        frame_type: Type of data payload
        timestamp: When the frame was captured (monotonic clock, seconds)
        wall_time: Wall clock time for human reference
        sequence_number: Monotonically increasing frame counter
        data: The actual sensor data (numpy array)
        metadata: Additional frame-specific metadata
    """
    sensor_id: str
    sensor_type: SensorType
    frame_type: FrameType
    timestamp: float
    wall_time: datetime
    sequence_number: int
    data: np.ndarray
    metadata: dict = field(default_factory=dict)
    
    @property
    def shape(self) -> tuple:
        """Shape of the data payload."""
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Data type of the payload."""
        return self.data.dtype
    
    @property
    def nbytes(self) -> int:
        """Size of the data payload in bytes."""
        return self.data.nbytes
    
    def __repr__(self) -> str:
        return (
            f"SensorFrame(sensor_id={self.sensor_id!r}, "
            f"type={self.frame_type.value}, "
            f"shape={self.shape}, "
            f"seq={self.sequence_number}, "
            f"ts={self.timestamp:.6f})"
        )
