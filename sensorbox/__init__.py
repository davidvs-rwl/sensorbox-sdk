"""
SensorBox SDK: Unified sensor ingestion for Arducams and RPLIDAR.

Quick Start:
    from sensorbox import ArducamSensor, discover_cameras
    
    # Find available cameras
    cameras = discover_cameras()
    print(f"Found {len(cameras)} cameras")
    
    # Stream from a camera
    with ArducamSensor(device_index=0) as camera:
        for frame in camera.stream(duration=5.0):
            print(f"Frame {frame.sequence_number}: {frame.shape}")
"""

__version__ = "0.1.0"

from .core import (
    SensorFrame,
    SensorMetadata,
    SensorType,
    FrameType,
    Sensor,
)
from .drivers import (
    ArducamSensor,
    discover_cameras,
)

__all__ = [
    "__version__",
    "SensorFrame",
    "SensorMetadata",
    "SensorType",
    "FrameType",
    "Sensor",
    "ArducamSensor",
    "discover_cameras",
]
