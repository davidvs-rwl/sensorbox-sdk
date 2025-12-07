"""SensorBox SDK - Multi-sensor data collection for NVIDIA Jetson."""

__version__ = "0.1.0"

from .core.frame import SensorFrame, SensorMetadata, SensorType, FrameType
from .core.sensor import Sensor

from .drivers import (
    CSICamera,
    MultiCamera,
    RPLidarSensor,
    OakDPro,
    discover_rplidars,
    discover_oakd_devices,
)

from .sync import (
    TimestampManager,
    SyncedSensorFusion,
)

from .storage import (
    HDF5Writer,
    HDF5Reader,
)

__all__ = [
    "SensorFrame",
    "SensorMetadata",
    "SensorType",
    "FrameType",
    "Sensor",
    "CSICamera",
    "MultiCamera",
    "RPLidarSensor",
    "OakDPro",
    "discover_rplidars",
    "discover_oakd_devices",
    "TimestampManager",
    "SyncedSensorFusion",
    "HDF5Writer",
    "HDF5Reader",
]
