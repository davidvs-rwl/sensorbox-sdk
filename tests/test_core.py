"""
Tests for SensorBox SDK core components.

Run with: pytest tests/ -v
"""

import pytest
import numpy as np
from datetime import datetime

from sensorbox.core.frame import (
    SensorFrame,
    SensorMetadata,
    SensorType,
    FrameType,
)


class TestSensorFrame:
    def test_create_image_frame(self):
        data = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = SensorFrame(
            sensor_id="camera_0",
            sensor_type=SensorType.CAMERA,
            frame_type=FrameType.IMAGE,
            timestamp=1.234,
            wall_time=datetime.now(),
            sequence_number=0,
            data=data,
        )
        assert frame.sensor_id == "camera_0"
        assert frame.sensor_type == SensorType.CAMERA
        assert frame.shape == (480, 640, 3)
    
    def test_frame_properties(self):
        data = np.ones((100, 100), dtype=np.float32)
        frame = SensorFrame(
            sensor_id="test",
            sensor_type=SensorType.LIDAR,
            frame_type=FrameType.SCAN,
            timestamp=0.0,
            wall_time=datetime.now(),
            sequence_number=42,
            data=data,
        )
        assert frame.shape == (100, 100)
        assert frame.dtype == np.float32
        assert frame.nbytes == 100 * 100 * 4


class TestSensorMetadata:
    def test_create_metadata(self):
        meta = SensorMetadata(
            sensor_id="camera_0",
            sensor_type=SensorType.CAMERA,
            manufacturer="Arducam",
            model="B0205",
        )
        assert meta.sensor_id == "camera_0"
        assert meta.manufacturer == "Arducam"


class TestArducamSensorMock:
    def test_sensor_creation(self):
        from sensorbox.drivers.camera import ArducamSensor
        sensor = ArducamSensor(device_index=0)
        assert sensor.sensor_id == "camera_0"
        assert sensor.is_connected is False
    
    def test_read_without_connect_raises(self):
        from sensorbox.drivers.camera import ArducamSensor
        sensor = ArducamSensor(device_index=0)
        with pytest.raises(RuntimeError, match="not connected"):
            sensor.read()
