"""
Arducam Driver: USB camera driver using OpenCV.

Arducams are UVC-compatible USB cameras that work with standard
video capture interfaces. This driver uses OpenCV's VideoCapture
for cross-platform compatibility.
"""

from typing import Optional, List
import cv2
import numpy as np

from ..core.sensor import Sensor
from ..core.frame import (
    SensorFrame,
    SensorMetadata,
    SensorType,
    FrameType,
)


class ArducamSensor(Sensor):
    """
    Driver for Arducam and other UVC-compatible USB cameras.
    
    Example:
        camera = ArducamSensor(device_index=0)
        camera.connect()
        frame = camera.read()
        print(f"Captured {frame.shape} image")
        camera.disconnect()
    """
    
    def __init__(
        self,
        device_index: Optional[int] = None,
        device_path: Optional[str] = None,
        sensor_id: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        if device_index is None and device_path is None:
            device_index = 0
        
        self._device_index = device_index
        self._device_path = device_path
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps
        
        if sensor_id is None:
            if device_path:
                sensor_id = f"camera_{device_path.replace('/', '_')}"
            else:
                sensor_id = f"camera_{device_index}"
        
        super().__init__(sensor_id=sensor_id, sensor_type=SensorType.CAMERA)
        
        self._capture: Optional[cv2.VideoCapture] = None
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_fps: float = 0.0
    
    @property
    def width(self) -> int:
        return self._actual_width
    
    @property
    def height(self) -> int:
        return self._actual_height
    
    @property
    def fps(self) -> float:
        return self._actual_fps
    
    def connect(self) -> None:
        if self._connected:
            return
        
        if self._device_path:
            self._capture = cv2.VideoCapture(self._device_path)
        else:
            self._capture = cv2.VideoCapture(self._device_index)
        
        if not self._capture.isOpened():
            device = self._device_path or f"index {self._device_index}"
            raise ConnectionError(f"Failed to open camera: {device}")
        
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        self._capture.set(cv2.CAP_PROP_FPS, self._requested_fps)
        
        self._actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
        
        self._metadata = SensorMetadata(
            sensor_id=self._sensor_id,
            sensor_type=SensorType.CAMERA,
            manufacturer="Arducam",
            model="USB Camera",
            config={
                "width": self._actual_width,
                "height": self._actual_height,
                "fps": self._actual_fps,
                "device_index": self._device_index,
                "device_path": self._device_path,
            },
        )
        
        self._connected = True
        self._sequence_number = 0
        self._time_offset = None
    
    def disconnect(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._connected = False
    
    def read(self) -> Optional[SensorFrame]:
        if not self._connected or self._capture is None:
            raise RuntimeError(f"Camera {self._sensor_id} is not connected")
        
        ret, frame = self._capture.read()
        
        if not ret or frame is None:
            return None
        
        timestamp, wall_time = self._get_timestamp()
        
        return SensorFrame(
            sensor_id=self._sensor_id,
            sensor_type=SensorType.CAMERA,
            frame_type=FrameType.IMAGE,
            timestamp=timestamp,
            wall_time=wall_time,
            sequence_number=self._next_sequence(),
            data=frame,
            metadata={
                "format": "BGR",
                "dtype": str(frame.dtype),
            },
        )
    
    def read_rgb(self) -> Optional[SensorFrame]:
        frame = self.read()
        if frame is None:
            return None
        
        rgb_data = cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB)
        
        return SensorFrame(
            sensor_id=frame.sensor_id,
            sensor_type=frame.sensor_type,
            frame_type=frame.frame_type,
            timestamp=frame.timestamp,
            wall_time=frame.wall_time,
            sequence_number=frame.sequence_number,
            data=rgb_data,
            metadata={"format": "RGB", "dtype": str(rgb_data.dtype)},
        )


def discover_cameras(max_index: int = 10) -> List[dict]:
    """Discover available cameras by probing device indices."""
    cameras = []
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append({
                "device_index": i,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "backend": cap.getBackendName(),
            })
            cap.release()
    
    return cameras
