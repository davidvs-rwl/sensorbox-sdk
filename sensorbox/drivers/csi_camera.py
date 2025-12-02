"""CSI Camera Driver for NVIDIA Jetson."""

from typing import Optional
import cv2

from ..core.sensor import Sensor
from ..core.frame import SensorFrame, SensorMetadata, SensorType, FrameType


class CSICamera(Sensor):
    """Driver for CSI cameras on NVIDIA Jetson (CAM0, CAM1)."""
    
    def __init__(
        self,
        sensor_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip_method: int = 0,
    ):
        self._csi_sensor_id = sensor_id
        self._width = width
        self._height = height
        self._fps = fps
        self._flip_method = flip_method
        
        super().__init__(
            sensor_id=f"csi_camera_{sensor_id}",
            sensor_type=SensorType.CAMERA
        )
        self._capture = None
    
    def _build_gstreamer_pipeline(self) -> str:
        return (
            f"nvarguscamerasrc sensor-id={self._csi_sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={self._width}, height={self._height}, "
            f"framerate={self._fps}/1 ! "
            f"nvvidconv flip-method={self._flip_method} ! "
            f"video/x-raw, width={self._width}, height={self._height}, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
    
    @property
    def width(self) -> int:
        return self._width
    
    @property
    def height(self) -> int:
        return self._height
    
    @property
    def fps(self) -> int:
        return self._fps
    
    def connect(self) -> None:
        if self._connected:
            return
        
        pipeline = self._build_gstreamer_pipeline()
        self._capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self._capture.isOpened():
            raise ConnectionError(f"Failed to open CSI camera {self._csi_sensor_id}")
        
        self._metadata = SensorMetadata(
            sensor_id=self._sensor_id,
            sensor_type=SensorType.CAMERA,
            manufacturer="Arducam",
            model="IMX219 CSI",
            config={"width": self._width, "height": self._height, "fps": self._fps},
        )
        self._connected = True
        self._sequence_number = 0
        self._time_offset = None
    
    def disconnect(self) -> None:
        if self._capture:
            self._capture.release()
            self._capture = None
        self._connected = False
    
    def read(self) -> Optional[SensorFrame]:
        if not self._connected:
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
            metadata={"format": "BGR"},
        )
