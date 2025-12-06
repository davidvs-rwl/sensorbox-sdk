"""CSI Camera Driver for NVIDIA Jetson with buffer flush."""

from typing import Optional, Generator
import time
import cv2
import logging

from ..core.sensor import Sensor
from ..core.frame import SensorFrame, SensorMetadata, SensorType, FrameType

logger = logging.getLogger(__name__)

RESOLUTIONS = {
    "4K": (3280, 2464, 21),
    "1080p": (1920, 1080, 30),
    "720p": (1280, 720, 60),
    "480p": (640, 480, 90),
}


class CameraError(Exception):
    pass

class CameraConnectionError(CameraError):
    pass

class CameraReadError(CameraError):
    pass


class CSICamera(Sensor):
    """CSI camera driver with buffer flushing."""
    
    def __init__(
        self,
        sensor_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip_method: int = 2,
        resolution: Optional[str] = None,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        max_consecutive_failures: int = 10,
        flush_frames: int = 5,  # Number of frames to discard on connect
    ):
        if resolution:
            if resolution not in RESOLUTIONS:
                raise ValueError(f"Unknown resolution '{resolution}'. Options: {list(RESOLUTIONS.keys())}")
            width, height, fps = RESOLUTIONS[resolution]
        
        self._csi_sensor_id = sensor_id
        self._width = width
        self._height = height
        self._fps = fps
        self._flip_method = flip_method
        self._flush_frames = flush_frames
        
        self._auto_reconnect = auto_reconnect
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._max_consecutive_failures = max_consecutive_failures
        
        self._consecutive_failures = 0
        self._total_frames = 0
        self._dropped_frames = 0
        
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
    
    @property
    def gstreamer_pipeline(self) -> str:
        return self._build_gstreamer_pipeline()
    
    @property
    def stats(self) -> dict:
        return {
            "total_frames": self._total_frames,
            "dropped_frames": self._dropped_frames,
            "drop_rate": self._dropped_frames / max(1, self._total_frames),
        }
    
    def connect(self) -> None:
        if self._connected:
            return
        
        last_error = None
        for attempt in range(self._max_reconnect_attempts):
            try:
                self._do_connect()
                return
            except Exception as e:
                last_error = e
                if attempt < self._max_reconnect_attempts - 1:
                    time.sleep(self._reconnect_delay)
        
        raise CameraConnectionError(
            f"Failed to connect to camera {self._csi_sensor_id} after {self._max_reconnect_attempts} attempts: {last_error}"
        )
    
    def _do_connect(self) -> None:
        pipeline = self._build_gstreamer_pipeline()
        self._capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self._capture.isOpened():
            raise CameraConnectionError(
                f"Failed to open CSI camera {self._csi_sensor_id}"
            )
        
        # FLUSH OLD FRAMES FROM BUFFER
        for _ in range(self._flush_frames):
            self._capture.read()
        
        self._metadata = SensorMetadata(
            sensor_id=self._sensor_id,
            sensor_type=SensorType.CAMERA,
            manufacturer="Arducam",
            model="CSI Camera",
            config={
                "width": self._width,
                "height": self._height,
                "fps": self._fps,
                "csi_sensor_id": self._csi_sensor_id,
            },
        )
        
        self._connected = True
        self._sequence_number = 0
        self._time_offset = None
        self._consecutive_failures = 0
    
    def disconnect(self) -> None:
        if self._capture:
            self._capture.release()
            self._capture = None
        self._connected = False
    
    def reconnect(self) -> bool:
        self.disconnect()
        try:
            self.connect()
            return True
        except CameraConnectionError:
            return False
    
    def read(self) -> Optional[SensorFrame]:
        if not self._connected:
            raise RuntimeError(f"Camera {self._sensor_id} is not connected")
        
        ret, frame = self._capture.read()
        
        if not ret or frame is None:
            self._consecutive_failures += 1
            self._dropped_frames += 1
            
            if self._consecutive_failures >= self._max_consecutive_failures:
                if self._auto_reconnect:
                    if self.reconnect():
                        self._consecutive_failures = 0
                        return self.read()
                    else:
                        raise CameraReadError(f"Camera {self._csi_sensor_id} failed")
                else:
                    raise CameraReadError(f"Camera {self._csi_sensor_id} exceeded max failures")
            
            return None
        
        self._consecutive_failures = 0
        self._total_frames += 1
        
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
    
    def stream(
        self,
        duration: Optional[float] = None,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
    ) -> Generator[SensorFrame, None, None]:
        if not self._connected:
            raise RuntimeError(f"Sensor {self._sensor_id} is not connected")
        
        start_time = time.monotonic()
        frame_count = 0
        frame_interval = 1.0 / target_fps if target_fps else None
        last_frame_time = 0.0
        
        while True:
            if duration is not None:
                if time.monotonic() - start_time >= duration:
                    break
            
            if max_frames is not None and frame_count >= max_frames:
                break
            
            if frame_interval is not None:
                current_time = time.monotonic()
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
            
            try:
                frame = self.read()
                if frame is not None:
                    frame_count += 1
                    last_frame_time = time.monotonic()
                    yield frame
            except CameraReadError:
                break


def list_resolutions() -> dict:
    return RESOLUTIONS.copy()
