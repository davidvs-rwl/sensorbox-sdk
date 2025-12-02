"""CSI Camera Driver for NVIDIA Jetson."""

from typing import Optional, Generator
import time
import cv2

from ..core.sensor import Sensor
from ..core.frame import SensorFrame, SensorMetadata, SensorType, FrameType


# Common resolution presets
RESOLUTIONS = {
    "4K": (3280, 2464, 21),      # Max resolution, 21 FPS
    "1080p": (1920, 1080, 30),   # Full HD
    "720p": (1280, 720, 60),     # HD, high frame rate
    "480p": (640, 480, 90),      # Low res, very high FPS
}


class CSICamera(Sensor):
    """
    Driver for CSI cameras on NVIDIA Jetson (CAM0, CAM1).
    
    Example:
        # Basic usage
        with CSICamera(sensor_id=0) as cam:
            for frame in cam.stream(duration=5.0):
                process(frame)
        
        # With resolution preset
        with CSICamera(sensor_id=0, resolution="1080p") as cam:
            for frame in cam.stream(duration=5.0):
                process(frame)
        
        # With target FPS (throttled)
        with CSICamera(sensor_id=0) as cam:
            for frame in cam.stream(duration=5.0, target_fps=15):
                process(frame)
    """
    
    def __init__(
        self,
        sensor_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip_method: int = 0,
        resolution: Optional[str] = None,
    ):
        """
        Initialize CSI camera.
        
        Args:
            sensor_id: CSI camera index (0 for CAM0, 1 for CAM1)
            width: Capture width (ignored if resolution is set)
            height: Capture height (ignored if resolution is set)
            fps: Capture FPS (ignored if resolution is set)
            flip_method: Image flip (0=none, 2=180°)
            resolution: Preset name ("4K", "1080p", "720p", "480p")
        """
        # Apply resolution preset if provided
        if resolution:
            if resolution not in RESOLUTIONS:
                raise ValueError(f"Unknown resolution '{resolution}'. Options: {list(RESOLUTIONS.keys())}")
            width, height, fps = RESOLUTIONS[resolution]
        
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
        """Build the GStreamer pipeline string."""
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
    
    def connect(self) -> None:
        """Connect to the camera."""
        if self._connected:
            return
        
        pipeline = self._build_gstreamer_pipeline()
        self._capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self._capture.isOpened():
            raise ConnectionError(
                f"Failed to open CSI camera {self._csi_sensor_id}. "
                f"Check connection and run: nvgstcapture-1.0 --sensor-id={self._csi_sensor_id}"
            )
        
        self._metadata = SensorMetadata(
            sensor_id=self._sensor_id,
            sensor_type=SensorType.CAMERA,
            manufacturer="Arducam",
            model="IMX219 CSI",
            config={
                "width": self._width,
                "height": self._height,
                "fps": self._fps,
                "csi_sensor_id": self._csi_sensor_id,
                "flip_method": self._flip_method,
            },
        )
        self._connected = True
        self._sequence_number = 0
        self._time_offset = None
    
    def disconnect(self) -> None:
        """Disconnect from the camera."""
        if self._capture:
            self._capture.release()
            self._capture = None
        self._connected = False
    
    def read(self) -> Optional[SensorFrame]:
        """Read a single frame from the camera."""
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
    
    def stream(
        self,
        duration: Optional[float] = None,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
    ) -> Generator[SensorFrame, None, None]:
        """
        Stream frames from the camera with optional FPS limiting.
        
        Args:
            duration: Maximum duration in seconds (None = infinite)
            max_frames: Maximum number of frames (None = infinite)
            target_fps: Target frame rate (None = no limit, use camera FPS)
        
        Yields:
            SensorFrame for each captured frame
        
        Example:
            # Stream at full speed
            for frame in cam.stream(duration=5.0):
                process(frame)
            
            # Stream at 10 FPS (throttled)
            for frame in cam.stream(duration=5.0, target_fps=10):
                process(frame)
        """
        if not self._connected:
            raise RuntimeError(f"Sensor {self._sensor_id} is not connected")
        
        start_time = time.monotonic()
        frame_count = 0
        frame_interval = 1.0 / target_fps if target_fps else None
        last_frame_time = 0.0
        
        while True:
            # Check duration limit
            if duration is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= duration:
                    break
            
            # Check frame limit
            if max_frames is not None and frame_count >= max_frames:
                break
            
            # FPS throttling
            if frame_interval is not None:
                current_time = time.monotonic()
                time_since_last = current_time - last_frame_time
                if time_since_last < frame_interval:
                    # Sleep for remaining time (minus a small buffer)
                    sleep_time = frame_interval - time_since_last - 0.001
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue
            
            # Read frame
            frame = self.read()
            if frame is not None:
                frame_count += 1
                last_frame_time = time.monotonic()
                yield frame


def list_resolutions() -> dict:
    """List available resolution presets."""
    return RESOLUTIONS.copy()
