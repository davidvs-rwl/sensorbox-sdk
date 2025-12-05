"""CSI Camera Driver for NVIDIA Jetson with error handling."""

from typing import Optional, Generator
import time
import cv2
import logging

from ..core.sensor import Sensor
from ..core.frame import SensorFrame, SensorMetadata, SensorType, FrameType

logger = logging.getLogger(__name__)

# Common resolution presets
RESOLUTIONS = {
    "4K": (3280, 2464, 21),
    "1080p": (1920, 1080, 30),
    "720p": (1280, 720, 60),
    "480p": (640, 480, 90),
}


class CameraError(Exception):
    """Base exception for camera errors."""
    pass


class CameraConnectionError(CameraError):
    """Raised when camera connection fails."""
    pass


class CameraReadError(CameraError):
    """Raised when frame read fails."""
    pass


class CSICamera(Sensor):
    """
    Driver for CSI cameras on NVIDIA Jetson with error handling.
    
    Features:
        - Automatic reconnection on failure
        - Dropped frame detection
        - Configurable retry behavior
    
    Example:
        with CSICamera(sensor_id=0, auto_reconnect=True) as cam:
            for frame in cam.stream(duration=60.0):
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
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        max_consecutive_failures: int = 10,
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
            auto_reconnect: Automatically reconnect on failure
            max_reconnect_attempts: Max reconnection attempts
            reconnect_delay: Delay between reconnection attempts (seconds)
            max_consecutive_failures: Max consecutive read failures before error
        """
        if resolution:
            if resolution not in RESOLUTIONS:
                raise ValueError(f"Unknown resolution '{resolution}'. Options: {list(RESOLUTIONS.keys())}")
            width, height, fps = RESOLUTIONS[resolution]
        
        self._csi_sensor_id = sensor_id
        self._width = width
        self._height = height
        self._fps = fps
        self._flip_method = flip_method
        
        # Error handling settings
        self._auto_reconnect = auto_reconnect
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._max_consecutive_failures = max_consecutive_failures
        
        # Stats
        self._consecutive_failures = 0
        self._total_frames = 0
        self._dropped_frames = 0
        
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
    
    @property
    def stats(self) -> dict:
        """Get camera statistics."""
        return {
            "total_frames": self._total_frames,
            "dropped_frames": self._dropped_frames,
            "drop_rate": self._dropped_frames / max(1, self._total_frames),
        }
    
    def connect(self) -> None:
        """Connect to the camera with retry logic."""
        if self._connected:
            return
        
        last_error = None
        for attempt in range(self._max_reconnect_attempts):
            try:
                self._do_connect()
                logger.info(f"Camera {self._csi_sensor_id} connected successfully")
                return
            except Exception as e:
                last_error = e
                logger.warning(f"Camera {self._csi_sensor_id} connection attempt {attempt + 1} failed: {e}")
                if attempt < self._max_reconnect_attempts - 1:
                    time.sleep(self._reconnect_delay)
        
        raise CameraConnectionError(
            f"Failed to connect to camera {self._csi_sensor_id} after {self._max_reconnect_attempts} attempts: {last_error}"
        )
    
    def _do_connect(self) -> None:
        """Internal connection logic."""
        pipeline = self._build_gstreamer_pipeline()
        self._capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if not self._capture.isOpened():
            raise CameraConnectionError(
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
        self._consecutive_failures = 0
    
    def disconnect(self) -> None:
        """Disconnect from the camera."""
        if self._capture:
            self._capture.release()
            self._capture = None
        self._connected = False
        logger.info(f"Camera {self._csi_sensor_id} disconnected. Stats: {self.stats}")
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the camera.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        logger.info(f"Attempting to reconnect camera {self._csi_sensor_id}...")
        self.disconnect()
        try:
            self.connect()
            return True
        except CameraConnectionError:
            return False
    
    def read(self) -> Optional[SensorFrame]:
        """Read a single frame with error handling."""
        if not self._connected:
            raise RuntimeError(f"Camera {self._sensor_id} is not connected")
        
        ret, frame = self._capture.read()
        
        if not ret or frame is None:
            self._consecutive_failures += 1
            self._dropped_frames += 1
            
            logger.warning(f"Camera {self._csi_sensor_id} frame read failed "
                          f"({self._consecutive_failures} consecutive)")
            
            # Try reconnection if enabled
            if self._consecutive_failures >= self._max_consecutive_failures:
                if self._auto_reconnect:
                    if self.reconnect():
                        self._consecutive_failures = 0
                        return self.read()  # Retry after reconnect
                    else:
                        raise CameraReadError(
                            f"Camera {self._csi_sensor_id} failed after reconnection attempt"
                        )
                else:
                    raise CameraReadError(
                        f"Camera {self._csi_sensor_id} exceeded max consecutive failures "
                        f"({self._max_consecutive_failures})"
                    )
            
            return None
        
        # Success - reset failure counter
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
        """Stream frames with FPS limiting and error recovery."""
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
                    sleep_time = frame_interval - time_since_last - 0.001
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue
            
            # Read frame with error handling
            try:
                frame = self.read()
                if frame is not None:
                    frame_count += 1
                    last_frame_time = time.monotonic()
                    yield frame
            except CameraReadError as e:
                logger.error(f"Stream error: {e}")
                break


def list_resolutions() -> dict:
    """List available resolution presets."""
    return RESOLUTIONS.copy()
