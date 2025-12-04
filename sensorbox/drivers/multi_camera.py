"""Multi-camera streaming for NVIDIA Jetson."""

from typing import Optional, List, Dict, Generator
from dataclasses import dataclass
from datetime import datetime
import threading
import queue
import time

from .csi_camera import CSICamera
from ..core.frame import SensorFrame


@dataclass
class MultiFrame:
    """A synchronized frame from multiple cameras."""
    timestamp: float
    wall_time: datetime
    frames: Dict[int, SensorFrame]  # sensor_id -> frame
    
    def __getitem__(self, sensor_id: int) -> Optional[SensorFrame]:
        return self.frames.get(sensor_id)
    
    def __len__(self) -> int:
        return len(self.frames)


class MultiCamera:
    """
    Stream from multiple CSI cameras simultaneously.
    
    Example:
        with MultiCamera([0, 1]) as cams:
            for multi_frame in cams.stream(duration=5.0):
                frame0 = multi_frame[0]  # CAM0
                frame1 = multi_frame[1]  # CAM1
                print(f"CAM0: {frame0.shape}, CAM1: {frame1.shape}")
    """
    
    def __init__(
        self,
        sensor_ids: List[int],
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        """
        Initialize multi-camera capture.
        
        Args:
            sensor_ids: List of CSI sensor IDs (e.g., [0, 1])
            width: Capture width for all cameras
            height: Capture height for all cameras
            fps: Capture FPS for all cameras
        """
        self._sensor_ids = sensor_ids
        self._width = width
        self._height = height
        self._fps = fps
        self._cameras: Dict[int, CSICamera] = {}
        self._connected = False
    
    @property
    def sensor_ids(self) -> List[int]:
        return self._sensor_ids.copy()
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    def connect(self) -> None:
        """Connect to all cameras."""
        if self._connected:
            return
        
        for sid in self._sensor_ids:
            cam = CSICamera(
                sensor_id=sid,
                width=self._width,
                height=self._height,
                fps=self._fps,
            )
            cam.connect()
            self._cameras[sid] = cam
        
        self._connected = True
    
    def disconnect(self) -> None:
        """Disconnect all cameras."""
        for cam in self._cameras.values():
            cam.disconnect()
        self._cameras.clear()
        self._connected = False
    
    def read(self) -> MultiFrame:
        """Read one frame from each camera."""
        if not self._connected:
            raise RuntimeError("MultiCamera is not connected")
        
        frames = {}
        for sid, cam in self._cameras.items():
            frame = cam.read()
            if frame:
                frames[sid] = frame
        
        # Use timestamp from first frame
        ts = list(frames.values())[0].timestamp if frames else 0.0
        wt = list(frames.values())[0].wall_time if frames else datetime.now()
        
        return MultiFrame(timestamp=ts, wall_time=wt, frames=frames)
    
    def stream(
        self,
        duration: Optional[float] = None,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
    ) -> Generator[MultiFrame, None, None]:
        """
        Stream frames from all cameras.
        
        Args:
            duration: Maximum duration in seconds
            max_frames: Maximum number of frame sets
            target_fps: Target frame rate (throttled)
        
        Yields:
            MultiFrame containing one frame from each camera
        """
        if not self._connected:
            raise RuntimeError("MultiCamera is not connected")
        
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
            
            # Read from all cameras
            multi_frame = self.read()
            if multi_frame.frames:
                frame_count += 1
                last_frame_time = time.monotonic()
                yield multi_frame
    
    def __enter__(self) -> "MultiCamera":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
