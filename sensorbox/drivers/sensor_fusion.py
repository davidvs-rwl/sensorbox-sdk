"""
Multi-sensor streaming for synchronized camera and LIDAR capture.
"""

from typing import Optional, List, Dict, Generator
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue
import time

from .csi_camera import CSICamera
from .rplidar import RPLidarSensor
from ..core.frame import SensorFrame


@dataclass
class FusedFrame:
    """A synchronized frame from multiple sensors."""
    timestamp: float
    wall_time: datetime
    cameras: Dict[int, SensorFrame] = field(default_factory=dict)
    lidar: Optional[SensorFrame] = None
    
    @property
    def has_lidar(self) -> bool:
        return self.lidar is not None
    
    @property
    def num_cameras(self) -> int:
        return len(self.cameras)
    
    def camera(self, sensor_id: int) -> Optional[SensorFrame]:
        return self.cameras.get(sensor_id)


class SensorFusion:
    """
    Stream from multiple cameras and LIDAR simultaneously.
    
    Example:
        fusion = SensorFusion(
            camera_ids=[0, 1],
            lidar_port='/dev/ttyUSB0',
        )
        
        with fusion:
            for frame in fusion.stream(duration=10.0):
                # Access camera frames
                cam0 = frame.camera(0)
                cam1 = frame.camera(1)
                
                # Access LIDAR scan
                if frame.has_lidar:
                    scan = frame.lidar.data
                
                print(f"Cameras: {frame.num_cameras}, LIDAR: {frame.has_lidar}")
    """
    
    def __init__(
        self,
        camera_ids: Optional[List[int]] = None,
        lidar_port: Optional[str] = None,
        camera_width: int = 1280,
        camera_height: int = 720,
        camera_fps: int = 30,
    ):
        """
        Initialize sensor fusion.
        
        Args:
            camera_ids: List of CSI camera IDs (e.g., [0, 1])
            lidar_port: Serial port for RPLIDAR (e.g., '/dev/ttyUSB0')
            camera_width: Camera capture width
            camera_height: Camera capture height
            camera_fps: Camera capture FPS
        """
        self._camera_ids = camera_ids or []
        self._lidar_port = lidar_port
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._camera_fps = camera_fps
        
        self._cameras: Dict[int, CSICamera] = {}
        self._lidar: Optional[RPLidarSensor] = None
        self._connected = False
        
        # Threading for async LIDAR capture
        self._lidar_thread: Optional[threading.Thread] = None
        self._lidar_queue: queue.Queue = queue.Queue(maxsize=10)
        self._stop_event = threading.Event()
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def camera_ids(self) -> List[int]:
        return self._camera_ids.copy()
    
    @property
    def has_lidar(self) -> bool:
        return self._lidar_port is not None
    
    def connect(self) -> None:
        """Connect to all sensors."""
        if self._connected:
            return
        
        # Connect cameras
        for cam_id in self._camera_ids:
            cam = CSICamera(
                sensor_id=cam_id,
                width=self._camera_width,
                height=self._camera_height,
                fps=self._camera_fps,
            )
            cam.connect()
            self._cameras[cam_id] = cam
        
        # Connect LIDAR
        if self._lidar_port:
            self._lidar = RPLidarSensor(self._lidar_port)
            self._lidar.connect()
            
            # Start LIDAR thread
            self._stop_event.clear()
            self._lidar_thread = threading.Thread(target=self._lidar_worker, daemon=True)
            self._lidar_thread.start()
        
        self._connected = True
    
    def disconnect(self) -> None:
        """Disconnect all sensors."""
        # Stop LIDAR thread
        self._stop_event.set()
        if self._lidar_thread:
            self._lidar_thread.join(timeout=2.0)
            self._lidar_thread = None
        
        # Disconnect cameras
        for cam in self._cameras.values():
            cam.disconnect()
        self._cameras.clear()
        
        # Disconnect LIDAR
        if self._lidar:
            self._lidar.disconnect()
            self._lidar = None
        
        # Clear queue
        while not self._lidar_queue.empty():
            try:
                self._lidar_queue.get_nowait()
            except queue.Empty:
                break
        
        self._connected = False
    
    def _lidar_worker(self) -> None:
        """Background thread for LIDAR capture."""
        if not self._lidar:
            return
        
        try:
            for frame in self._lidar.stream():
                if self._stop_event.is_set():
                    break
                
                # Put frame in queue, drop old if full
                try:
                    self._lidar_queue.put_nowait(frame)
                except queue.Full:
                    try:
                        self._lidar_queue.get_nowait()
                        self._lidar_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
        except Exception as e:
            print(f"LIDAR worker error: {e}")
    
    def _get_latest_lidar(self) -> Optional[SensorFrame]:
        """Get the most recent LIDAR frame (non-blocking)."""
        latest = None
        while True:
            try:
                latest = self._lidar_queue.get_nowait()
            except queue.Empty:
                break
        return latest
    
    def read(self) -> FusedFrame:
        """Read one frame from all sensors."""
        if not self._connected:
            raise RuntimeError("SensorFusion is not connected")
        
        timestamp = time.monotonic()
        wall_time = datetime.now()
        
        # Read from cameras
        cameras = {}
        for cam_id, cam in self._cameras.items():
            frame = cam.read()
            if frame:
                cameras[cam_id] = frame
        
        # Get latest LIDAR
        lidar_frame = self._get_latest_lidar()
        
        return FusedFrame(
            timestamp=timestamp,
            wall_time=wall_time,
            cameras=cameras,
            lidar=lidar_frame,
        )
    
    def stream(
        self,
        duration: Optional[float] = None,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
    ) -> Generator[FusedFrame, None, None]:
        """
        Stream fused frames from all sensors.
        
        Args:
            duration: Maximum duration in seconds
            max_frames: Maximum number of frames
            target_fps: Target frame rate
        
        Yields:
            FusedFrame containing data from all sensors
        """
        if not self._connected:
            raise RuntimeError("SensorFusion is not connected")
        
        start_time = time.monotonic()
        frame_count = 0
        frame_interval = 1.0 / target_fps if target_fps else None
        last_frame_time = 0.0
        
        while True:
            # Check duration
            if duration is not None:
                if time.monotonic() - start_time >= duration:
                    break
            
            # Check frame limit
            if max_frames is not None and frame_count >= max_frames:
                break
            
            # FPS throttling
            if frame_interval:
                current = time.monotonic()
                if current - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
            
            # Read all sensors
            fused = self.read()
            frame_count += 1
            last_frame_time = time.monotonic()
            
            yield fused
    
    def __enter__(self) -> "SensorFusion":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
