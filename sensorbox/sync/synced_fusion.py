"""Simple synchronized multi-sensor fusion."""

from typing import Optional, List, Dict, Generator
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue
import time

from ..drivers.csi_camera import CSICamera
from ..drivers.rplidar import RPLidarSensor
from ..core.frame import SensorFrame


@dataclass
class SyncedFrame:
    """Frame set with common timestamp."""
    timestamp: float
    wall_time: datetime
    cameras: Dict[int, SensorFrame] = field(default_factory=dict)
    lidar: Optional[SensorFrame] = None
    
    def camera(self, cam_id: int) -> Optional[SensorFrame]:
        return self.cameras.get(cam_id)
    
    @property
    def has_lidar(self) -> bool:
        return self.lidar is not None


class SyncedSensorFusion:
    """Simple synchronized multi-sensor capture."""
    
    def __init__(
        self,
        camera_ids: Optional[List[int]] = None,
        lidar_port: Optional[str] = None,
        camera_width: int = 1280,
        camera_height: int = 720,
        camera_fps: int = 30,
    ):
        self._camera_ids = camera_ids or []
        self._lidar_port = lidar_port
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._camera_fps = camera_fps
        
        self._cameras: Dict[int, CSICamera] = {}
        self._lidar: Optional[RPLidarSensor] = None
        self._lidar_queue: queue.Queue = queue.Queue(maxsize=5)
        self._lidar_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._connected = False
        self._start_time: Optional[float] = None
    
    def connect(self) -> None:
        if self._connected:
            return
        
        self._start_time = time.monotonic()
        self._stop_event.clear()
        
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
            
            self._lidar_thread = threading.Thread(target=self._lidar_worker, daemon=True)
            self._lidar_thread.start()
        
        self._connected = True
    
    def disconnect(self) -> None:
        self._stop_event.set()
        
        if self._lidar_thread:
            self._lidar_thread.join(timeout=2.0)
        
        for cam in self._cameras.values():
            cam.disconnect()
        self._cameras.clear()
        
        if self._lidar:
            self._lidar.disconnect()
            self._lidar = None
        
        self._connected = False
    
    def _lidar_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._lidar.read()
                if frame:
                    try:
                        self._lidar_queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            self._lidar_queue.get_nowait()
                            self._lidar_queue.put_nowait(frame)
                        except:
                            pass
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"LIDAR error: {e}")
                break
    
    def _get_lidar(self) -> Optional[SensorFrame]:
        try:
            return self._lidar_queue.get_nowait()
        except queue.Empty:
            return None
    
    def read(self) -> SyncedFrame:
        if not self._connected:
            raise RuntimeError("Not connected")
        
        ts = time.monotonic() - self._start_time
        wall = datetime.now()
        
        cameras = {}
        for cam_id, cam in self._cameras.items():
            frame = cam.read()
            if frame:
                cameras[cam_id] = frame
        
        lidar = self._get_lidar()
        
        return SyncedFrame(timestamp=ts, wall_time=wall, cameras=cameras, lidar=lidar)
    
    def stream(
        self,
        duration: Optional[float] = None,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
    ) -> Generator[SyncedFrame, None, None]:
        if not self._connected:
            raise RuntimeError("Not connected")
        
        start = time.monotonic()
        count = 0
        interval = 1.0 / target_fps if target_fps else 0.033
        last = 0.0
        
        while True:
            if duration and (time.monotonic() - start) >= duration:
                break
            if max_frames and count >= max_frames:
                break
            
            now = time.monotonic()
            if now - last < interval:
                time.sleep(0.005)
                continue
            
            frame = self.read()
            count += 1
            last = now
            yield frame
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.disconnect()
