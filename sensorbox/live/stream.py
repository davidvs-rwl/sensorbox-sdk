"""Live sensor streaming server."""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue
import time
import numpy as np

from ..drivers.oakd import OakDPro
from ..drivers.csi_camera import CSICamera
from ..core.pointcloud import depth_to_pointcloud


@dataclass
class LiveFrame:
    """A live frame from sensors."""
    timestamp: float
    wall_time: datetime
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    depth_colorized: Optional[np.ndarray] = None
    pointcloud: Optional[np.ndarray] = None
    imu: Optional[Dict[str, Any]] = None
    csi_frames: Dict[int, np.ndarray] = field(default_factory=dict)
    fps: float = 0.0
    depth_valid_pct: float = 0.0


class LiveStreamManager:
    """Manages live sensor streaming with background capture."""
    
    def __init__(
        self,
        oakd: bool = True,
        csi_cameras: list = None,
        oakd_fps: float = 15.0,
        enable_pointcloud: bool = True,
        pointcloud_subsample: int = 4,
        depth_quality: str = "balanced",
        use_ir: bool = False,
    ):
        self._oakd_enabled = oakd
        self._csi_cameras = csi_cameras or []
        self._oakd_fps = oakd_fps
        self._enable_pointcloud = enable_pointcloud
        self._pc_subsample = pointcloud_subsample
        self._depth_quality = depth_quality
        self._use_ir = use_ir
        
        self._oakd: Optional[OakDPro] = None
        self._csi: Dict[int, CSICamera] = {}
        
        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
        self._frame_count = 0
        self._start_time = 0.0
        self._last_fps_time = 0.0
        self._fps_frame_count = 0
        self._current_fps = 0.0
    
    def _get_depth_settings(self) -> dict:
        """Get depth settings based on quality preset."""
        presets = {
            "fast": {
                "depth_preset": "DEFAULT",
                "depth_size": (640, 400),
                "median_filter": 0,
                "confidence_threshold": 50,
            },
            "balanced": {
                "depth_preset": "DEFAULT",
                "depth_size": (640, 400),
                "median_filter": 0,
                "confidence_threshold": 100,
            },
            "high": {
                "depth_preset": "DEFAULT",
                "depth_size": (640, 400),
                "median_filter": 0,
                "confidence_threshold": 150,
            },
        }
        return presets.get(self._depth_quality, presets["balanced"])
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def fps(self) -> float:
        return self._current_fps
    
    def start(self) -> None:
        """Start live streaming."""
        if self._running:
            return
        
        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._last_fps_time = self._start_time
        
        if self._oakd_enabled:
            depth_settings = self._get_depth_settings()
            self._oakd = OakDPro(
                rgb_size=(1280, 720),
                depth_enabled=True,
                imu_enabled=True,
                fps=self._oakd_fps,
                enable_ir=self._use_ir,
                ir_brightness=800 if self._use_ir else 0,
                **depth_settings,
            )
            self._oakd.connect()
        
        for cam_id in self._csi_cameras:
            cam = CSICamera(sensor_id=cam_id)
            cam.connect()
            self._csi[cam_id] = cam
        
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        self._running = True
    
    def stop(self) -> None:
        """Stop live streaming."""
        self._stop_event.set()
        
        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        
        if self._oakd:
            self._oakd.disconnect()
            self._oakd = None
        
        for cam in self._csi.values():
            cam.disconnect()
        self._csi.clear()
        
        self._running = False
    
    def get_latest(self) -> Optional[LiveFrame]:
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_frame(self, timeout: float = 1.0) -> Optional[LiveFrame]:
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._capture_frame()
                if frame:
                    self._frame_count += 1
                    self._fps_frame_count += 1
                    
                    now = time.monotonic()
                    if now - self._last_fps_time >= 1.0:
                        self._current_fps = self._fps_frame_count / (now - self._last_fps_time)
                        self._fps_frame_count = 0
                        self._last_fps_time = now
                    
                    frame.fps = self._current_fps
                    
                    try:
                        self._frame_queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            self._frame_queue.get_nowait()
                            self._frame_queue.put_nowait(frame)
                        except:
                            pass
                
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
    
    def _capture_frame(self) -> Optional[LiveFrame]:
        timestamp = time.monotonic() - self._start_time
        wall_time = datetime.now()
        
        rgb = None
        depth = None
        depth_colorized = None
        pointcloud = None
        imu = None
        csi_frames = {}
        depth_valid_pct = 0.0
        
        if self._oakd:
            oakd_frame = self._oakd.read()
            if oakd_frame and oakd_frame.rgb is not None:
                rgb = oakd_frame.rgb
                
                if oakd_frame.depth is not None:
                    # Flip depth horizontally to match RGB camera orientation
                    depth = oakd_frame.depth
                    depth_colorized = self._colorize_depth(depth)
                    
                    valid = depth > 0
                    depth_valid_pct = (valid.sum() / valid.size) * 100
                    
                    if self._enable_pointcloud:
                        pointcloud = depth_to_pointcloud(
                            depth, 
                            subsample=self._pc_subsample,
                            max_depth=5000,
                        )
                
                if oakd_frame.imu:
                    imu = oakd_frame.imu
        
        for cam_id, cam in self._csi.items():
            frame = cam.read()
            if frame:
                csi_frames[cam_id] = frame.data
        
        if rgb is None and not csi_frames:
            return None
        
        return LiveFrame(
            timestamp=timestamp,
            wall_time=wall_time,
            rgb=rgb,
            depth=depth,
            depth_colorized=depth_colorized,
            pointcloud=pointcloud,
            imu=imu,
            csi_frames=csi_frames,
            fps=self._current_fps,
            depth_valid_pct=depth_valid_pct,
        )
    
    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        import cv2
        
        valid = depth > 0
        if not np.any(valid):
            return np.zeros((*depth.shape, 3), dtype=np.uint8)
        
        depth_norm = np.zeros_like(depth, dtype=np.float32)
        depth_norm[valid] = depth[valid]
        
        max_val = np.percentile(depth_norm[valid], 95)
        depth_norm = np.clip(depth_norm / max_val * 255, 0, 255).astype(np.uint8)
        
        colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        
        return colored
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, *args):
        self.stop()
