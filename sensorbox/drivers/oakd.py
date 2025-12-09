"""OAK-D Pro Driver for RGB, Depth, and IMU data (DepthAI v3)."""

from typing import Optional, Generator, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import time
import numpy as np
import depthai as dai

from ..core.sensor import Sensor
from ..core.frame import SensorFrame, SensorMetadata, SensorType, FrameType


@dataclass
class OakDFrame:
    """Combined frame from OAK-D Pro."""
    timestamp: float
    wall_time: datetime
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    imu: Optional[Dict[str, Any]] = None


class OakDProError(Exception):
    pass


class OakDPro(Sensor):
    """
    OAK-D Pro driver for RGB, stereo depth, and IMU (DepthAI v3).
    """
    
    def __init__(
        self,
        rgb_size: tuple = (1280, 720),
        depth_enabled: bool = True,
        imu_enabled: bool = True,
        fps: float = 30.0,
        depth_preset: str = "DEFAULT",
        depth_size: tuple = (640, 400),
        median_filter: int = 0,
        confidence_threshold: int = 200,
        enable_ir: bool = False,
        ir_brightness: int = 800,
    ):
        self._rgb_size = rgb_size
        self._depth_enabled = depth_enabled
        self._imu_enabled = imu_enabled
        self._fps = fps
        self._depth_preset = depth_preset
        self._depth_size = depth_size
        self._median_filter = median_filter
        self._confidence_threshold = confidence_threshold
        self._enable_ir = enable_ir
        self._ir_brightness = ir_brightness
        
        self._pipeline: Optional[dai.Pipeline] = None
        self._rgb_queue = None
        self._depth_queue = None
        self._imu_queue = None
        
        super().__init__(
            sensor_id="oakd_pro",
            sensor_type=SensorType.CAMERA,
        )
    
    def _get_depth_preset(self):
        """Get stereo depth preset."""
        presets = {
            "DEFAULT": dai.node.StereoDepth.PresetMode.DEFAULT,
            "FAST_ACCURACY": dai.node.StereoDepth.PresetMode.FAST_ACCURACY,
            "FAST_DENSITY": dai.node.StereoDepth.PresetMode.FAST_DENSITY,
            "HIGH_DETAIL": dai.node.StereoDepth.PresetMode.HIGH_DETAIL,
            "ROBOTICS": dai.node.StereoDepth.PresetMode.ROBOTICS,
        }
        return presets.get(self._depth_preset, presets["DEFAULT"])
    
    def _get_median_filter(self):
        """Get median filter setting."""
        filters = {
            0: dai.MedianFilter.MEDIAN_OFF,
            3: dai.MedianFilter.KERNEL_3x3,
            5: dai.MedianFilter.KERNEL_5x5,
            7: dai.MedianFilter.KERNEL_7x7,
        }
        return filters.get(self._median_filter, dai.MedianFilter.MEDIAN_OFF)
    
    def connect(self) -> None:
        if self._connected:
            return
        
        try:
            self._pipeline = dai.Pipeline()
            
            # RGB Camera
            cam = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            rgb_out = cam.requestOutput(self._rgb_size, dai.ImgFrame.Type.BGR888p, fps=self._fps)
            self._rgb_queue = rgb_out.createOutputQueue()
            
            # Stereo Depth
            if self._depth_enabled:
                stereo = self._pipeline.create(dai.node.StereoDepth).build(
                    autoCreateCameras=True,
                    presetMode=self._get_depth_preset(),
                    size=self._depth_size,
                    fps=self._fps,
                )
                
                stereo.initialConfig.setMedianFilter(self._get_median_filter())
                stereo.initialConfig.setConfidenceThreshold(self._confidence_threshold)
                
                self._depth_queue = stereo.depth.createOutputQueue()
            
            # IMU
            if self._imu_enabled:
                imu = self._pipeline.create(dai.node.IMU)
                imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
                imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 100)
                imu.setBatchReportThreshold(1)
                imu.setMaxBatchReports(10)
                self._imu_queue = imu.out.createOutputQueue()
            
            # Start pipeline
            self._pipeline.start()
            
            # Enable IR projector
            if self._enable_ir and self._depth_enabled:
                try:
                    device = self._pipeline.getDefaultDevice()
                    device.setIrLaserDotProjectorIntensity(self._ir_brightness / 1200.0)
                except Exception as e:
                    print(f"IR projector not available: {e}")
            
            self._metadata = SensorMetadata(
                sensor_id=self._sensor_id,
                sensor_type=SensorType.CAMERA,
                manufacturer="Luxonis",
                model="OAK-D Pro",
                config={
                    "rgb_size": self._rgb_size,
                    "depth_enabled": self._depth_enabled,
                    "depth_size": self._depth_size,
                    "depth_preset": self._depth_preset,
                    "median_filter": self._median_filter,
                    "confidence_threshold": self._confidence_threshold,
                    "imu_enabled": self._imu_enabled,
                    "fps": self._fps,
                    "ir_enabled": self._enable_ir,
                },
            )
            
            self._connected = True
            self._time_offset = None
            
        except Exception as e:
            raise OakDProError(f"Failed to connect to OAK-D Pro: {e}")
    
    def disconnect(self) -> None:
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        self._rgb_queue = None
        self._depth_queue = None
        self._imu_queue = None
        self._connected = False
    
    def read(self) -> Optional[OakDFrame]:
        """Read a frame from OAK-D Pro."""
        if not self._connected:
            raise RuntimeError("OAK-D Pro is not connected")
        
        timestamp, wall_time = self._get_timestamp()
        
        rgb = None
        rgb_msg = self._rgb_queue.tryGet()
        if rgb_msg:
            rgb = rgb_msg.getCvFrame()
        
        depth = None
        if self._depth_queue:
            depth_msg = self._depth_queue.tryGet()
            if depth_msg:
                # Rotate 180° to align with RGB camera orientation
                depth = np.flipud(depth_msg.getFrame())
        
        imu_data = None
        if self._imu_queue:
            imu_msg = self._imu_queue.tryGet()
            if imu_msg:
                packets = imu_msg.packets
                if packets:
                    latest = packets[-1]
                    accel = latest.acceleroMeter
                    gyro = latest.gyroscope
                    imu_data = {
                        "accelerometer": {"x": accel.x, "y": accel.y, "z": accel.z},
                        "gyroscope": {"x": gyro.x, "y": gyro.y, "z": gyro.z},
                    }
        
        if rgb is None:
            return None
        
        return OakDFrame(timestamp=timestamp, wall_time=wall_time, rgb=rgb, depth=depth, imu=imu_data)
    
    def stream(
        self,
        duration: Optional[float] = None,
        max_frames: Optional[int] = None,
        target_fps: Optional[float] = None,
    ) -> Generator[OakDFrame, None, None]:
        """Stream frames from OAK-D Pro."""
        if not self._connected:
            raise RuntimeError("OAK-D Pro is not connected")
        
        start_time = time.monotonic()
        frame_count = 0
        frame_interval = 1.0 / target_fps if target_fps else 0.0
        last_frame_time = 0.0
        
        while True:
            if duration and (time.monotonic() - start_time) >= duration:
                break
            if max_frames and frame_count >= max_frames:
                break
            
            if frame_interval:
                current = time.monotonic()
                if current - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
            
            frame = self.read()
            if frame and frame.rgb is not None:
                frame_count += 1
                last_frame_time = time.monotonic()
                yield frame


def discover_oakd_devices():
    """Find all connected OAK-D devices."""
    devices = dai.Device.getAllAvailableDevices()
    return [{"name": dev.name, "state": dev.state.name} for dev in devices]
