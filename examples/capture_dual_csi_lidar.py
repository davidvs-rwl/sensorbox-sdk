"""
Synchronized Dual CSI Camera + RPLIDAR Capture

Captures images from CAM0 and CAM1 along with RPLIDAR scan data.
Saves synchronized data with timestamps.

Usage:
    python examples/capture_dual_csi_lidar.py -o ./captures -n 10 --interval 1.0
"""

import argparse
import cv2
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional
import threading


@dataclass
class LidarPoint:
    angle: float
    distance: float
    quality: int


@dataclass
class SensorFrame:
    """Single synchronized capture from all sensors."""
    timestamp: str
    timestamp_ms: int
    cam0_path: Optional[str] = None
    cam1_path: Optional[str] = None
    lidar_path: Optional[str] = None
    lidar_points: int = 0


class DualCSICapture:
    """Dual CSI camera capture using GStreamer."""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam0 = None
        self.cam1 = None
    
    def _gstreamer_pipeline(self, sensor_id: int) -> str:
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate={self.fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink drop=1"
        )
    
    def start(self) -> bool:
        """Initialize both cameras."""
        try:
            self.cam0 = cv2.VideoCapture(self._gstreamer_pipeline(0), cv2.CAP_GSTREAMER)
            self.cam1 = cv2.VideoCapture(self._gstreamer_pipeline(1), cv2.CAP_GSTREAMER)
            
            if not self.cam0.isOpened():
                print("❌ Failed to open CAM0")
                return False
            print("✓ CAM0 initialized")
            
            if not self.cam1.isOpened():
                print("⚠ CAM1 not available (single camera mode)")
                self.cam1 = None
            else:
                print("✓ CAM1 initialized")
            
            # Warm up
            for _ in range(5):
                self.cam0.read()
                if self.cam1:
                    self.cam1.read()
            
            return True
            
        except Exception as e:
            print(f"❌ Camera init error: {e}")
            return False
    
    def capture(self) -> tuple:
        """Capture frames from both cameras."""
        frame0 = None
        frame1 = None
        
        if self.cam0:
            ret, frame0 = self.cam0.read()
            if not ret:
                frame0 = None
        
        if self.cam1:
            ret, frame1 = self.cam1.read()
            if not ret:
                frame1 = None
        
        return frame0, frame1
    
    def stop(self):
        """Release cameras."""
        if self.cam0:
            self.cam0.release()
        if self.cam1:
            self.cam1.release()


class RPLidarCapture:
    """RPLIDAR A1 capture."""
    
    def __init__(self, port: str = "/dev/ttyUSB0"):
        self.port = port
        self.lidar = None
        self._scan_data: List[LidarPoint] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
    
    def start(self) -> bool:
        """Initialize LIDAR."""
        try:
            from rplidar import RPLidar
            self.lidar = RPLidar(self.port)
            self.lidar.clear_input()
            
            info = self.lidar.get_info()
            print(f"✓ RPLIDAR connected: {info.get('model', 'Unknown')}")
            
            # Start background scanning
            self._running = True
            self._thread = threading.Thread(target=self._scan_loop, daemon=True)
            self._thread.start()
            
            # Wait for first scan
            time.sleep(1.0)
            return True
            
        except Exception as e:
            print(f"❌ RPLIDAR error: {e}")
            return False
    
    def _scan_loop(self):
        """Background scan loop."""
        try:
            for scan in self.lidar.iter_scans():
                if not self._running:
                    break
                
                points = [
                    LidarPoint(angle=m[1], distance=m[2], quality=m[0])
                    for m in scan if m[2] > 0
                ]
                
                with self._lock:
                    self._scan_data = points
                    
        except Exception as e:
            if self._running:
                print(f"Scan error: {e}")
    
    def capture(self) -> List[LidarPoint]:
        """Get current scan data."""
        with self._lock:
            return self._scan_data.copy()
    
    def stop(self):
        """Stop LIDAR."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()
            except:
                pass


def save_lidar_csv(points: List[LidarPoint], path: Path):
    """Save LIDAR scan to CSV."""
    with open(path, 'w') as f:
        f.write("angle,distance,quality\n")
        for p in points:
            f.write(f"{p.angle},{p.distance},{p.quality}\n")


def main():
    parser = argparse.ArgumentParser(description="Dual CSI + RPLIDAR synchronized capture")
    parser.add_argument("-o", "--output", default="./captures", help="Output directory")
    parser.add_argument("-n", "--num-captures", type=int, default=10, help="Number of captures")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval between captures (seconds)")
    parser.add_argument("--lidar-port", default="/dev/ttyUSB0", help="RPLIDAR serial port")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--no-lidar", action="store_true", help="Skip RPLIDAR")
    parser.add_argument("--no-cam1", action="store_true", help="Skip CAM1")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    timestamp_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DUAL CSI + RPLIDAR CAPTURE")
    print("=" * 60)
    print(f"Output: {timestamp_dir}")
    print(f"Captures: {args.num_captures}")
    print(f"Interval: {args.interval}s")
    print()
    
    # Initialize sensors
    cameras = DualCSICapture(width=args.width, height=args.height)
    if not cameras.start():
        print("Failed to initialize cameras")
        return
    
    lidar = None
    if not args.no_lidar:
        lidar = RPLidarCapture(port=args.lidar_port)
        if not lidar.start():
            print("⚠ Continuing without LIDAR")
            lidar = None
    
    # Capture loop
    frames: List[SensorFrame] = []
    
    print()
    print("Starting captures...")
    print()
    
    try:
        for i in range(args.num_captures):
            ts = datetime.now()
            ts_str = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            ts_ms = int(ts.timestamp() * 1000)
            
            frame = SensorFrame(timestamp=ts_str, timestamp_ms=ts_ms)
            
            # Capture cameras
            cam0_frame, cam1_frame = cameras.capture()
            
            if cam0_frame is not None:
                cam0_path = timestamp_dir / f"cam0_{ts_str}.jpg"
                cv2.imwrite(str(cam0_path), cam0_frame)
                frame.cam0_path = cam0_path.name
                print(f"  [{i+1}/{args.num_captures}] CAM0: ✓", end="")
            else:
                print(f"  [{i+1}/{args.num_captures}] CAM0: ✗", end="")
            
            if cam1_frame is not None and not args.no_cam1:
                cam1_path = timestamp_dir / f"cam1_{ts_str}.jpg"
                cv2.imwrite(str(cam1_path), cam1_frame)
                frame.cam1_path = cam1_path.name
                print(f" | CAM1: ✓", end="")
            else:
                print(f" | CAM1: -", end="")
            
            # Capture LIDAR
            if lidar:
                scan_data = lidar.capture()
                if scan_data:
                    lidar_path = timestamp_dir / f"lidar_{ts_str}.csv"
                    save_lidar_csv(scan_data, lidar_path)
                    frame.lidar_path = lidar_path.name
                    frame.lidar_points = len(scan_data)
                    print(f" | LIDAR: {len(scan_data)} pts")
                else:
                    print(f" | LIDAR: ✗")
            else:
                print()
            
            frames.append(frame)
            
            if i < args.num_captures - 1:
                time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n\nCapture interrupted")
    
    finally:
        cameras.stop()
        if lidar:
            lidar.stop()
    
    # Save manifest
    manifest = {
        "capture_session": timestamp_dir.name,
        "total_frames": len(frames),
        "settings": {
            "width": args.width,
            "height": args.height,
            "interval": args.interval,
            "lidar_port": args.lidar_port
        },
        "frames": [asdict(f) for f in frames]
    }
    
    manifest_path = timestamp_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print()
    print("=" * 60)
    print("CAPTURE COMPLETE")
    print("=" * 60)
    print(f"Saved {len(frames)} frames to: {timestamp_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
