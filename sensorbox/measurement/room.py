"""Room measurement combining RPLIDAR and OAK-D data."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import time

from .wall_detector import WallDetector, Wall
from .plane_detector import PlaneDetector, Plane


@dataclass
class RoomDimensions:
    """Measured room dimensions."""
    length: float  # meters
    width: float   # meters
    height: float  # meters
    area: float    # square meters
    volume: float  # cubic meters
    
    # Confidence metrics
    wall_confidence: float  # 0-1
    height_confidence: float  # 0-1
    
    # Raw data
    walls: List[Wall] = None
    floor_plane: Plane = None
    ceiling_plane: Plane = None
    
    def __str__(self):
        return (
            f"Room Dimensions:\n"
            f"  Length: {self.length:.2f}m\n"
            f"  Width:  {self.width:.2f}m\n"
            f"  Height: {self.height:.2f}m\n"
            f"  Area:   {self.area:.2f}m²\n"
            f"  Volume: {self.volume:.2f}m³\n"
            f"  Confidence: walls={self.wall_confidence:.0%}, height={self.height_confidence:.0%}"
        )


class RoomMeasurement:
    """
    Measure room dimensions using RPLIDAR (2D floor plan) and OAK-D (height).
    """
    
    def __init__(
        self,
        lidar_port: str = "/dev/ttyUSB0",
        camera_height: float = 0.5,  # Height of OAK-D from floor
    ):
        self.lidar_port = lidar_port
        self.camera_height = camera_height
        
        self.wall_detector = WallDetector()
        self.plane_detector = PlaneDetector()
        
        self._lidar = None
        self._oakd = None
        
        # Captured data
        self._lidar_points: Optional[np.ndarray] = None
        self._depth_points: Optional[np.ndarray] = None
        self._rgb_image: Optional[np.ndarray] = None
        self._walls: List[Wall] = []
        self._planes: List[Plane] = []
    
    def connect(self):
        """Connect to sensors."""
        from ..drivers.rplidar import RPLidarSensor
        from ..drivers.oakd import OakDPro
        
        print("Connecting to RPLIDAR...")
        self._lidar = RPLidarSensor(port=self.lidar_port)
        self._lidar.connect()
        
        print("Connecting to OAK-D Pro...")
        self._oakd = OakDPro(
            rgb_size=(1280, 720),
            depth_enabled=True,
            confidence_threshold=50,
            enable_ir=True,
            ir_brightness=800,
        )
        self._oakd.connect()
        time.sleep(0.5)
        
        print("Sensors connected!")
    
    def disconnect(self):
        """Disconnect sensors."""
        if self._lidar:
            self._lidar.disconnect()
            self._lidar = None
        if self._oakd:
            self._oakd.disconnect()
            self._oakd = None
    
    def capture_lidar_scan(
        self, 
        num_scans: int = 5,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Capture RPLIDAR scans and convert to 2D points."""
        if not self._lidar:
            raise RuntimeError("RPLIDAR not connected")
        
        if show_progress:
            print("Capturing RPLIDAR scans...")
        
        all_points = []
        
        for i in range(num_scans):
            # read() returns a SensorFrame with data as (angle, distance, quality)
            frame = self._lidar.read()
            
            if frame is None or frame.data is None:
                continue
            
            scan_data = frame.data  # numpy array: (angle, distance, quality)
            
            if show_progress:
                print(f"  Scan {i+1}/{num_scans}: {len(scan_data)} points")
            
            # Extract angles (degrees) and distances (mm)
            angles = scan_data[:, 0]  # degrees
            distances = scan_data[:, 1] / 1000.0  # mm to meters
            
            # Convert to radians
            angles_rad = np.radians(angles)
            
            # Convert to cartesian
            points = self.wall_detector.scan_to_cartesian(angles_rad, distances)
            all_points.append(points)
        
        if not all_points:
            raise RuntimeError("No valid RPLIDAR scans captured")
        
        # Combine all scans
        self._lidar_points = np.vstack(all_points)
        
        if show_progress:
            print(f"Total RPLIDAR points: {len(self._lidar_points)}")
        
        return self._lidar_points
    
    def capture_depth_for_height(
        self,
        num_frames: int = 10,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Capture OAK-D depth frames for height measurement."""
        points, _ = self.capture_depth_and_rgb(num_frames, show_progress)
        return points
    
    def capture_depth_and_rgb(
        self,
        num_frames: int = 10,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Capture OAK-D depth frames and RGB image."""
        if not self._oakd:
            raise RuntimeError("OAK-D not connected")
        
        if show_progress:
            print("Capturing OAK-D depth frames...")
            print("(Point camera to see both floor and ceiling for best results)")
        
        from ..core.pointcloud import depth_to_pointcloud
        
        all_points = []
        rgb_image = None
        
        for i in range(num_frames * 3):  # Oversample to get good frames
            frame = self._oakd.read()
            
            if frame and frame.depth is not None:
                valid_pct = (frame.depth > 0).sum() / frame.depth.size * 100
                
                if valid_pct > 10:  # Only use frames with enough data
                    points = depth_to_pointcloud(frame.depth, subsample=2, max_depth=10000)
                    all_points.append(points)
                    
                    # Capture RGB from the best frame
                    if frame.rgb is not None and (rgb_image is None or valid_pct > 30):
                        rgb_image = frame.rgb.copy()
                    
                    if show_progress and len(all_points) <= num_frames:
                        print(f"  Frame {len(all_points)}/{num_frames}: {len(points)} points ({valid_pct:.1f}% valid)")
                
                if len(all_points) >= num_frames:
                    break
            
            time.sleep(0.05)
        
        if not all_points:
            raise RuntimeError("No valid depth frames captured")
        
        # Combine all points
        self._depth_points = np.vstack(all_points)
        self._rgb_image = rgb_image
        
        if show_progress:
            print(f"Total depth points: {len(self._depth_points)}")
        
        return self._depth_points, self._rgb_image
    
    def detect_walls(self, show_progress: bool = True) -> List[Wall]:
        """Detect walls from RPLIDAR data."""
        if self._lidar_points is None:
            raise RuntimeError("No RPLIDAR data. Call capture_lidar_scan() first.")
        
        if show_progress:
            print("Detecting walls...")
        
        self._walls = self.wall_detector.detect_walls(self._lidar_points)
        
        if show_progress:
            print(f"Found {len(self._walls)} walls:")
            for i, wall in enumerate(self._walls):
                print(f"  Wall {i+1}: {wall.length:.2f}m at {np.degrees(wall.angle):.1f}°, "
                      f"distance={wall.distance:.2f}m, {wall.num_points} points")
        
        return self._walls
    
    def detect_planes(self, show_progress: bool = True) -> List[Plane]:
        """Detect planes from OAK-D data."""
        if self._depth_points is None:
            raise RuntimeError("No depth data. Call capture_depth_for_height() first.")
        
        if show_progress:
            print("Detecting planes...")
        
        self._planes = self.plane_detector.detect_planes(self._depth_points)
        
        if show_progress:
            print(f"Found {len(self._planes)} planes:")
            for i, plane in enumerate(self._planes):
                plane_type = "horizontal" if plane.is_horizontal else "vertical"
                print(f"  Plane {i+1}: {plane_type}, height={plane.height:.2f}m, "
                      f"{plane.num_points} points")
        
        return self._planes
    
    def compute_dimensions(self, show_progress: bool = True) -> RoomDimensions:
        """Compute room dimensions from captured data."""
        # Detect features if not already done
        if not self._walls and self._lidar_points is not None:
            self.detect_walls(show_progress)
        
        if not self._planes and self._depth_points is not None:
            self.detect_planes(show_progress)
        
        # Get floor plan dimensions from RPLIDAR
        if self._walls:
            length, width = self.wall_detector.estimate_room_dimensions(self._walls)
            wall_confidence = min(1.0, len(self._walls) / 4)  # 4 walls = 100%
        else:
            # Fallback: use RPLIDAR point extent
            if self._lidar_points is not None:
                x_range = self._lidar_points[:, 0].max() - self._lidar_points[:, 0].min()
                y_range = self._lidar_points[:, 1].max() - self._lidar_points[:, 1].min()
                length, width = max(x_range, y_range), min(x_range, y_range)
                wall_confidence = 0.3
            else:
                length, width = 0, 0
                wall_confidence = 0
        
        # Get height from OAK-D
        if self._depth_points is not None:
            height = self.plane_detector.estimate_room_height(self._depth_points, self._planes)
            
            floor, ceiling = self.plane_detector.find_floor_ceiling(self._planes)
            height_confidence = 0.8 if (floor and ceiling) else 0.5
        else:
            height = 0
            height_confidence = 0
        
        # Build result
        floor, ceiling = (None, None)
        if self._planes:
            floor, ceiling = self.plane_detector.find_floor_ceiling(self._planes)
        
        dimensions = RoomDimensions(
            length=length,
            width=width,
            height=height,
            area=length * width,
            volume=length * width * height,
            wall_confidence=wall_confidence,
            height_confidence=height_confidence,
            walls=self._walls,
            floor_plane=floor,
            ceiling_plane=ceiling,
        )
        
        if show_progress:
            print("\n" + str(dimensions))
        
        return dimensions
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.disconnect()
