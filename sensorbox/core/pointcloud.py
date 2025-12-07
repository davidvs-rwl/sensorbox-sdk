"""Point cloud generation from depth data."""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int
    height: int


# Default OAK-D Pro depth camera intrinsics (approximate for 640x400)
OAKD_DEPTH_INTRINSICS = CameraIntrinsics(
    fx=380.0,
    fy=380.0,
    cx=320.0,
    cy=200.0,
    width=640,
    height=400,
)


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: Optional[CameraIntrinsics] = None,
    max_depth: float = 10000.0,  # mm
    subsample: int = 1,
) -> np.ndarray:
    """
    Convert depth image to 3D point cloud.
    
    Args:
        depth: Depth image (H, W) in millimeters
        intrinsics: Camera intrinsic parameters
        max_depth: Maximum depth to include (mm)
        subsample: Subsample factor (1 = all points, 2 = every other, etc.)
    
    Returns:
        Point cloud as (N, 3) array with X, Y, Z in meters
    """
    if intrinsics is None:
        # Auto-detect based on image size
        h, w = depth.shape
        intrinsics = CameraIntrinsics(
            fx=w * 0.6,
            fy=w * 0.6,
            cx=w / 2,
            cy=h / 2,
            width=w,
            height=h,
        )
    
    h, w = depth.shape
    
    # Create pixel coordinate grids
    u = np.arange(0, w, subsample)
    v = np.arange(0, h, subsample)
    u, v = np.meshgrid(u, v)
    
    # Get depth values (subsampled)
    z = depth[::subsample, ::subsample].astype(np.float32)
    
    # Filter invalid depths
    valid = (z > 0) & (z < max_depth)
    
    # Convert to meters
    z = z / 1000.0
    
    # Back-project to 3D
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    
    # Stack and filter
    points = np.stack([x, y, z], axis=-1)
    points = points[valid]
    
    return points


def depth_to_colored_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Optional[CameraIntrinsics] = None,
    max_depth: float = 10000.0,
    subsample: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image to colored 3D point cloud.
    
    Args:
        depth: Depth image (H, W) in millimeters
        rgb: RGB image (H, W, 3) - will be resized to match depth
        intrinsics: Camera intrinsic parameters
        max_depth: Maximum depth to include (mm)
        subsample: Subsample factor
    
    Returns:
        Tuple of (points (N, 3), colors (N, 3))
    """
    import cv2
    
    if intrinsics is None:
        h, w = depth.shape
        intrinsics = CameraIntrinsics(
            fx=w * 0.6,
            fy=w * 0.6,
            cx=w / 2,
            cy=h / 2,
            width=w,
            height=h,
        )
    
    h, w = depth.shape
    
    # Resize RGB to match depth if needed
    if rgb.shape[:2] != (h, w):
        rgb_resized = cv2.resize(rgb, (w, h))
    else:
        rgb_resized = rgb
    
    # Create pixel coordinate grids
    u = np.arange(0, w, subsample)
    v = np.arange(0, h, subsample)
    u, v = np.meshgrid(u, v)
    
    # Get depth and color values (subsampled)
    z = depth[::subsample, ::subsample].astype(np.float32)
    colors = rgb_resized[::subsample, ::subsample]
    
    # Filter invalid depths
    valid = (z > 0) & (z < max_depth)
    
    # Convert depth to meters
    z = z / 1000.0
    
    # Back-project to 3D
    x = (u - intrinsics.cx) * z / intrinsics.fx
    y = (v - intrinsics.cy) * z / intrinsics.fy
    
    # Stack and filter
    points = np.stack([x, y, z], axis=-1)
    points = points[valid]
    colors = colors[valid]
    
    # Normalize colors to 0-1 range
    colors = colors.astype(np.float32) / 255.0
    
    return points, colors


def pointcloud_to_ply(
    points: np.ndarray,
    filepath: str,
    colors: Optional[np.ndarray] = None,
) -> None:
    """
    Save point cloud to PLY file.
    
    Args:
        points: (N, 3) array of XYZ coordinates
        filepath: Output .ply file path
        colors: Optional (N, 3) array of RGB colors (0-1 range)
    """
    n_points = len(points)
    
    with open(filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        
        f.write("end_header\n")
        
        # Data
        for i in range(n_points):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = (colors[i] * 255).astype(np.uint8)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
