"""Point cloud generation from depth data."""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    
    def scale(self, new_width: int, new_height: int) -> "CameraIntrinsics":
        """Scale intrinsics to a different resolution."""
        sx = new_width / self.width
        sy = new_height / self.height
        return CameraIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            width=new_width,
            height=new_height,
        )


# OAK-D Pro stereo intrinsics at 640x400 (from calibration)
OAKD_MONO_640x400 = CameraIntrinsics(
    fx=400.0, fy=400.0, cx=312.0, cy=192.0, width=640, height=400
)


def get_intrinsics_for_depth(depth_shape: Tuple[int, int]) -> CameraIntrinsics:
    """Get intrinsics scaled to actual depth resolution."""
    h, w = depth_shape
    return OAKD_MONO_640x400.scale(w, h)


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: Optional[CameraIntrinsics] = None,
    max_depth: float = 10000.0,
    subsample: int = 1,
) -> np.ndarray:
    """
    Convert depth image to 3D point cloud.
    
    Args:
        depth: Depth image (H, W) in millimeters (already aligned to RGB)
        intrinsics: Camera intrinsics (auto-detected if None)
        max_depth: Maximum depth to include (mm)
        subsample: Subsample factor
    
    Returns:
        Point cloud as (N, 3) array with X, Y, Z in meters
    """
    h, w = depth.shape
    
    if intrinsics is None:
        intrinsics = get_intrinsics_for_depth(depth.shape)
    
    # Create pixel coordinate grids
    u = np.arange(0, w, subsample)
    v = np.arange(0, h, subsample)
    u, v = np.meshgrid(u, v)
    
    # Get depth values (subsampled)
    z = depth[::subsample, ::subsample].astype(np.float32)
    
    # Filter invalid depths
    valid = (z > 0) & (z < max_depth)
    
    # Convert to meters
    z_m = z / 1000.0
    
    # Back-project to 3D
    x = (u - intrinsics.cx) * z_m / intrinsics.fx
    y = (v - intrinsics.cy) * z_m / intrinsics.fy
    
    # Rotate 180° to match RGB orientation (depth was flipped with flipud)
    # x = -x  # Not needed
    y = -y
    
    # Stack and filter
    points = np.stack([x, y, z_m], axis=-1)
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
    """
    import cv2
    
    h, w = depth.shape
    
    if intrinsics is None:
        intrinsics = get_intrinsics_for_depth(depth.shape)
    
    # Resize RGB to match depth
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
    
    # Convert to meters
    z_m = z / 1000.0
    
    # Back-project to 3D
    x = (u - intrinsics.cx) * z_m / intrinsics.fx
    y = (v - intrinsics.cy) * z_m / intrinsics.fy
    
    # Rotate 180° to match RGB orientation
    # x = -x  # Not needed
    y = -y
    
    # Stack and filter
    points = np.stack([x, y, z_m], axis=-1)
    points = points[valid]
    colors = colors[valid]
    colors = colors.astype(np.float32) / 255.0
    
    return points, colors


def pointcloud_to_ply(
    points: np.ndarray,
    filepath: str,
    colors: Optional[np.ndarray] = None,
) -> None:
    """Save point cloud to PLY file."""
    n_points = len(points)
    
    with open(filepath, 'w') as f:
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
        
        for i in range(n_points):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = (colors[i] * 255).astype(np.uint8)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
