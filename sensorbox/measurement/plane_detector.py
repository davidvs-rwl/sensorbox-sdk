"""Plane detection from OAK-D 3D point clouds."""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Plane:
    """Detected plane in 3D space."""
    normal: np.ndarray  # Unit normal vector (a, b, c)
    distance: float  # Distance from origin
    centroid: np.ndarray  # Center of the plane points
    num_points: int
    
    @property
    def is_horizontal(self) -> bool:
        """Check if plane is roughly horizontal (floor/ceiling)."""
        return abs(self.normal[1]) > 0.9
    
    @property
    def is_vertical(self) -> bool:
        """Check if plane is roughly vertical (wall)."""
        return abs(self.normal[1]) < 0.3
    
    @property
    def height(self) -> float:
        """Get height of horizontal plane (Y coordinate)."""
        if self.is_horizontal:
            return self.centroid[1]
        return 0.0


class PlaneDetector:
    """Detect planes from 3D point clouds using RANSAC."""
    
    def __init__(
        self,
        distance_threshold: float = 0.05,  # 5cm tolerance
        ransac_iterations: int = 100,
        min_plane_points: int = 500,
        max_points_for_ransac: int = 10000,  # Subsample if more
    ):
        self.distance_threshold = distance_threshold
        self.ransac_iterations = ransac_iterations
        self.min_plane_points = min_plane_points
        self.max_points_for_ransac = max_points_for_ransac
    
    def _subsample_points(self, points: np.ndarray) -> np.ndarray:
        """Subsample points if too many."""
        if len(points) <= self.max_points_for_ransac:
            return points
        
        indices = np.random.choice(len(points), self.max_points_for_ransac, replace=False)
        return points[indices]
    
    def detect_planes(
        self, 
        points: np.ndarray,
        max_planes: int = 3,
    ) -> List[Plane]:
        """Detect multiple planes in point cloud."""
        if len(points) < self.min_plane_points:
            return []
        
        # Subsample for efficiency
        working_points = self._subsample_points(points)
        
        planes = []
        remaining_points = working_points.copy()
        
        for _ in range(max_planes):
            if len(remaining_points) < self.min_plane_points:
                break
            
            plane, inlier_mask = self._fit_plane_ransac(remaining_points)
            
            if plane is None:
                break
            
            planes.append(plane)
            
            # Remove inlier points
            remaining_points = remaining_points[~inlier_mask]
        
        return planes
    
    def _fit_plane_ransac(
        self,
        points: np.ndarray,
    ) -> Tuple[Optional[Plane], np.ndarray]:
        """Fit a plane using RANSAC."""
        best_plane = None
        best_inlier_mask = np.zeros(len(points), dtype=bool)
        best_num_inliers = 0
        
        n_points = len(points)
        
        for _ in range(self.ransac_iterations):
            # Random sample 3 points
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]
            
            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            
            normal = normal / norm
            
            # Distance of all points to plane
            d = -np.dot(normal, p1)
            distances = np.abs(points @ normal + d)
            
            # Find inliers
            inlier_mask = distances < self.distance_threshold
            num_inliers = inlier_mask.sum()
            
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inlier_mask = inlier_mask
                
                inlier_points = points[inlier_mask]
                centroid = inlier_points.mean(axis=0)
                
                # Ensure consistent normal direction
                if normal[1] < 0:
                    normal = -normal
                
                best_plane = Plane(
                    normal=normal,
                    distance=d,
                    centroid=centroid,
                    num_points=num_inliers,
                )
        
        if best_plane and best_plane.num_points < self.min_plane_points:
            return None, np.zeros(len(points), dtype=bool)
        
        return best_plane, best_inlier_mask
    
    def find_floor_ceiling(
        self,
        planes: List[Plane],
    ) -> Tuple[Optional[Plane], Optional[Plane]]:
        """Find floor and ceiling planes."""
        horizontal_planes = [p for p in planes if p.is_horizontal]
        
        if not horizontal_planes:
            return None, None
        
        # Sort by height (Y coordinate)
        horizontal_planes.sort(key=lambda p: p.height)
        
        floor = horizontal_planes[0]
        ceiling = horizontal_planes[-1] if len(horizontal_planes) > 1 else None
        
        # Validate: floor should be below camera (negative Y)
        if floor.height > 0.5:
            floor = None
        if ceiling and ceiling.height < 0.5:
            ceiling = None
        
        return floor, ceiling
    
    def estimate_room_height(
        self,
        points: np.ndarray,
        planes: Optional[List[Plane]] = None,
    ) -> float:
        """Estimate room height from point cloud."""
        if planes is None:
            planes = self.detect_planes(points)
        
        floor, ceiling = self.find_floor_ceiling(planes)
        
        if floor and ceiling:
            return abs(ceiling.height - floor.height)
        
        # Fallback: use Y extent of points (subsampled)
        sample = self._subsample_points(points)
        y_values = sample[:, 1]
        
        y_min = np.percentile(y_values, 5)
        y_max = np.percentile(y_values, 95)
        
        return abs(y_max - y_min)
