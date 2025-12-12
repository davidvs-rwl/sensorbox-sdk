"""Wall detection from RPLIDAR 2D scans."""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Wall:
    """Detected wall segment."""
    start: Tuple[float, float]  # (x, y) in meters
    end: Tuple[float, float]
    length: float
    angle: float  # radians from x-axis
    distance: float  # perpendicular distance from origin
    num_points: int
    
    @property
    def midpoint(self) -> Tuple[float, float]:
        return (
            (self.start[0] + self.end[0]) / 2,
            (self.start[1] + self.end[1]) / 2,
        )


class WallDetector:
    """Detect walls from RPLIDAR 2D scan data."""
    
    def __init__(
        self,
        min_wall_length: float = 0.5,  # Minimum wall length in meters
        angle_threshold: float = 10.0,  # Degrees - max deviation for same wall
        distance_threshold: float = 0.1,  # Meters - max gap in wall
        ransac_iterations: int = 100,
    ):
        self.min_wall_length = min_wall_length
        self.angle_threshold = np.radians(angle_threshold)
        self.distance_threshold = distance_threshold
        self.ransac_iterations = ransac_iterations
    
    def scan_to_cartesian(
        self, 
        angles: np.ndarray, 
        distances: np.ndarray,
        max_distance: float = 12.0,
    ) -> np.ndarray:
        """Convert polar scan to cartesian points."""
        # Filter invalid readings
        valid = (distances > 0.1) & (distances < max_distance)
        angles = angles[valid]
        distances = distances[valid]
        
        # Convert to cartesian
        x = distances * np.cos(angles)
        y = distances * np.sin(angles)
        
        return np.column_stack([x, y])
    
    def detect_walls(self, points: np.ndarray) -> List[Wall]:
        """Detect walls from 2D points using RANSAC line fitting."""
        if len(points) < 10:
            return []
        
        walls = []
        remaining_points = points.copy()
        
        # Iteratively find walls
        for _ in range(10):  # Max 10 walls
            if len(remaining_points) < 20:
                break
            
            wall, inliers = self._fit_wall_ransac(remaining_points)
            
            if wall is None or wall.length < self.min_wall_length:
                break
            
            walls.append(wall)
            
            # Remove inlier points
            mask = np.ones(len(remaining_points), dtype=bool)
            mask[inliers] = False
            remaining_points = remaining_points[mask]
        
        return walls
    
    def _fit_wall_ransac(
        self, 
        points: np.ndarray,
    ) -> Tuple[Optional[Wall], np.ndarray]:
        """Fit a wall using RANSAC."""
        best_wall = None
        best_inliers = np.array([])
        best_score = 0
        
        n_points = len(points)
        
        for _ in range(self.ransac_iterations):
            # Random sample 2 points
            idx = np.random.choice(n_points, 2, replace=False)
            p1, p2 = points[idx]
            
            # Skip if points too close
            if np.linalg.norm(p2 - p1) < 0.1:
                continue
            
            # Line parameters
            direction = p2 - p1
            direction = direction / np.linalg.norm(direction)
            normal = np.array([-direction[1], direction[0]])
            
            # Distance of all points to line
            diff = points - p1
            distances = np.abs(diff @ normal)
            
            # Find inliers
            inliers = np.where(distances < self.distance_threshold)[0]
            
            if len(inliers) > best_score:
                best_score = len(inliers)
                best_inliers = inliers
                
                # Compute wall from inliers
                inlier_points = points[inliers]
                
                # Project points onto line to find extent
                projections = diff[inliers] @ direction
                min_proj = projections.min()
                max_proj = projections.max()
                
                start = p1 + min_proj * direction
                end = p1 + max_proj * direction
                length = max_proj - min_proj
                
                angle = np.arctan2(direction[1], direction[0])
                dist_to_origin = abs(np.cross(p1, direction))
                
                best_wall = Wall(
                    start=tuple(start),
                    end=tuple(end),
                    length=length,
                    angle=angle,
                    distance=dist_to_origin,
                    num_points=len(inliers),
                )
        
        return best_wall, best_inliers
    
    def find_room_corners(self, walls: List[Wall]) -> List[Tuple[float, float]]:
        """Find corners where walls intersect."""
        corners = []
        
        for i, wall1 in enumerate(walls):
            for wall2 in walls[i+1:]:
                # Check if walls are roughly perpendicular (60-120 degrees)
                angle_diff = abs(wall1.angle - wall2.angle)
                angle_diff = min(angle_diff, np.pi - angle_diff)
                
                if np.pi/6 < angle_diff < 5*np.pi/6:  # 30-150 degrees
                    corner = self._line_intersection(wall1, wall2)
                    if corner is not None:
                        corners.append(corner)
        
        return corners
    
    def _line_intersection(
        self, 
        wall1: Wall, 
        wall2: Wall,
    ) -> Optional[Tuple[float, float]]:
        """Find intersection of two wall lines."""
        # Line 1: p1 + t * d1
        p1 = np.array(wall1.start)
        d1 = np.array(wall1.end) - p1
        
        # Line 2: p2 + s * d2
        p2 = np.array(wall2.start)
        d2 = np.array(wall2.end) - p2
        
        # Solve for intersection
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-10:
            return None  # Parallel lines
        
        diff = p2 - p1
        t = (diff[0] * d2[1] - diff[1] * d2[0]) / cross
        
        intersection = p1 + t * d1
        
        return tuple(intersection)
    
    def estimate_room_dimensions(
        self, 
        walls: List[Wall],
    ) -> Tuple[float, float]:
        """Estimate room length and width from detected walls."""
        if len(walls) < 2:
            return (0.0, 0.0)
        
        # Group walls by angle (parallel walls)
        wall_groups = self._group_parallel_walls(walls)
        
        if len(wall_groups) < 2:
            return (0.0, 0.0)
        
        # Find two perpendicular groups with most points
        sorted_groups = sorted(wall_groups, key=lambda g: sum(w.num_points for w in g), reverse=True)
        
        # Get dimensions from wall distances
        dimensions = []
        for group in sorted_groups[:2]:
            if len(group) >= 2:
                # Distance between parallel walls
                distances = [w.distance for w in group]
                dim = max(distances) - min(distances)
                if dim > 0.5:  # Minimum room dimension
                    dimensions.append(dim)
            elif len(group) == 1:
                # Single wall - estimate from wall length
                dimensions.append(group[0].length)
        
        if len(dimensions) < 2:
            return (0.0, 0.0)
        
        # Return as (length, width) with length >= width
        dimensions.sort(reverse=True)
        return (dimensions[0], dimensions[1])
    
    def _group_parallel_walls(self, walls: List[Wall]) -> List[List[Wall]]:
        """Group walls that are roughly parallel."""
        if not walls:
            return []
        
        groups = []
        used = set()
        
        for i, wall in enumerate(walls):
            if i in used:
                continue
            
            group = [wall]
            used.add(i)
            
            for j, other in enumerate(walls):
                if j in used:
                    continue
                
                # Check if parallel (angles within threshold)
                angle_diff = abs(wall.angle - other.angle)
                angle_diff = min(angle_diff, np.pi - angle_diff)
                
                if angle_diff < self.angle_threshold:
                    group.append(other)
                    used.add(j)
            
            groups.append(group)
        
        return groups
