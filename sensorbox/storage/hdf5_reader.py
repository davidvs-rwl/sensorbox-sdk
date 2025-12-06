"""HDF5 reader for multi-sensor data."""

from typing import Optional, Dict, List, Generator, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import h5py
import json


@dataclass
class RecordingInfo:
    """Information about a recording."""
    filepath: str
    created: str
    duration_seconds: float
    frame_count: int
    cameras: List[int]
    has_lidar: bool
    lidar_scan_count: int
    user_metadata: Dict[str, Any]


@dataclass 
class PlaybackFrame:
    """A frame during playback."""
    timestamp: float
    cameras: Dict[int, np.ndarray]
    lidar: Optional[np.ndarray] = None


class HDF5Reader:
    """
    Read sensor data from HDF5 files.
    
    Example:
        with HDF5Reader("recording.h5") as reader:
            print(reader.info)
            
            for frame in reader.playback():
                cam0 = frame.cameras[0]
                if frame.lidar is not None:
                    scan = frame.lidar
    """
    
    def __init__(self, filepath: str):
        self._filepath = Path(filepath)
        self._file: Optional[h5py.File] = None
    
    @property
    def filepath(self) -> Path:
        return self._filepath
    
    def open(self) -> None:
        """Open the HDF5 file for reading."""
        if not self._filepath.exists():
            raise FileNotFoundError(f"Recording not found: {self._filepath}")
        
        self._file = h5py.File(self._filepath, "r")
    
    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file:
            self._file.close()
            self._file = None
    
    @property
    def info(self) -> RecordingInfo:
        """Get recording information."""
        if self._file is None:
            raise RuntimeError("File not open")
        
        # Get camera IDs
        cameras = []
        if "cameras" in self._file:
            for name in self._file["cameras"]:
                cam_id = int(name.split("_")[1])
                cameras.append(cam_id)
        
        # Get LIDAR info
        has_lidar = "lidar" in self._file and "scans" in self._file["lidar"]
        lidar_count = 0
        if has_lidar:
            lidar_count = len(self._file["lidar"]["timestamps"])
        
        # Get user metadata
        user_metadata = {}
        if "user_metadata" in self._file.attrs:
            user_metadata = json.loads(self._file.attrs["user_metadata"])
        
        return RecordingInfo(
            filepath=str(self._filepath),
            created=self._file.attrs.get("created", "unknown"),
            duration_seconds=self._file.attrs.get("duration_seconds", 0),
            frame_count=self._file.attrs.get("frame_count", 0),
            cameras=cameras,
            has_lidar=has_lidar,
            lidar_scan_count=lidar_count,
            user_metadata=user_metadata,
        )
    
    def get_frame(self, index: int) -> PlaybackFrame:
        """Get a specific frame by index."""
        if self._file is None:
            raise RuntimeError("File not open")
        
        cameras = {}
        timestamp = 0.0
        
        for name in self._file["cameras"]:
            cam_id = int(name.split("_")[1])
            cam_group = self._file["cameras"][name]
            
            if index < cam_group["frames"].shape[0]:
                cameras[cam_id] = cam_group["frames"][index]
                timestamp = cam_group["timestamps"][index]
        
        # Find matching LIDAR scan (closest timestamp)
        lidar = None
        if "lidar" in self._file and "timestamps" in self._file["lidar"]:
            lidar_ts = self._file["lidar"]["timestamps"][:]
            if len(lidar_ts) > 0:
                closest_idx = np.argmin(np.abs(lidar_ts - timestamp))
                if abs(lidar_ts[closest_idx] - timestamp) < 0.1:
                    lidar = self._get_lidar_scan(closest_idx)
        
        return PlaybackFrame(timestamp=timestamp, cameras=cameras, lidar=lidar)
    
    def _get_lidar_scan(self, scan_index: int) -> np.ndarray:
        """Get a specific LIDAR scan by index."""
        lidar_group = self._file["lidar"]
        scan_lengths = lidar_group["scan_lengths"][:]
        
        # Calculate start position
        start = int(np.sum(scan_lengths[:scan_index]))
        length = int(scan_lengths[scan_index])
        
        return lidar_group["scans"][start:start + length]
    
    def playback(
        self,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> Generator[PlaybackFrame, None, None]:
        """
        Playback frames in order.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp (None = until end)
        
        Yields:
            PlaybackFrame for each frame
        """
        if self._file is None:
            raise RuntimeError("File not open")
        
        info = self.info
        
        for i in range(info.frame_count):
            frame = self.get_frame(i)
            
            if frame.timestamp < start_time:
                continue
            if end_time and frame.timestamp > end_time:
                break
            
            yield frame
    
    def get_all_camera_frames(self, cam_id: int) -> np.ndarray:
        """Get all frames for a camera as a numpy array."""
        if self._file is None:
            raise RuntimeError("File not open")
        
        return self._file["cameras"][f"cam_{cam_id}"]["frames"][:]
    
    def get_all_lidar_scans(self) -> List[np.ndarray]:
        """Get all LIDAR scans as a list of arrays."""
        if self._file is None:
            raise RuntimeError("File not open")
        
        if "lidar" not in self._file:
            return []
        
        scans = []
        scan_lengths = self._file["lidar"]["scan_lengths"][:]
        all_points = self._file["lidar"]["scans"][:]
        
        start = 0
        for length in scan_lengths:
            scans.append(all_points[start:start + length])
            start += length
        
        return scans
    
    def __enter__(self) -> "HDF5Reader":
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
