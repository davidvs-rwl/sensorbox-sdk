"""HDF5 writer for multi-sensor data."""

from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import numpy as np
import h5py
import json

from ..core.frame import SensorFrame, SensorType


class HDF5Writer:
    """
    Write synchronized sensor data to HDF5 format.
    
    File structure:
        /metadata           - Session metadata (JSON)
        /cameras/
            cam_0/
                frames      - Image data (N, H, W, C)
                timestamps  - Frame timestamps
            cam_1/
                ...
        /lidar/
            scans           - Variable-length scan data
            timestamps      - Scan timestamps
            scan_lengths    - Number of points per scan
    
    Example:
        with HDF5Writer("recording.h5") as writer:
            writer.set_metadata({"location": "warehouse"})
            
            for frame in fusion.stream(duration=60.0):
                writer.write_cameras(frame.cameras)
                if frame.has_lidar:
                    writer.write_lidar(frame.lidar)
    """
    
    def __init__(
        self,
        filepath: str,
        compression: str = "gzip",
        compression_level: int = 4,
        chunk_size: int = 30,
    ):
        """
        Initialize HDF5 writer.
        
        Args:
            filepath: Output file path (.h5)
            compression: Compression algorithm ("gzip", "lzf", or None)
            compression_level: Compression level (1-9, higher = smaller file)
            chunk_size: Number of frames per chunk (affects read performance)
        """
        self._filepath = Path(filepath)
        self._compression = compression
        self._compression_level = compression_level
        self._chunk_size = chunk_size
        
        self._file: Optional[h5py.File] = None
        self._camera_datasets: Dict[int, dict] = {}
        self._lidar_data: list = []
        self._lidar_timestamps: list = []
        self._frame_count = 0
        self._start_time: Optional[datetime] = None
    
    @property
    def filepath(self) -> Path:
        return self._filepath
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    def open(self) -> None:
        """Open the HDF5 file for writing."""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._filepath, "w")
        self._start_time = datetime.now()
        
        # Create groups
        self._file.create_group("cameras")
        self._file.create_group("lidar")
        
        # Store basic metadata
        self._file.attrs["created"] = self._start_time.isoformat()
        self._file.attrs["sdk_version"] = "0.1.0"
    
    def close(self) -> None:
        """Close the HDF5 file and finalize data."""
        if self._file is None:
            return
        
        # Write any remaining LIDAR data
        self._flush_lidar()
        
        # Update metadata
        self._file.attrs["frame_count"] = self._frame_count
        self._file.attrs["duration_seconds"] = (
            datetime.now() - self._start_time
        ).total_seconds() if self._start_time else 0
        
        self._file.close()
        self._file = None
    
    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set session metadata."""
        if self._file is None:
            raise RuntimeError("File not open")
        
        self._file.attrs["user_metadata"] = json.dumps(metadata)
    
    def _get_camera_dataset(self, cam_id: int, frame_shape: tuple) -> dict:
        """Get or create datasets for a camera."""
        if cam_id in self._camera_datasets:
            return self._camera_datasets[cam_id]
        
        # Create camera group
        cam_group = self._file["cameras"].create_group(f"cam_{cam_id}")
        
        h, w, c = frame_shape
        
        # Create resizable datasets
        frames_ds = cam_group.create_dataset(
            "frames",
            shape=(0, h, w, c),
            maxshape=(None, h, w, c),
            dtype=np.uint8,
            chunks=(min(self._chunk_size, 10), h, w, c),
            compression=self._compression,
            compression_opts=self._compression_level if self._compression == "gzip" else None,
        )
        
        timestamps_ds = cam_group.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
        )
        
        # Store camera info
        cam_group.attrs["width"] = w
        cam_group.attrs["height"] = h
        cam_group.attrs["channels"] = c
        
        self._camera_datasets[cam_id] = {
            "frames": frames_ds,
            "timestamps": timestamps_ds,
            "group": cam_group,
        }
        
        return self._camera_datasets[cam_id]
    
    def write_cameras(self, cameras: Dict[int, SensorFrame]) -> None:
        """Write camera frames."""
        if self._file is None:
            raise RuntimeError("File not open")
        
        for cam_id, frame in cameras.items():
            ds = self._get_camera_dataset(cam_id, frame.data.shape)
            
            # Resize and append
            current_size = ds["frames"].shape[0]
            ds["frames"].resize(current_size + 1, axis=0)
            ds["timestamps"].resize(current_size + 1, axis=0)
            
            ds["frames"][current_size] = frame.data
            ds["timestamps"][current_size] = frame.timestamp
        
        self._frame_count += 1
    
    def write_lidar(self, frame: SensorFrame) -> None:
        """Write LIDAR scan."""
        if self._file is None:
            raise RuntimeError("File not open")
        
        self._lidar_data.append(frame.data)
        self._lidar_timestamps.append(frame.timestamp)
        
        # Flush periodically
        if len(self._lidar_data) >= 100:
            self._flush_lidar()
    
    def _flush_lidar(self) -> None:
        """Flush buffered LIDAR data to disk."""
        if not self._lidar_data:
            return
        
        lidar_group = self._file["lidar"]
        
        # Concatenate all scans
        all_points = np.vstack(self._lidar_data)
        scan_lengths = [len(scan) for scan in self._lidar_data]
        timestamps = np.array(self._lidar_timestamps)
        
        if "scans" not in lidar_group:
            # Create datasets
            lidar_group.create_dataset(
                "scans",
                data=all_points,
                maxshape=(None, 3),
                compression=self._compression,
            )
            lidar_group.create_dataset(
                "timestamps",
                data=timestamps,
                maxshape=(None,),
            )
            lidar_group.create_dataset(
                "scan_lengths",
                data=np.array(scan_lengths, dtype=np.int32),
                maxshape=(None,),
            )
        else:
            # Append to existing
            scans_ds = lidar_group["scans"]
            ts_ds = lidar_group["timestamps"]
            lens_ds = lidar_group["scan_lengths"]
            
            old_size = scans_ds.shape[0]
            scans_ds.resize(old_size + len(all_points), axis=0)
            scans_ds[old_size:] = all_points
            
            old_ts_size = ts_ds.shape[0]
            ts_ds.resize(old_ts_size + len(timestamps), axis=0)
            ts_ds[old_ts_size:] = timestamps
            
            old_lens_size = lens_ds.shape[0]
            lens_ds.resize(old_lens_size + len(scan_lengths), axis=0)
            lens_ds[old_lens_size:] = scan_lengths
        
        # Clear buffers
        self._lidar_data.clear()
        self._lidar_timestamps.clear()
    
    def __enter__(self) -> "HDF5Writer":
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
