"""HDF5 reader for OAK-D Pro data."""

from typing import Optional, Dict, List, Generator, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import h5py
import json


@dataclass
class OakDRecordingInfo:
    """Information about an OAK-D recording."""
    filepath: str
    created: str
    duration_seconds: float
    frame_count: int
    depth_count: int
    imu_count: int
    rgb_size: tuple
    depth_size: tuple
    user_metadata: Dict[str, Any]


@dataclass
class OakDPlaybackFrame:
    """A frame during playback."""
    index: int
    timestamp: float
    rgb: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    imu: Optional[Dict[str, Any]] = None


class OakDHDF5Reader:
    """Read OAK-D Pro data from HDF5 files."""
    
    def __init__(self, filepath: str):
        self._filepath = Path(filepath)
        self._file: Optional[h5py.File] = None
    
    def open(self) -> None:
        if not self._filepath.exists():
            raise FileNotFoundError(f"Recording not found: {self._filepath}")
        self._file = h5py.File(self._filepath, "r")
    
    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
    
    @property
    def info(self) -> OakDRecordingInfo:
        if self._file is None:
            raise RuntimeError("File not open")
        
        rgb_size = (0, 0)
        if "rgb" in self._file and "frames" in self._file["rgb"]:
            shape = self._file["rgb"]["frames"].shape
            rgb_size = (shape[2], shape[1])  # (width, height)
        
        depth_size = (0, 0)
        if "depth" in self._file and "frames" in self._file["depth"]:
            shape = self._file["depth"]["frames"].shape
            depth_size = (shape[2], shape[1])
        
        user_metadata = {}
        if "user_metadata" in self._file.attrs:
            user_metadata = json.loads(self._file.attrs["user_metadata"])
        
        return OakDRecordingInfo(
            filepath=str(self._filepath),
            created=self._file.attrs.get("created", "unknown"),
            duration_seconds=self._file.attrs.get("duration_seconds", 0),
            frame_count=self._file.attrs.get("frame_count", 0),
            depth_count=self._file.attrs.get("depth_count", 0),
            imu_count=self._file.attrs.get("imu_count", 0),
            rgb_size=rgb_size,
            depth_size=depth_size,
            user_metadata=user_metadata,
        )
    
    def get_frame(self, index: int) -> OakDPlaybackFrame:
        if self._file is None:
            raise RuntimeError("File not open")
        
        rgb = None
        timestamp = 0.0
        
        if "rgb" in self._file and index < self._file["rgb"]["frames"].shape[0]:
            rgb = self._file["rgb"]["frames"][index]
            timestamp = self._file["rgb"]["timestamps"][index]
        
        depth = None
        if "depth" in self._file and index < self._file["depth"]["frames"].shape[0]:
            depth = self._file["depth"]["frames"][index]
        
        imu = None
        if "imu" in self._file and "accelerometer" in self._file["imu"]:
            if index < self._file["imu"]["accelerometer"].shape[0]:
                acc = self._file["imu"]["accelerometer"][index]
                gyro = self._file["imu"]["gyroscope"][index]
                imu = {
                    "accelerometer": {"x": acc[0], "y": acc[1], "z": acc[2]},
                    "gyroscope": {"x": gyro[0], "y": gyro[1], "z": gyro[2]},
                }
        
        return OakDPlaybackFrame(index=index, timestamp=timestamp, rgb=rgb, depth=depth, imu=imu)
    
    def playback(self) -> Generator[OakDPlaybackFrame, None, None]:
        if self._file is None:
            raise RuntimeError("File not open")
        
        for i in range(self.info.frame_count):
            yield self.get_frame(i)
    
    def __enter__(self) -> "OakDHDF5Reader":
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
