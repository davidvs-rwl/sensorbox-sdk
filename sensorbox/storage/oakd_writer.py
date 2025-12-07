"""HDF5 writer for OAK-D Pro data."""

from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import numpy as np
import h5py
import json

from ..drivers.oakd import OakDFrame


class OakDHDF5Writer:
    """
    Write OAK-D Pro data to HDF5 format.
    
    File structure:
        /metadata           - Session metadata
        /rgb/
            frames          - RGB images (N, H, W, 3)
            timestamps      - Frame timestamps
        /depth/
            frames          - Depth maps (N, H, W)
            timestamps      - Frame timestamps
        /imu/
            accelerometer   - (N, 3) x, y, z
            gyroscope       - (N, 3) x, y, z
            timestamps      - IMU timestamps
    """
    
    def __init__(
        self,
        filepath: str,
        compression: str = None,  # No compression for speed
    ):
        self._filepath = Path(filepath)
        self._compression = compression
        
        self._file: Optional[h5py.File] = None
        self._rgb_ds = None
        self._depth_ds = None
        self._rgb_ts_ds = None
        self._depth_ts_ds = None
        self._imu_accel = []
        self._imu_gyro = []
        self._imu_ts = []
        
        self._frame_count = 0
        self._depth_count = 0
        self._start_time: Optional[datetime] = None
    
    @property
    def filepath(self) -> Path:
        return self._filepath
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    def open(self) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._filepath, "w")
        self._start_time = datetime.now()
        
        self._file.create_group("rgb")
        self._file.create_group("depth")
        self._file.create_group("imu")
        
        self._file.attrs["created"] = self._start_time.isoformat()
        self._file.attrs["sensor"] = "OAK-D Pro"
        self._file.attrs["sdk_version"] = "0.1.0"
    
    def close(self) -> None:
        if self._file is None:
            return
        
        # Write IMU data
        if self._imu_accel:
            imu_group = self._file["imu"]
            imu_group.create_dataset("accelerometer", data=np.array(self._imu_accel))
            imu_group.create_dataset("gyroscope", data=np.array(self._imu_gyro))
            imu_group.create_dataset("timestamps", data=np.array(self._imu_ts))
        
        self._file.attrs["frame_count"] = self._frame_count
        self._file.attrs["depth_count"] = self._depth_count
        self._file.attrs["imu_count"] = len(self._imu_accel)
        self._file.attrs["duration_seconds"] = (
            datetime.now() - self._start_time
        ).total_seconds() if self._start_time else 0
        
        self._file.close()
        self._file = None
    
    def write(self, frame: OakDFrame) -> None:
        """Write an OAK-D frame."""
        if self._file is None:
            raise RuntimeError("File not open")
        
        # Write RGB
        if frame.rgb is not None:
            self._write_rgb(frame.rgb, frame.timestamp)
            self._frame_count += 1
        
        # Write Depth
        if frame.depth is not None:
            self._write_depth(frame.depth, frame.timestamp)
            self._depth_count += 1
        
        # Buffer IMU
        if frame.imu:
            acc = frame.imu['accelerometer']
            gyro = frame.imu['gyroscope']
            self._imu_accel.append([acc['x'], acc['y'], acc['z']])
            self._imu_gyro.append([gyro['x'], gyro['y'], gyro['z']])
            self._imu_ts.append(frame.timestamp)
    
    def _write_rgb(self, rgb: np.ndarray, timestamp: float) -> None:
        rgb_group = self._file["rgb"]
        
        if self._rgb_ds is None:
            h, w, c = rgb.shape
            self._rgb_ds = rgb_group.create_dataset(
                "frames",
                shape=(0, h, w, c),
                maxshape=(None, h, w, c),
                dtype=np.uint8,
                chunks=(1, h, w, c),
                compression=self._compression,
            )
            self._rgb_ts_ds = rgb_group.create_dataset(
                "timestamps",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float64,
            )
            rgb_group.attrs["width"] = w
            rgb_group.attrs["height"] = h
        
        idx = self._rgb_ds.shape[0]
        self._rgb_ds.resize(idx + 1, axis=0)
        self._rgb_ts_ds.resize(idx + 1, axis=0)
        self._rgb_ds[idx] = rgb
        self._rgb_ts_ds[idx] = timestamp
    
    def _write_depth(self, depth: np.ndarray, timestamp: float) -> None:
        depth_group = self._file["depth"]
        
        if self._depth_ds is None:
            h, w = depth.shape
            self._depth_ds = depth_group.create_dataset(
                "frames",
                shape=(0, h, w),
                maxshape=(None, h, w),
                dtype=np.uint16,
                chunks=(1, h, w),
                compression=self._compression,
            )
            self._depth_ts_ds = depth_group.create_dataset(
                "timestamps",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float64,
            )
            depth_group.attrs["width"] = w
            depth_group.attrs["height"] = h
        
        idx = self._depth_ds.shape[0]
        self._depth_ds.resize(idx + 1, axis=0)
        self._depth_ts_ds.resize(idx + 1, axis=0)
        self._depth_ds[idx] = depth
        self._depth_ts_ds[idx] = timestamp
    
    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        if self._file is None:
            raise RuntimeError("File not open")
        self._file.attrs["user_metadata"] = json.dumps(metadata)
    
    def __enter__(self) -> "OakDHDF5Writer":
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
