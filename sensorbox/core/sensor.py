"""
Base Sensor class: Abstract interface for all sensor drivers.

All sensor drivers (camera, LIDAR, etc.) inherit from this class
and implement the required methods.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generator, Optional
import time

from .frame import SensorFrame, SensorMetadata, SensorType


class Sensor(ABC):
    """
    Abstract base class for all sensor drivers.
    
    Lifecycle:
        1. Create sensor instance
        2. Call connect() to establish connection
        3. Use read() for single frames or stream() for continuous
        4. Call disconnect() when done
    """
    
    def __init__(self, sensor_id: str, sensor_type: SensorType):
        self._sensor_id = sensor_id
        self._sensor_type = sensor_type
        self._connected = False
        self._sequence_number = 0
        self._metadata: Optional[SensorMetadata] = None
        self._time_offset: Optional[float] = None
    
    @property
    def sensor_id(self) -> str:
        return self._sensor_id
    
    @property
    def sensor_type(self) -> SensorType:
        return self._sensor_type
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def metadata(self) -> Optional[SensorMetadata]:
        return self._metadata
    
    def _get_timestamp(self) -> tuple[float, datetime]:
        mono = time.monotonic()
        wall = datetime.now()
        if self._time_offset is None:
            self._time_offset = mono
        return (mono - self._time_offset, wall)
    
    def _next_sequence(self) -> int:
        seq = self._sequence_number
        self._sequence_number += 1
        return seq
    
    @abstractmethod
    def connect(self) -> None:
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        pass
    
    @abstractmethod
    def read(self) -> Optional[SensorFrame]:
        pass
    
    def stream(
        self,
        duration: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> Generator[SensorFrame, None, None]:
        if not self._connected:
            raise RuntimeError(f"Sensor {self._sensor_id} is not connected")
        
        start_time = time.monotonic()
        frame_count = 0
        
        while True:
            if duration is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= duration:
                    break
            
            if max_frames is not None and frame_count >= max_frames:
                break
            
            frame = self.read()
            if frame is not None:
                frame_count += 1
                yield frame
    
    def __enter__(self) -> "Sensor":
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()
    
    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"{self.__class__.__name__}(id={self._sensor_id!r}, {status})"
