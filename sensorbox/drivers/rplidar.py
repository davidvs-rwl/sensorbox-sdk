"""RPLIDAR Driver for 360° laser scanners."""

from typing import Optional, Generator
import numpy as np
from rplidar import RPLidar as RPLidarDevice

from ..core.sensor import Sensor
from ..core.frame import SensorFrame, SensorMetadata, SensorType, FrameType


class RPLidarSensor(Sensor):
    """
    Driver for RPLIDAR A1/A2/A3 laser scanners.
    
    Each scan contains 360° of distance measurements.
    Data format: numpy array of shape (N, 3) where columns are:
        - angle (degrees, 0-360)
        - distance (mm)
        - quality (0-15)
    
    Example:
        with RPLidarSensor('/dev/ttyUSB0') as lidar:
            for frame in lidar.stream(max_frames=10):
                scan = frame.data  # (N, 3) array
                angles = scan[:, 0]
                distances = scan[:, 1]
                print(f"Scan with {len(scan)} points")
    """
    
    def __init__(
        self,
        port: str = '/dev/ttyUSB0',
        sensor_id: Optional[str] = None,
    ):
        """
        Initialize RPLIDAR sensor.
        
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0')
            sensor_id: Custom sensor ID (auto-generated if not provided)
        """
        self._port = port
        
        if sensor_id is None:
            sensor_id = f"rplidar_{port.replace('/', '_')}"
        
        super().__init__(sensor_id=sensor_id, sensor_type=SensorType.LIDAR)
        
        self._lidar: Optional[RPLidarDevice] = None
        self._scan_iterator = None
    
    @property
    def port(self) -> str:
        return self._port
    
    def connect(self) -> None:
        """Connect to the RPLIDAR."""
        if self._connected:
            return
        
        self._lidar = RPLidarDevice(self._port)
        
        # Get device info
        info = self._lidar.get_info()
        health = self._lidar.get_health()
        
        self._metadata = SensorMetadata(
            sensor_id=self._sensor_id,
            sensor_type=SensorType.LIDAR,
            manufacturer="Slamtec",
            model=f"RPLIDAR (model {info['model']})",
            serial_number=info['serialnumber'],
            firmware_version=f"{info['firmware'][0]}.{info['firmware'][1]}",
            config={
                "port": self._port,
                "hardware": info['hardware'],
                "health": health[0],
            },
        )
        
        self._connected = True
        self._sequence_number = 0
        self._time_offset = None
    
    def disconnect(self) -> None:
        """Disconnect from the RPLIDAR."""
        if self._lidar is not None:
            self._lidar.stop()
            self._lidar.stop_motor()
            self._lidar.disconnect()
            self._lidar = None
        self._scan_iterator = None
        self._connected = False
    
    def read(self) -> Optional[SensorFrame]:
        """
        Read a single 360° scan.
        
        Returns:
            SensorFrame containing scan data as numpy array,
            or None if no complete scan available
        """
        if not self._connected or self._lidar is None:
            raise RuntimeError(f"RPLIDAR {self._sensor_id} is not connected")
        
        # Start iterator if needed
        if self._scan_iterator is None:
            self._scan_iterator = self._lidar.iter_scans()
        
        try:
            scan = next(self._scan_iterator)
        except StopIteration:
            return None
        
        # Convert to numpy array: (quality, angle, distance)
        # Reorder to: (angle, distance, quality)
        scan_array = np.array([(angle, distance, quality) 
                               for quality, angle, distance in scan],
                              dtype=np.float32)
        
        timestamp, wall_time = self._get_timestamp()
        
        return SensorFrame(
            sensor_id=self._sensor_id,
            sensor_type=SensorType.LIDAR,
            frame_type=FrameType.SCAN,
            timestamp=timestamp,
            wall_time=wall_time,
            sequence_number=self._next_sequence(),
            data=scan_array,
            metadata={
                "num_points": len(scan_array),
                "columns": ["angle_deg", "distance_mm", "quality"],
            },
        )
    
    def get_info(self) -> dict:
        """Get RPLIDAR device info."""
        if not self._connected or self._lidar is None:
            raise RuntimeError("RPLIDAR is not connected")
        return self._lidar.get_info()
    
    def get_health(self) -> tuple:
        """Get RPLIDAR health status."""
        if not self._connected or self._lidar is None:
            raise RuntimeError("RPLIDAR is not connected")
        return self._lidar.get_health()


def discover_rplidars() -> list:
    """
    Discover connected RPLIDAR devices.
    
    Returns:
        List of serial port paths that have RPLIDAR devices
    """
    import glob
    
    ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    rplidars = []
    
    for port in ports:
        try:
            lidar = RPLidarDevice(port)
            info = lidar.get_info()
            rplidars.append({
                "port": port,
                "model": info['model'],
                "serial": info['serialnumber'],
                "firmware": f"{info['firmware'][0]}.{info['firmware'][1]}",
            })
            lidar.stop()
            lidar.disconnect()
        except Exception:
            pass
    
    return rplidars
