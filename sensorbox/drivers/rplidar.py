"""RPLIDAR Driver with error handling."""

from typing import Optional, Generator
import time
import logging
import numpy as np
from rplidar import RPLidar as RPLidarDevice, RPLidarException

from ..core.sensor import Sensor
from ..core.frame import SensorFrame, SensorMetadata, SensorType, FrameType

logger = logging.getLogger(__name__)


class LidarError(Exception):
    """Base exception for LIDAR errors."""
    pass


class LidarConnectionError(LidarError):
    """Raised when LIDAR connection fails."""
    pass


class LidarReadError(LidarError):
    """Raised when scan read fails."""
    pass


class RPLidarSensor(Sensor):
    """
    Driver for RPLIDAR with error handling and auto-reconnection.
    
    Example:
        with RPLidarSensor('/dev/ttyUSB0', auto_reconnect=True) as lidar:
            for frame in lidar.stream(max_frames=100):
                scan = frame.data
                print(f"Scan: {len(scan)} points")
    """
    
    def __init__(
        self,
        port: str = '/dev/ttyUSB0',
        sensor_id: Optional[str] = None,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
        reconnect_delay: float = 2.0,
        max_consecutive_failures: int = 5,
    ):
        """
        Initialize RPLIDAR sensor.
        
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0')
            sensor_id: Custom sensor ID
            auto_reconnect: Automatically reconnect on failure
            max_reconnect_attempts: Max reconnection attempts
            reconnect_delay: Delay between attempts (seconds)
            max_consecutive_failures: Max failures before error
        """
        self._port = port
        
        if sensor_id is None:
            sensor_id = f"rplidar_{port.replace('/', '_')}"
        
        # Error handling settings
        self._auto_reconnect = auto_reconnect
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._max_consecutive_failures = max_consecutive_failures
        
        # Stats
        self._consecutive_failures = 0
        self._total_scans = 0
        self._failed_scans = 0
        
        super().__init__(sensor_id=sensor_id, sensor_type=SensorType.LIDAR)
        
        self._lidar: Optional[RPLidarDevice] = None
        self._scan_iterator = None
    
    @property
    def port(self) -> str:
        return self._port
    
    @property
    def stats(self) -> dict:
        """Get LIDAR statistics."""
        return {
            "total_scans": self._total_scans,
            "failed_scans": self._failed_scans,
            "failure_rate": self._failed_scans / max(1, self._total_scans),
        }
    
    def connect(self) -> None:
        """Connect with retry logic."""
        if self._connected:
            return
        
        last_error = None
        for attempt in range(self._max_reconnect_attempts):
            try:
                self._do_connect()
                logger.info(f"RPLIDAR connected on {self._port}")
                return
            except Exception as e:
                last_error = e
                logger.warning(f"RPLIDAR connection attempt {attempt + 1} failed: {e}")
                if attempt < self._max_reconnect_attempts - 1:
                    time.sleep(self._reconnect_delay)
        
        raise LidarConnectionError(
            f"Failed to connect to RPLIDAR on {self._port} after {self._max_reconnect_attempts} attempts: {last_error}"
        )
    
    def _do_connect(self) -> None:
        """Internal connection logic."""
        self._lidar = RPLidarDevice(self._port)
        
        info = self._lidar.get_info()
        health = self._lidar.get_health()
        
        if health[0] != 'Good':
            logger.warning(f"RPLIDAR health: {health[0]} (error code: {health[1]})")
        
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
        self._consecutive_failures = 0
        self._scan_iterator = None
    
    def disconnect(self) -> None:
        """Disconnect from the RPLIDAR."""
        if self._lidar is not None:
            try:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._lidar.disconnect()
            except Exception as e:
                logger.warning(f"Error during RPLIDAR disconnect: {e}")
            self._lidar = None
        self._scan_iterator = None
        self._connected = False
        logger.info(f"RPLIDAR disconnected. Stats: {self.stats}")
    
    def reconnect(self) -> bool:
        """Attempt to reconnect."""
        logger.info(f"Attempting to reconnect RPLIDAR on {self._port}...")
        self.disconnect()
        try:
            time.sleep(self._reconnect_delay)
            self.connect()
            return True
        except LidarConnectionError:
            return False
    
    def read(self) -> Optional[SensorFrame]:
        """Read a single scan with error handling."""
        if not self._connected or self._lidar is None:
            raise RuntimeError(f"RPLIDAR {self._sensor_id} is not connected")
        
        try:
            if self._scan_iterator is None:
                self._scan_iterator = self._lidar.iter_scans()
            
            scan = next(self._scan_iterator)
            
            # Success
            self._consecutive_failures = 0
            self._total_scans += 1
            
            scan_array = np.array(
                [(angle, distance, quality) for quality, angle, distance in scan],
                dtype=np.float32
            )
            
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
            
        except (StopIteration, RPLidarException) as e:
            self._consecutive_failures += 1
            self._failed_scans += 1
            
            logger.warning(f"RPLIDAR scan failed ({self._consecutive_failures} consecutive): {e}")
            
            if self._consecutive_failures >= self._max_consecutive_failures:
                if self._auto_reconnect:
                    if self.reconnect():
                        self._consecutive_failures = 0
                        return self.read()
                    else:
                        raise LidarReadError(f"RPLIDAR failed after reconnection attempt")
                else:
                    raise LidarReadError(
                        f"RPLIDAR exceeded max consecutive failures ({self._max_consecutive_failures})"
                    )
            
            return None
    
    def get_info(self) -> dict:
        """Get device info."""
        if not self._connected or self._lidar is None:
            raise RuntimeError("RPLIDAR is not connected")
        return self._lidar.get_info()
    
    def get_health(self) -> tuple:
        """Get health status."""
        if not self._connected or self._lidar is None:
            raise RuntimeError("RPLIDAR is not connected")
        return self._lidar.get_health()


def discover_rplidars() -> list:
    """Discover connected RPLIDAR devices."""
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
