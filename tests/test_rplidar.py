"""Tests for RPLIDAR driver."""

import pytest
from unittest.mock import Mock, patch
from sensorbox.drivers.rplidar import (
    RPLidarSensor,
    LidarConnectionError,
    discover_rplidars,
)


class TestRPLidarUnit:
    """Unit tests (no hardware required)."""
    
    def test_sensor_id_generation(self):
        """Test sensor ID is generated from port."""
        lidar = RPLidarSensor(port="/dev/ttyUSB0")
        assert lidar.sensor_id == "rplidar__dev_ttyUSB0"
    
    def test_custom_sensor_id(self):
        """Test custom sensor ID."""
        lidar = RPLidarSensor(port="/dev/ttyUSB0", sensor_id="my_lidar")
        assert lidar.sensor_id == "my_lidar"
    
    def test_port_property(self):
        """Test port property."""
        lidar = RPLidarSensor(port="/dev/ttyUSB1")
        assert lidar.port == "/dev/ttyUSB1"
    
    def test_stats_initial_values(self):
        """Test initial stats are zero."""
        lidar = RPLidarSensor(port="/dev/ttyUSB0")
        stats = lidar.stats
        
        assert stats["total_scans"] == 0
        assert stats["failed_scans"] == 0
        assert stats["failure_rate"] == 0.0
    
    def test_read_without_connect_raises(self):
        """Test that reading without connecting raises error."""
        lidar = RPLidarSensor(port="/dev/ttyUSB0")
        
        with pytest.raises(RuntimeError, match="not connected"):
            lidar.read()


class TestRPLidarIntegration:
    """Integration tests (require hardware)."""
    
    @pytest.mark.hardware
    def test_lidar_connect_disconnect(self):
        """Test LIDAR connection cycle."""
        lidar = RPLidarSensor(port="/dev/ttyUSB0")
        
        lidar.connect()
        assert lidar.is_connected
        assert lidar.metadata is not None
        
        lidar.disconnect()
        assert not lidar.is_connected
    
    @pytest.mark.hardware
    def test_lidar_read_scan(self):
        """Test reading a single scan."""
        with RPLidarSensor("/dev/ttyUSB0") as lidar:
            frame = lidar.read()
            
            assert frame is not None
            assert frame.sensor_type.value == "lidar"
            assert len(frame.data) > 0
            assert frame.data.shape[1] == 3  # angle, distance, quality
    
    @pytest.mark.hardware
    def test_lidar_health(self):
        """Test health check."""
        with RPLidarSensor("/dev/ttyUSB0") as lidar:
            health = lidar.get_health()
            assert health[0] == "Good"
    
    @pytest.mark.hardware
    def test_discover_rplidars(self):
        """Test RPLIDAR discovery."""
        devices = discover_rplidars()
        assert len(devices) >= 1
        assert "port" in devices[0]
        assert "model" in devices[0]
