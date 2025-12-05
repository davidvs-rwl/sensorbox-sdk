"""Tests for sensor fusion."""

import pytest
from sensorbox.drivers.sensor_fusion import SensorFusion, FusedFrame


class TestFusedFrame:
    """Tests for FusedFrame dataclass."""
    
    def test_empty_frame(self):
        """Test empty fused frame."""
        from datetime import datetime
        
        frame = FusedFrame(
            timestamp=0.0,
            wall_time=datetime.now(),
            cameras={},
            lidar=None,
        )
        
        assert frame.num_cameras == 0
        assert not frame.has_lidar
        assert frame.camera(0) is None
    
    def test_frame_with_cameras(self):
        """Test frame with camera data."""
        from datetime import datetime
        from unittest.mock import Mock
        
        mock_cam0 = Mock()
        mock_cam1 = Mock()
        
        frame = FusedFrame(
            timestamp=1.0,
            wall_time=datetime.now(),
            cameras={0: mock_cam0, 1: mock_cam1},
            lidar=None,
        )
        
        assert frame.num_cameras == 2
        assert frame.camera(0) == mock_cam0
        assert frame.camera(1) == mock_cam1
        assert frame.camera(2) is None


class TestSensorFusionUnit:
    """Unit tests (no hardware required)."""
    
    def test_initialization(self):
        """Test SensorFusion initialization."""
        fusion = SensorFusion(
            camera_ids=[0, 1],
            lidar_port="/dev/ttyUSB0",
        )
        
        assert fusion.camera_ids == [0, 1]
        assert fusion.has_lidar
        assert not fusion.is_connected
    
    def test_no_lidar(self):
        """Test initialization without LIDAR."""
        fusion = SensorFusion(
            camera_ids=[0],
            lidar_port=None,
        )
        
        assert not fusion.has_lidar
    
    def test_read_without_connect_raises(self):
        """Test reading without connecting raises error."""
        fusion = SensorFusion(camera_ids=[0])
        
        with pytest.raises(RuntimeError, match="not connected"):
            fusion.read()


class TestSensorFusionIntegration:
    """Integration tests (require hardware)."""
    
    @pytest.mark.hardware
    def test_fusion_cameras_only(self):
        """Test fusion with cameras only."""
        fusion = SensorFusion(
            camera_ids=[0, 1],
            lidar_port=None,
        )
        
        with fusion:
            frames = list(fusion.stream(max_frames=5, target_fps=10))
            
            assert len(frames) == 5
            for frame in frames:
                assert frame.num_cameras == 2
    
    @pytest.mark.hardware
    def test_fusion_all_sensors(self):
        """Test fusion with cameras and LIDAR."""
        fusion = SensorFusion(
            camera_ids=[0, 1],
            lidar_port="/dev/ttyUSB0",
        )
        
        with fusion:
            frames = list(fusion.stream(duration=2.0, target_fps=10))
            
            assert len(frames) > 0
            # At least some frames should have LIDAR data
            lidar_frames = [f for f in frames if f.has_lidar]
            assert len(lidar_frames) > 0
