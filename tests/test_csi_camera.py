"""Tests for CSI camera driver."""

import pytest
from unittest.mock import Mock, patch
from sensorbox.drivers.csi_camera import (
    CSICamera,
    CameraConnectionError,
    list_resolutions,
    RESOLUTIONS,
)


class TestCSICameraUnit:
    """Unit tests (no hardware required)."""
    
    def test_resolution_presets(self):
        """Test resolution presets are valid."""
        resolutions = list_resolutions()
        assert "720p" in resolutions
        assert "1080p" in resolutions
        assert "4K" in resolutions
        
        for name, (w, h, fps) in resolutions.items():
            assert w > 0
            assert h > 0
            assert fps > 0
    
    def test_invalid_resolution_raises(self):
        """Test that invalid resolution raises ValueError."""
        with pytest.raises(ValueError, match="Unknown resolution"):
            CSICamera(sensor_id=0, resolution="invalid")
    
    def test_gstreamer_pipeline_format(self):
        """Test GStreamer pipeline is properly formatted."""
        cam = CSICamera(sensor_id=0, width=1280, height=720, fps=30)
        pipeline = cam.gstreamer_pipeline
        
        assert "nvarguscamerasrc" in pipeline
        assert "sensor-id=0" in pipeline
        assert "width=1280" in pipeline
        assert "height=720" in pipeline
        assert "framerate=30/1" in pipeline
    
    def test_camera_properties(self):
        """Test camera property accessors."""
        cam = CSICamera(sensor_id=1, width=1920, height=1080, fps=60)
        
        assert cam.sensor_id == "csi_camera_1"
        assert cam.width == 1920
        assert cam.height == 1080
        assert cam.fps == 60
    
    def test_resolution_preset_overrides_dimensions(self):
        """Test that resolution preset overrides manual dimensions."""
        cam = CSICamera(sensor_id=0, width=100, height=100, resolution="720p")
        
        assert cam.width == 1280
        assert cam.height == 720
    
    def test_stats_initial_values(self):
        """Test initial stats are zero."""
        cam = CSICamera(sensor_id=0)
        stats = cam.stats
        
        assert stats["total_frames"] == 0
        assert stats["dropped_frames"] == 0
        assert stats["drop_rate"] == 0.0
    
    def test_read_without_connect_raises(self):
        """Test that reading without connecting raises error."""
        cam = CSICamera(sensor_id=0)
        
        with pytest.raises(RuntimeError, match="not connected"):
            cam.read()


class TestCSICameraIntegration:
    """Integration tests (require hardware)."""
    
    @pytest.mark.hardware
    def test_camera_connect_disconnect(self):
        """Test camera connection cycle."""
        cam = CSICamera(sensor_id=0)
        
        cam.connect()
        assert cam.is_connected
        
        cam.disconnect()
        assert not cam.is_connected
    
    @pytest.mark.hardware
    def test_camera_read_frame(self):
        """Test reading a single frame."""
        with CSICamera(sensor_id=0) as cam:
            frame = cam.read()
            
            assert frame is not None
            assert frame.shape == (720, 1280, 3)
            assert frame.sensor_type.value == "camera"
    
    @pytest.mark.hardware
    def test_camera_stream(self):
        """Test streaming frames."""
        with CSICamera(sensor_id=0) as cam:
            frames = list(cam.stream(max_frames=5))
            
            assert len(frames) == 5
            for i, frame in enumerate(frames):
                assert frame.sequence_number == i
