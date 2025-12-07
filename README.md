# SensorBox SDK

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python SDK for synchronized multi-sensor data capture on NVIDIA Jetson. Supports CSI cameras and RPLIDAR laser scanners.

## Features

- 🎥 **Dual CSI Cameras** — IMX219, IMX477 support via GStreamer
- 📡 **RPLIDAR** — A1/A2/A3 360° laser scanners
- 🔄 **Sensor Fusion** — Synchronized multi-sensor streaming
- 🛡️ **Error Handling** — Auto-reconnection, dropped frame detection
- ⏱️ **FPS Control** — Configurable target frame rates

## Installation
```bash
git clone https://github.com/davidvs-rwl/sensorbox-sdk.git
cd sensorbox-sdk
pip install -e .
```

**Note:** On Jetson, use system OpenCV (not pip) for GStreamer support:
```bash
pip uninstall opencv-python
sudo apt install python3-opencv
pip install "numpy<2"
```

## Quick Start

### Single Camera
```python
from sensorbox.drivers.csi_camera import CSICamera

with CSICamera(sensor_id=0, resolution="720p") as cam:
    for frame in cam.stream(duration=5.0, target_fps=30):
        print(f"Frame {frame.sequence_number}: {frame.shape}")
```

### Dual Cameras
```python
from sensorbox.drivers.multi_camera import MultiCamera

with MultiCamera([0, 1]) as cams:
    for multi_frame in cams.stream(duration=5.0):
        cam0 = multi_frame[0]
        cam1 = multi_frame[1]
```

### RPLIDAR
```python
from sensorbox.drivers.rplidar import RPLidarSensor

with RPLidarSensor('/dev/ttyUSB0') as lidar:
    for frame in lidar.stream(max_frames=10):
        scan = frame.data  # (N, 3) array: angle, distance, quality
        print(f"Scan: {len(scan)} points")
```

### Multi-Sensor Fusion
```python
from sensorbox.drivers.sensor_fusion import SensorFusion

fusion = SensorFusion(
    camera_ids=[0, 1],
    lidar_port='/dev/ttyUSB0',
)

with fusion:
    for frame in fusion.stream(duration=10.0, target_fps=10):
        cam0 = frame.camera(0)
        cam1 = frame.camera(1)
        if frame.has_lidar:
            scan = frame.lidar.data
```

## Command Line Examples
```bash
# Discover cameras
python examples/csi_capture.py --list-resolutions

# Capture from CSI camera
python examples/csi_capture.py --duration 5 --resolution 720p

# Capture from RPLIDAR
python examples/rplidar_capture.py --scans 20

# Multi-sensor capture
python examples/multi_sensor_capture.py --duration 10 --output ./captures
```

## Hardware Setup

### Jetson Camera Configuration

Enable cameras using:
```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

Select the appropriate configuration:
- **Camera IMX219-A** — Single IMX219 on CAM0
- **Camera IMX219 Dual** — Two IMX219 cameras
- **Camera IMX219-A and IMX477-C** — IMX219 on CAM0, IMX477 on CAM1

### RPLIDAR

Connect via USB. The device appears as `/dev/ttyUSB0`.

Add user to dialout group for serial access:
```bash
sudo usermod -a -G dialout $USER
```

## API Reference

### CSICamera
```python
CSICamera(
    sensor_id=0,              # CSI port (0 or 1)
    resolution="720p",        # "4K", "1080p", "720p", "480p"
    auto_reconnect=True,      # Reconnect on failure
)
```

### RPLidarSensor
```python
RPLidarSensor(
    port='/dev/ttyUSB0',      # Serial port
    auto_reconnect=True,      # Reconnect on failure
)
```

### SensorFusion
```python
SensorFusion(
    camera_ids=[0, 1],        # Camera IDs to use
    lidar_port='/dev/ttyUSB0', # LIDAR port (None to disable)
    camera_width=1280,
    camera_height=720,
)
```

## Testing
```bash
# Unit tests only
pytest tests/ -m "not hardware" -v

# All tests (requires hardware)
pytest tests/ -v
```

## Project Structure
```
sensorbox-sdk/
├── sensorbox/
│   ├── core/
│   │   ├── frame.py          # SensorFrame dataclass
│   │   └── sensor.py         # Base Sensor class
│   └── drivers/
│       ├── csi_camera.py     # CSI camera driver
│       ├── rplidar.py        # RPLIDAR driver
│       ├── multi_camera.py   # Multi-camera streaming
│       └── sensor_fusion.py  # Camera + LIDAR fusion
├── examples/
│   ├── csi_capture.py
│   ├── rplidar_capture.py
│   └── multi_sensor_capture.py
└── tests/
```

## License

MIT

## Quick Start Guide

### 1. Record Data
```bash
cd ~/sensorbox-sdk

# Basic recording (10 seconds)
sensorbox record -t 10 -o my_recording.h5

# Longer recording with name
sensorbox record -t 60 --name "warehouse" -o warehouse.h5

# Fast recording (no compression)
sensorbox record -t 30 --no-compression -o fast.h5

# Cameras only (no LIDAR)
sensorbox record -t 10 --no-lidar -o cameras.h5
```

### 2. Check Recording
```bash
# Show recording info
sensorbox playback my_recording.h5 --info

# View frames
sensorbox playback my_recording.h5 --frames 20
```

### 3. Launch Dashboard
```bash
cd ~/sensorbox-sdk
streamlit run sensorbox/dashboard/app.py --server.headless true --server.port 8501
```

Then open in browser: **http://192.168.50.163:8501**

Press `Ctrl+C` to stop the dashboard.

### CLI Options Reference

| Command | Description |
|---------|-------------|
| `sensorbox record -t 30` | Record 30 seconds |
| `sensorbox record --fps 5` | Record at 5 FPS |
| `sensorbox record --no-compression` | Faster, larger files |
| `sensorbox record --no-lidar` | Cameras only |
| `sensorbox playback file.h5 --info` | Show file info |
| `sensorbox info` | Show connected sensors |
| `sensorbox export file.h5 -o ./images` | Export to JPGs/CSV |

## Sensor Configurations

The SensorBox SDK supports multiple sensor configurations. Due to USB bandwidth and processing constraints on the Jetson, we recommend using one of these configurations:

### Configuration 1: CSI Cameras + OAK-D Pro (Recommended)

Best for: RGB-D applications, depth sensing, indoor navigation
```bash
# Record with CSI cameras and OAK-D Pro
python3 -m sensorbox.cli record -t 30 --no-lidar -o session.h5

# Or via Python
from sensorbox.sync import SyncedSensorFusion

with SyncedSensorFusion(camera_ids=[0, 1], oakd_enabled=True) as fusion:
    for frame in fusion.stream(duration=30.0, target_fps=10):
        csi_cam0 = frame.camera(0)
        csi_cam1 = frame.camera(1)
        if frame.has_oakd:
            rgb = frame.oakd.rgb        # (720, 1280, 3)
            depth = frame.oakd.depth    # (200, 320) depth map
            imu = frame.oakd.imu        # accelerometer + gyroscope
```

### Configuration 2: CSI Cameras + LIDAR

Best for: 360° scanning, outdoor navigation, SLAM
```bash
# Record with CSI cameras and LIDAR
sensorbox record -t 30 -o session.h5

# Or via Python
from sensorbox.sync import SyncedSensorFusion

with SyncedSensorFusion(camera_ids=[0, 1], lidar_port='/dev/ttyUSB0') as fusion:
    for frame in fusion.stream(duration=30.0, target_fps=10):
        csi_cam0 = frame.camera(0)
        csi_cam1 = frame.camera(1)
        if frame.has_lidar:
            scan = frame.lidar.data  # (N, 3): angle, distance, quality
```

### Configuration 3: OAK-D Pro Only

Best for: Compact setups, RGB-D only applications
```python
from sensorbox.drivers.oakd import OakDPro

with OakDPro(rgb_size=(1280, 720), depth_enabled=True, imu_enabled=True) as oak:
    for frame in oak.stream(duration=30.0, target_fps=10):
        rgb = frame.rgb
        depth = frame.depth
        if frame.imu:
            accel = frame.imu['accelerometer']
            gyro = frame.imu['gyroscope']
```

### ⚠️ Not Recommended: All Sensors Together

Running CSI cameras + LIDAR + OAK-D Pro simultaneously may cause LIDAR buffer overflows due to USB bandwidth constraints. Use one of the configurations above for best results.

## Hardware Setup

### Sensors Supported

| Sensor | Interface | Description |
|--------|-----------|-------------|
| IMX477 (CAM0) | CSI | 12.3MP HQ camera |
| IMX219 (CAM1) | CSI | 8MP camera |
| RPLIDAR A1 | USB | 360° laser scanner |
| OAK-D Pro | USB | RGB + Stereo Depth + IMU |

### OAK-D Pro Setup

1. Connect via USB-C (use USB 3.0 port for best performance)
2. Install udev rules:
```bash
   echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
   sudo udevadm control --reload-rules && sudo udevadm trigger
```
3. Install DepthAI:
```bash
   pip install depthai --user --break-system-packages
```

### RPLIDAR Setup

1. Connect via USB (data cable, not power-only)
2. Add user to dialout group:
```bash
   sudo usermod -a -G dialout $USER
```
3. Device appears at `/dev/ttyUSB0`
