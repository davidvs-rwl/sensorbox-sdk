# SensorBox SDK

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python SDK for ingesting synchronized data from Arducams and RPLIDAR sensors. Built for robotics, autonomous systems, and multi-sensor data collection.

## Features

- 🎥 **Arducam Support** — USB UVC cameras via OpenCV
- 📡 **RPLIDAR Support** — A1/A2/A3 laser scanners *(coming soon)*
- 🔄 **Unified Frame Format** — Consistent data structure across all sensor types
- ⏱️ **Time Synchronization** — Aligned timestamps for multi-sensor fusion
- 💾 **HDF5 Storage** — Efficient, chunked dataset storage *(coming soon)*

## Installation

```bash
git clone https://github.com/davidvs-rwl/sensorbox-sdk.git
cd sensorbox-sdk
pip install -e .
```

## Quick Start

```python
from sensorbox import ArducamSensor, discover_cameras

# Discover cameras
cameras = discover_cameras()
print(f"Found {len(cameras)} cameras")

# Stream from a camera
with ArducamSensor(device_index=0) as camera:
    for frame in camera.stream(duration=5.0):
        print(f"Frame {frame.sequence_number}: {frame.shape}")
```

## Command Line

```bash
# Discover cameras
python examples/camera_capture.py --discover

# Capture for 10 seconds
python examples/camera_capture.py --duration 10

# Save frames
python examples/camera_capture.py --duration 5 --output ./captures
```

## License

MIT
