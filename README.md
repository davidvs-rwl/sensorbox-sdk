# SensorBox SDK

Multi-sensor data capture and room measurement platform for Jetson Orin Nano.

## Hardware Support

| Sensor | Status | Description |
|--------|--------|-------------|
| OAK-D Pro | ✅ | RGB, Stereo Depth, IMU |
| RPLIDAR A1 | ✅ | 360° 2D LiDAR |
| CSI Cameras | ✅ | IMX219/IMX477 |

## Quick Start

### Room Measurement Dashboard
```bash
./run_dashboard.sh
# Or directly:
python3 -m streamlit run sensorbox/dashboard/room_measurement.py --server.port 8503
```

Access at `http://<jetson-ip>:8503`

### Live Streaming Dashboard
```bash
python3 -m streamlit run sensorbox/dashboard/live.py --server.port 8502
```

### Recording Data
```bash
# Record OAK-D data
python examples/record_oakd.py -t 30 --fps 15

# Record RPLIDAR data  
python examples/record_rplidar.py -t 30
```

## Room Measurement

The room measurement system combines:
- **RPLIDAR**: 360° scan for floor plan (length × width)
- **OAK-D Pro**: Depth sensing for height measurement

### Usage

1. Click **Connect Sensors**
2. Position sensors with clear view of room
3. Click **Capture Data**
4. Click **Measure Room**

### Accuracy

| Range | Accuracy |
|-------|----------|
| < 3m | ±5cm |
| 3-5m | ±10cm |
| > 5m | ±15cm+ |

## Project Structure
```
sensorbox/
├── drivers/          # Sensor drivers
│   ├── oakd.py       # OAK-D Pro (RGB, Depth, IMU)
│   ├── rplidar.py    # RPLIDAR A1
│   └── csi_camera.py # CSI cameras
├── core/             # Core utilities
│   ├── pointcloud.py # Depth to 3D conversion
│   └── storage.py    # HDF5 recording
├── measurement/      # Room measurement
│   ├── room.py       # Main measurement class
│   ├── wall_detector.py
│   └── plane_detector.py
├── dashboard/        # Streamlit dashboards
│   ├── room_measurement.py  # Main dashboard
│   └── live.py       # Live streaming
└── live/             # Live streaming module
    └── stream.py
```

## Performance

Baseline (CPU, Jetson Orin Nano):

| Metric | Value |
|--------|-------|
| RGB Capture | 26 FPS |
| Depth Capture | 23 FPS |
| Point Cloud | 0.2-2ms |
| Full Pipeline | 23 FPS |

## Requirements

- JetPack 6.0+
- Python 3.10+
- DepthAI v3
- Streamlit
- NumPy, OpenCV

## License

MIT
