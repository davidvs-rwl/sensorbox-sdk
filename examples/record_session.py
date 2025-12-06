#!/usr/bin/env python3
"""Record multi-sensor data to HDF5."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sensorbox.sync import SyncedSensorFusion
from sensorbox.storage import HDF5Writer


def main():
    parser = argparse.ArgumentParser(description="Record sensor data to HDF5")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file (default: recording_TIMESTAMP.h5)")
    parser.add_argument("--duration", "-t", type=float, default=10.0,
                        help="Recording duration in seconds")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Target frame rate")
    parser.add_argument("--cameras", "-c", type=int, nargs="+", default=[0, 1],
                        help="Camera IDs")
    parser.add_argument("--lidar", "-l", type=str, default="/dev/ttyUSB0",
                        help="LIDAR port")
    parser.add_argument("--no-lidar", action="store_true",
                        help="Disable LIDAR")
    parser.add_argument("--no-compression", action="store_true",
                        help="Disable compression (faster recording)")
    parser.add_argument("--metadata", "-m", type=str, default=None,
                        help="Metadata as JSON string")
    args = parser.parse_args()
    
    # Generate output filename
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"recording_{timestamp}.h5"
    
    lidar_port = None if args.no_lidar else args.lidar
    compression = None if args.no_compression else "gzip"
    
    print(f"=== SensorBox Recording ===")
    print(f"Output: {args.output}")
    print(f"Duration: {args.duration}s @ {args.fps} FPS")
    print(f"Cameras: {args.cameras}")
    print(f"LIDAR: {lidar_port or 'disabled'}")
    print(f"Compression: {compression or 'none'}")
    print()
    
    # Parse metadata
    metadata = {"recorded_at": datetime.now().isoformat()}
    if args.metadata:
        import json
        metadata.update(json.loads(args.metadata))
    
    # Record
    print("Starting recording...")
    with SyncedSensorFusion(
        camera_ids=args.cameras,
        lidar_port=lidar_port,
    ) as fusion:
        with HDF5Writer(args.output, compression=compression) as writer:
            writer.set_metadata(metadata)
            
            frame_count = 0
            lidar_count = 0
            
            for frame in fusion.stream(duration=args.duration, target_fps=args.fps):
                writer.write_cameras(frame.cameras)
                frame_count += 1
                
                if frame.has_lidar:
                    writer.write_lidar(frame.lidar)
                    lidar_count += 1
                
                # Progress every 10 frames
                if frame_count % 10 == 0:
                    print(f"  Recorded {frame_count} frames, {lidar_count} LIDAR scans...")
            
            print()
            print(f"=== Recording Complete ===")
            print(f"Frames: {frame_count}")
            print(f"LIDAR scans: {lidar_count}")
    
    # Show file info
    file_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"File size: {file_size:.1f} MB")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
