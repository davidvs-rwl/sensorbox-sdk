#!/usr/bin/env python3
"""Record CSI cameras + LIDAR data (CONF01)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sensorbox.sync import SyncedSensorFusion
from sensorbox.storage import HDF5Writer
from sensorbox.core.config import SensorConfig, generate_filename


def main():
    parser = argparse.ArgumentParser(description="Record CSI + LIDAR data")
    parser.add_argument("-t", "--duration", type=float, default=10.0, help="Duration (seconds)")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file")
    parser.add_argument("-c", "--cameras", type=int, nargs="+", default=[0, 1], help="CSI camera IDs")
    parser.add_argument("-l", "--lidar", type=str, default="/dev/ttyUSB0", help="LIDAR port")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = generate_filename(SensorConfig.CONF01)
    
    print(f"=== CSI + LIDAR Recording (CONF01) ===")
    print(f"Output: {args.output}")
    print(f"Duration: {args.duration}s @ {args.fps} FPS")
    print(f"CSI Cameras: {args.cameras}")
    print(f"LIDAR: {args.lidar}")
    print()
    
    with SyncedSensorFusion(
        camera_ids=args.cameras,
        lidar_port=args.lidar,
        oakd_enabled=False,
    ) as fusion:
        with HDF5Writer(args.output, compression=None) as writer:
            writer.set_metadata({
                "config": "CONF01",
                "config_description": SensorConfig.CONF01.value,
            })
            
            print("Recording... (Ctrl+C to stop)")
            frame_count = 0
            lidar_count = 0
            
            try:
                for frame in fusion.stream(duration=args.duration, target_fps=args.fps):
                    writer.write_cameras(frame.cameras)
                    frame_count += 1
                    
                    if frame.has_lidar:
                        writer.write_lidar(frame.lidar)
                        lidar_count += 1
                    
                    print(f"  CSI: {frame_count}, LIDAR: {lidar_count}", end="\r")
            except KeyboardInterrupt:
                print("\nStopped by user")
            
            print(f"\n\n=== Complete ===")
            print(f"CSI Frames: {frame_count}")
            print(f"LIDAR Scans: {lidar_count}")
    
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
