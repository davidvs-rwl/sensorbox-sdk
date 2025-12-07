#!/usr/bin/env python3
"""Record CSI cameras + OAK-D Pro data (CONF02)."""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sensorbox.sync import SyncedSensorFusion
from sensorbox.storage import HDF5Writer
from sensorbox.storage.oakd_writer import OakDHDF5Writer
from sensorbox.core.config import SensorConfig, generate_filename


def main():
    parser = argparse.ArgumentParser(description="Record CSI + OAK-D data")
    parser.add_argument("-t", "--duration", type=float, default=10.0, help="Duration (seconds)")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file")
    parser.add_argument("-c", "--cameras", type=int, nargs="+", default=[0, 1], help="CSI camera IDs")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = generate_filename(SensorConfig.CONF02)
    
    print(f"=== CSI + OAK-D Recording (CONF02) ===")
    print(f"Output: {args.output}")
    print(f"Duration: {args.duration}s @ {args.fps} FPS")
    print(f"CSI Cameras: {args.cameras}")
    print()
    
    with SyncedSensorFusion(
        camera_ids=args.cameras,
        lidar_port=None,
        oakd_enabled=True,
    ) as fusion:
        with HDF5Writer(args.output, compression=None) as writer:
            writer.set_metadata({
                "config": "CONF02",
                "config_description": SensorConfig.CONF02.value,
            })
            
            print("Recording... (Ctrl+C to stop)")
            frame_count = 0
            oakd_count = 0
            
            try:
                for frame in fusion.stream(duration=args.duration, target_fps=args.fps):
                    writer.write_cameras(frame.cameras)
                    frame_count += 1
                    
                    if frame.has_oakd:
                        oakd_count += 1
                    
                    print(f"  CSI: {frame_count}, OAK-D: {oakd_count}", end="\r")
            except KeyboardInterrupt:
                print("\nStopped by user")
            
            print(f"\n\n=== Complete ===")
            print(f"CSI Frames: {frame_count}")
            print(f"OAK-D Frames: {oakd_count}")
    
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
