#!/usr/bin/env python3
"""Record OAK-D Pro data with automatic filename."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sensorbox.drivers.oakd import OakDPro
from sensorbox.storage.oakd_writer import OakDHDF5Writer
from sensorbox.core.config import SensorConfig, generate_filename


def main():
    parser = argparse.ArgumentParser(description="Record OAK-D Pro data")
    parser.add_argument("-t", "--duration", type=float, default=10.0, help="Duration (seconds)")
    parser.add_argument("--fps", type=float, default=10.0, help="Target FPS")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file (auto-generated if not specified)")
    parser.add_argument("--no-depth", action="store_true", help="Disable depth")
    parser.add_argument("--no-imu", action="store_true", help="Disable IMU")
    args = parser.parse_args()
    
    # Generate filename if not provided
    if args.output is None:
        args.output = generate_filename(SensorConfig.CONF03)
    
    print(f"=== OAK-D Pro Recording (CONF03) ===")
    print(f"Output: {args.output}")
    print(f"Duration: {args.duration}s @ {args.fps} FPS")
    print(f"Depth: {'enabled' if not args.no_depth else 'disabled'}")
    print(f"IMU: {'enabled' if not args.no_imu else 'disabled'}")
    print()
    
    with OakDPro(
        rgb_size=(1280, 720),
        depth_enabled=not args.no_depth,
        imu_enabled=not args.no_imu,
    ) as oak:
        with OakDHDF5Writer(args.output) as writer:
            writer.set_metadata({
                "config": "CONF03",
                "config_description": SensorConfig.CONF03.value,
            })
            
            print("Recording... (Ctrl+C to stop)")
            try:
                for frame in oak.stream(duration=args.duration, target_fps=args.fps):
                    writer.write(frame)
                    print(f"  Frames: {writer.frame_count}", end="\r")
            except KeyboardInterrupt:
                print("\nStopped by user")
            
            print(f"\n\n=== Complete ===")
            print(f"RGB Frames: {writer.frame_count}")
    
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
