#!/usr/bin/env python3
"""
Example: RPLIDAR capture on NVIDIA Jetson.

This script demonstrates:
1. Discovering RPLIDAR devices
2. Streaming scan data
3. Basic point cloud analysis
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sensorbox.drivers.rplidar import RPLidarSensor, discover_rplidars


def main():
    parser = argparse.ArgumentParser(description="Capture scans from RPLIDAR")
    parser.add_argument(
        "--port", "-p",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--scans", "-n",
        type=int,
        default=10,
        help="Number of scans to capture (default: 10)",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover RPLIDAR devices and exit",
    )
    args = parser.parse_args()
    
    if args.discover:
        print("Discovering RPLIDAR devices...")
        devices = discover_rplidars()
        if not devices:
            print("No RPLIDAR devices found.")
        else:
            for d in devices:
                print(f"\n  Port: {d['port']}")
                print(f"  Model: {d['model']}")
                print(f"  Serial: {d['serial']}")
                print(f"  Firmware: {d['firmware']}")
        return
    
    print(f"Connecting to RPLIDAR on {args.port}...")
    
    try:
        with RPLidarSensor(args.port) as lidar:
            print(f"Connected: {lidar.metadata.model}")
            print(f"Serial: {lidar.metadata.serial_number}")
            print(f"\nCapturing {args.scans} scans...\n")
            
            total_points = 0
            for frame in lidar.stream(max_frames=args.scans):
                scan = frame.data
                total_points += len(scan)
                
                min_dist = scan[:, 1].min()
                max_dist = scan[:, 1].max()
                avg_dist = scan[:, 1].mean()
                
                print(f"  Scan {frame.sequence_number:3d} | "
                      f"Points: {len(scan):3d} | "
                      f"Range: {min_dist:5.0f} - {max_dist:5.0f}mm | "
                      f"Avg: {avg_dist:5.0f}mm")
            
            print(f"\nTotal: {total_points} points from {args.scans} scans")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
