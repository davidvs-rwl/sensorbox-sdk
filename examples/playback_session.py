#!/usr/bin/env python3
"""Playback recorded sensor data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sensorbox.storage import HDF5Reader


def main():
    parser = argparse.ArgumentParser(description="Playback HDF5 recording")
    parser.add_argument("file", type=str, help="HDF5 recording file")
    parser.add_argument("--info", "-i", action="store_true",
                        help="Show info only")
    parser.add_argument("--frames", "-n", type=int, default=10,
                        help="Number of frames to show")
    args = parser.parse_args()
    
    with HDF5Reader(args.file) as reader:
        info = reader.info
        
        print(f"=== Recording Info ===")
        print(f"File: {info.filepath}")
        print(f"Created: {info.created}")
        print(f"Duration: {info.duration_seconds:.1f}s")
        print(f"Frames: {info.frame_count}")
        print(f"Cameras: {info.cameras}")
        print(f"LIDAR scans: {info.lidar_scan_count}")
        print(f"Metadata: {info.user_metadata}")
        
        if args.info:
            return
        
        print(f"\n=== Playback (first {args.frames} frames) ===")
        for i, frame in enumerate(reader.playback()):
            if i >= args.frames:
                break
            
            cam_shapes = {k: v.shape for k, v in frame.cameras.items()}
            lidar_info = f"{len(frame.lidar)} pts" if frame.lidar is not None else "---"
            
            print(f"  Frame {i:3d} | ts={frame.timestamp:.2f}s | "
                  f"cams={cam_shapes} | lidar={lidar_info}")


if __name__ == "__main__":
    main()
