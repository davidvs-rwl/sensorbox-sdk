#!/usr/bin/env python3
"""
Example: Basic camera capture with the SensorBox SDK.

This script demonstrates:
1. Discovering available cameras
2. Connecting to a camera
3. Streaming frames
4. Saving frames to disk
"""

import argparse
import cv2
from pathlib import Path

from sensorbox import ArducamSensor, discover_cameras


def main():
    parser = argparse.ArgumentParser(description="Capture frames from an Arducam")
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--duration", "-t",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for saved frames (optional)",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover and list available cameras, then exit",
    )
    args = parser.parse_args()
    
    if args.discover:
        print("Discovering cameras...")
        cameras = discover_cameras()
        if not cameras:
            print("No cameras found.")
        else:
            print(f"Found {len(cameras)} camera(s):\n")
            for cam in cameras:
                print(f"  Index {cam['device_index']}:")
                print(f"    Resolution: {cam['width']}x{cam['height']}")
                print(f"    FPS: {cam['fps']}")
                print(f"    Backend: {cam['backend']}")
                print()
        return
    
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to: {output_dir}")
    
    print(f"Connecting to camera {args.device}...")
    
    with ArducamSensor(device_index=args.device) as camera:
        print(f"Connected: {camera.width}x{camera.height} @ {camera.fps:.1f} FPS")
        print(f"Streaming for {args.duration} seconds...")
        print()
        
        frame_count = 0
        for frame in camera.stream(duration=args.duration):
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(
                    f"  Frame {frame.sequence_number:4d} | "
                    f"Time: {frame.timestamp:6.2f}s | "
                    f"Size: {frame.nbytes / 1024:.0f} KB"
                )
            
            if output_dir:
                filename = output_dir / f"frame_{frame.sequence_number:06d}.jpg"
                cv2.imwrite(str(filename), frame.data)
        
        print()
        print(f"Captured {frame_count} frames")
        
        if output_dir:
            print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
