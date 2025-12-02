#!/usr/bin/env python3
"""
Example: CSI camera capture on NVIDIA Jetson.

This script demonstrates:
1. Connecting to a CSI camera (CAM0/CAM1)
2. Using resolution presets
3. FPS limiting
4. Saving frames to disk
"""

import argparse
import cv2
from pathlib import Path
import sys

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

from sensorbox.drivers.csi_camera import CSICamera, list_resolutions


def main():
    parser = argparse.ArgumentParser(description="Capture frames from a Jetson CSI camera")
    parser.add_argument(
        "--sensor-id", "-s",
        type=int,
        default=0,
        help="CSI sensor ID (0 for CAM0, 1 for CAM1)",
    )
    parser.add_argument(
        "--resolution", "-r",
        type=str,
        choices=["4K", "1080p", "720p", "480p"],
        default="720p",
        help="Resolution preset (default: 720p)",
    )
    parser.add_argument(
        "--duration", "-t",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Target FPS for throttling (default: no limit)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for saved frames (optional)",
    )
    parser.add_argument(
        "--flip",
        type=int,
        default=0,
        help="Flip method: 0=none, 2=180° (default: 0)",
    )
    parser.add_argument(
        "--list-resolutions",
        action="store_true",
        help="List available resolutions and exit",
    )
    args = parser.parse_args()
    
    # List resolutions mode
    if args.list_resolutions:
        print("Available resolution presets:\n")
        for name, (w, h, fps) in list_resolutions().items():
            print(f"  {name:8s}: {w}x{h} @ {fps} FPS")
        return
    
    # Setup output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to: {output_dir}")
    
    # Connect and stream
    print(f"Connecting to CSI camera {args.sensor_id} ({args.resolution})...")
    
    try:
        with CSICamera(
            sensor_id=args.sensor_id,
            resolution=args.resolution,
            flip_method=args.flip,
        ) as camera:
            print(f"Connected: {camera.width}x{camera.height} @ {camera.fps} FPS")
            
            if args.target_fps:
                print(f"Throttling to {args.target_fps} FPS")
            
            print(f"Streaming for {args.duration} seconds...")
            print()
            
            frame_count = 0
            start_time = None
            
            for frame in camera.stream(duration=args.duration, target_fps=args.target_fps):
                frame_count += 1
                
                if start_time is None:
                    start_time = frame.timestamp
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    elapsed = frame.timestamp - start_time if start_time else 0
                    actual_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(
                        f"  Frame {frame.sequence_number:4d} | "
                        f"Time: {frame.timestamp:6.2f}s | "
                        f"FPS: {actual_fps:.1f} | "
                        f"Size: {frame.nbytes / 1024:.0f} KB"
                    )
                
                # Save frame if output directory specified
                if output_dir:
                    filename = output_dir / f"frame_{frame.sequence_number:06d}.jpg"
                    cv2.imwrite(str(filename), frame.data)
            
            print()
            elapsed = frame.timestamp - start_time if start_time and frame_count > 0 else args.duration
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Captured {frame_count} frames ({actual_fps:.1f} FPS actual)")
            
            if output_dir:
                print(f"Saved to: {output_dir}")
                
    except ConnectionError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
