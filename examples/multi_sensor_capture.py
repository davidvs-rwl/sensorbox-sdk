#!/usr/bin/env python3
"""
Example: Multi-sensor capture (cameras + LIDAR).
"""

import argparse
import sys
import cv2
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from sensorbox.drivers.sensor_fusion import SensorFusion


def main():
    parser = argparse.ArgumentParser(description="Capture from cameras + LIDAR")
    parser.add_argument("--cameras", "-c", type=int, nargs="+", default=[0, 1],
                        help="Camera IDs (default: 0 1)")
    parser.add_argument("--lidar", "-l", type=str, default="/dev/ttyUSB0",
                        help="LIDAR port (default: /dev/ttyUSB0)")
    parser.add_argument("--no-lidar", action="store_true",
                        help="Disable LIDAR")
    parser.add_argument("--duration", "-t", type=float, default=5.0,
                        help="Duration in seconds (default: 5.0)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Target FPS (default: 10.0)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for saved data")
    args = parser.parse_args()
    
    lidar_port = None if args.no_lidar else args.lidar
    
    print(f"=== Multi-Sensor Capture ===")
    print(f"Cameras: {args.cameras}")
    print(f"LIDAR: {lidar_port or 'disabled'}")
    print(f"Duration: {args.duration}s @ {args.fps} FPS\n")
    
    # Setup output
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        for cam_id in args.cameras:
            (output_dir / f"cam{cam_id}").mkdir(exist_ok=True)
        print(f"Saving to: {output_dir}\n")
    
    fusion = SensorFusion(
        camera_ids=args.cameras,
        lidar_port=lidar_port,
    )
    
    with fusion:
        print("Streaming...\n")
        
        frame_count = 0
        lidar_count = 0
        
        for frame in fusion.stream(duration=args.duration, target_fps=args.fps):
            frame_count += 1
            
            # Save camera frames
            if output_dir:
                for cam_id in args.cameras:
                    cam_frame = frame.camera(cam_id)
                    if cam_frame:
                        path = output_dir / f"cam{cam_id}" / f"frame_{frame_count:06d}.jpg"
                        cv2.imwrite(str(path), cam_frame.data)
            
            # Count LIDAR frames
            if frame.has_lidar:
                lidar_count += 1
            
            # Progress
            if frame_count % 10 == 0:
                lidar_status = f"{len(frame.lidar.data)}pts" if frame.has_lidar else "---"
                print(f"  Frame {frame_count:4d} | LIDAR: {lidar_status}")
        
        print(f"\n=== Complete ===")
        print(f"Frames: {frame_count}")
        print(f"LIDAR scans: {lidar_count}")
        print(f"Effective FPS: {frame_count / args.duration:.1f}")


if __name__ == "__main__":
    main()
