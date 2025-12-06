#!/usr/bin/env python3
"""SensorBox CLI - Multi-sensor data collection toolkit."""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import json


def cmd_record(args):
    """Record sensor data to HDF5."""
    from ..sync import SyncedSensorFusion
    from ..storage import HDF5Writer
    
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
    print()
    
    metadata = {"recorded_at": datetime.now().isoformat()}
    if args.name:
        metadata["name"] = args.name
    
    print("Recording... (Ctrl+C to stop early)")
    try:
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
                    
                    if frame_count % 10 == 0:
                        print(f"  {frame_count} frames, {lidar_count} LIDAR scans", end="\r")
                
                print()
                print(f"\n=== Complete ===")
                print(f"Frames: {frame_count}")
                print(f"LIDAR scans: {lidar_count}")
    
    except KeyboardInterrupt:
        print("\n\nRecording stopped by user.")
    
    if Path(args.output).exists():
        size_mb = Path(args.output).stat().st_size / (1024 * 1024)
        print(f"Saved: {args.output} ({size_mb:.1f} MB)")


def cmd_playback(args):
    """Playback recorded data."""
    from ..storage import HDF5Reader
    
    with HDF5Reader(args.file) as reader:
        info = reader.info
        
        print(f"=== {Path(args.file).name} ===")
        print(f"Created: {info.created}")
        print(f"Duration: {info.duration_seconds:.1f}s")
        print(f"Frames: {info.frame_count}")
        print(f"Cameras: {info.cameras}")
        print(f"LIDAR scans: {info.lidar_scan_count}")
        
        if info.user_metadata:
            print(f"Metadata: {info.user_metadata}")
        
        if args.info:
            return
        
        print(f"\n=== Frames ===")
        for i, frame in enumerate(reader.playback()):
            if i >= args.frames:
                break
            
            lidar = f"{len(frame.lidar)}pts" if frame.lidar is not None else "---"
            print(f"  {i:4d} | t={frame.timestamp:.2f}s | cams={list(frame.cameras.keys())} | lidar={lidar}")


def cmd_info(args):
    """Show info about sensors or recordings."""
    if args.file:
        # Show recording info
        from ..storage import HDF5Reader
        
        with HDF5Reader(args.file) as reader:
            info = reader.info
            print(json.dumps({
                "file": str(info.filepath),
                "created": info.created,
                "duration_seconds": info.duration_seconds,
                "frame_count": info.frame_count,
                "cameras": info.cameras,
                "lidar_scans": info.lidar_scan_count,
                "metadata": info.user_metadata,
            }, indent=2))
    else:
        # Show available sensors
        print("=== Available Sensors ===\n")
        
        print("Cameras:")
        try:
            import subprocess
            result = subprocess.run(["ls", "/dev/video*"], capture_output=True, text=True, shell=True)
            videos = [v.strip() for v in result.stdout.strip().split() if v]
            if videos:
                for v in videos:
                    print(f"  {v}")
            else:
                print("  No cameras found")
        except:
            print("  Could not detect cameras")
        
        print("\nLIDAR:")
        try:
            from ..drivers.rplidar import discover_rplidars
            lidars = discover_rplidars()
            if lidars:
                for l in lidars:
                    print(f"  {l['port']} - Model {l['model']} (SN: {l['serial'][:8]}...)")
            else:
                print("  No RPLIDAR found")
        except Exception as e:
            print(f"  Could not detect LIDAR: {e}")


def cmd_export(args):
    """Export recording to other formats."""
    from ..storage import HDF5Reader
    import cv2
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with HDF5Reader(args.file) as reader:
        info = reader.info
        
        print(f"Exporting {info.frame_count} frames to {output_dir}/")
        
        # Create subdirectories
        for cam_id in info.cameras:
            (output_dir / f"cam_{cam_id}").mkdir(exist_ok=True)
        
        if info.has_lidar:
            (output_dir / "lidar").mkdir(exist_ok=True)
        
        # Export frames
        lidar_scans = reader.get_all_lidar_scans() if info.has_lidar else []
        lidar_idx = 0
        
        for i, frame in enumerate(reader.playback()):
            # Save camera frames
            for cam_id, img in frame.cameras.items():
                path = output_dir / f"cam_{cam_id}" / f"frame_{i:06d}.jpg"
                cv2.imwrite(str(path), img)
            
            # Save LIDAR if present
            if frame.lidar is not None and lidar_idx < len(lidar_scans):
                import numpy as np
                path = output_dir / "lidar" / f"scan_{lidar_idx:06d}.csv"
                np.savetxt(path, frame.lidar, delimiter=",", header="angle,distance,quality", comments="")
                lidar_idx += 1
            
            if (i + 1) % 10 == 0:
                print(f"  Exported {i + 1}/{info.frame_count} frames", end="\r")
        
        print(f"\nExported to: {output_dir}/")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sensorbox",
        description="SensorBox - Multi-sensor data collection toolkit",
    )
    parser.add_argument("--version", action="version", version="sensorbox 0.1.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Record command
    rec = subparsers.add_parser("record", help="Record sensor data")
    rec.add_argument("-o", "--output", type=str, help="Output file (.h5)")
    rec.add_argument("-t", "--duration", type=float, default=30.0, help="Duration (seconds)")
    rec.add_argument("--fps", type=float, default=10.0, help="Target FPS")
    rec.add_argument("-c", "--cameras", type=int, nargs="+", default=[0, 1], help="Camera IDs")
    rec.add_argument("-l", "--lidar", type=str, default="/dev/ttyUSB0", help="LIDAR port")
    rec.add_argument("--no-lidar", action="store_true", help="Disable LIDAR")
    rec.add_argument("--no-compression", action="store_true", help="Disable compression")
    rec.add_argument("-n", "--name", type=str, help="Recording name")
    rec.set_defaults(func=cmd_record)
    
    # Playback command
    play = subparsers.add_parser("playback", help="Playback recording")
    play.add_argument("file", type=str, help="HDF5 file")
    play.add_argument("-i", "--info", action="store_true", help="Show info only")
    play.add_argument("-n", "--frames", type=int, default=20, help="Frames to show")
    play.set_defaults(func=cmd_playback)
    
    # Info command
    info = subparsers.add_parser("info", help="Show sensor/recording info")
    info.add_argument("file", type=str, nargs="?", help="HDF5 file (optional)")
    info.set_defaults(func=cmd_info)
    
    # Export command
    exp = subparsers.add_parser("export", help="Export to images/CSV")
    exp.add_argument("file", type=str, help="HDF5 file")
    exp.add_argument("-o", "--output", type=str, default="./export", help="Output directory")
    exp.set_defaults(func=cmd_export)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
