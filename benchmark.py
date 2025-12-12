"""Performance benchmark for SensorBox SDK."""

import time
import numpy as np
import psutil
import os
import json

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_cpu_info():
    """Show CPU information."""
    print("=" * 60)
    print("SYSTEM INFO")
    print("=" * 60)
    print(f"  CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
    print(f"  CPU usage: {psutil.cpu_percent(interval=1)}%")
    print(f"  RAM: {psutil.virtual_memory().total / 1024**3:.1f}GB total, {psutil.virtual_memory().available / 1024**3:.1f}GB available")
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  GPU: {result.stdout.strip()}")
    except:
        print("  GPU: nvidia-smi not available")


def benchmark_oakd_capture(duration=10, fps=30):
    """Benchmark OAK-D capture performance with blocking reads."""
    from sensorbox.drivers.oakd import OakDPro
    
    print("\n" + "=" * 60)
    print("BENCHMARK: OAK-D Capture (Synchronized)")
    print("=" * 60)
    
    oak = OakDPro(
        rgb_size=(1280, 720),
        depth_enabled=True,
        imu_enabled=True,
        fps=fps,
        confidence_threshold=100,
    )
    oak.connect()
    
    # Warmup - wait for pipeline to stabilize
    print("  Warming up...")
    time.sleep(1.0)
    for _ in range(30):
        oak.read()
        time.sleep(0.01)
    
    frame_times = []
    rgb_count = 0
    depth_count = 0
    both_count = 0
    
    start_time = time.perf_counter()
    start_mem = get_memory_usage()
    
    while (time.perf_counter() - start_time) < duration:
        t0 = time.perf_counter()
        frame = oak.read()
        t1 = time.perf_counter()
        
        if frame:
            has_rgb = frame.rgb is not None
            has_depth = frame.depth is not None
            
            if has_rgb:
                rgb_count += 1
            if has_depth:
                depth_count += 1
            if has_rgb and has_depth:
                both_count += 1
                frame_times.append((t1 - t0) * 1000)
        
        time.sleep(0.001)
    
    oak.disconnect()
    end_mem = get_memory_usage()
    
    elapsed = time.perf_counter() - start_time
    
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  RGB frames: {rgb_count} ({rgb_count/elapsed:.1f} FPS)")
    print(f"  Depth frames: {depth_count} ({depth_count/elapsed:.1f} FPS)")
    print(f"  Synced frames (both): {both_count} ({both_count/elapsed:.1f} FPS)")
    if frame_times:
        print(f"  Read latency: {np.mean(frame_times):.2f}ms avg, {np.max(frame_times):.2f}ms max")
    print(f"  Memory: {start_mem:.1f}MB -> {end_mem:.1f}MB (+{end_mem-start_mem:.1f}MB)")
    
    return {
        "rgb_fps": rgb_count / elapsed,
        "depth_fps": depth_count / elapsed,
        "synced_fps": both_count / elapsed,
        "read_latency_ms": np.mean(frame_times) if frame_times else 0,
        "memory_mb": end_mem,
    }


def benchmark_depth_processing(iterations=100):
    """Benchmark depth colorization and point cloud generation."""
    import cv2
    from sensorbox.drivers.oakd import OakDPro
    from sensorbox.core.pointcloud import depth_to_pointcloud
    
    print("\n" + "=" * 60)
    print("BENCHMARK: Depth Processing (CPU)")
    print("=" * 60)
    
    # Capture a sample depth frame
    oak = OakDPro(depth_enabled=True, confidence_threshold=50)
    oak.connect()
    time.sleep(0.5)
    
    depth = None
    for _ in range(50):
        frame = oak.read()
        if frame and frame.depth is not None:
            valid_pct = (frame.depth > 0).sum() / frame.depth.size * 100
            if valid_pct > 5:  # Get a good frame
                depth = frame.depth.copy()
                break
        time.sleep(0.02)
    
    oak.disconnect()
    
    if depth is None:
        print("  ERROR: Could not capture depth frame")
        return None
    
    valid_pct = (depth > 0).sum() / depth.size * 100
    print(f"  Depth shape: {depth.shape}, valid: {valid_pct:.1f}%")
    
    # Benchmark colorization
    def colorize_depth(d):
        valid = d > 0
        depth_norm = np.zeros_like(d, dtype=np.float32)
        if valid.any():
            depth_norm[valid] = d[valid]
            max_val = np.percentile(depth_norm[valid], 95)
            depth_norm = np.clip(depth_norm / max_val * 255, 0, 255).astype(np.uint8)
            colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
            colored[~valid] = 0
            return colored
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(20):
        colorize_depth(depth)
        depth_to_pointcloud(depth, subsample=4)
    
    # Benchmark colorization
    times_color = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        colorize_depth(depth)
        times_color.append((time.perf_counter() - t0) * 1000)
    
    print(f"\n  Colorization:")
    print(f"    Time: {np.mean(times_color):.2f}ms avg, {np.std(times_color):.2f}ms std")
    print(f"    Throughput: {1000/np.mean(times_color):.1f} FPS")
    
    # Benchmark point cloud at different subsample rates
    print(f"\n  Point Cloud Generation:")
    pc_results = {}
    for subsample in [1, 2, 4]:
        times = []
        point_counts = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            points = depth_to_pointcloud(depth, subsample=subsample)
            times.append((time.perf_counter() - t0) * 1000)
            point_counts.append(len(points))
        
        avg_points = int(np.mean(point_counts))
        avg_time = np.mean(times)
        print(f"    Subsample={subsample}: {avg_time:.2f}ms, {avg_points:,} points, {1000/avg_time:.1f} FPS")
        pc_results[f"pointcloud_sub{subsample}_ms"] = avg_time
        pc_results[f"pointcloud_sub{subsample}_fps"] = 1000 / avg_time
    
    return {
        "colorize_ms": np.mean(times_color),
        "colorize_fps": 1000 / np.mean(times_color),
        **pc_results,
    }


def benchmark_full_pipeline(duration=10):
    """Benchmark the full capture + processing pipeline."""
    import cv2
    from sensorbox.drivers.oakd import OakDPro
    from sensorbox.core.pointcloud import depth_to_pointcloud
    
    print("\n" + "=" * 60)
    print("BENCHMARK: Full Pipeline (Capture + Colorize + PointCloud)")
    print("=" * 60)
    
    oak = OakDPro(
        rgb_size=(1280, 720),
        depth_enabled=True,
        fps=30,
        confidence_threshold=50,
    )
    oak.connect()
    time.sleep(0.5)
    
    frame_count = 0
    depth_frame_count = 0
    capture_times = []
    colorize_times = []
    pointcloud_times = []
    total_times = []
    
    start_time = time.perf_counter()
    
    while (time.perf_counter() - start_time) < duration:
        t_start = time.perf_counter()
        
        # Capture
        t0 = time.perf_counter()
        frame = oak.read()
        t1 = time.perf_counter()
        
        if frame and frame.rgb is not None:
            frame_count += 1
            capture_times.append((t1 - t0) * 1000)
            
            if frame.depth is not None:
                depth = frame.depth
                depth_frame_count += 1
                
                # Colorize
                t2 = time.perf_counter()
                valid = depth > 0
                if valid.any():
                    depth_norm = np.zeros_like(depth, dtype=np.float32)
                    depth_norm[valid] = depth[valid]
                    max_val = np.percentile(depth_norm[valid], 95)
                    depth_norm = np.clip(depth_norm / max_val * 255, 0, 255).astype(np.uint8)
                    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
                t3 = time.perf_counter()
                
                # Point cloud
                points = depth_to_pointcloud(depth, subsample=4)
                t4 = time.perf_counter()
                
                colorize_times.append((t3 - t2) * 1000)
                pointcloud_times.append((t4 - t3) * 1000)
                total_times.append((t4 - t_start) * 1000)
        
        time.sleep(0.001)
    
    oak.disconnect()
    
    elapsed = time.perf_counter() - start_time
    
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Total frames: {frame_count} ({frame_count/elapsed:.1f} FPS)")
    print(f"  Depth frames: {depth_frame_count} ({depth_frame_count/elapsed:.1f} FPS)")
    
    if capture_times:
        print(f"  Capture: {np.mean(capture_times):.2f}ms avg")
    if colorize_times:
        print(f"  Colorize: {np.mean(colorize_times):.2f}ms avg")
    if pointcloud_times:
        print(f"  PointCloud: {np.mean(pointcloud_times):.2f}ms avg")
    if total_times:
        print(f"  Total per frame: {np.mean(total_times):.2f}ms avg")
        print(f"  Max achievable FPS: {1000/np.mean(total_times):.1f}")
    
    return {
        "total_fps": frame_count / elapsed,
        "depth_fps": depth_frame_count / elapsed,
        "capture_ms": np.mean(capture_times) if capture_times else 0,
        "colorize_ms": np.mean(colorize_times) if colorize_times else 0,
        "pointcloud_ms": np.mean(pointcloud_times) if pointcloud_times else 0,
        "total_ms": np.mean(total_times) if total_times else 0,
    }


def main():
    print("\n" + "=" * 60)
    print("  SENSORBOX PERFORMANCE BENCHMARK")
    print("=" * 60 + "\n")
    
    benchmark_cpu_info()
    
    results = {}
    
    # Run benchmarks
    r = benchmark_oakd_capture(duration=10, fps=30)
    if r:
        results.update(r)
    
    r = benchmark_depth_processing(iterations=100)
    if r:
        results.update(r)
    
    r = benchmark_full_pipeline(duration=10)
    if r:
        results.update(r)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (Baseline - CPU)")
    print("=" * 60)
    for key, value in sorted(results.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Save results
    with open("benchmark_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_baseline.json")


if __name__ == "__main__":
    main()
