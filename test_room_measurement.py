"""Test room measurement with RPLIDAR + OAK-D."""

from sensorbox.measurement import RoomMeasurement

def main():
    print("=" * 60)
    print("ROOM MEASUREMENT TEST")
    print("=" * 60)
    print()
    print("Instructions:")
    print("1. Place RPLIDAR horizontally with clear view of walls")
    print("2. Point OAK-D camera to see floor and ceiling if possible")
    print("3. Stand clear of the sensors")
    print()
    input("Press Enter to begin...")
    
    with RoomMeasurement(camera_height=0.5) as rm:
        print("\n--- Step 1: Capture RPLIDAR scan ---")
        rm.capture_lidar_scan(num_scans=5)
        
        print("\n--- Step 2: Capture OAK-D depth ---")
        rm.capture_depth_for_height(num_frames=10)
        
        print("\n--- Step 3: Detect walls ---")
        rm.detect_walls()
        
        print("\n--- Step 4: Detect planes ---")
        rm.detect_planes()
        
        print("\n--- Step 5: Compute dimensions ---")
        dimensions = rm.compute_dimensions()
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(dimensions)


if __name__ == "__main__":
    main()
