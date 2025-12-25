"""
Dual CSI Camera + RPLIDAR Live Streaming Dashboard
"""

import streamlit as st
import numpy as np
import cv2
import time
import threading
from dataclasses import dataclass
from typing import Optional, List

st.set_page_config(
    page_title="Dual CSI + RPLIDAR",
    page_icon="📷",
    layout="wide"
)


@dataclass
class LidarPoint:
    angle: float
    distance: float
    quality: int


class CSICamera:
    """CSI Camera using nvarguscamerasrc pipeline with full auto-exposure."""
    
    def __init__(self, sensor_id: int = 0, width: int = 640, height: int = 480):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.cap = None
        self._running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None
    
    def _gst_pipeline(self) -> str:
        # Full auto mode - no constraints on exposure/gain
        # flip-method=2 for 180 degree rotation
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            f"nvvidconv flip-method=2 ! "
            f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink drop=1"
        )
    
    def start(self) -> bool:
        try:
            pipeline = self._gst_pipeline()
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                return False
            
            # Let camera warm up and auto-adjust
            for _ in range(10):
                self.cap.read()
            
            ret, _ = self.cap.read()
            if not ret:
                self.cap.release()
                return False
            
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True
            
        except Exception as e:
            print(f"Camera {self.sensor_id} error: {e}")
            return False
    
    def _capture_loop(self):
        while self._running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            time.sleep(0.001)
    
    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None


class RPLidarDriver:
    def __init__(self, port: str = "/dev/ttyUSB0"):
        self.port = port
        self.lidar = None
        self._running = False
        self._scan_data: List[LidarPoint] = []
        self._lock = threading.Lock()
        self._thread = None
    
    def start(self) -> bool:
        try:
            from rplidar import RPLidar
            self.lidar = RPLidar(self.port)
            
            # Reset to clear any stale data
            self.lidar.stop()
            self.lidar.stop_motor()
            time.sleep(0.5)
            self.lidar.start_motor()
            time.sleep(1.0)
            
            info = self.lidar.get_info()
            st.success(f"RPLIDAR connected: {info.get('model', 'Unknown')}")
            
            self._running = True
            self._thread = threading.Thread(target=self._scan_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as e:
            st.error(f"RPLIDAR error: {e}")
            return False
    
    def _scan_loop(self):
        try:
            for scan in self.lidar.iter_scans():
                if not self._running:
                    break
                points = [
                    LidarPoint(angle=m[1], distance=m[2], quality=m[0])
                    for m in scan if m[2] > 0
                ]
                with self._lock:
                    self._scan_data = points
        except Exception as e:
            if self._running:
                print(f"Scan error: {e}")
    
    def get_scan(self) -> List[LidarPoint]:
        with self._lock:
            return self._scan_data.copy()
    
    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.lidar:
            try:
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()
            except:
                pass


def create_cartesian_plot(scan_data: List[LidarPoint], max_distance: float = 6000) -> np.ndarray:
    size = 500
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (14, 17, 23)
    center = size // 2
    scale = (size // 2 - 20) / max_distance
    for r in [1000, 2000, 3000, 4000, 5000, 6000]:
        radius = int(r * scale)
        cv2.circle(img, (center, center), radius, (40, 40, 40), 1)
    cv2.line(img, (center, 0), (center, size), (40, 40, 40), 1)
    cv2.line(img, (0, center), (size, center), (40, 40, 40), 1)
    if scan_data:
        for p in scan_data:
            angle_rad = np.radians(p.angle - 90)
            x = int(center + p.distance * scale * np.cos(angle_rad))
            y = int(center + p.distance * scale * np.sin(angle_rad))
            if 0 <= x < size and 0 <= y < size:
                color_val = int(255 * (1 - p.distance / max_distance))
                color = (0, color_val, 255 - color_val)
                cv2.circle(img, (x, y), 2, color, -1)
    cv2.circle(img, (center, center), 5, (0, 255, 0), -1)
    return img


def main():
    st.title("📷 Dual CSI Camera + RPLIDAR Dashboard")
    st.markdown("Stream CAM0, CAM1, and RPLIDAR data in real-time")
    
    st.sidebar.header("⚙️ Settings")
    lidar_port = st.sidebar.text_input("RPLIDAR Port", "/dev/ttyUSB0")
    cam0_id = st.sidebar.number_input("CAM0 Sensor ID", 0, 3, 0)
    cam1_id = st.sidebar.number_input("CAM1 Sensor ID", 0, 3, 1)
    max_lidar_range = st.sidebar.slider("LIDAR Max Range (m)", 2, 12, 6) * 1000
    
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    if 'cam0' not in st.session_state:
        st.session_state.cam0 = None
    if 'cam1' not in st.session_state:
        st.session_state.cam1 = None
    if 'lidar' not in st.session_state:
        st.session_state.lidar = None
    
    col_start, col_stop, col_capture = st.columns(3)
    
    with col_start:
        if st.button("🚀 Start Streaming", type="primary", disabled=st.session_state.streaming):
            st.session_state.streaming = True
            st.session_state.cam0 = CSICamera(sensor_id=cam0_id)
            st.session_state.cam1 = CSICamera(sensor_id=cam1_id)
            st.session_state.lidar = RPLidarDriver(port=lidar_port)
            
            if st.session_state.cam0.start():
                st.success("CAM0 started")
            else:
                st.error("Failed to open CAM0")
            
            if st.session_state.cam1.start():
                st.success("CAM1 started")
            else:
                st.warning("CAM1 not available")
            
            st.session_state.lidar.start()
    
    with col_stop:
        if st.button("⏹️ Stop Streaming", disabled=not st.session_state.streaming):
            st.session_state.streaming = False
            if st.session_state.cam0:
                st.session_state.cam0.stop()
            if st.session_state.cam1:
                st.session_state.cam1.stop()
            if st.session_state.lidar:
                st.session_state.lidar.stop()
            st.rerun()
    
    with col_capture:
        if st.button("📸 Capture Snapshot", disabled=not st.session_state.streaming):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if st.session_state.cam0:
                frame0 = st.session_state.cam0.read()
                if frame0 is not None:
                    cv2.imwrite(f"cam0_{timestamp}.jpg", frame0)
                    st.success(f"Saved cam0_{timestamp}.jpg")
            if st.session_state.cam1:
                frame1 = st.session_state.cam1.read()
                if frame1 is not None:
                    cv2.imwrite(f"cam1_{timestamp}.jpg", frame1)
                    st.success(f"Saved cam1_{timestamp}.jpg")
            if st.session_state.lidar:
                scan = st.session_state.lidar.get_scan()
                if scan:
                    with open(f"lidar_{timestamp}.csv", 'w') as f:
                        f.write("angle,distance,quality\n")
                        for p in scan:
                            f.write(f"{p.angle},{p.distance},{p.quality}\n")
                    st.success(f"Saved lidar_{timestamp}.csv")
    
    st.divider()
    
    if st.session_state.streaming:
        cam_col1, cam_col2 = st.columns(2)
        with cam_col1:
            st.subheader("📷 CAM0")
            cam0_placeholder = st.empty()
        with cam_col2:
            st.subheader("📷 CAM1")
            cam1_placeholder = st.empty()
        
        lidar_col1, lidar_col2 = st.columns([2, 1])
        with lidar_col1:
            st.subheader("🔴 RPLIDAR Scan")
            lidar_placeholder = st.empty()
        with lidar_col2:
            st.subheader("📊 Stats")
            stats_placeholder = st.empty()
        
        while st.session_state.streaming:
            if st.session_state.cam0:
                frame0 = st.session_state.cam0.read()
                if frame0 is not None:
                    cam0_placeholder.image(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            if st.session_state.cam1:
                frame1 = st.session_state.cam1.read()
                if frame1 is not None:
                    cam1_placeholder.image(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            if st.session_state.lidar:
                scan_data = st.session_state.lidar.get_scan()
                lidar_img = create_cartesian_plot(scan_data, max_lidar_range)
                lidar_placeholder.image(lidar_img, use_container_width=True)
                if scan_data:
                    distances = [p.distance for p in scan_data]
                    stats_placeholder.markdown(f"""
                    **Points:** {len(scan_data)}  
                    **Min Distance:** {min(distances)/1000:.2f} m  
                    **Max Distance:** {max(distances)/1000:.2f} m  
                    **Avg Distance:** {np.mean(distances)/1000:.2f} m  
                    """)
            
            time.sleep(0.05)
    else:
        st.info("👆 Click 'Start Streaming' to begin")


if __name__ == "__main__":
    main()
