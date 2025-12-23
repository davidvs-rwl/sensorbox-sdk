"""
Dual CSI Camera + RPLIDAR Live Streaming Dashboard

Stream CAM0 and CAM1 RGB images alongside RPLIDAR scan data in browser.

Usage:
    python3 -m streamlit run sensorbox/dashboard/dual_csi_lidar.py --server.port 8504
"""

import streamlit as st
import numpy as np
import cv2
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, List
from queue import Queue
import matplotlib.pyplot as plt
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Dual CSI + RPLIDAR",
    page_icon="📷",
    layout="wide"
)


@dataclass
class LidarPoint:
    """Single LIDAR measurement."""
    angle: float  # degrees
    distance: float  # mm
    quality: int


class CSICamera:
    """CSI Camera driver using GStreamer pipeline."""
    
    def __init__(self, sensor_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self._running = False
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None
    
    def _gstreamer_pipeline(self) -> str:
        """Build GStreamer pipeline for CSI camera."""
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate={self.fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink drop=1"
        )
    
    def start(self) -> bool:
        """Start camera capture."""
        try:
            pipeline = self._gstreamer_pipeline()
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                st.error(f"Failed to open CAM{self.sensor_id}")
                return False
            
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return True
            
        except Exception as e:
            st.error(f"CAM{self.sensor_id} error: {e}")
            return False
    
    def _capture_loop(self):
        """Background capture loop."""
        while self._running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            time.sleep(0.001)
    
    def read(self) -> Optional[np.ndarray]:
        """Get latest frame."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None
    
    def stop(self):
        """Stop camera capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None


class RPLidarDriver:
    """RPLIDAR A1 driver."""
    
    def __init__(self, port: str = "/dev/ttyUSB0"):
        self.port = port
        self.lidar = None
        self._running = False
        self._scan_data: List[LidarPoint] = []
        self._lock = threading.Lock()
        self._thread = None
    
    def start(self) -> bool:
        """Start LIDAR scanning."""
        try:
            from rplidar import RPLidar
            self.lidar = RPLidar(self.port)
            self.lidar.clear_input()
            
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
        """Background scan loop."""
        try:
            for scan in self.lidar.iter_scans():
                if not self._running:
                    break
                
                points = [
                    LidarPoint(angle=m[1], distance=m[2], quality=m[0])
                    for m in scan if m[2] > 0  # Filter zero distances
                ]
                
                with self._lock:
                    self._scan_data = points
                    
        except Exception as e:
            if self._running:
                print(f"Scan error: {e}")
    
    def get_scan(self) -> List[LidarPoint]:
        """Get latest scan data."""
        with self._lock:
            return self._scan_data.copy()
    
    def stop(self):
        """Stop LIDAR."""
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


def create_lidar_plot(scan_data: List[LidarPoint], max_distance: float = 6000) -> np.ndarray:
    """Create LIDAR visualization as numpy image."""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    if scan_data:
        angles = np.array([np.radians(p.angle) for p in scan_data])
        distances = np.array([p.distance for p in scan_data])
        qualities = np.array([p.quality for p in scan_data])
        
        # Color by quality
        colors = plt.cm.viridis(qualities / 15.0)
        
        ax.scatter(angles, distances, c=colors, s=2, alpha=0.8)
    
    ax.set_rmax(max_distance)
    ax.set_rticks([1000, 2000, 3000, 4000, 5000, 6000])
    ax.set_yticklabels(['1m', '2m', '3m', '4m', '5m', '6m'], color='white', fontsize=8)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Convert to image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='#0e1117', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    
    img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def create_cartesian_plot(scan_data: List[LidarPoint], max_distance: float = 6000) -> np.ndarray:
    """Create Cartesian LIDAR visualization."""
    size = 600
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (14, 17, 23)  # Dark background
    
    center = size // 2
    scale = (size // 2 - 20) / max_distance
    
    # Draw grid circles
    for r in [1000, 2000, 3000, 4000, 5000, 6000]:
        radius = int(r * scale)
        cv2.circle(img, (center, center), radius, (40, 40, 40), 1)
    
    # Draw crosshairs
    cv2.line(img, (center, 0), (center, size), (40, 40, 40), 1)
    cv2.line(img, (0, center), (size, center), (40, 40, 40), 1)
    
    # Draw points
    if scan_data:
        for p in scan_data:
            angle_rad = np.radians(p.angle - 90)  # Rotate so 0° is up
            x = int(center + p.distance * scale * np.cos(angle_rad))
            y = int(center + p.distance * scale * np.sin(angle_rad))
            
            if 0 <= x < size and 0 <= y < size:
                # Color by distance
                color_val = int(255 * (1 - p.distance / max_distance))
                color = (0, color_val, 255 - color_val)
                cv2.circle(img, (x, y), 2, color, -1)
    
    # Draw robot position
    cv2.circle(img, (center, center), 5, (0, 255, 0), -1)
    
    return img


def main():
    st.title("📷 Dual CSI Camera + RPLIDAR Dashboard")
    st.markdown("Stream CAM0, CAM1, and RPLIDAR data in real-time")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    
    lidar_port = st.sidebar.text_input("RPLIDAR Port", "/dev/ttyUSB0")
    cam_resolution = st.sidebar.selectbox(
        "Camera Resolution",
        ["640x480", "1280x720", "1920x1080"],
        index=0
    )
    cam_fps = st.sidebar.slider("Camera FPS", 10, 60, 30)
    max_lidar_range = st.sidebar.slider("LIDAR Max Range (m)", 2, 12, 6) * 1000
    lidar_view = st.sidebar.radio("LIDAR View", ["Cartesian", "Polar"])
    
    # Parse resolution
    res_w, res_h = map(int, cam_resolution.split('x'))
    
    # Initialize session state
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    if 'cam0' not in st.session_state:
        st.session_state.cam0 = None
    if 'cam1' not in st.session_state:
        st.session_state.cam1 = None
    if 'lidar' not in st.session_state:
        st.session_state.lidar = None
    
    # Control buttons
    col_start, col_stop, col_capture = st.columns(3)
    
    with col_start:
        if st.button("🚀 Start Streaming", type="primary", disabled=st.session_state.streaming):
            st.session_state.streaming = True
            
            # Initialize cameras
            st.session_state.cam0 = CSICamera(sensor_id=0, width=res_w, height=res_h, fps=cam_fps)
            st.session_state.cam1 = CSICamera(sensor_id=1, width=res_w, height=res_h, fps=cam_fps)
            st.session_state.lidar = RPLidarDriver(port=lidar_port)
            
            cam0_ok = st.session_state.cam0.start()
            cam1_ok = st.session_state.cam1.start()
            lidar_ok = st.session_state.lidar.start()
            
            if cam0_ok:
                st.success("CAM0 started")
            if cam1_ok:
                st.success("CAM1 started")
            if lidar_ok:
                st.success("RPLIDAR started")
    
    with col_stop:
        if st.button("⏹️ Stop Streaming", disabled=not st.session_state.streaming):
            st.session_state.streaming = False
            
            if st.session_state.cam0:
                st.session_state.cam0.stop()
            if st.session_state.cam1:
                st.session_state.cam1.stop()
            if st.session_state.lidar:
                st.session_state.lidar.stop()
            
            st.info("Streaming stopped")
    
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
    
    # Display area
    if st.session_state.streaming:
        # Create placeholders
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
        
        # Streaming loop
        while st.session_state.streaming:
            # Update CAM0
            if st.session_state.cam0:
                frame0 = st.session_state.cam0.read()
                if frame0 is not None:
                    cam0_placeholder.image(
                        cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )
            
            # Update CAM1
            if st.session_state.cam1:
                frame1 = st.session_state.cam1.read()
                if frame1 is not None:
                    cam1_placeholder.image(
                        cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )
            
            # Update LIDAR
            if st.session_state.lidar:
                scan_data = st.session_state.lidar.get_scan()
                
                if lidar_view == "Cartesian":
                    lidar_img = create_cartesian_plot(scan_data, max_lidar_range)
                else:
                    lidar_img = create_lidar_plot(scan_data, max_lidar_range)
                
                lidar_placeholder.image(lidar_img, use_container_width=True)
                
                # Update stats
                if scan_data:
                    distances = [p.distance for p in scan_data]
                    stats_placeholder.markdown(f"""
                    **Points:** {len(scan_data)}  
                    **Min Distance:** {min(distances)/1000:.2f} m  
                    **Max Distance:** {max(distances)/1000:.2f} m  
                    **Avg Distance:** {np.mean(distances)/1000:.2f} m  
                    """)
            
            time.sleep(0.033)  # ~30 FPS update
    
    else:
        st.info("👆 Click 'Start Streaming' to begin")
        
        # Show placeholder layout
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### CAM0")
            st.image(np.zeros((240, 320, 3), dtype=np.uint8), use_container_width=True)
        with col2:
            st.markdown("### CAM1")
            st.image(np.zeros((240, 320, 3), dtype=np.uint8), use_container_width=True)
        
        st.markdown("### RPLIDAR")
        st.image(create_cartesian_plot([], max_lidar_range), use_container_width=True)


if __name__ == "__main__":
    main()
