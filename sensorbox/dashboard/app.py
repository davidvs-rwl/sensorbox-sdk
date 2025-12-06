"""SensorBox Dashboard - Web-based visualization."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sensorbox.storage import HDF5Reader


def main():
    st.set_page_config(
        page_title="SensorBox Dashboard",
        page_icon="📦",
        layout="wide",
    )
    
    st.title("📦 SensorBox Dashboard")
    
    # Sidebar - File selection
    st.sidebar.header("Recording")
    
    # Find HDF5 files
    h5_files = list(Path(".").glob("*.h5"))
    h5_files = [str(f) for f in h5_files][:20]
    
    if not h5_files:
        st.warning("No .h5 recordings found in current directory.")
        st.info("Record data with: `sensorbox record -t 10 -o my_recording.h5`")
        return
    
    selected_file = st.sidebar.selectbox("Select Recording", h5_files)
    
    if selected_file:
        show_recording(selected_file)


def show_recording(filepath: str):
    """Display recording data."""
    
    try:
        reader = HDF5Reader(filepath)
        reader.open()
        info = reader.info
    except Exception as e:
        st.error(f"Error opening file: {e}")
        return
    
    # Info panel
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", info.frame_count)
    col2.metric("Duration", f"{info.duration_seconds:.1f}s")
    col3.metric("Cameras", len(info.cameras))
    col4.metric("LIDAR Scans", info.lidar_scan_count)
    
    st.divider()
    
    # Frame viewer
    st.subheader("🎥 Frame Viewer")
    
    if info.frame_count == 0:
        st.warning("No frames in recording")
        reader.close()
        return
    
    frame_idx = st.slider("Frame", 0, max(0, info.frame_count - 1), 0)
    
    frame = reader.get_frame(frame_idx)
    
    # Show cameras
    if frame.cameras:
        cam_cols = st.columns(len(frame.cameras))
        for i, (cam_id, img) in enumerate(frame.cameras.items()):
            with cam_cols[i]:
                st.image(img, caption=f"Camera {cam_id}", channels="BGR")
    
    # Show LIDAR
    if frame.lidar is not None:
        st.subheader("📡 LIDAR Scan")
        show_lidar_plot(frame.lidar)
    
    # Metadata
    with st.expander("📋 Recording Metadata"):
        st.json({
            "file": filepath,
            "created": info.created,
            "duration": info.duration_seconds,
            "cameras": info.cameras,
            "lidar_scans": info.lidar_scan_count,
            "user_metadata": info.user_metadata,
        })
    
    reader.close()


def show_lidar_plot(scan: np.ndarray):
    """Show LIDAR scan as scatter plot."""
    
    angles = scan[:, 0]
    distances = scan[:, 1] / 1000  # meters
    
    angles_rad = np.radians(angles)
    x = distances * np.cos(angles_rad)
    y = distances * np.sin(angles_rad)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=5, color='blue'),
        name='Points'
    ))
    
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=12, color='red'),
        name='LIDAR'
    ))
    
    fig.update_layout(
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        yaxis_scaleanchor="x",
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
