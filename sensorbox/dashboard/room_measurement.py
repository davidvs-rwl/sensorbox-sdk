"""Room measurement visualization dashboard."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    st.set_page_config(
        page_title="Room Measurement",
        page_icon="📐",
        layout="wide",
    )
    
    st.title("📐 Room Measurement")
    
    # Session state
    if 'measurement' not in st.session_state:
        st.session_state.measurement = None
    if 'lidar_points' not in st.session_state:
        st.session_state.lidar_points = None
    if 'depth_points' not in st.session_state:
        st.session_state.depth_points = None
    if 'rgb_image' not in st.session_state:
        st.session_state.rgb_image = None
    if 'walls' not in st.session_state:
        st.session_state.walls = None
    if 'dimensions' not in st.session_state:
        st.session_state.dimensions = None
    
    # Sidebar controls
    st.sidebar.header("Capture Settings")
    num_lidar_scans = st.sidebar.slider("RPLIDAR Scans", 3, 20, 10)
    num_depth_frames = st.sidebar.slider("Depth Frames", 5, 30, 15)
    camera_height = st.sidebar.slider("Camera Height (m)", 0.2, 1.5, 0.5, 0.1)
    
    st.sidebar.header("Actions")
    
    if st.sidebar.button("🔌 Connect Sensors", use_container_width=True):
        connect_sensors()
    
    if st.sidebar.button("📡 Capture Data", use_container_width=True):
        capture_data(num_lidar_scans, num_depth_frames)
    
    if st.sidebar.button("📏 Measure Room", use_container_width=True):
        measure_room()
    
    if st.sidebar.button("🔓 Disconnect", use_container_width=True):
        disconnect_sensors()
    
    # Results (show at top if available)
    if st.session_state.dimensions is not None:
        st.subheader("📊 Room Dimensions")
        dim = st.session_state.dimensions
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Length", f"{dim.length:.2f} m")
        col2.metric("Width", f"{dim.width:.2f} m")
        col3.metric("Height", f"{dim.height:.2f} m")
        col4.metric("Area", f"{dim.area:.2f} m²")
        col5.metric("Volume", f"{dim.volume:.2f} m³")
        
        col1, col2 = st.columns(2)
        col1.progress(dim.wall_confidence, text=f"Wall Confidence: {dim.wall_confidence:.0%}")
        col2.progress(dim.height_confidence, text=f"Height Confidence: {dim.height_confidence:.0%}")
        
        st.divider()
    
    # Top row - LIDAR and Point Cloud side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📡 RPLIDAR 2D Scan (Top-Down)")
        if st.session_state.lidar_points is not None:
            fig = plot_lidar_scan(
                st.session_state.lidar_points,
                st.session_state.walls
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Click 'Capture Data' to capture RPLIDAR scan")
    
    with col2:
        st.subheader("🎯 OAK-D Point Cloud")
        if st.session_state.depth_points is not None:
            view = st.radio("View", ["Front (Y vs X)", "Top (Z vs X)", "Side (Y vs Z)"], horizontal=True)
            fig = plot_depth_cloud_2d(st.session_state.depth_points, view)
            st.plotly_chart(fig, use_container_width=True)
            
            # Point cloud stats
            pts = st.session_state.depth_points
            st.caption(f"Points: {len(pts):,} | "
                      f"X: {pts[:,0].min():.2f} to {pts[:,0].max():.2f}m | "
                      f"Y: {pts[:,1].min():.2f} to {pts[:,1].max():.2f}m | "
                      f"Z: {pts[:,2].min():.2f} to {pts[:,2].max():.2f}m")
        else:
            st.info("Click 'Capture Data' to capture depth data")
    
    # Bottom row - RGB image centered
    st.subheader("📷 RGB Camera View")
    if st.session_state.rgb_image is not None:
        # Center the image using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(st.session_state.rgb_image, channels="BGR", use_container_width=True)
    else:
        st.info("Click 'Capture Data' to capture RGB image")


def connect_sensors():
    """Connect to sensors."""
    from sensorbox.measurement.room import RoomMeasurement
    
    with st.spinner("Connecting to sensors..."):
        try:
            rm = RoomMeasurement(camera_height=0.5)
            rm.connect()
            st.session_state.measurement = rm
            st.success("Sensors connected!")
        except Exception as e:
            st.error(f"Connection failed: {e}")


def disconnect_sensors():
    """Disconnect sensors."""
    if st.session_state.measurement:
        st.session_state.measurement.disconnect()
        st.session_state.measurement = None
        st.info("Sensors disconnected")


def capture_data(num_lidar_scans: int, num_depth_frames: int):
    """Capture sensor data."""
    rm = st.session_state.measurement
    
    if rm is None:
        st.error("Connect sensors first!")
        return
    
    with st.spinner("Capturing RPLIDAR scans..."):
        try:
            st.session_state.lidar_points = rm.capture_lidar_scan(
                num_scans=num_lidar_scans, 
                show_progress=False
            )
            st.success(f"Captured {len(st.session_state.lidar_points)} RPLIDAR points")
        except Exception as e:
            st.error(f"RPLIDAR capture failed: {e}")
    
    with st.spinner("Capturing RGB and depth frames..."):
        try:
            st.session_state.depth_points, st.session_state.rgb_image = rm.capture_depth_and_rgb(
                num_frames=num_depth_frames,
                show_progress=False
            )
            st.success(f"Captured {len(st.session_state.depth_points)} depth points")
        except Exception as e:
            st.error(f"Depth capture failed: {e}")


def measure_room():
    """Run room measurement."""
    rm = st.session_state.measurement
    
    if rm is None:
        st.error("Connect sensors first!")
        return
    
    if st.session_state.lidar_points is None:
        st.error("Capture data first!")
        return
    
    with st.spinner("Detecting walls..."):
        st.session_state.walls = rm.detect_walls(show_progress=False)
    
    with st.spinner("Computing dimensions..."):
        st.session_state.dimensions = rm.compute_dimensions(show_progress=False)
    
    st.success("Measurement complete!")
    st.rerun()


def plot_lidar_scan(points: np.ndarray, walls=None) -> go.Figure:
    """Plot 2D RPLIDAR scan with detected walls."""
    fig = go.Figure()
    
    # Plot raw points
    fig.add_trace(go.Scatter(
        x=points[:, 0],
        y=points[:, 1],
        mode='markers',
        marker=dict(size=3, color='cyan', opacity=0.6),
        name='RPLIDAR Points',
    ))
    
    # Plot detected walls
    if walls:
        colors = px.colors.qualitative.Set1
        for i, wall in enumerate(walls):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=[wall.start[0], wall.end[0]],
                y=[wall.start[1], wall.end[1]],
                mode='lines+markers',
                line=dict(color=color, width=4),
                marker=dict(size=8),
                name=f'Wall {i+1}: {wall.length:.2f}m',
            ))
    
    # Plot sensor position
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Sensor',
    ))
    
    fig.update_layout(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        height=450,
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)'),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_dark",
        margin=dict(l=50, r=20, t=20, b=50),
    )
    
    return fig


def plot_depth_cloud_2d(points: np.ndarray, view: str) -> go.Figure:
    """Plot point cloud in 2D projection."""
    # Subsample for performance
    max_points = 10000
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    if view == "Front (Y vs X)":
        plot_x, plot_y = x, -y
        color_data = z
        x_label, y_label = "X (m) - Left/Right", "Y (m) - Height"
        color_label = "Depth (m)"
    elif view == "Top (Z vs X)":
        plot_x, plot_y = x, z
        color_data = -y
        x_label, y_label = "X (m) - Left/Right", "Z (m) - Depth"
        color_label = "Height (m)"
    else:  # Side (Y vs Z)
        plot_x, plot_y = z, -y
        color_data = x
        x_label, y_label = "Z (m) - Depth", "Y (m) - Height"
        color_label = "X (m)"
    
    fig = go.Figure(data=go.Scatter(
        x=plot_x, 
        y=plot_y,
        mode='markers',
        marker=dict(
            size=3,
            color=color_data,
            colorscale='Viridis',
            colorbar=dict(title=color_label, thickness=15),
            opacity=0.7,
        ),
    ))
    
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        template="plotly_dark",
        margin=dict(l=50, r=20, t=20, b=50),
    )
    
    return fig


if __name__ == "__main__":
    main()
