"""SensorBox Dashboard - Web-based visualization with Point Cloud support."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sensorbox.storage import HDF5Reader
from sensorbox.storage.oakd_reader import OakDHDF5Reader
from sensorbox.core.pointcloud import depth_to_colored_pointcloud
from sensorbox.core.config import parse_filename


def main():
    st.set_page_config(
        page_title="SensorBox Dashboard",
        page_icon="📦",
        layout="wide",
    )
    
    st.title("📦 SensorBox Dashboard")
    
    # Sidebar
    st.sidebar.header("Recording")
    
    # Find HDF5 files
    h5_files = list(Path(".").glob("*.h5"))
    h5_files = sorted([str(f) for f in h5_files], reverse=True)[:20]
    
    if not h5_files:
        st.warning("No .h5 recordings found in current directory.")
        st.info("Record data with: `python examples/record_oakd.py -t 10`")
        return
    
    # Format display names
    file_display = {}
    for f in h5_files:
        info = parse_filename(f)
        if info["config"]:
            display = f"{info['config']} - {info['formatted_date']}"
        else:
            display = f
        file_display[display] = f
    
    selected_display = st.sidebar.selectbox("Select Recording", list(file_display.keys()))
    selected_file = file_display[selected_display]
    
    if selected_file:
        if is_oakd_recording(selected_file):
            show_oakd_recording(selected_file)
        else:
            show_csi_recording(selected_file)


def is_oakd_recording(filepath: str) -> bool:
    """Check if file is an OAK-D recording."""
    import h5py
    try:
        with h5py.File(filepath, "r") as f:
            return f.attrs.get("sensor", "") == "OAK-D Pro"
    except:
        return False


def show_oakd_recording(filepath: str):
    """Display OAK-D Pro recording with point cloud."""
    
    try:
        reader = OakDHDF5Reader(filepath)
        reader.open()
        info = reader.info
    except Exception as e:
        st.error(f"Error opening file: {e}")
        return
    
    # Parse filename for config info
    file_info = parse_filename(filepath)
    config_str = f"{file_info['config']} ({file_info['config_description']})" if file_info['config'] else "OAK-D Pro"
    
    st.subheader(f"🎥 {config_str} Recording")
    
    # Info panel
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RGB Frames", info.frame_count)
    col2.metric("Depth Frames", info.depth_count)
    col3.metric("IMU Samples", info.imu_count)
    col4.metric("Duration", f"{info.duration_seconds:.1f}s")
    
    st.divider()
    
    if info.frame_count == 0:
        st.warning("No frames in recording")
        reader.close()
        return
    
    # Frame slider
    frame_idx = st.slider("Frame", 0, max(0, info.frame_count - 1), 0, key="oakd_frame")
    frame = reader.get_frame(frame_idx)
    
    # View mode selector
    view_mode = st.radio(
        "View Mode", 
        ["RGB + Depth", "RGB + Point Cloud (3D)", "RGB + Point Cloud (2D)", "All Views"], 
        horizontal=True
    )
    
    if view_mode == "RGB + Depth":
        show_rgb_depth(frame)
    elif view_mode == "RGB + Point Cloud (3D)":
        show_rgb_pointcloud_3d(frame)
    elif view_mode == "RGB + Point Cloud (2D)":
        show_rgb_pointcloud_2d(frame)
    else:
        show_all_views(frame)
    
    # IMU data
    if frame.imu:
        with st.expander("📊 IMU Data"):
            col1, col2 = st.columns(2)
            with col1:
                acc = frame.imu['accelerometer']
                st.write("**Accelerometer (m/s²)**")
                st.write(f"X: {acc['x']:.3f}  Y: {acc['y']:.3f}  Z: {acc['z']:.3f}")
            with col2:
                gyro = frame.imu['gyroscope']
                st.write("**Gyroscope (rad/s)**")
                st.write(f"X: {gyro['x']:.3f}  Y: {gyro['y']:.3f}  Z: {gyro['z']:.3f}")
    
    # Metadata
    with st.expander("📋 Recording Info"):
        st.json({
            "file": filepath,
            "config": file_info.get("config"),
            "created": info.created,
            "duration": info.duration_seconds,
            "rgb_size": info.rgb_size,
            "depth_size": info.depth_size,
            "metadata": info.user_metadata,
        })
    
    reader.close()


def show_rgb_depth(frame):
    """Show RGB and depth side by side."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RGB")
        if frame.rgb is not None:
            st.image(frame.rgb, channels="BGR", use_container_width=True)
    
    with col2:
        st.subheader("Depth")
        if frame.depth is not None:
            depth_viz = colorize_depth(frame.depth)
            st.image(depth_viz, use_container_width=True)
            st.caption(f"Depth range: {frame.depth.min()} - {frame.depth.max()} mm")


def show_rgb_pointcloud_3d(frame):
    """Show RGB and 3D point cloud side by side."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RGB")
        if frame.rgb is not None:
            st.image(frame.rgb, channels="BGR", use_container_width=True)
    
    with col2:
        st.subheader("Point Cloud (3D)")
        if frame.depth is not None and frame.rgb is not None:
            show_pointcloud_3d(frame.depth, frame.rgb)
        else:
            st.warning("No depth data available")


def show_rgb_pointcloud_2d(frame):
    """Show RGB and 2D point cloud projections."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RGB")
        if frame.rgb is not None:
            st.image(frame.rgb, channels="BGR", use_container_width=True)
    
    with col2:
        st.subheader("Point Cloud (Top-Down View)")
        if frame.depth is not None:
            show_pointcloud_2d(frame.depth, frame.rgb)
        else:
            st.warning("No depth data available")


def show_all_views(frame):
    """Show RGB, depth, and point cloud."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RGB")
        if frame.rgb is not None:
            st.image(frame.rgb, channels="BGR", use_container_width=True)
        
        st.subheader("Depth")
        if frame.depth is not None:
            depth_viz = colorize_depth(frame.depth)
            st.image(depth_viz, use_container_width=True)
    
    with col2:
        st.subheader("Point Cloud (Top-Down)")
        if frame.depth is not None:
            show_pointcloud_2d(frame.depth, frame.rgb, height=300)
        
        st.subheader("Point Cloud (Side View)")
        if frame.depth is not None:
            show_pointcloud_side(frame.depth, frame.rgb, height=300)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Convert depth to colorized visualization."""
    import cv2
    
    valid = depth > 0
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    
    depth_norm = np.zeros_like(depth, dtype=np.float32)
    depth_norm[valid] = depth[valid]
    
    max_val = np.percentile(depth_norm[valid], 95)
    depth_norm = np.clip(depth_norm / max_val * 255, 0, 255).astype(np.uint8)
    
    colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    
    return colored


def show_pointcloud_2d(depth: np.ndarray, rgb: np.ndarray = None, height: int = 400):
    """Show point cloud as 2D top-down scatter plot (X vs Z)."""
    
    subsample = 4
    
    if rgb is not None:
        points, colors = depth_to_colored_pointcloud(depth, rgb, subsample=subsample, max_depth=5000)
        colors_rgb = colors[:, ::-1]  # BGR to RGB
    else:
        from sensorbox.core.pointcloud import depth_to_pointcloud
        points = depth_to_pointcloud(depth, subsample=subsample, max_depth=5000)
        colors_rgb = None
    
    if len(points) == 0:
        st.warning("No valid depth points")
        return
    
    x = points[:, 0]  # Left/right
    z = points[:, 2]  # Depth (forward)
    
    if colors_rgb is not None:
        color_strs = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors_rgb]
    else:
        color_strs = 'blue'
    
    fig = go.Figure(data=go.Scatter(
        x=x,
        y=z,
        mode='markers',
        marker=dict(size=3, color=color_strs, opacity=0.7),
    ))
    
    fig.update_layout(
        xaxis_title='X (m) ← Left | Right →',
        yaxis_title='Z (m) - Distance',
        height=height,
        margin=dict(l=40, r=20, t=20, b=40),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(points):,} points | Top-down view (looking from above)")


def show_pointcloud_side(depth: np.ndarray, rgb: np.ndarray = None, height: int = 400):
    """Show point cloud as 2D side view scatter plot (Z vs Y)."""
    
    subsample = 4
    
    if rgb is not None:
        points, colors = depth_to_colored_pointcloud(depth, rgb, subsample=subsample, max_depth=5000)
        colors_rgb = colors[:, ::-1]
    else:
        from sensorbox.core.pointcloud import depth_to_pointcloud
        points = depth_to_pointcloud(depth, subsample=subsample, max_depth=5000)
        colors_rgb = None
    
    if len(points) == 0:
        st.warning("No valid depth points")
        return
    
    z = points[:, 2]  # Depth
    y = -points[:, 1]  # Height (inverted for display)
    
    if colors_rgb is not None:
        color_strs = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors_rgb]
    else:
        color_strs = 'blue'
    
    fig = go.Figure(data=go.Scatter(
        x=z,
        y=y,
        mode='markers',
        marker=dict(size=3, color=color_strs, opacity=0.7),
    ))
    
    fig.update_layout(
        xaxis_title='Z (m) - Distance',
        yaxis_title='Y (m) ← Down | Up →',
        height=height,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(points):,} points | Side view (looking from the side)")


def show_pointcloud_3d(depth: np.ndarray, rgb: np.ndarray = None, height: int = 500):
    """Show interactive 3D point cloud (requires WebGL)."""
    
    subsample = 4
    
    if rgb is not None:
        points, colors = depth_to_colored_pointcloud(depth, rgb, subsample=subsample, max_depth=5000)
        colors_rgb = colors[:, ::-1] * 255
    else:
        from sensorbox.core.pointcloud import depth_to_pointcloud
        points = depth_to_pointcloud(depth, subsample=subsample, max_depth=5000)
        z = points[:, 2]
        colors_rgb = np.zeros((len(points), 3))
        colors_rgb[:, 2] = (1 - z / z.max()) * 255
        colors_rgb[:, 0] = (z / z.max()) * 255
    
    if len(points) == 0:
        st.warning("No valid depth points")
        return
    
    color_strs = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in colors_rgb]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=color_strs, opacity=0.8),
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(eye=dict(x=0, y=-2, z=0.5)),
        ),
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(points):,} points displayed (requires WebGL)")


def show_csi_recording(filepath: str):
    """Display CSI camera recording."""
    
    try:
        reader = HDF5Reader(filepath)
        reader.open()
        info = reader.info
    except Exception as e:
        st.error(f"Error opening file: {e}")
        return
    
    file_info = parse_filename(filepath)
    config_str = f"{file_info['config']} ({file_info['config_description']})" if file_info['config'] else "CSI Camera"
    
    st.subheader(f"🎥 {config_str} Recording")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Frames", info.frame_count)
    col2.metric("Duration", f"{info.duration_seconds:.1f}s")
    col3.metric("Cameras", len(info.cameras))
    col4.metric("LIDAR Scans", info.lidar_scan_count)
    
    st.divider()
    
    if info.frame_count == 0:
        st.warning("No frames in recording")
        reader.close()
        return
    
    frame_idx = st.slider("Frame", 0, max(0, info.frame_count - 1), 0, key="csi_frame")
    frame = reader.get_frame(frame_idx)
    
    if frame.cameras:
        cols = st.columns(len(frame.cameras))
        for i, (cam_id, img) in enumerate(frame.cameras.items()):
            with cols[i]:
                st.image(img, caption=f"Camera {cam_id}", channels="BGR", use_container_width=True)
    
    if frame.lidar is not None:
        st.subheader("📡 LIDAR Scan")
        show_lidar_plot(frame.lidar)
    
    with st.expander("📋 Recording Info"):
        st.json({
            "file": filepath,
            "config": file_info.get("config"),
            "created": info.created,
            "duration": info.duration_seconds,
            "cameras": info.cameras,
            "lidar_scans": info.lidar_scan_count,
            "metadata": info.user_metadata,
        })
    
    reader.close()


def show_lidar_plot(scan: np.ndarray):
    """Show LIDAR scan as scatter plot."""
    angles = scan[:, 0]
    distances = scan[:, 1] / 1000
    
    angles_rad = np.radians(angles)
    x = distances * np.cos(angles_rad)
    y = distances * np.sin(angles_rad)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=5, color='blue')))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=12, color='red')))
    fig.update_layout(xaxis_title="X (m)", yaxis_title="Y (m)", yaxis_scaleanchor="x", height=400)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
