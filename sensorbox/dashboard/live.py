"""Live streaming dashboard."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sensorbox.live.stream import LiveStreamManager


def main():
    st.set_page_config(
        page_title="SensorBox Live",
        page_icon="📡",
        layout="wide",
    )
    
    st.title("📡 SensorBox Live Stream")
    
    if 'stream_manager' not in st.session_state:
        st.session_state.stream_manager = None
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    
    # Sidebar controls
    st.sidebar.header("Stream Settings")
    
    oakd_enabled = st.sidebar.checkbox("OAK-D Pro", value=True)
    enable_pointcloud = st.sidebar.checkbox("Point Cloud", value=True)
    fps_target = st.sidebar.slider("Target FPS", 5, 30, 15)
    
    depth_quality = st.sidebar.select_slider(
        "Depth Quality",
        options=["fast", "balanced", "high"],
        value="fast",
        help="Fast: More points. High: Less noise."
    )
    
    # IR Toggle
    use_ir = st.sidebar.checkbox(
        "🔦 IR Dot Projector", 
        value=False,
        help="Enable for low-light or textureless surfaces (white walls). Disable for bright scenes."
    )
    
    st.sidebar.header("Point Cloud View")
    pc_view = st.sidebar.radio(
        "Projection",
        ["Top-Down (Z vs X)", "Front (Y vs X)", "Side (Z vs Y)"],
        index=0,
    )
    
    st.sidebar.header("Depth Filter")
    min_depth = st.sidebar.slider("Min Depth (m)", 0.1, 2.0, 0.2, 0.1)
    max_depth = st.sidebar.slider("Max Depth (m)", 0.5, 5.0, 1.5, 0.1)
    
    col1, col2 = st.sidebar.columns(2)
    start_btn = col1.button("▶ Start", use_container_width=True)
    stop_btn = col2.button("⏹ Stop", use_container_width=True)
    
    if start_btn and not st.session_state.streaming:
        st.session_state.stream_manager = LiveStreamManager(
            oakd=oakd_enabled,
            oakd_fps=fps_target,
            enable_pointcloud=enable_pointcloud,
            depth_quality=depth_quality,
            use_ir=use_ir,
        )
        st.session_state.stream_manager.start()
        st.session_state.streaming = True
        st.rerun()
    
    if stop_btn and st.session_state.streaming:
        if st.session_state.stream_manager:
            st.session_state.stream_manager.stop()
            st.session_state.stream_manager = None
        st.session_state.streaming = False
        st.rerun()
    
    if st.session_state.streaming:
        st.sidebar.success("🟢 Streaming")
    else:
        st.sidebar.info("⚪ Stopped")
        st.info("Click **Start** to begin live streaming. Adjust settings in sidebar, then click Start.")
        return
    
    manager = st.session_state.stream_manager
    if manager is None:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RGB")
        rgb_placeholder = st.empty()
        
        st.subheader("Depth Heatmap")
        depth_placeholder = st.empty()
    
    with col2:
        view_titles = {
            "Top-Down (Z vs X)": "Point Cloud (Top-Down)",
            "Front (Y vs X)": "Point Cloud (Front View)",
            "Side (Z vs Y)": "Point Cloud (Side View)",
        }
        st.subheader(view_titles[pc_view])
        pc_placeholder = st.empty()
        
        st.subheader("Info")
        info_placeholder = st.empty()
    
    # Streaming loop
    filtered_points = []
    while st.session_state.streaming:
        frame = manager.get_frame(timeout=0.5)
        
        if frame:
            if frame.rgb is not None:
                rgb_placeholder.image(frame.rgb, channels="BGR", use_container_width=True)
            
            if frame.depth is not None:
                fig = create_depth_heatmap(frame.depth, min_depth, max_depth)
                depth_placeholder.plotly_chart(fig, use_container_width=True, key=f"depth_{time.time()}")
            
            if frame.pointcloud is not None and len(frame.pointcloud) > 0:
                z = frame.pointcloud[:, 2]
                mask = (z >= min_depth) & (z <= max_depth)
                filtered_points = frame.pointcloud[mask]
                
                if len(filtered_points) > 0:
                    fig = create_pointcloud_plot(filtered_points, pc_view, min_depth, max_depth)
                    pc_placeholder.plotly_chart(fig, use_container_width=True, key=f"pc_{time.time()}")
                else:
                    pc_placeholder.warning("No points in depth range")
            
            n_points_total = len(frame.pointcloud) if frame.pointcloud is not None else 0
            n_points_filtered = len(filtered_points) if len(filtered_points) > 0 else 0
            
            info_text = f"""
**FPS:** {frame.fps:.1f}  
**Timestamp:** {frame.timestamp:.2f}s  
**Points (filtered):** {n_points_filtered:,} / {n_points_total:,}  
**Depth Valid:** {frame.depth_valid_pct:.1f}%  
**Depth Range:** {min_depth:.1f} - {max_depth:.1f} m
"""
            if frame.imu:
                acc = frame.imu['accelerometer']
                info_text += f"\n**Accel:** ({acc['x']:.2f}, {acc['y']:.2f}, {acc['z']:.2f})"
            
            info_placeholder.markdown(info_text)
        
        time.sleep(0.05)


def create_depth_heatmap(depth: np.ndarray, min_depth_m: float, max_depth_m: float) -> go.Figure:
    depth_m = depth.astype(np.float32) / 1000.0
    depth_m[depth == 0] = np.nan
    depth_m[depth_m < min_depth_m] = np.nan
    depth_m[depth_m > max_depth_m] = np.nan
    
    fig = go.Figure(data=go.Heatmap(
        z=depth_m,
        colorscale='Turbo',
        zmin=min_depth_m,
        zmax=max_depth_m,
        colorbar=dict(title=dict(text='m', side='right'), tickformat='.2f'),
        hoverongaps=False,
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor='x'),
    )
    return fig


def create_pointcloud_plot(points: np.ndarray, view: str, min_depth: float, max_depth: float) -> go.Figure:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    if view == "Top-Down (Z vs X)":
        plot_x, plot_y = x, z
        x_label, y_label = "X (m)", "Z (m)"
    elif view == "Front (Y vs X)":
        plot_x, plot_y = x, -y
        x_label, y_label = "X (m)", "Y (m)"
    else:
        plot_x, plot_y = z, -y
        x_label, y_label = "Z (m)", "Y (m)"
    
    fig = go.Figure(data=go.Scatter(
        x=plot_x, y=plot_y,
        mode='markers',
        marker=dict(
            size=5,
            color=z,
            colorscale='Turbo',
            cmin=min_depth,
            cmax=max_depth,
            colorbar=dict(title=dict(text='m'), tickformat='.2f', thickness=15),
        ),
    ))
    
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=300,
        margin=dict(l=50, r=70, t=10, b=40),
        yaxis=dict(scaleanchor="x"),
    )
    return fig


if __name__ == "__main__":
    main()
