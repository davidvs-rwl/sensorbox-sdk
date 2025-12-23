#!/bin/bash
# Run Dual CSI Camera + RPLIDAR Dashboard
#
# Usage: ./run_dual_csi_lidar.sh [port]
#
# Default port: 8504

PORT=${1:-8504}

echo "============================================"
echo "  Dual CSI Camera + RPLIDAR Dashboard"
echo "============================================"
echo ""
echo "Starting on port $PORT..."
echo "Access at: http://$(hostname -I | awk '{print $1}'):$PORT"
echo ""

# Check for required hardware
echo "Checking hardware..."

# Check CSI cameras
if [ -e /dev/video0 ]; then
    echo "✓ CSI camera interface available"
else
    echo "⚠ Warning: /dev/video0 not found - CSI cameras may not work"
fi

# Check RPLIDAR
if [ -e /dev/ttyUSB0 ]; then
    echo "✓ RPLIDAR found at /dev/ttyUSB0"
else
    echo "⚠ Warning: /dev/ttyUSB0 not found - check RPLIDAR connection"
fi

echo ""
echo "Starting Streamlit..."
echo "Press Ctrl+C to stop"
echo ""

python3 -m streamlit run sensorbox/dashboard/dual_csi_lidar.py \
    --server.port $PORT \
    --server.address 0.0.0.0 \
    --server.headless true
