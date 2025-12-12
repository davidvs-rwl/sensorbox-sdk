#!/bin/bash
# SensorBox Room Measurement Dashboard Launcher

cd ~/sensorbox-sdk

echo "Starting SensorBox Room Measurement Dashboard..."
echo "Access at: http://$(hostname -I | awk '{print $1}'):8503"
echo ""

python3 -m streamlit run sensorbox/dashboard/room_measurement.py \
    --server.headless true \
    --server.port 8503
