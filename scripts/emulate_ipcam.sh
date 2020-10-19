#!/bin/bash

sudo killall rtsp-simple-server
sudo killall ffmpeg

./bin/rtsp-simple-server &
echo "done started rtsp-server"

ffmpeg -re -stream_loop -1 -i demo/Input_StrangerIntrusion.mp4 -c copy -f rtsp rtsp://localhost:8554/AlertStranger &
ffmpeg -re -stream_loop -1 -i demo/Input_VehicleMonitor.mp4 -c copy -f rtsp rtsp://localhost:8554/MonitorVehicle &
ffmpeg -re -stream_loop -1 -i demo/Input_TrafficStat.mp4 -c copy -f rtsp rtsp://localhost:8554/EventStat &
echo "done executing ffmpeg stream"