#!/bin/bash
echo "Starting Evento Application"

sudo killall streamlit
sudo killall python3

python3 evento/app_server.py --video_feed $1 &
echo "Evento Application Engine Started"

sleep 10

streamlit run evento/ui_server.py -- --app_config_dir "config/"
echo "Evento UI Engine Started"
