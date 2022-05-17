## Evento

This Utility helps in identifying and alerting events of interests in real-time.  The possibilities are breaching into restricted areas, monitoring asset presence, and counting the number of interesting objects in the specified region. This Utility can be used in use cases such as detecting strangers, tracking the vehicle presence, and vehicle counting for traffic control.

## Getting Started

This Utility uses the PaddlePaddle Object detection, Streamlit framework, and Redis in-memory database to accomplish these use cases. As of now, PP-Yolo is the object detection model.

* Detailed article - https://medium.com/@monifooz/evento-is-the-ai-system-a-magic-wand-3523ae2e5344
* Youtube Video - https://www.youtube.com/watch?v=oPvlByQpre4

### Prerequisites

* Python version is 3.6.9.  
* Ubuntu 18 with GPU
* As the tested system is an Intel i7 based Ubuntu 18 system that equips NVIDIA GPU RTX2080Ti, Cuda 10.2 version is a must for the PaddlePaddle framework. 
* In-memory database Redis version shall be above 5, and the setup uses 5.0.9. 
* Streamlit framework is used for a user interface for configuring the application settings and auditing purpose.
* Simple RTSP server is needed to emulate IP cam traffic for testing purposes.


### Installation Procedure

* As installing Redis server 5.0.9 (with version  5.x.x) is a must, it may be required to uninstall the old version. Use the below link for it.

```
https://itnext.io/redis-5-x-under-the-hood-1-downloading-and-installing-redis-locally-3373fe67a154
```
* Install all the package dependencies available in requirements.txt

```
$ pip3 install -r requirements.txt
```
* Open a terminal, clone this repo, and go to the Evento folder where you placed.
 
```
$ mkdir Documents/projects/
$ cd Documents/projects/
$ git clone git clone https://github.com/makemypoc/python-utils.git
$ cd Evento
```
* Download the simple RTSP server and PP-Yolo model by running the following script.

```
$ sh scripts/download_essential.sh
``` 
Note: PP-Yolo model can be downloaded by an alternative method specified in the PaddleDetection repo. This repo must be in the local system to export the model and config file.

```
# export model, --exclude_nms to prune NMS part, model will be save in output/ppyolo as default
python tools/export_model.py -c configs/ppyolo/ppyolo.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --exclude_nms
```

Refer to this link - https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.4/configs/ppyolo/README.md

### Testing the application

* Test the Evento application by running the following script. Be in the parent folder when running this bash script.

```
# Run the application to test with video by giving the CLI arguments in position $1.
$ sh app.sh demo/Input_StrangerIntrusion.mp4

# Run the application with RTSP stream by specifying the RTSP URL
$ sh app.sh rtsp://localhost:8554/AlertStranger
```
* Emulate the IP camera stream by running the below bash script.

```
# Run the IP camera emulation of three streams.
$ sh scripts/emulate_ipcam.sh
```

Note:
The RTSP streaming URLs are as follows.
rtsp://localhost:8554/AlertStranger
rtsp://localhost:8554/MonitorVehicle
rtsp://localhost:8554/EventStat
Local mp4 video file can be configured  in emulate_ipcam.sh script
 

### Deployment

Just run the app.sh script with the RTSP URL.

```

# Run the application with RTSP stream by specifying the RTSP URL
$ sh app.sh rtsp://Ip_cam_IP:Ip_cam_port/your_Ip_cam_url_endpoint
```

## Built With

* [Streamlit](https://github.com/streamlit/streamlit) - The Streamlit framework used for settings configuration and auditing the deployment scenario.
* [PaddlePaddle](https://github.com/PaddlePaddle/PaddleDetection) - Deep learning Framework used for object detection.

## Authors

* **Mujammil Ali A S** - *Initial work* - [makemypoc](https://github.com/makemypoc)

## License

The following modules licensed under the Apache License, Version 2.0.

1. paddle_detect.py
2. visualize.py

You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

The following modules licensed under the Apache License, Version 2.0 with Commons Clause

1. app_server.py
2. event_alert.py
3. ui_server.py

You may obtain a copy of the License at https://commonsclause.com/ and http://www.apache.org/licenses/LICENSE-2.0

## Acknowledgments

* [Simple RTSP server](https://github.com/aler9/rtsp-simple-server) - It is used along with ffmpeg for emulating the camera feed to test the code.
* [Lion Gate Bridge Footage](https://www.videvo.net/video/lions-gate-bridge/4713/) - This video footage is for traffic stat use case. Just download it, resize it to 360p,  change the name as Input_TrafficStat.mp4 and place it under demo/ folder.
* [PaddlePaddle Detection inference](https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.4/deploy/python) - This repo inference code is for finding the bounding box of the detected object.
