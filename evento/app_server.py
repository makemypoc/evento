# Copyright (c) 2020 Mujammil Ali A S. All Rights Reserved.
# 
# This module is Licensed under the Apache License, Version 2.0 with Commons
# Clause (the "License") as defined below, subject to the following condition.
#
#     https://commonsclause.com/
#
# Commons Clause License Condition v1.0
#
# Without limiting other conditions in the License, the grant of rights 
# under the License will not include, and the License does not grant to you, 
# the right to Sell the Software.


"""Evento Video Analytics Pipeline Handler Module

This module is a video analytics pipeline that handles
    - Get the user arguments based on argparse
    - Capturing the frame in a video feed
    - connect with Redis server
    - configure region of interest based on application mode and 
    event type
    - Apply object detection algorithm
    - Visualalize the object as a overlaid rectangle 
    - Analyze the event of interest, get the metadata and processed frame
    - Send the metadata and processed frame to the Redis Server
    - Repeat the process untill the video feed is alive

This module contains the following functions:
    - predict_publish - This function handles the entire video analytics 
    pipeline for the selected configuration.
    - print_arguments -- It prints the command line arguments for this module

Additionally it has entry point __main__ where it collects the user arguments
based on argparse.
"""

import os
import argparse
import redis
import base64
import ast
import json

import cv2
import numpy as np
from visualize import visualize_box_mask
import paddle_detect
import event_alert

prd = redis.Redis(host='localhost', port=6379, db=0)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

def predict_publish():
    """ 
    This Function handles the video analytics pipeline that captures
    the image in a video feed, apply the onject detection, analyze the 
    interest of event and notify visually. This function also sends the
    metadata and processed frame to the Redis. It runs forever.

    Args:
        Nothing

    Returns:
        Nothing
 
    """

    index = 1
    detector = paddle_detect.Detector(FLAGS.model_dir, use_gpu=FLAGS.use_gpu, run_mode=FLAGS.run_mode)
    event_type = int(prd.get("event_type"))
    app_mode = int(prd.get("app_mode"))
    event_manager = event_alert.EventManager(FLAGS.app_config_dir, event_type) 
    eventName = event_manager.getEventName(event_type)
    event_type_prev = event_type 
    app_mode_prev = app_mode  

    capture = cv2.VideoCapture(FLAGS.video_feed, cv2.CAP_FFMPEG)

    while True:

        success, image = capture.read()
        print('Reading')

        if not success:
            break

        print('Reading Successful')
        while True:
            app_mode = int(prd.get("app_mode"))
            event_type = int(prd.get("event_type"))

            if event_type != event_type_prev or app_mode != app_mode_prev:
                event_manager.updateROISettings(event_type)
                eventName = event_manager.getEventName(event_type)

            retval, buffer = cv2.imencode('.jpg', image)
            pic_str = base64.b64encode(buffer)
            pic_str = pic_str.decode()
            prd.set(eventName, pic_str)
            event_type_prev = event_type
            app_mode_prev = app_mode

            if app_mode != 1:
                break 
            
        print('detect frame:%d' % (index))
        index += 1
        results = detector.predict(image, FLAGS.threshold)
        #print(results['boxes'])

        im = visualize_box_mask(
            image,
            results,
            detector.config.labels,
            mask_resolution=detector.config.mask_resolution)
        im = np.array(im)
        img = im


        if event_type == 2:
            img, result, appMetadata = event_manager.monitorVehicle(results['boxes'],im)
        elif event_type == 1:
            img, result, appMetadata = event_manager.alertStrangerIntrusion(results['boxes'],im)
        elif event_type == 0:
            img, result, appMetadata = event_manager.findEventStat(results['boxes'],im, vehicleThres=5)
        else:
            img, result, appMetadata = event_manager.findEventStat(results['boxes'],im, vehicleThres=5)

        retval, buffer = cv2.imencode('.jpg', img)
     
        pic_str = base64.b64encode(buffer)
        pic_str = pic_str.decode()
        appMetadatastr = json.dumps(appMetadata)

        msg = {
            'appMetadata': appMetadatastr,
            'image': pic_str
        }
        _id = prd.xadd('camera:0', msg, maxlen=1000)

        event_type_prev = event_type

        if FLAGS.local_debug == True:
            cv2.imshow("Evento", img)
            cv2.waitKey(10)


def print_arguments(args):
    """ 
    This Function prints the command line arguments. 
 
     Args:
        args (dict): list of user arguments passed via 
            argparse
    Returns:
        Nothing

    """
    print('-----------  Running Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')


if __name__ == '__main__':
    """ 
    This Function is the entry point for this module. It collects the
    user arguments based on argparse library. The list of options are

        --model_dir - Directory where the object detection model is placed.
        --app_config_dir - Directory where the json files that has the ROI
            settings information. Every event has each file of json.
        --video_feed - It is either rtsp url or video file path. Deployed 
            application has to has rtsp url.
        --run_mode - run mode for PaddlePaddle framework.
        --use_gpu - True if the running host has GPU.
        --local_debug - It helps in vidualizing the processed frame in the 
            running host itself [for debug purpose].
        --threshold - Confidence threshold above which the object detection 
            finds or filters the obejects as final list.    
    
    How to run: python3 evento/app_server.py [Optionally with intended options]

    Finally it triggers the video analytics pipeline.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--app_config_dir",
        type=str,
        default='config',
        help=("Directory include:'AlertStranger.json', 'EventStat.json', "
              "'MonitorVehicle.json', created by ui_server.py."),
        required=False)
    parser.add_argument(
        "--model_dir",
        type=str,
        default='models/ppyolo',
        help=("Directory include:'__model__', '__params__', "
              "'infer_cfg.yml', created by tools/export_model.py."),
        required=False)
    parser.add_argument(
        "--video_feed", 
        type=str, 
        default='demo/Input_EventSeries.mp4', #'rtsp://192.168.43.23:8554/AlertStranger',
        help="rtsp url or any kind of video file path") 
    parser.add_argument(
        "--run_mode",
        type=str,
        default='fluid',
        help="mode of running(fluid/trt_fp32/trt_fp16)")
    parser.add_argument(
        "--use_gpu",
        type=ast.literal_eval,
        default=True,
        help="Whether to predict with GPU.")
    parser.add_argument(
        "--local_debug",
        type=ast.literal_eval,
        default=False,
        help="used for local visual debug")
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5, 
        help="Threshold of score.")

    FLAGS = parser.parse_args()
    print_arguments(FLAGS)

    # Settings redis for all the keys for first time

    if not prd.exists("app_mode"):
        prd.set("app_mode", 1)

    if not prd.exists("event_type"):
        prd.set("event_type", 0)

    image = cv2.imread("demo/Evento.jpg")
    image = cv2.resize(image,(640,320))
    retval, buffer = cv2.imencode('.jpg', image)

    pic_str = base64.b64encode(buffer)
    pic_str = pic_str.decode()
    appMetadata = {"ROI 1":0,"ROI 2":0,"ROI 3":0,"ROI 4":0,"ROI 5":0}
    appMetadatastr = json.dumps(appMetadata)

    if not prd.exists("camera:0"):
        msg = {
            'appMetadata': appMetadatastr,
            'image': pic_str
        }
        _id = prd.xadd('camera:0', msg, maxlen=1000)

    if not prd.exists("img"):
        prd.set("img", pic_str)

    if not prd.exists("AlertStranger"):
        prd.set("AlertStranger", pic_str)

    if not prd.exists("EventStat"):
        prd.set("EventStat", pic_str)

    if not prd.exists("MonitorVehicle"):
        prd.set("MonitorVehicle", pic_str)

    # Running the video analytics forever

    if FLAGS.video_feed != '':
        predict_publish()