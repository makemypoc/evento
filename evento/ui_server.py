# Copyright (c) 2020 Mujammil Ali A S. All Rights Reserved.
# 
# This module is Licensed under the Apache License, Version 2.0 with Commons
# Clause (the "License") as defined below, subject to the following condition.
#
#     https://commonsclause.com/
#
# “Commons Clause” License Condition v1.0
#
# Without limiting other conditions in the License, the grant of rights 
# under the License will not include, and the License does not grant to you, 
# the right to Sell the Software.


"""Streamlit based Evento Deployment Audit Module

This module acts as a web-based user interface and does the following stuff
    - Connects with Redis server and it is Evento protocol aware.
    - Capable of configuring ROI settings and checking the alert stream.
    - Capabel of handling five ROI regions.
    - Capable of handling three different event Type
    - Reads the ROI settings file with last saved settings and save with the
    latest change in ROI settings.

This module contains the following functions:
    - runStreamlitServer(): Runs the streamlit server and does either ROI
        settings configuration or alert stream based on selected application 
        mode and event type.
    - connectRedis(): Connect with redis server instance once in the entire 
        streamlit session by caching
    - configureROI(): Reads the ROI settings file and save it again with new
        information if altered. If the selected application mode is "Event 
        Alert Stream", then this function is invoked.
    - streamAlert(): Once ROI settings configuration is in place, it streams
        the processed frame from Evento Video Analytics Pipeline Module based 
        on the selected event type If the selected application mode is "ROI 
        Configuration", then this function is invoked.
"""

import streamlit as st
import time
import numpy as np
import redis
import base64
import cv2
import json
import pandas as pd
import os
import argparse

contourROI = {}
ROIColor = {"ROI 1": (0,255,0), "ROI 2": (255,0,255), "ROI 3": (255,255,0), "ROI 4": (255,0,0), "ROI 5": (50,30,70)}

my_json = st.empty()

contourROI = {}

settingsFileName = {""}

@st.cache(allow_output_mutation=True)
def connectRedis():
    """This function connects with the Redis Server and cache the connection for the
    entire streamlit session.

    Args:
        Nothing
    Returns:
        prd (redis): Returns the redis server instance.
    """

    prd = redis.Redis(host='localhost', port=6379,charset="utf-8", decode_responses=True)

    if not prd.ping():
        raise Exception('Redis unavailable')

    return prd

def configureROI(prd, contourROI, eventType, FLAGS):
    """This function configures the ROI settings based on eventType. It parses the keys that obeys the Redis 
    Evento protocol to get the latest or relevant frame/scene for which the ROI settings has to be done. It 
    helps in creating new point to form contour for the selected ROI, delete the last point (x,y), or deletes 
    the entire config using the slider and buttons. It finally stores the changes as json file for each event.

    Args:
        prd (redis): Returns the redis server instance.
        contourROI (dict): contains the last saved/currently modified json settings value as dict.
        eventType (int): Mapped integer value for type of the event.
        FLAGS: contains the command line arguments that holds the path of config file to be stored.

    Returns:
        Nothing. However, it stores the ROI settings as json file for each event.
    """
    
    image_placeholder = st.empty()

    #resp = prd.xrevrange(camera,count=1)
    #image_string = resp[0][1]['image']

    image_string = prd.get(eventType)

    img_data = base64.b64decode(image_string)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width, channel = img_np.shape
    overlay_np = img_np.copy()
    final_np = img_np.copy()

    whichROI = st.sidebar.selectbox('Select RoI',('ROI 1', 'ROI 2', 'ROI 3', 'ROI 4', 'ROI 5'))

    x = st.sidebar.slider('X Direction', 0, width-1)
    y = st.sidebar.slider('Y Direction', 0, height-1)

    alpha = st.sidebar.slider('Transperancy', 0.0, 1.0, 0.15, 0.005)

    if st.sidebar.button('<<<<<<Create Points>>>>>>'):
        if [x, y] not in list(contourROI[whichROI]):
            contourROI[whichROI].append([x, y])

    if st.sidebar.button('<<<<<Delete last Points>>>>'):
        if not list(contourROI[whichROI]) == []:
            del contourROI[whichROI][-1]

    if st.sidebar.button('<<<<<<<<Clear All>>>>>>>>'):
        contourROI = {'ROI 1':[],'ROI 2':[],'ROI 3':[], 'ROI 4':[], 'ROI 5':[]}

    for index, whichROI in enumerate(contourROI.keys()):
        if not list(contourROI[whichROI]) == []:
            
            ctr = np.array(contourROI[whichROI]).reshape((-1,1,2)).astype(np.int32)

            cv2.drawContours(overlay_np,[ctr],0,ROIColor[whichROI],-1)
            cv2.drawContours(overlay_np,[ctr],0,(255,255,255),4)
            cv2.addWeighted(overlay_np, alpha, final_np, 1 - alpha, 0, final_np)
            moment = cv2.moments(ctr)

            if (moment["m10"] != moment["m00"])  and  (moment["m01"] != moment["m00"]):
                
                centerX = int(moment["m10"] / moment["m00"])
                centerY = int(moment["m01"] / moment["m00"])
                cv2.putText(final_np, str(index + 1), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

    with open(os.path.join(FLAGS.app_config_dir, eventType + '.json'), 'w') as json_file:
        json.dump(contourROI, json_file)

    img_np = cv2.circle(final_np, (x, y),10,(255,0,0),-1)

    image_placeholder.image(final_np, channels="BGR")

    #df = pd.DataFrame.from_dict(contourROI)
    df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in contourROI.items() ]))

    st.subheader('ROI Points')

    st.write(df)

def streamAlert(app_mode, prd, contourROI):
    """This function stream the processed video as images to the image container. It gets the video feed from Redis 
    Stream of Evento video Analytics Pipeline.

    Args:
        app_mode (int): Integer value mapped to application mode. 1 - ROI Configuration and 2 - Event Alert Stream 
        prd (redis): Returns the redis server instance.
        contourROI (dict): contains the last saved/currently modified json settings value as dict.

    Returns:
        Nothing. It streams forever until the application mode is changed to "ROI Configuration". Yes, it abuses the 
         while loop usage which breaks the Steamlit execution model at the time of developint this.
    """
    
    image_placeholder_ias = st.empty()
    st.write(" ")
    st.write(" ")

    videoResolution = st.sidebar.selectbox('Select Resolution',('Original', '360*240'))

    st.sidebar.title(" ")
    st.sidebar.title(" ")
    st.sidebar.title(" ")
    st.sidebar.title(" ")
    st.sidebar.title(" ")
    st.sidebar.title(" ")

    while True:

        if app_mode != "Event Alert Stream":
            break

        resp = prd.xrevrange("camera:0",count=1)
        image_string = resp[0][1]['image']

        appMetadata = json.loads(resp[0][1]['appMetadata'])
        
        img_data = base64.b64decode(image_string)
        nparr = np.fromstring(img_data, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if videoResolution == "360*240":
            img_np = cv2.resize(img_np,(360,240))

        image_placeholder_ias.image(img_np, channels="BGR")

        time.sleep(0.025)
        
        
def runStreamlitServer(FLAGS):
    """This function starts the streamlit server with all UI components and it allows user to change the application
    mode and event Type. It behaves differently based on the changes in corresponding UI components. It reads
    the ROI settings Json file and store it in dict. It also handles the Evento Protocol control signal.

    Args:
        FLAGS: contains the command line arguments that holds the path of config file to be stored.

    Returns:
        Nothing. It streams forever until the application mode is changed to "ROI Configuration". Yes, it abuses the 
         while loop usage which breaks the Steamlit execution model at the time of developint this.
    """
    dummy = 0

    st.markdown("<h1 style='text-align: center; color: red;'>Event Alert Manager</h1>", unsafe_allow_html=True)

    st.subheader('Video Stream')

    st.sidebar.title('Control Panel')

    app_mode = st.sidebar.selectbox("Application Mode",
        ("ROI Configuration", "Event Alert Stream"))
 
    eventType = st.sidebar.selectbox("Event",
    ("EventStat", "MonitorVehicle", "AlertStranger"))

    configFile = os.path.join(FLAGS.app_config_dir,eventType + '.json')
    
    if not os.path.exists(configFile):
        contourROI = {'ROI 1':[],'ROI 2':[],'ROI 3':[], 'ROI 4':[], 'ROI 5':[]}
        with open(configFile, 'w') as json_file:
            json.dump(contourROI, json_file)
    else:
        contourROI = json.load(open(configFile))

    prd = connectRedis()
    

    if app_mode == "ROI Configuration":

        prd.set("app_mode", 1)

        configureROI(prd, contourROI, eventType, FLAGS)

    elif app_mode == "Event Alert Stream":

        prd.set("app_mode", 2)


        if eventType == "EventStat":
            prd.set("event_type", 0)

        elif eventType == "AlertStranger":
            prd.set("event_type", 1)

        elif eventType == "MonitorVehicle":
            prd.set("event_type", 2)

        streamAlert(app_mode, prd, contourROI)

    else:

        prd.set("app_mode", 0)   

            
if __name__ == '__main__':
#   """ 
#   This Function is the entry point for this module. It collects the
#   user arguments based on argparse library. The list of options are
#
#       --app_config_dir - Directory where the json files that has the ROI
#            settings information. Every event has each file of json.  
#
#   How to run in CLI: streamlit run evento/ui_server.py -- --app_config_dir "config/"  
#
#   Finally it triggers the streamlit server with web interface.
#   """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--app_config_dir",
        type=str,
        default='config',
        help=("Directory include:'AlertStranger.json', 'EventStat.json', "
              "'MonitorVehicle.json', created by ui_server.py."),
        required=False)

    FLAGS = parser.parse_args()

    runStreamlitServer(FLAGS)