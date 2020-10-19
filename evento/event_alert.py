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


"""Event of interest analysis manager Module

This module is an event analysis manager for event of interest that handles 
the follwing functionalities.
    - Get the plain image with bounding boxes of overall detected objects
    - Read the ROI settings file for each event of interest
    - It is capable of handling and analysing events such as  traffic stat, 
    vehicle monitoring and restricted area intrusion.
    - Process the information and overlay on the plaing image for visual 
    indication

This module contains the following class:
    - EventManager - Does all the jobs illustrated above.

The following is a simple usage example for using the EventManager class:
    # Creating an event manager instance
    event_manager = event_alert.EventManager(app_config_directory, event_type)

    # Getting the event name
    eventName = event_manager.getEventName(event_type)

    # Updating the ROI settings file on the fly if event type changes. In this 
    #case event manager shall update the event name as well
    event_manager.updateROISettings(event_type)
    eventName = event_manager.getEventName(event_type)

    # For monitoring the presence of vehicle
    processed_image, result, appMetadata = event_manager.monitorVehicle
        (bb_objects_numpy_arr,plain_image)

    # For monitoring the traffic stat
    processed_image, result, appMetadata = event_manager.findEventStat
        (bb_objects_numpy_arr,plain_image, traffic_vehicle_count_thresh_reach)

    # For alerting intrusion into restricted area
    processed_image, result, appMetadata = event_manager.alertStrangerIntrusion
        (bb_objects_numpy_arr,plain_image)
"""

import json
import cv2
import numpy as np
import os

class EventManager():
    """Event Manager class analyzes the event of interest and its input 
    configuration settings.

    Members:
        configPath (str): path of the directory where all the ROI json settings 
            file resides.
        contourROI (dict): Reads the json of ROI settings file and hold that 
            info as dict.
        colorBlink (int): counter for controlling visual alert changes in 
            processed image.
        eventType (int): Number that is mapped to event name. See also eventName
        eventName (dict): Dict that has event name {0:"EventStat", 
            1:"AlertStranger", 2:"MonitorVehicle"}.
        objects (dict): Objects that is required for event of interest. Now the 
            available events supports person, bicycle, car and motorcycle.
        ROIColor (dict): contains color map for visual representation for 
            ROI 1,2,3,4 and 5.

    Methods:
        __init__() - Constructor.
        updateROISettings() - Updates the settings of ROI by reading 
            the json again.
        getEventName() - Returns the event name in string for mapped eventType.
        findEventStat() - Analyzes the count of objects present in each ROI.
        alertStrangerIntrusion() - Alerts the intrusion of object into 
            restricted area.
        monitorVehicle() - Monitors the presence of object in ROI.
    """
    
    def __init__(self, config, eventType):
        """Constructor of EventManager

        Args:
            config (str): path of the directory where all the ROI json settings 
                file resides.
            eventType (int): Number that is mapped to event name. See also eventName.
        Returns:
            Nothing
        """

        self.configPath = config
        self.contourROI = {}
        self.colorBlink = 0
        self.eventType = eventType
        self.eventName = {0:"EventStat", 1:"AlertStranger", 2:"MonitorVehicle"}
        self.objects = {0:"person",1:"bicycle",2:"car",3:"motorcycle"}
        self.contourROI = json.load(open(os.path.join(self.configPath, self.eventName[self.eventType] + '.json')))
        self.ROIColor = {"ROI 1": (0,255,0), "ROI 2": (255,0,255), "ROI 3": (255,255,0), "ROI 4": (255,0,0), "ROI 5": (50,30,70)}

    def updateROISettings(self, eventType):
        """This method updates the ROI settings in contourROI. getEventName() shall be
        called following this method call. 

        Args:
            eventType (int): Number that is mapped to event name. See also eventName.
        Returns:
            Nothing
        """

        self.eventType = eventType
        self.contourROI = json.load(open(os.path.join(self.configPath, self.eventName[self.eventType] + '.json')))

    def getEventName(self, eventType):
        """This method returns the eventName mapped with eventType, in string.

        Args:
            eventType (int): Number that is mapped to event name. See also eventName.
        Returns:
            Nothing
        """
        
        return self.eventName[eventType]

    def findEventStat(self, np_boxes, img=None, threshold=0.5, alpha = 0.15, drawContour = True, vehicleThres = 5):
        """This method anlyses the number of obejcts present in each ROI, visually process the image
         and returns that metadata info with processed image.

        Args:
            np_boxes (np.ndarray): shape:[N,6], N: number of box，matix element:[class, 
                score, x_min, y_min, x_max, y_max].
            img (np.ndarray): np.ndarray of image read by cv2.
            threshold (float): Object detection's Confidence score threshold, object with this 
                score above which are filtered.
            alpha (float): Alpha value is used to specify the level of alpha channel for overlaid 
                processed image stuff.
            drawContour (bool): It defines whether to draw the processed info to alpha channel or not.
            vehicleThres (int): No of object threshold above which visual color change indication starts.

        Returns:
            final_np_1 (np.ndarray): np.ndarray of processed image with overlaid info.
            results (np.ndarray): shape:[N,7], N: number of filtered objects based on thres and eventType，
                matix element:[class, score, x_min, y_min, x_max, y_max, isObjectInsideROI]
            varApp (dict): That holds the number of event occurence inside that particular ROI.
        """

        results = {}
        results['boxes'] = []
        varApp = {}
        varApp["ROI 1"] = 0
        varApp["ROI 2"] = 0
        varApp["ROI 3"] = 0
        varApp["ROI 4"] = 0
        varApp["ROI 5"] = 0
        varAppThresh = vehicleThres

        if not img is None:
            overlay_np = img.copy()
            final_np = img.copy()
            final_np_1 = img.copy()

        if np_boxes.size != 0:
            expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
            np_boxes = np_boxes[expect_boxes, :]

        y = 20

        for indexROI, whichROI in enumerate(self.contourROI.keys()):
            if not list(self.contourROI[whichROI]) == []:

                cnt = np.array(self.contourROI[whichROI]).reshape((-1,1,2)).astype(np.int32)

                for indexBox, box in enumerate(np_boxes):

                    # Left Top >(x1,y1) as (box[2], boxp[3]) and right bottom (x2,y2) as (box[4],box[5])
                    # (poinX = x1 + [(x2 - x1) / 2] and pointY = y1 + [(y2-y2) / 2] 
                    pointX = box[2] + ((box[4] - box[2]) / 2)
                    pointY = box[3] + ((box[5] - box[3]) / 2)

                    isIntrusion = cv2.pointPolygonTest(cnt,(int(pointX),int(pointY)),False)


                    if isIntrusion == 1 or isIntrusion == 0:
                        print("Region {} is having {}".format(str(indexROI + 1), str(box[0])))
                        results['boxes'].append(np.append(np_boxes[indexBox], indexROI+1).tolist())
                        varApp[whichROI] += 1

                    elif isIntrusion == -1:
                        #print("Point 1 is outside the Region ", index + 1)
                        pass


                if varApp[whichROI] != 0:



                    if self.colorBlink < 5 and varApp[whichROI] >= varAppThresh: 
                        bColor = (0,0,255)
                    else:
                        bColor = self.ROIColor[whichROI]

                    stringValue = "Lane {} : {}".format(indexROI+1, varApp[whichROI])
                    cv2.putText(final_np, stringValue, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bColor, 2)
                    y = y + 20
                
                
                if not img is None and drawContour == True:
                    
                    
                    if self.colorBlink < 5 and varApp[whichROI] >= varAppThresh: 
                        bColor = (0,0,255)
                    else:
                        bColor = self.ROIColor[whichROI]
                    print(whichROI, self.colorBlink, bColor, varApp[whichROI])
                    cv2.drawContours(overlay_np,[cnt],0, bColor,-1)
                    cv2.drawContours(overlay_np,[cnt],0,(255,255,255),4)
                    moment = cv2.moments(cnt)
                    if (moment["m10"] != moment["m00"])  and  (moment["m01"] != moment["m00"]):
                        centerX = int(moment["m10"] / moment["m00"])
                        centerY = int(moment["m01"] / moment["m00"])
                        cv2.putText(final_np, str(indexROI + 1), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                    
                    cv2.addWeighted(overlay_np, alpha, final_np, 1 - alpha, 0, final_np_1)

        self.colorBlink += 1
        if self.colorBlink == 10:
            self.colorBlink = 0

        return final_np_1, results, varApp


    def alertStrangerIntrusion(self, np_boxes, img=None, threshold=0.5, alpha = 0.15, drawContour = True):
        """This method alerts when some interesting object inruded in each ROI, visually process the image
        and returns that metadata info with processed image.

        Args:
            np_boxes (np.ndarray): shape:[N,6], N: number of box，matix element:[class, 
                score, x_min, y_min, x_max, y_max].
            img (np.ndarray): np.ndarray of image read by cv2.
            threshold (float): Object detection's Confidence score threshold, object with this 
                score above which are filtered.
            alpha (float): Alpha value is used to specify the level of alpha channel for overlaid 
                processed image stuff.
            drawContour (bool): It defines whether to draw the processed info to alpha channel or not.

        Returns:
            final_np_1 (np.ndarray): np.ndarray of processed image with overlaid info.
            results (np.ndarray): shape:[N,7], N: number of filtered intruded objects based on thres 
                and eventType，matix element:[class, score, x_min, y_min, x_max, y_max, isObjectInsideROI]
            varApp (dict): That holds the number of event occurence inside that particular ROI.
        """
 
        results = {}
        results['boxes'] = []
        varApp = {}
        varApp["ROI 1"] = 0
        varApp["ROI 2"] = 0
        varApp["ROI 3"] = 0
        varApp["ROI 4"] = 0
        varApp["ROI 5"] = 0

        if not img is None:
            overlay_np = img.copy()
            final_np = img.copy()
            final_np_1 = img.copy()

        if np_boxes.size != 0:
            expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] == 0)
            np_boxes = np_boxes[expect_boxes, :]

        y = 20

        for indexROI, whichROI in enumerate(self.contourROI.keys()):
            if not list(self.contourROI[whichROI]) == []:

                cnt = np.array(self.contourROI[whichROI]).reshape((-1,1,2)).astype(np.int32)

                for indexBox, box in enumerate(np_boxes):

                    # Left Top >(x1,y1) as (box[2], boxp[3]) and right bottom (x2,y2) as (box[4],box[5])
                    # (poinX = x1 + [(x2 - x1) / 2] and pointY = y1 + [(y2-y2) / 2] 
                    pointX = box[2] + ((box[4] - box[2]) / 2)
                    pointY = box[3] + ((box[5] - box[3]) / 2)

                    isIntrusion = cv2.pointPolygonTest(cnt,(int(pointX),int(pointY)),False)


                    if isIntrusion == 1 or isIntrusion == 0:
                        print("Region {} is having some {}".format(str(indexROI + 1), str(box[0])))
                        results['boxes'].append(np.append(np_boxes[indexBox], indexROI+1).tolist())
                        varApp[whichROI] += 1

                    elif isIntrusion == -1:
                        #print("Point 1 is outside the Region ", index + 1)
                        pass

                if varApp[whichROI] != 0:

                    if self.colorBlink < 5: 
                        bColor = (0,0,255)
                    else:
                        bColor = self.ROIColor[whichROI]

                    stringValue = "{} Stranger Intrusion in Location {}".format(varApp[whichROI], indexROI+1)
                    cv2.putText(final_np, stringValue, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bColor, 2)
                    y = y + 20
                                
                if not img is None and drawContour == True:
                    
                    if self.colorBlink < 5 and varApp[whichROI] != 0: 
                        bColor = (0,0,255)
                    else:
                        bColor = self.ROIColor[whichROI]
                    print(whichROI, self.colorBlink, bColor, varApp[whichROI])
                    
                    cv2.drawContours(overlay_np,[cnt],0, bColor,-1)
                    cv2.drawContours(overlay_np,[cnt],0,(255,255,255),4)
                    moment = cv2.moments(cnt)
                    if (moment["m10"] != moment["m00"])  and  (moment["m01"] != moment["m00"]):
                        centerX = int(moment["m10"] / moment["m00"])
                        centerY = int(moment["m01"] / moment["m00"])
                        cv2.putText(final_np, str(indexROI + 1), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                    
                    cv2.addWeighted(overlay_np, alpha, final_np, 1 - alpha, 0, final_np_1)


        self.colorBlink += 1
        if self.colorBlink == 10:
            self.colorBlink = 0

        return final_np_1, results, varApp

    def monitorVehicle(self, np_boxes, img=None, threshold=0.5, alpha = 0.15, drawContour = True):
        """This method alerts when the vehicle is not present in the configured ROI, visually process the image
        and returns that metadata info with processed image.

        Args:
            np_boxes (np.ndarray): shape:[N,6], N: number of box，matix element:[class, 
                score, x_min, y_min, x_max, y_max].
            img (np.ndarray): np.ndarray of image read by cv2.
            threshold (float): Object detection's Confidence score threshold, object with this 
                score above which are filtered.
            alpha (float): Alpha value is used to specify the level of alpha channel for overlaid 
                processed image stuff.
            drawContour (bool): It defines whether to draw the processed info to alpha channel or not.

        Returns:
            final_np_1 (np.ndarray): np.ndarray of processed image with overlaid info.
            results (np.ndarray): shape:[N,7], N: number of captured event for the vehicle based on thres 
                and eventType，matix element:[class, score, x_min, y_min, x_max, y_max, isObjectInsideROI]
            varApp (dict): That holds the number of event occurence inside that particular ROI.
        """
 
        results = {}
        results['boxes'] = []
        varApp = {}
        varApp["ROI 1"] = 0
        varApp["ROI 2"] = 0
        varApp["ROI 3"] = 0
        varApp["ROI 4"] = 0
        varApp["ROI 5"] = 0

        if not img is None:
            overlay_np = img.copy()
            final_np = img.copy()
            final_np_1 = img.copy()

        if np_boxes.size != 0:
            expect_boxes = (np_boxes[:, 1] > threshold) & ((np_boxes[:, 0] == 1) | (np_boxes[:, 0] == 2) | (np_boxes[:, 0] == 3))
            np_boxes = np_boxes[expect_boxes, :]

        y = 20

        for indexROI, whichROI in enumerate(self.contourROI.keys()):
            if not list(self.contourROI[whichROI]) == []:

                cnt = np.array(self.contourROI[whichROI]).reshape((-1,1,2)).astype(np.int32)

                for indexBox, box in enumerate(np_boxes):

                    # Left Top >(x1,y1) as (box[2], boxp[3]) and right bottom (x2,y2) as (box[4],box[5])
                    # (poinX = x1 + [(x2 - x1) / 2] and pointY = y1 + [(y2-y2) / 2] 
                    pointX = box[2] + ((box[4] - box[2]) / 2)
                    pointY = box[3] + ((box[5] - box[3]) / 2)

                    isIntrusion = cv2.pointPolygonTest(cnt,(int(pointX),int(pointY)),False)
                     
                    

                    if isIntrusion == 1 or isIntrusion == 0:
                        print("My {} in Location {}".format(self.objects.get(box[0],"Unknown"), str(indexROI + 1)))
                        results['boxes'].append(np.append(np_boxes[indexBox], indexROI+1).tolist())
                        varApp[whichROI] += 1

                    elif isIntrusion == -1:
                        #print("Point 1 is outside the Region ", index + 1)
                        pass

                if varApp[whichROI] == 0:

                    if self.colorBlink < 5: 
                        bColor = (0,0,255)
                    else:
                        bColor = self.ROIColor[whichROI]

                    stringValue = "!!!No or missing vehicle Alert in Location {}!!!".format(indexROI+1)
                    cv2.putText(final_np, stringValue, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bColor, 2)
                    y = y + 20

                elif varApp[whichROI] != 0:
                    bColor = self.ROIColor[whichROI]

                    stringValue = "My {} in Location {}".format(self.objects.get(box[0],"Unknown"), indexROI+1)
                    cv2.putText(final_np, stringValue, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bColor, 2)
                    y = y + 20
                                
                if not img is None and drawContour == True:
                    
                    if self.colorBlink < 5 and varApp[whichROI] == 0: 
                        bColor = (0,0,255)
                    else:
                        bColor = self.ROIColor[whichROI]
                    print(whichROI, self.colorBlink, bColor, varApp[whichROI])
                    cv2.drawContours(overlay_np,[cnt],0, bColor,-1)
                    cv2.drawContours(overlay_np,[cnt],0,(255,255,255),4)
                    moment = cv2.moments(cnt)
                    if (moment["m10"] != moment["m00"])  and  (moment["m01"] != moment["m00"]):
                        centerX = int(moment["m10"] / moment["m00"])
                        centerY = int(moment["m01"] / moment["m00"])
                        cv2.putText(final_np, str(indexROI + 1), (centerX, centerY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                    
                    cv2.addWeighted(overlay_np, alpha, final_np, 1 - alpha, 0, final_np_1)


        self.colorBlink += 1
        if self.colorBlink == 10:
            self.colorBlink = 0

        return final_np_1, results, varApp
