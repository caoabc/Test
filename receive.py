#!/usr/bin/env python
#!coding=utf-8
#right code !
#write by leo at 2018.04.26
#function: 
#display the frame from another node.
#ROS
from __future__ import division, print_function, absolute_import
import rospy
import numpy as np
from sensor_msgs.msg import Image as Image2
from cv_bridge import CvBridge, CvBridgeError
import cv2
#deep_sort_yolov3
#from __future__ import division, print_function, absolute_import                                                                 
import os
from timeit import time
import warnings
import sys
#import cv2
#import numpy as np
from PIL import Image
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0

# deep_sort
model_filename = '/home/nvidia/hello_rospy/src/beginner_tutorials/scripts/model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)
writeVideo_flag = True

#video_capture = cv2.VideoCapture(0)

if writeVideo_flag:
        # Define the codec and create VideoWriter object
        #w = int(video_capture.get(3))
        #h = int(video_capture.get(4))
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
    list_file = open('detection.txt', 'w')
    frame_index = -1

fps = 0.0
yolo=YOLO()

def callback(data):
    scaling_factor = 0.5
    global count, bridge, nms_max_overlap, encoder, tracker, writeVideo_flag, list_file, frame_index, fps, yolo
    count = count + 1
    if count == 1:
        count = 0
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
        #frame = bridge.imgmsg_to_cv2(data, "rgb8")
        #cv2.imshow("frame", frame)
        t1 = time.time()
        #print(type(frame))
        image = Image.fromarray(frame)
        #image.save("1.jpg")
        #frame=cv2.imread("1.jpg")
        #cv2.imshow("frame",frame)
        #ii2=Image.fromarray(frame)
        boxs = yolo.detect_image(image)
        #print(boxs)
        #frame=np.array(frame)
        features = encoder(frame, boxs)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        tracker.predict()
        tracker.update(detections)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        cv2.imshow('result', frame)
        if writeVideo_flag:
            #out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index) + ' ')
            if len(boxs) != 0:
                for i in range(0, len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(boxs[i][3]) + ' ')
            list_file.write('\n')
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

        cv2.waitKey(3)
    else:
        pass





if __name__=='__main__':
    #ROS
    rospy.init_node('webcam_display', anonymous=True)
    count = 0
    bridge = CvBridge()
    #ROS
    #rospy.init_node('webcam_display', anonymous=True)
    #count = 0
    #bridge = CvBridge()
    rospy.Subscriber('dji_sdk/image_raw', Image2, callback)
    rospy.spin()
