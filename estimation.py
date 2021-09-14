"""
哑铃动作标准化计数
"""

import math
import time
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import paddlehub as hub
from PIL import Image

pose_estimation = hub.Module(name="human_pose_estimation_resnet50_mpii")

count = 0
det_1 = 0
angle_last = 90.0

train_content = input("请问您训练的是左臂还是右臂：") #左臂还是右臂

def Cal_Ang(point_1, point_2, point_3):
    """
    根据三点坐标计算夹角
    :param point_1: 点1坐标
    :param point_2: 点2坐标
    :param point_3: 点3坐标
    :return: 点2的夹角(中心点)
    """
    a=math.sqrt((point_2[0]-point_3[0]) * (point_2[0]-point_3[0])+(point_2[1]-point_3[1]) * (point_2[1]-point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0]) * (point_1[0]-point_3[0])+(point_1[1]-point_3[1]) * (point_1[1]-point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0]) * (point_1[0]-point_2[0])+(point_1[1]-point_2[1]) * (point_1[1]-point_2[1]))
    B=math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
    return B

def aitrainer(picture):

    consequence = []
    result = pose_estimation.keypoint_detection(images=[cv2.imread(picture)])

    predict = dict(result[0]["data"])

    right_wrist = predict['right_wrist']
    consequence.append(right_wrist)

    right_elbow = predict['right_elbow']
    consequence.append(right_elbow)

    right_shoulder = predict['right_shoulder']
    consequence.append(right_shoulder)

    left_shoulder = predict['left_shoulder']
    consequence.append(left_shoulder)

    left_elbow = predict['left_elbow']
    consequence.append(left_elbow)

    left_wrist = predict['left_wrist']
    consequence.append(left_wrist)

    return consequence


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("AiTrainer/curls.mp4")

while(1):
    # get a frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))

    print("训练将在5秒后进行，请您做好准备！")

    for i in range(5):
        time.sleep(1)
        m = 5 - i
        print(m)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
    cv2.imwrite("camera.jpg", frame)

    point = aitrainer('camera.jpg')

    if train_content == '左臂':

        left_wrist = point[5]
        left_elbow = point[4]
        left_shoulder = point[3]

        angle_left = Cal_Ang(left_wrist,left_elbow,left_shoulder)

        diff = angle_left - angle_last

        if diff >=0 and det_1 == 0:
            count += 0.5
            det_1 = 1 
    
        if diff <=0 and det_1 ==1:
            count += 0.5
            det_1 = 2

        cv2.line(frame, tuple(left_wrist), tuple(left_elbow),(0, 0, 255), 3)
        cv2.line(frame, tuple(left_elbow), tuple(left_shoulder),(0, 0, 225), 3)

        if count % 1.0 == 0.0:
            det_1 = 0
            print("count:%.2f"%count)

            cv2.rectangle(frame, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 10,
            (255, 0, 0), 20)

        cv2.imshow('OpenPose using OpenCV', frame)

    if train_content == '右臂':

        right_wrist = point[0]
        right_elbow = point[1]
        right_shoulder = point[2]

        angle_right = Cal_Ang(right_wrist,right_elbow,right_shoulder)

        diff = angle_right - angle_last

        if diff >=0 and det_1 == 0:
            count += 0.5
            det_1 = 1 
    
        if diff <=0 and det_1 ==1:
            count += 0.5
            det_1 = 2

        cv2.line(frame, tuple(right_wrist), tuple(right_elbow),(0, 0, 255), 3)
        cv2.line(frame, tuple(right_elbow), tuple(right_shoulder),(0, 0, 225), 3)

        if count % 1.0 == 0.0:
            det_1 = 0
            print("count:%.2f"%count)

            cv2.rectangle(frame, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 10,
            (255, 0, 0), 20)

        cv2.imshow('OpenPose using OpenCV', frame)

cap.release()

