# AItrainer

## 基于Paddlehub的哑铃抬举检测及自动计数

### PaddleHub 简介

使用Paddlehub能够便捷地获取PaddlePaddle生态下的预训练模型，完成模型的管理和一键预测。配合使用Fine-tune API，可以基于大规模预训练模型快速完成迁移学习，让预训练模型能更好地服务于用户特定场景的应用。

本项目采用Paddlehub, 基于[human_pose_estimation_resnet50_mpii](https://www.paddlepaddle.org.cn/hubdetail?name=human_pose_estimation_resnet50_mpii&en_category=KeyPointDetection)人体关键点检测模型。

```
# 使用前先安装模型
hub install human_pose_estimation_resnet50_mpii==1.1.1
```

### 模型概述

人体骨骼关键点检测(Pose Estimation) 是计算机视觉的基础性算法之一，在诸多计算机视觉任务起到了基础性的作用，如行为识别、人物跟踪、步态识别等相关领域。具体应用主要集中在智能视频监控，病人监护系统，人机交互，虚拟现实，人体动画，智能家居，智能安防，运动员辅助训练等等。 该模型的论文《Simple Baselines for Human Pose Estimation and Tracking》由 MSRA 发表于 ECCV18，使用 MPII 数据集训练完成。

### 项目思路

本项目实现在健身过程的哑铃抬举的过程中，对进行此过程的六个关节点分别检测，即左右手腕、左右手肘、左右肩部，并分别计算以手肘为基点，手腕、手肘、肩部所构成的向量夹角。以夹角为核心 判断哑铃动作是否到位，完成两次的【0°-90°】向量夹角变化算作一次哑铃抬举动作。

### 项目流程

**Step one**

使用human_pose_estimation_resnet50_mpii模型对受众人体关键点进行检测，并从中抽取出项目所需的六个关节点，即：【right_wrist、right_elbow、right_shoulder、left_wrist、left_elbow、left_shoulder】,并将其存放进列表中：

```
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

```

**Step two**

前文提及，本项目的核心在于【以手肘为中心点，对手腕、手肘、肩部所构成的夹角进行计算，以夹角为判断依据，从而对哑铃抬举动作进行监督并且计数】。

```
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
```

**Step three**

使用OpenCV打开本地摄像头获取动态页面，并将项目检测结果实时展示在页面中：
```
cap = cv2.VideoCapture(0)

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
```
【注：这里区分了左臂or右臂】

### 效果展示【[bnilibili链接](https://www.bilibili.com/video/BV1cU4y1N7Yo/)】

![image](https://user-images.githubusercontent.com/58030051/133214015-b9211570-4b9f-48a5-9edb-f2dac00b0d18.png)


