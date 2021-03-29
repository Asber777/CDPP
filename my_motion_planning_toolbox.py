# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
import yaml
def speed(route):
    step = len(route)-1
    L = 0
    for i in range(step):
        L += straight_distance(route[i],route[i+1])
    return L/step
def generate_yaml_doc(yaml_file,py_object):
    file = open(yaml_file, 'w', encoding='utf-8')
    yaml.dump(py_object, file)
    file.close()
def get_yaml_data(yaml_file):
    # 打开yaml文件
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    # 将字符串转化为字典或列表
    data = yaml.safe_load(file_data)
    print(data)
    return data
def straight_distance(point_a, point_b):
    if point_b==None or point_a==None:
        return 0
    x1,y1 = point_a
    x2,y2 = point_b
    return ( (x1-x2)**2 + (y1-y2)**2 )**0.5
# def improved_straight_distance(point_a, point_b):
#     x1,y1 = point_a
#     x2,y2 = point_b
#     length = ( (x1-x2)**2 + (y1-y2)**2 )**0.5
#     return int(length/100)*100
def find_nearest_obstacle_distance(array,value):#这个速度快一点
    dis_array = (array-value)**2
    min_distance = (dis_array[:,1]+dis_array[:,0]).min()
    return min_distance**0.5

'''
vertex：顶点集
adjacency_mat：邻接矩阵
start_index：一般是0 代表顶点集第一个是起点
goal_index：一般是1 代表顶点集第二个是起点
'''
current_path = os.path.abspath(".")
yaml_path = os.path.join(current_path, "my_motion_planning_toolbox.yaml")
data = get_yaml_data(yaml_path)
D_MAX = data["D_MAX"]
OBSTACLE_MAX = data["OBSTACLE_MAX"]
APF_WAY = data["APF_WAY"]
ROU = data["ROU"]
#这个会有终点不可达效应:
#https://blog.csdn.net/junshen1314/article/details/50472410
def APF_function(D):
    if D > D_MAX:
        return 0
    elif D > 0:
        return OBSTACLE_MAX * (1.0 / D - 1.0 / D_MAX) / (D**2)
    else:
        return OBSTACLE_MAX
#这个会有终点不可达效应
# 3 rou的时候接近0
def my_APF_function(D):
    if D > D_MAX:
        return 0
    else:
        return OBSTACLE_MAX * np.exp(- D ** 2 / (2 * ROU ** 2))
'''解决了终点不可达效应,但是在动态障碍物的情况下显得有一些贪心,时常会发生碰撞'''
'''improved_APF_function的障碍物影响力下降的太快,不能很好的避障'''
def improved_APF_function(D,D_goal):
    if D > D_MAX:
        return 0
    elif D > 0:
        uo =  (1.0 / D - 1.0 / D_MAX)  / (D ** 2) * (D_goal ** 2)
        uo +=  ((1.0 / D - 1.0 / D_MAX )**2) * D_goal
        return uo
    else:
        return OBSTACLE_MAX
def my_improved_APF_function(D,D_goal):
    if D > D_MAX:
        return 0
    else:
        uo =  np.exp(- D ** 2 / (2 * ROU ** 2))  / (D ** 2) * (D_goal ** 2)
        uo += np.exp(- D ** 2 / (2 * ROU ** 2))**2 * D_goal
        return uo
def visualize_apf(img,mr):
    #assert isinstance(mr,MotionRoadmap)
    #  visualize apf
    for i,r in enumerate(mr.robot_list):
        x, y = r.getLocation()
        for h in range(x - 50, x + 50):
            for u in range(y - 50, y + 50):
                apf = mr.apf_cul((h, u), r.current_goal,i)# modified to Robot's subgoal
                cv2.circle(img, (h, u), 1, (apf, apf, apf), -1)
        if r.collision:
            cv2.circle(img, (x, y), 10, (100, 200, 150), -1)
        else:
            cv2.circle(img, (x, y), 10, (255, 255, 0), -1)
        cv2.circle(img, r.current_goal, 5, (255, 100, 0), 1)# modified to Robot's subgoal
    return img



