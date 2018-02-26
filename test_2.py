model_dir = 'model/experiment9'
pic_dir = 'pic/experiment9'
data_dir = 'data/experiment9'

import math
from enum import Enum
import os
import cv2
import numpy as np
import math
def imresize(src, height):
    ratio = src.shape[0] * 1.0 / height
    width = int(src.shape[1] * 1.0 / ratio)
    return cv2.resize(src, (width, height))
import matplotlib.pyplot as plt
def calc_distance(martix):
    martix = martix.reshape(-1,4)
    return np.sqrt((martix[:,0]- martix[:,2])**2+(martix[:,1]- martix[:,3])**2)
def point_disttance(y,x,arr):
    x1 = arr[0]
    y1 = arr[1]
    x2 = arr[2]
    y2 = arr[3]
    cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)
    d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    r = cross / d2
    px = x1 + (x2 - x1) * r
    py = y1 + (y2 - y1) * r
    return math.sqrt((x - px) * (x - px) + (py - y1) * (py - y1))
if __name__ == '__main__':
    print(point_disttance(211,205,[305.791015625, 242.8429718017578, 419.61041259765625, 309.84844970703125]))