import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib import rc
import lark
import math
import sys
import time
import numpy.linalg as LA
sys.path.append('''F:\python_learning\pythonMylib''')
import dbtool
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
model_dir = 'model/experiment18'
pic_dir = 'pic/experiment18'
data_dir = 'data/experiment13'
items = ['a.jpg','c.png']
def calc_distance(martix):
    martix = martix.reshape(-1,4)
    return np.sqrt((martix[:,0]- martix[:,2])**2+(martix[:,1]- martix[:,3])**2)

def point_disttance(x,y,arr):
    x1 = arr[0]
    y1 = arr[1]
    x2 = arr[2]
    y2 = arr[3]
    d = (math.fabs((y2 - y1) * x + (x1 - x2) * y + ((x2 * y1) - (x1 * y2)))) / (math.sqrt(pow(y2 - y1, 2) + pow(x1 - x2, 2)));
    return d

def angle(martix):
    martix = martix.reshape(-1, 4)
    return  np.arctan((martix[:,3]- martix[:,1])/(martix[:,0]- martix[:,2])) * 180 / np.pi
# 定义一个检测器
lsd = cv2.createLineSegmentDetector(0)
def Getfilter_line(img):
    height, width = img.shape
    # select_h = int(height * 0.18)
    # select_w = int(width * 0.18)
    # img_gary_roi = gray[select_h:height - select_h, select_w: width - select_w]
    lines, width, prec, nfa = lsd.detect(img)

    alldistance = calc_distance(lines)
    newlines = lines[alldistance > 150]
    alldistance = alldistance[alldistance > 150]
    new_list = newlines.reshape(-1, 4).tolist()

    results = map(lambda x: point_disttance(int(img.shape[1] / 2), int(img.shape[0] / 2), x), new_list)
    d = np.array(list(results))

    select = d< 30
    new_distance = alldistance[select]
    newlines = newlines[select]
    d = d[select]
    sumlength = np.sum(new_distance)
    sumd = np.sum(d)
    k = calc_k(new_distance,d,sumlength,sumd)
    angeles = angle(newlines)
    result = np.dot(k,angeles.T)
    return result, newlines

def calc_k(length, d, lengthsum, dsum):
    return (length/lengthsum)*0.9 + (d/dsum)*0.1

if __name__ =='__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    # 读取图片
    for i,item in enumerate(items):
        img = cv2.imread(os.path.join(data_dir, item))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result, newlines = Getfilter_line(gray)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = lsd.drawSegments(img, newlines)
        cv2.putText(img, str(result)[0:7], (350, 300), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(img, (0, int(gray.shape[0]/2)), (gray.shape[0], int(gray.shape[1]/2)), (0,255,255), 2)
        cv2.imwrite(os.path.join(pic_dir, str(i)+'.jpg'), img)

    # cv2.imshow("xxxx",img)
    # cv2.waitKey()