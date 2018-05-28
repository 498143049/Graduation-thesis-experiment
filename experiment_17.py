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
sys.path.append('''.\lib''')
import dbtool
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
model_dir = 'model/experiment17'
pic_dir = 'pic/experiment17'
data_dir = 'data/experiment13'
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

if __name__ =='__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    # 读取图片
    img = cv2.imread(os.path.join(data_dir,'g.jpg'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    select_h = int(height*0.18)
    select_w = int(width*0.18)
    img_roi = img[select_h:height-select_h, select_w: width-select_w]
    img_gary_roi = gray[select_h:height - select_h, select_w: width - select_w]
    lsd = cv2.createLineSegmentDetector(0)
    lines, width, prec, nfa = lsd.detect(img_gary_roi)
    drawn_img = lsd.drawSegments(img_roi.copy(), lines)
    cv2.imwrite(os.path.join(pic_dir,"LSD_result.png"), drawn_img)
    ans = calc_distance(lines)
    newlines = lines[ans>100]
    drawn_img_1 = lsd.drawSegments(img_roi.copy(), newlines)
    new_list = newlines.reshape(-1,4).tolist();
    results = map(lambda x: point_disttance(int(drawn_img_1.shape[1]/2), int(drawn_img_1.shape[0]/2),x),new_list)
    newlines =  newlines[np.array(list(results))<30]
    # cv2.circle(drawn_img_1, (, 20, (55, 255, 155), -1)  # 修改最后一个参数
    drawn_img_2 = lsd.drawSegments(img_roi.copy(), newlines)
    cv2.imwrite(os.path.join(pic_dir, "LSD_filter_result_1.png"), drawn_img_1)
    cv2.imwrite(os.path.join(pic_dir, "LSD_filter_result_3.png"), drawn_img_2)
    # cv2.imshow("xxxx",drawn_img_2)
    cv2.waitKey()