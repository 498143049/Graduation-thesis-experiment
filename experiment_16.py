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
sys.path.append('''F:\python_learning\pythonMylib''')
import dbtool
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
model_dir = 'model/experiment16'
pic_dir = 'pic/experiment16'
data_dir = 'data/experiment13'
if __name__ =='__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    # 读取图片
    img = cv2.imread(os.path.join(data_dir, 'c.png'))
    height, width, depth = img.shape
    select_h = int(height*0.18)
    select_w = int(width*0.18)
    cv2.rectangle(img, (select_h, select_w), (height-select_h, width-select_w), (0, 255, 255), 5)
    cv2.imwrite(os.path.join(pic_dir,"result.png"), img)