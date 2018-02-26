import sys
sys.path.append('''F:\python_learning\pythonMylib''')
import dbtool
import config
#加载通用模块
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats
from sklearn.externals import joblib
import uuid
### 定义配置文件
import os
from sklearn.externals import joblib
train_data_dir = 'data/train_data'
test_dir = 'data/test_data'
model_output = 'model/train'
log_output = 'log/train'
pic_output = 'pic/train'
temp_pic = 'data/temp_data'
def deal(HSV_SRC):
    HSV_datam = HSV_SRC.reshape(-1, 3)
    HSV_data = ([dbtool.normalization_n(HSV_datam[:, 0], 0, 180),
                dbtool.normalization_n(HSV_datam[:, 1], 0, 255),
                dbtool.normalization_n(HSV_datam[:, 2], 0, 255)])
    clf = joblib.load(os.path.join(model_output, "HSV_IF.model"))
    result = clf.predict(np.array(HSV_data).T)
    result =  result.reshape(HSV_SRC.shape[0], HSV_SRC.shape[1])
    result = np.where(result == -1, 0, 255)
    result = result.astype(np.uint8)
    dbtool.save_img(temp_pic, str(uuid.uuid1())+".jpg", result)
    clf_result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 2)))
    dbtool.save_img(temp_pic, str(uuid.uuid1()) + ".jpg", clf_result)
    return  clf_result
def filter_area(width, height):
    rect_area_min = 0.0003 * width * height;
    rect_area_max = 0.4 * width * height;
    max_height = 0.4 * height;
    max_width = 0.75 * width;
    def myfilter(x):
        area = x[2]*x[3]
        rate = x[2]/x[3]
        print(rate)
        return area > rect_area_min and area < rect_area_max and max_width > x[2] and max_height > x[3] and rate<8
    return myfilter
def deal_fun(HSV_SRC):
    clf_result = deal(HSV_src)
    im2, contours, hierarchy = cv2.findContours(clf_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = [cv2.boundingRect(item) for item in contours]
    box = filter(filter_area(HSV_SRC.shape[0], HSV_SRC.shape[1]), box)
    return box;

if __name__ == "__main__":
    #  加载一个测试样本
    imgsrc = cv2.imread(os.path.join(test_dir,"op2.jpg"))
    # 策略修改大小
    imgsrc = dbtool.resize(imgsrc, 600)
    HSV_src = cv2.cvtColor(imgsrc, cv2.COLOR_BGR2HSV)
    clf_result = deal(HSV_src)
    im2, contours, hierarchy = cv2.findContours(clf_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = [cv2.boundingRect(item) for item in contours]
    print(imgsrc.shape[0], imgsrc.shape[1])
    box = filter(filter_area(imgsrc.shape[0],imgsrc.shape[1]), box)
    #
    for item in box:
        imgsrc = cv2.rectangle(imgsrc, (item[0], item[1]), (item[0] + item[2], item[1] + item[3]), (0, 255, 0), 2)
    dbtool.save_img(temp_pic, str(uuid.uuid1()) + ".jpg", imgsrc)
    cv2.imshow("xxx",imgsrc);
    cv2.waitKey()