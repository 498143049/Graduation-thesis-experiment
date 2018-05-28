import sys
sys.path.append('''.\lib''')
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
def test_hsv(fileName, num):
    imgsrc = cv2.imread(fileName);
    imgsrc = dbtool.resize(imgsrc, imgsrc.shape[1] / num);
    # imgsrc = cv2.fastNlMeansDenoisingColored(imgsrc, None, 10, 10, 7, 21)
    imgsrc = cv2.medianBlur(imgsrc, 5)

    img_equ = dbtool.hisEqulColor(imgsrc.copy());
    hsv = cv2.cvtColor(img_equ, cv2.COLOR_BGR2HSV);
    selectPixel = hsv.reshape(-1, 3);
    return  imgsrc,selectPixel, imgsrc.shape;
if __name__ == "__main__":
    #  加载一个测试样本
    imgsrc, test_data, shape = test_hsv("data/test_data/1.jpg",8)
    test_data_n1ormalize = dbtool.normalization(test_data)
    # 噪点比较多
    clf = joblib.load('data/model/if.model')
    th =  np.load("data/model/th.npy")
    clf_result = clf.predict(test_data_n1ormalize)
    clf_result = clf_result.reshape(shape[0], shape[1])
    clf_result = np.where(clf_result==-1,0,255);
    clf_result = clf_result.astype(np.uint8)
    # 提取出部分区域
    clf_result = cv2.morphologyEx(clf_result, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(13,1)))
    print(clf_result)
    im2, contours, hierarchy = cv2.findContours(clf_result, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    box = [cv2.boundingRect(item) for item in contours]
    for item in box:
        imgsrc = cv2.rectangle(imgsrc, (item[0], item[1]), (item[0] + item[2], item[1] + item[3]), (0, 255, 0), 2)
    # for item in contours:
    #     x, y, w, h = cv2.boundingRect(item)
    #     ## 这里需要几何过滤
    #     if w*h>100:

