## 验证各种算法对闭合面的闭合情况，s
import sys
sys.path.append('''F:\python_learning\pythonMylib''')
import dbtool
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import  numpy as np
import matplotlib.font_manager
from scipy import stats
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.externals import joblib
from matplotlib import rc
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import datetime
import time
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
# 读取图像
list_img = {
    "img1" :['src.jpg', 'bg.jpg'],
    "img2" :['src2.jpg', 'bg2.jpg']
}
model_dir = 'model/experiment5'
pic_dir = 'pic/experiment5'
data_dir = 'data/experiment5'
scacle = [4,8,16]
select_model = ["Isolation Forest", "Local Outlier Factor"]

def load_mask(fileName,scaleNum):
    mask = cv2.imread(fileName,0)
    mask = dbtool.resize(mask, mask.shape[1] / scaleNum)
    ret,thresh1=cv2.threshold(mask,80,255,cv2.THRESH_BINARY)
    thresh1 = (thresh1/255).astype('int').reshape(-1)
    return thresh1
# os.path.join(data_dir, list_img[0])
def load_data(fileName,scaleNum):
    imgsrc = cv2.imread(fileName)
    imgsrc = dbtool.resize(imgsrc, imgsrc.shape[1] / scaleNum)
    HSV_src = cv2.cvtColor(imgsrc, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    YUV_src = cv2.cvtColor(imgsrc, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    HSV_data = [dbtool.normalization_n(HSV_src[:, 0], 0, 180),
                dbtool.normalization_n(HSV_src[:, 1], 0, 255),
                dbtool.normalization_n(HSV_src[:, 2], 0, 255)]
    HS_data = [dbtool.normalization_n(HSV_src[:, 0], 0, 180),
               dbtool.normalization_n(HSV_src[:, 1], 0, 255)]
    YUV_data = [dbtool.normalization_n(YUV_src[:, 0], 0, 255),
                dbtool.normalization_n(YUV_src[:, 1], 0, 255),
                dbtool.normalization_n(YUV_src[:, 2], 0, 255)]
    UV_data = [dbtool.normalization_n(YUV_src[:, 1], 0, 255),
               dbtool.normalization_n(YUV_src[:, 2], 0, 255)]

    data_sourse = {"HSV": np.array(HSV_data).T, "HS": np.array(HS_data).T, "YUV": np.array(YUV_data).T,
                   "UV": np.array(UV_data).T}
    return  data_sourse
if __name__ =='__main__':
## 环境监测
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    for img_name,img_data in list_img.items():
        for scacle_item in scacle:
            print("current img:{0} scacle:{1}".format(img_name,scacle_item))
        ### 准备数据
            thresh1 = load_mask(os.path.join(data_dir, img_data[1]), scacle_item)
            data_sourse = load_data(os.path.join(data_dir, img_data[0]), scacle_item)
        ### 画图工具
            plt.figure(figsize=(6, 4))
            plt.rcParams.update(params)
            plt.subplots_adjust(right=0.96, left=0.1, bottom=0.1, top=0.94, wspace=0.2, hspace=0.3)
            lines = []
            labels = []
        ### 遍历绘图
            for data_name, data  in data_sourse.items():
                for model_name in select_model:
                    print("select data: ",data_name,"Select model:", model_name)
                    if(len(data_name)==3):
                        or_path = "model/experiment4"
                    else:
                        or_path = "model/experiment3"

                    clf = joblib.load(os.path.join(or_path, data_name+model_name+".model"));
                    start = time.clock()
                    if model_name == "Local Outlier Factor":
                        Z = clf._decision_function(data)
                    else:
                        Z = clf.decision_function(data)
                    end = time.clock()
                    costtime =  end-start
                    min = Z.min()
                    max = Z.max()
                    Z = (Z - min)/(max - min)
                    th_Z  = (clf.threshold_ - min)/(max - min)
                    Z = Z / th_Z
                    Z = np.clip(Z, 0, 1)
                    average_precision = average_precision_score(thresh1, Z)
                    print('Average precision-recall score: {0:0.2f} cost time: {1:0.2f}'.format(average_precision, costtime))
                    print('=======================================================')
                    precision, recall, _ = precision_recall_curve(thresh1, Z)
                    l, = plt.plot(recall, precision)
                    lines.append(l)
                    labels.append('Precision-recall for {0} (area = {1:0.2f})'.format(data_name+" "+model_name, average_precision))

            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(' Precision-Recall curve:')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.legend(lines, labels,loc=3)
            plt.savefig(os.path.join(pic_dir,img_name+str(scacle_item)+".png"))
    plt.show()