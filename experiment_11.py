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
sys.path.append('''F:\python_learning\pythonMylib''')
import dbtool
#  三种降维方法
#  实验失败。。。。
from lpproj import LocalityPreservingProjection
from lpproj2 import LocalityPreservingProjection2
from sklearn.decomposition import PCA

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
model_dir = 'model/experiment11'
pic_dir = 'pic/experiment11'
data_dir = 'data/experiment9'
data_item = ["6.jpg","7.jpg"]
zoom_size = 20
step_size = 1
win_size = 11
range_size = 3
max_components = 2
t = 0.96
alpha = 0.995
reduce={
    "PCA": PCA(n_components=max_components),
    # "LPP": LocalityPreservingProjection(n_components=max_components),
    # "MY_LPP":LocalityPreservingProjection2(n_components=max_components),
}
def get_feature_bylark(gray,step_size,window_size=5):
    ds, unlark, LARK = lark._raw_larks(gray,wsize=window_size,interval=step_size);
    class p:
        pass
    p.x = math.ceil(gray.shape[0] /step_size);
    p.y = math.ceil(gray.shape[1] /step_size);
    x, y, z = LARK.T.shape
    transe_LARK =  LARK.T.reshape(x * y, z, order='F').copy()
    p.descriptors = transe_LARK
    p.ds = ds
    p.unlark =unlark
    return p
def extra_pack(fileName):
    img_object = cv2.imread(os.path.join(data_dir, fileName))
    img_object = dbtool.resize(img_object, width=img_object.shape[1] / zoom_size);
    gray_object = cv2.cvtColor(img_object, cv2.COLOR_BGR2GRAY)
    des_object = get_feature_bylark(gray_object, step_size, win_size)
    return img_object, gray_object, des_object
if __name__ =='__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    feature_list =  [extra_pack(item) for item in data_item]
    for data_name, reducer in reduce.items():
        reducer.fit(feature_list[0][2].descriptors)
        feature_object = reducer.transform(feature_list[0][2].descriptors).reshape([feature_list[0][2].x, feature_list[0][2].y, max_components])
        feature_bg = reducer.transform(feature_list[1][2].descriptors).reshape([feature_list[1][2].x, feature_list[1][2].y, max_components])
        dis = dbtool.fftconvle(feature_bg, feature_object)
        squrt = np.sqrt(dis)
        dis = dis / (1 - dis)


        showTarget = feature_list[1][0].copy()
        showTarget = cv2.cvtColor(showTarget,cv2.COLOR_BGR2RGB)
        # 获取框 个数
        xnum = math.ceil(dis.shape[0] * dis.shape[1] * (1 - alpha))
        dis_one = np.reshape(dis, dis.shape[0] * dis.shape[1])
        result = dis_one.argsort()[-xnum:]
        apoint = []
        for i in result:
            x_i = (i % dis.shape[1]) * step_size
            y_i = int(i / dis.shape[1]) * step_size
            apoint.append([x_i, y_i, x_i + feature_list[0][0].shape[1], y_i + feature_list[0][0].shape[0]])
            dbtool.drwabox(showTarget, feature_list[0][0], x_i, y_i,(0,255,0),1)

        target = np.array(apoint);
        keep = dbtool.py_cpu_nms(target, 0.4)

        for i in keep:
            dbtool.drwabox(showTarget,  feature_list[0][0], apoint[i][0], apoint[i][1],(255,255,0),3)

        plt.figure(figsize=(6.3, 4.7))
        plt.rcParams.update(params)
        plt.imshow(showTarget)
        plt.axis('off')
        plt.tight_layout()
        # plt.subplots_adjust(right=0.96, left=0.06, bottom=0.05, top=0.95, wspace=0.2, hspace=0.1)
    plt.savefig(os.path.join(pic_dir, "result.png"))
    plt.show()

