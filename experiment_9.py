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
sys.path.append('''.\lib''')
import dbtool
#  三种降维方法
#  实验失败。。。。
from lpproj import LocalityPreservingProjection
from lpproj2 import LocalityPreservingProjection2
from sklearn.decomposition import PCA

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
model_dir = 'model/experiment9'
pic_dir = 'pic/experiment9'
data_dir = 'data/experiment9'
data_item = ["a.jpg","2.jpg"]
zoom_size = 15
step_size = 1
win_size = 11
range_size = 3
max_components = 2
t = 0.96
alpha = 0.999
reduce={
    "PCA": PCA(n_components=max_components),
    "LPP": LocalityPreservingProjection(n_components=max_components),
    "MY_LPP":LocalityPreservingProjection2(n_components=max_components),
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
    img_object = dbtool.resize(img_object, width=img_object.shape[1] / zoom_size)
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
        if data_name == "MY_LPP":
            reducer.SetW(dbtool.resturct_w(range_size,win_size,feature_list[0][2].unlark))
        reducer.fit(feature_list[0][2].descriptors)
        feature_object = reducer.transform(feature_list[0][2].descriptors).reshape([feature_list[0][2].x, feature_list[0][2].y, max_components])
        feature_bg = reducer.transform(feature_list[1][2].descriptors).reshape([feature_list[1][2].x, feature_list[1][2].y, max_components])
        plt.figure(figsize=(6, 4))
        plt.rcParams.update(params)
        plt.subplots_adjust(right=0.96, left=0.06, bottom=0.14, top=0.95, wspace=0.2, hspace=0.3)
        plt.imshow(feature_bg[:,:,0], cmap='jet')
        plt.colorbar()
        plt.title(data_name)
    plt.show()

