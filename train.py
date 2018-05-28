# 通过训练样本获取数据
# coding=utf-8
# 加载自定义模块
import sys
sys.path.append('''.\lib''')
import dbtool
import config
#加载通用模块
import cv2
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats
rng = np.random.RandomState(40)
from sklearn.externals import joblib
from mpl_toolkits.mplot3d import axes3d
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}


# 设定模型 参数
clf = IsolationForest(n_estimators=4, max_samples=30000 ,n_jobs=-1, contamination=0.05, random_state=rng)
train_scale = 2
test_scale = 8
prosibility = 0.85
# 设定训练数据
train_data = (("src.jpg","bg.jpg"),
              ("black_bg.jpg","bg.jpg"),
              ("light_bg.jpg","bg.jpg"))
###测试样本
test_data = ("a.jpg","c.jpg")


### 定义配置文件
train_data_dir = 'data/train_data'
test_dir = 'data/test_data'
model_output = 'model/train'
log_output = 'log/train'
pic_output = 'pic/train'

def load_data(fileName,scaleNum):
    imgsrc = cv2.imread(fileName)
    # 策略修改大小
    imgsrc = dbtool.resize(imgsrc, 600)
    HSV_src = cv2.cvtColor(imgsrc, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    HSV_data = [dbtool.normalization_n(HSV_src[:, 0], 0, 180),
                dbtool.normalization_n(HSV_src[:, 1], 0, 255),
                dbtool.normalization_n(HSV_src[:, 2], 0, 255)]
    return  np.array(HSV_data).T,imgsrc.shape


if __name__ == "__main__":
    # 检查目录
    if not os.path.exists(model_output):
        os.makedirs(model_output)
    if not os.path.exists(pic_output):
        os.makedirs(pic_output)
    if not os.path.exists(log_output):
        os.makedirs(log_output)

    #加载训练数据
    train_data = [dbtool.load_data(os.path.join(train_data_dir, item[0]),train_scale)["HSV"][dbtool.load_mask(os.path.join(train_data_dir, item[1]),train_scale)!=0].reshape(-1,3) for item in train_data]
    train_data = np.concatenate(tuple(train_data), axis=0)
    print(train_data.shape)
    test_data = [load_data(os.path.join(test_dir, item),test_scale) for item in test_data]
    print(test_data[0][1])
    #画出散点分布图
    plt.figure(figsize=(6, 4))
    ax = plt.gca(projection='3d')
    plt.rcParams.update(params)
    plt.subplots_adjust(right=0.96, left=0.06, bottom=0.14, top=0.975, wspace=0.2, hspace=0.3)
    # 训练
    print("begin fit")
    clf.fit(train_data)
    print(" fit ok")
    # 找到边界概率值
    # 加载均匀分布的样本
    xx, yy, zz = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    value_range = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    z_surf = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    z_surf = z_surf.reshape(xx.shape)
    isPrint = z_surf == 1
    ax.scatter(train_data[:, 0], train_data[:, 1], train_data[:, 2], c='b', marker='^', label="src")
    ax.scatter(xx[isPrint], yy[isPrint], zz[isPrint], c='r', marker='o', label="bound")
    # 全部概率化
    ### 重构ans
    min = z_surf.min()
    ans = (clf.threshold_ - min) * prosibility + min
    clf.threshold_ = ans;
    # 保持模型
    joblib.dump(clf, os.path.join(model_output, "HSV_IF.model"))
    for item in test_data:
        ans = clf.predict(item[0])
        plt.figure(figsize=(6, 4))
        ans = ans.reshape(item[1][0],item[1][1])
        plt.imshow(ans, cmap = 'gray')
    plt.show()