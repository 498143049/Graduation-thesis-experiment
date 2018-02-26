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
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
model_dir = 'model/experiment14'
pic_dir = 'pic/experiment14'
data_dir = 'data/experiment14'
zoom_size = 8
step_size = 1
win_size = 15
range_size = 3
max_components = 2
t = 0.96
alpha = 0.999
fitem = ["a.jpg","b.jpg","c.jpg","d.jpg","f.jpg","g.jpg","h.jpg","j.jpg","k.jpg"]
def show_this(lks, wsize_skr):
    _, M, N = lks.shape
    out_larks = np.zeros([M * wsize_skr, N * wsize_skr], dtype='float')
    for i in range(M):
        for j in range(N):
            out_larks[wsize_skr * i:(wsize_skr * i + wsize_skr), wsize_skr * j:(wsize_skr * j + wsize_skr)] = lks[:, i,j].reshape(
                wsize_skr, wsize_skr)
    return out_larks
def get_feature_bylark(gray,step_size,window_size=5):
    ds, unlark, LARK = lark._raw_larks(gray,wsize=window_size,interval=step_size);
    class p:
        pass
    p.x = math.ceil(gray.shape[0] /step_size);
    p.y = math.ceil(gray.shape[1] /step_size);
    x, y, z = LARK.T.shape
    transe_LARK =  LARK.T.reshape(x * y, z, order='F').copy()
    LARK = show_this(LARK, win_size)
    return LARK, transe_LARK
def extra_pack(fileName):
    img_object = cv2.imread(os.path.join(data_dir, fileName))
    img_src = cv2.cvtColor(img_object, cv2.COLOR_BGR2RGB)
    img_object = cv2.resize(img_object,(371,347),interpolation=cv2.INTER_CUBIC)
    img_object = dbtool.resize(img_object, width=img_object.shape[1] / zoom_size);
    gray_object = cv2.cvtColor(img_object, cv2.COLOR_BGR2GRAY)
    LARK, transe_LARK = get_feature_bylark(gray_object, step_size, win_size)
    return  LARK, transe_LARK,img_src
def distance(A, B):
    mse = ((A - B) ** 2).mean(axis=None)
    return mse
if __name__ =='__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    feature_list =  [extra_pack(filename)  for  filename in fitem]
    plt.figure(figsize=(6.3, 6.7))
    value = distance(feature_list[0][1], feature_list[0][1]) * 100000
    plt.subplot(3, 3, 1).set_title("MES: %0.6f * 10^5" % value)
    plt.subplot(3, 3, 1).imshow(feature_list[0][0],  cmap='jet')
    plt.axis('off')
    for i in range(1,len(feature_list)):
        value = distance(feature_list[0][1], feature_list[i][1]) * 100000
        plt.subplot(3, 3, i+1).imshow( feature_list[i][0],  cmap='jet')
        plt.subplot(3, 3, i + 1).set_title("MES: %0.6f * 10^5"%value)
        plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 0.98, 0.02, 0.02)
    plt.savefig(os.path.join(pic_dir, "result.png"))
    plt.figure(figsize=(6.3, 6))
    for i in range(0, len(feature_list)):
        plt.subplot(3, 3, i + 1).imshow(feature_list[i][2], cmap='jet')
        plt.axis('off')
    plt.subplots_adjust(0, 0, 1, 0.98, 0.02, 0.02)
    plt.savefig(os.path.join(pic_dir, "result2.png"))
    plt.show()