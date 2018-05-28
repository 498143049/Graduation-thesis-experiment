import sys
sys.path.append('''.\lib''')
import dbtool
import cv2
import os
import matplotlib.pyplot as plt
import  numpy as np
from sklearn.externals import joblib
from matplotlib import rc
from skimage import feature
import time
import lark
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
model_dir = 'model/experiment8'
pic_dir = 'pic/experiment8'
data_dir = 'data/experiment8'
testpic = '2.jpg'

def show_this(lks, wsize_skr):
    _, M, N = lks.shape
    out_larks = np.zeros([M * wsize_skr, N * wsize_skr], dtype='float')
    for i in range(M):
        for j in range(N):
            out_larks[wsize_skr * i:(wsize_skr * i + wsize_skr), wsize_skr * j:(wsize_skr * j + wsize_skr)] = lks[:, i,j].reshape(
                wsize_skr, wsize_skr)
    return out_larks

if __name__ =='__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    img = cv2.imread(os.path.join(data_dir,testpic))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    winsize = 5;
    ds, unlark, LARK = lark._raw_larks(img_gray, wsize=winsize, interval=5)
    ds = ds.reshape(LARK.shape)
    ds = show_this(ds, winsize)
    unlark = unlark.reshape(LARK.shape)
    unlark = show_this(unlark, winsize)
    LARK = show_this(LARK, winsize)
    # print(LARK)
    plt.figure(figsize=(6.3, 4.7))
    plt.subplot(2,2,1).imshow(img_rgb)
    plt.subplot(2, 2, 1).set_title("SRC")
    plt.axis('off')
    plt.subplot(2, 2, 2).imshow(ds)
    plt.axis('off')
    plt.subplot(2, 2, 2).set_title("Geodesic Distance")
    plt.subplot(2,2,3).imshow(unlark)
    plt.axis('off')
    plt.subplot(2, 2, 3).set_title("LARKS")
    plt.subplot(2, 2, 4).imshow(LARK)
    plt.axis('off')
    plt.subplot(2, 2, 4).set_title("Normalize LARKS")
    plt.rcParams.update(params)
    plt.subplots_adjust(0.02, 0.02, 0.96, 0.9,0,0.2)
    plt.suptitle('LARK Features show')
    plt.savefig(os.path.join(pic_dir, "result.png"))
    plt.show()