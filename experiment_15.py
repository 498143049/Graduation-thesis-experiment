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
sys.path.append('''.\lib''')
import dbtool
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
model_dir = 'model/experiment15'
pic_dir = 'pic/experiment15'
data_dir = 'data/experiment13'
feature_map={
    "SIFT":cv2.xfeatures2d.SIFT_create(),
    "SURF":cv2.xfeatures2d.SURF_create(),
    "ORB 5000":cv2.ORB_create(5000),
    "ORB 10000": cv2.ORB_create(10000),
    "AKAZE": cv2.AKAZE_create()
}

def get_std():
    pts2 = np.float32([[174, 369], [156, 208], [454, 142], [529, 301]])
    pts1 = np.float32([[143, 444], [118, 272], [449, 164], [540, 326]])
    M = cv2.getPerspectiveTransform(pts2, pts1)
    return M
def distance(A, B):
    mse = ((A - B) ** 2).mean(axis=None)
    return mse

def kaze_match(im1_path, im2_path):
    img1_src = cv2.imread(im1_path)  #模板图像
    img2_src = cv2.imread(im2_path)  #需要变换的图像
    img1 = cv2.cvtColor(img1_src, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_src, cv2.COLOR_BGR2GRAY)
    stdvalue = get_std()
    for feature_name, feature_instace in  feature_map.items():
        start = time.clock()
        kp1, des1 = feature_instace.detectAndCompute(img1, None)
        kp2, des2 = feature_instace.detectAndCompute(img2, None)
        if feature_name=='SIFT' or  feature_name=='SURF':
            bf = cv2.BFMatcher()
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        matches = bf.match(des1, des2)
        # 基于网格估计的过滤
        if feature_name == 'SIFT' or feature_name == 'SURF':
            good = matches
        else :
            good = cv2.xfeatures2d.matchGMS(img1.shape, img2.shape, kp1, kp2, matches)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 2)
        elapsed = (time.clock() - start)
        outline = np.sum(mask == 0)
        inline  = np.sum(mask == 1)
        print("%s :inline count %d, outline count %d, InLineRate%f cost time%f, MSE:%f"%(feature_name, inline, outline,inline/(inline+outline), elapsed, distance(M, stdvalue)))
import time
if __name__ =='__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    start = time.clock()
    kaze_match(os.path.join(data_dir,'a.jpg'),os.path.join(data_dir,'b.jpg'))
    end = time.clock()
    print( end-start)