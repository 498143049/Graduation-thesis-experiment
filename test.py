import cv2
import os
import  numpy as np
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
model_dir = 'model/experiment11'
pic_dir = 'pic/experiment11'
data_dir = 'data/experiment9'

def kaze_match(im1_path, im2_path):
    img1_src = cv2.imread(im1_path)  #模板图像
    img2_src = cv2.imread(im2_path)  #需要变换的图像
    img1 = cv2.cvtColor(img1_src, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_src, cv2.COLOR_BGR2GRAY)
    img1_src = cv2.cvtColor(img1_src, cv2.COLOR_BGR2RGB)
    img2_src = cv2.cvtColor(img2_src, cv2.COLOR_BGR2RGB)

    # Initiate SIFT detector ORB
    sift = cv2.AKAZE_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf =  cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    # 基于网格估计的过滤
    good = cv2.xfeatures2d.matchGMS(img1.shape, img2.shape,kp1,kp2,matches)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)
    matchesMask = mask.ravel().tolist()
    result = cv2.warpPerspective(img2_src, M, (img1_src.shape[1],img1_src.shape[0]))

    draw_params = dict(matchColor=(255, 0, 0), singlePointColor=None, matchesMask=None, flags=2)
    draw_params2 = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=None, flags=2)
    draw_params3 = dict(matchColor=(0, 0, 255), singlePointColor=None, matchesMask=matchesMask, flags=2)
    img3 = cv2.drawMatches(img1_src, kp1, img2_src, kp2, matches, None, **draw_params)
    img4 = cv2.drawMatches(img1_src, kp1, img2_src, kp2, good, None, **draw_params2)
    img5 = cv2.drawMatches(img1_src, kp1, img2_src, kp2, good, None, **draw_params3)
    cv2.imwrite()
    plt.show()
if __name__ =='__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    kaze_match(os.path.join(data_dir,'a.jpg'),os.path.join(data_dir,'b.jpg'))

