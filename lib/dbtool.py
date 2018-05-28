#!/usr/bin/env python
# coding=utf-8
import numpy as np
import cv2
import math
import pylab as plt
from numpy import linalg as la
import  scipy
from scipy import signal

import numpy as np
import cv2
import math
import pylab as plt
from numpy import linalg as la
import  scipy
from scipy import signal
import os
# 代匹配的特征向量 维度为n*64
def get_feature(gray,step_size):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
                                     for x in range(0, gray.shape[1], step_size)]
    keypoints, descriptors = sift.compute(gray, kp)
    mydescriptors = []
    for item in descriptors:
        n_item = []
        for i in range(16):
            n_item.append((item[8 * i + 0] + item[8 * i + 4]))
            n_item.append((item[8 * i + 1] + item[8 * i + 5]))
            n_item.append((item[8 * i + 2] + item[8 * i + 6]))
            n_item.append((item[8 * i + 3] + item[8 * i + 7]))
        mydescriptors.append(n_item);
    class p:
        pass;
    p.x = math.ceil(gray.shape[0] /step_size);
    p.y = math.ceil(gray.shape[1] /step_size);
    p.descriptors = np.array(mydescriptors);
    return p;




## 定义差值距离
def diff_num(vec1, vec2):
    return np.linalg.norm(vec1 - vec2);

def cosSimilar(inA,inB):
    inA=np.mat(inA)
    inB=np.mat(inB)
    num=np.dot(inA,inB.T)/la.norm(inA)/la.norm(inB)
    value = np.trace(num)
    value = value*value;
    return value/(1-value)

def cosSimilar_2(inA,inB):
    inA=np.mat(inA)
    inB=np.mat(inB)
    num=np.dot(inA,inB.T)/la.norm(inA)/la.norm(inB)
    value = np.trace(num)
    return value
### 只需要2个图像
### in x1 y1 8 x2 y2 8
###
def cnn_conv(in_image, filter_map):
    shape_image=np.shape(in_image)#[row,col,8]
    shape_filter=np.shape(filter_map)#[row,col,8]
    if shape_filter[0]>shape_image[0] or shape_filter[1]>shape_image[1]:
        raise Exception
    shape_out=((shape_image[0]-shape_filter[0]+1),(shape_image[1]-shape_filter[1]+1));
    out_feature=np.zeros(shape_out)
    x,y=np.shape(out_feature)
    for x_idx in range(0,x):
        for y_idx in range(0,y):
            for w_idx in range(0,shape_filter[0]):
                for h_idx in range(0,shape_filter[1]):
                    out_feature[x_idx][y_idx]+=cosSimilar(in_image[w_idx+x_idx][h_idx+y_idx],filter_map[w_idx][h_idx])
            # print(out_feature[x_idx][y_idx])
            # print(x_idx,y_idx)
    return out_feature
def cnn_conv_new(in_image, filter_map):
    shape_image=np.shape(in_image)#[row,col,8]
    shape_filter=np.shape(filter_map)#[row,col,8]
    shape_target = (shape_filter[0]*shape_filter[1],shape_filter[2]);
    filter_map_row = np.reshape(filter_map,shape_target);
    if shape_filter[0]>shape_image[0] or shape_filter[1]>shape_image[1]:
        raise Exception
    shape_out=((shape_image[0]-shape_filter[0]+1),(shape_image[1]-shape_filter[1]+1));
    out_feature=np.zeros(shape_out)
    x,y=np.shape(out_feature)
    print(x*y)
    for x_idx in range(0,x):
        for y_idx in range(0,y):
            f_temp = in_image[x_idx:x_idx+shape_filter[0],y_idx:y_idx+shape_filter[1],:];
            out_feature[x_idx][y_idx]=cosSimilar(np.reshape(f_temp,shape_target),filter_map_row)
            # print(out_feature[x_idx][y_idx])
            # print(x_idx,y_idx)
    return out_feature

def cnn_conv_new_2(in_image, filter_map):
    shape_image=np.shape(in_image)#[row,col,8]
    shape_filter=np.shape(filter_map)#[row,col,8]
    shape_target = (shape_filter[0]*shape_filter[1],shape_filter[2]);
    filter_map_row = np.reshape(filter_map,shape_target);
    if shape_filter[0]>shape_image[0] or shape_filter[1]>shape_image[1]:
        raise Exception
    shape_out=((shape_image[0]-shape_filter[0]+1),(shape_image[1]-shape_filter[1]+1));
    out_feature=np.zeros(shape_out)
    x,y=np.shape(out_feature)
    for x_idx in range(0,x):
        for y_idx in range(0,y):
            f_temp = in_image[x_idx:x_idx+shape_filter[0],y_idx:y_idx+shape_filter[1],:];
            out_feature[x_idx][y_idx]=cosSimilar_2(np.reshape(f_temp,shape_target),filter_map_row)
            # print(out_feature[x_idx][y_idx])
            # print(x_idx,y_idx)
    return out_feature

def cnn_conv_ff(in_image, filter_map,basic):
    shape_image=np.shape(in_image)#[row,col,8]
    shape_filter=np.shape(filter_map)#[row,col,8]
    shape_target = (shape_filter[0]*shape_filter[1],shape_filter[2]);
    filter_map_row = np.reshape(filter_map,shape_target);
    if shape_filter[0]>shape_image[0] or shape_filter[1]>shape_image[1]:
        raise Exception
    shape_out=((shape_image[0]-shape_filter[0]+1),(shape_image[1]-shape_filter[1]+1));
    out_feature=np.zeros(shape_out)
    x,y=np.shape(out_feature)
    la_norm = la.norm(filter_map)  # FQ的F范数
    for x_idx in range(0,x):
        for y_idx in range(0,y):
            f_temp = in_image[x_idx:x_idx+shape_filter[0],y_idx:y_idx+shape_filter[1],:];
            out_feature[x_idx][y_idx]= basic/la.norm(f_temp)*la_norm
            # print(out_feature[x_idx][y_idx])
            # print(x_idx,y_idx)
    return out_feature



def write(img_bg ,img_object,x,y):
    cv2.rectangle(img_bg, (x, y), (x+img_object.shape[1], y + img_object.shape[0]), (0, 255, 0), 3)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_bg,cmap ='gray')
    plt.show()

def my_show(img_bg):
    plt.figure(figsize=(10, 10))
    plt.imshow(img_bg, cmap='gray')
    plt.show()


def drwabox(img_bg ,one_apoint, color,font):
    cv2.rectangle(img_bg, (one_apoint[0], one_apoint[1]), (one_apoint[2], one_apoint[3]), color, font)
    return img_bg;

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    # 初始化缩放比例，并获取图像尺寸
    dim = None
    (h, w) = image.shape[:2]

    # 如果宽度和高度均为0，则返回原图
    if width is None and height is None:
        return image



    # 宽度是0
    if width is None:
        # 则根据高度计算缩放比例
        height = int(height)
        r = height / float(h)
        dim = (int(w * r), height)

    # 如果高度为0
    else:
        # 根据宽度计算缩放比例

        width = int(width)
        r = width / float(w)
        dim = (width, int(h * r))

    # 缩放图像
    resized = cv2.resize(image, dim, interpolation=inter)

    # 返回缩放后的图像
    return resized
# 计算显著区域，提高搜索速度
def ComputeSaliencyMap():
    pass


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 从大到小排列，取index
    order = np.array([x for x in range(dets.shape[0]-1,-1,-1)])
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，之前没有被过滤掉，肯定是要保留的
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所以窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # ind为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        # 下一次计算前要把窗口i去除，所有i对应的在order里的位置是0，所以剩下的加1
        order = order[inds + 1]

    return keep

def get_value(maxtix,x1,y1,x2,y2):
   return  maxtix[x2][y2]+maxtix[x1-1][y1-1]-maxtix[x2][y1-1]-maxtix[x1-1][y2];



def calc_bg_norm(maxtix,out_shape,object_shape):
    sum,qsum= cv2.integral2(maxtix);
    if(len(qsum.shape)>=3):
        qsum = np.sum(qsum,axis=2);
    f_b = np.zeros(out_shape)
    for i in range(0, out_shape[0]):
        for j in range(0, out_shape[1]):
            f_b[i][j] = get_value(qsum, i + 1, j + 1, i + object_shape[0], j + object_shape[1])
    return f_b;

def fftconvle(feature_bg, feature_object):
    feature_object_r = feature_object[:, :, ::-1]
    conv = scipy.signal.convolve(feature_object_r, feature_bg, mode='valid', method='fft');
    conv = np.squeeze(np.square(conv));  ## 卷积层
    f_f = np.square(la.norm(feature_object));  ##直接求出范数
    f_b = calc_bg_norm(feature_bg, conv.shape, feature_object.shape)
    denominator = f_f * f_b
    dis = conv / denominator
    return dis

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

def normalization_n(arr, min, max):
    arr = arr.astype('float')
    arr = (arr-min)/(max-min)
    return arr

def normalization(selectPixel):
    H = selectPixel[:,0]
    H = H.astype('float')
    H = (H-90)/90
    S =  selectPixel[:,1]
    S = S.astype('float')
    S = (S-127)/127
    return np.column_stack((H, S))
def mypredict(clf, X, threshold):
    is_inlier = np.ones(X.shape[0], dtype=int)
    is_inlier[clf.decision_function(X) <= threshold] = 0
    return is_inlier
def extra(fileName, templet, num=16):
    imgsrc =  cv2.imread(fileName)
    imgsrc = resize(imgsrc, imgsrc.shape[1] / num)
    # equ_src = dbtool.hisEqulColor(imgsrc.copy())
    equ_src = imgsrc
    HSV_src = cv2.cvtColor(equ_src, cv2.COLOR_BGR2HSV)
    YUV_src = cv2.cvtColor(equ_src, cv2.COLOR_BGR2YCrCb)
    return equ_src.reshape(-1, 3)[templet != 0], HSV_src.reshape(-1, 3)[templet != 0], YUV_src.reshape(-1, 3)[templet != 0]

def load_mask(fileName,scaleNum):
    mask = cv2.imread(fileName,0)
    mask = resize(mask, mask.shape[1] / scaleNum)
    ret,thresh1=cv2.threshold(mask,80,255,cv2.THRESH_BINARY)
    thresh1 = (thresh1/255).astype('int').reshape(-1)
    return thresh1
# os.path.join(data_dir, list_img[0])
def load_data(fileName,scaleNum):
    imgsrc = cv2.imread(fileName)
    imgsrc = resize(imgsrc, imgsrc.shape[1] / scaleNum)
    HSV_src = cv2.cvtColor(imgsrc, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    YUV_src = cv2.cvtColor(imgsrc, cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    HSV_data = [normalization_n(HSV_src[:, 0], 0, 180),
                normalization_n(HSV_src[:, 1], 0, 255),
                normalization_n(HSV_src[:, 2], 0, 255)]
    HS_data = [normalization_n(HSV_src[:, 0], 0, 180),
               normalization_n(HSV_src[:, 1], 0, 255)]
    YUV_data = [normalization_n(YUV_src[:, 0], 0, 255),
                normalization_n(YUV_src[:, 1], 0, 255),
                normalization_n(YUV_src[:, 2], 0, 255)]
    UV_data = [normalization_n(YUV_src[:, 1], 0, 255),
               normalization_n(YUV_src[:, 2], 0, 255)]

    data_sourse = {"HSV": np.array(HSV_data).T, "HS": np.array(HS_data).T, "YUV": np.array(YUV_data).T,
                   "UV": np.array(UV_data).T}
    return  data_sourse

def save_img(dir, name, mat):
    cv2.imwrite(os.path.join(dir,name), mat)

def constuct(wsize, r):
    win = int((wsize - 1) / 2)
    crad = math.ceil(win - 0.5)
    [x, y] = np.meshgrid(np.arange(-crad, crad + 1), np.arange(-crad, crad + 1))
    maxxy = np.maximum(abs(x), abs(y))
    return maxxy <= r

def resturct_w(rang_size, win_size, ds):
    list = []
    rangenum = int((rang_size - 1) / 2)
    mask = constuct(win_size, rangenum)
    # des_bg.ds = des_bg.ds.reshape(des_bg.x,des_bg.y,win_size,win_size)
    tshape = ds.shape
    # print(tshape)
    # print(mask)
    # print(ds[mask][:,0,0].shape)
    for i in range(tshape[2]):
        for j in range(tshape[3]):
            temp = ds[mask][:,i,j].reshape(rang_size, rang_size)
            atemp = np.lib.pad(temp, ((i, tshape[2] - i - 1), (j, tshape[3] - j - 1)), 'constant', constant_values=0)
            atemp = atemp[rangenum:-rangenum, rangenum:-rangenum].reshape(-1)
            list.append(atemp)
    ans = np.column_stack(list)
    return ans