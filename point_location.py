import cv2
import os
import numpy as np
import lark
import math
import sys
sys.path.append('''.\lib''')
import dbtool
from sklearn.decomposition import PCA

########### 配置参数 #################
zoom_size = 20
step_size = 3
win_size = 11
range_size = 3
max_components = 4
t = 0.96
alpha = 0.99
max_rectangle_num = 100
set_width_size =  547
reducer = PCA(n_components=max_components)
range_sizes = [1]
########### 配置的参数 #############################
########### 输入参数为模板的地址，以及图片的地址########
def get_feature_bylark(gray, step_size, window_size=5):
    ds, unlark, LARK = lark._raw_larks(gray, wsize=window_size, interval=step_size);

    class p:
        pass

    p.x = math.ceil(gray.shape[0] / step_size);
    p.y = math.ceil(gray.shape[1] / step_size);
    x, y, z = LARK.T.shape
    transe_LARK = LARK.T.reshape(x * y, z, order='F').copy()
    p.descriptors = transe_LARK
    p.ds = ds
    p.unlark = unlark
    return p


def extra_lark(fileName):
    img_object = cv2.imread(fileName)
    img_object = dbtool.resize(img_object, width=img_object.shape[1] / zoom_size)
    gray_object = cv2.cvtColor(img_object, cv2.COLOR_BGR2GRAY)
    des_object = get_feature_bylark(gray_object, step_size, win_size)
    return img_object, gray_object, des_object

def Get_location_postion(template_source, background_source):

    # 计算缩放的倍数
    zoom_scale = background_source.shape[1]/set_width_size
    print(zoom_scale)
    background_zoom = dbtool.resize(background_source, width=set_width_size)
    template_zoom = dbtool.resize(template_source, width=template_source.shape[1]/(zoom_scale*2))
    background_gary = cv2.cvtColor(background_zoom, cv2.COLOR_BGR2GRAY)
    template_gary = cv2.cvtColor(template_zoom, cv2.COLOR_BGR2GRAY)
    # 提取LARK特征
    print(template_gary.shape)
    background_lark = get_feature_bylark(background_gary, step_size, win_size)
    template_lark = get_feature_bylark(template_gary, step_size, win_size)
    # 降维
    reducer.fit(template_lark.descriptors)
    background_reduce_lark = reducer.transform(background_lark.descriptors).reshape([background_lark.x,background_lark.y, -1])
    template_reduce_lark = reducer.transform(template_lark.descriptors).reshape([template_lark.x, template_lark.y, -1])
    background_reduce_lark = dbtool.resize(background_reduce_lark, width=background_reduce_lark.shape[1] * 0.5)
    # 卷积 输出概率图 多尺度的问题
    # 这里进行多尺度缩放
    dis = dbtool.fftconvle(background_reduce_lark, template_reduce_lark)

    print(dis.shape)
    dis = dis / (1 - dis)
    # 分析概率图
    current_retangle_num  = math.ceil(dis.shape[0] * dis.shape[1] * (1 - alpha))
    current_retangle_num = min(current_retangle_num, max_rectangle_num)
    print(current_retangle_num)
    dis_one = np.reshape(dis, dis.shape[0] * dis.shape[1])
    result = dis_one.argsort()[-current_retangle_num:]
    apoint = []

    for i in result:
        x_i = int(i % dis.shape[1]) * step_size
        y_i = int(i / dis.shape[1]) * step_size
        apoint.append(list(map(lambda x: int(x*2*zoom_scale), [x_i, y_i, (x_i + template_gary.shape[1]), (y_i + template_gary.shape[0])])))
    print(len(apoint))
    retangle = dbtool.py_cpu_nms(np.array(apoint), 0.4)
    ### 分析图
    return retangle, apoint



if __name__ =='__main__':
    # 读取文件
    template_url = 'data/experiment9/b.jpg'
    background_url = 'data/experiment9/2.jpg'
    template_source =  cv2.imread(template_url)
    background_source = cv2.imread(background_url)
    result, apoint  = Get_location_postion(template_source, background_source)
    # for i in range(0,len(apoint)):
    #     dbtool.drwabox(background_source, apoint[i], (0, 255, 255), 5)
    for i in result:
        dbtool.drwabox(background_source, apoint[i], (0, 255, 0), 10)
    cv2.imwrite("xxx.jpg", background_source)