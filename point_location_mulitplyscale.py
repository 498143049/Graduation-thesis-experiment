import cv2
import os
import numpy as np
import lark
import math
import sys
sys.path.append('''F:\python_learning\pythonMylib''')
import dbtool
from lpproj import LocalityPreservingProjection
from sklearn.decomposition import PCA
import logging
import time
########### 配置参数 #################
step_size = 1
win_size = 11
max_components = 2
t = 0.5
alpha = 0.996
max_rectangle_num = 10
set_width_size = 547/3
set_width_size_1 = 123/3
reducer = PCA(n_components=max_components)
########### 配置的参数 #############################

##################配置logging####################################
logger = logging.getLogger("multiply_detection")
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
# formatter = logging.Formatter("%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s")
file_handler = logging.FileHandler("log/multiply_detection_"+time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime())+".log")
file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
file_handler.setLevel(logging.DEBUG)
# 控制台日志
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter  # 也可以直接给formatter赋值
console_handler.setLevel(logging.DEBUG)
# 为logger添加的日志处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# 指定日志的最低输出级别，默认为WARN级别
logger.setLevel(logging.DEBUG)
########### 输入参数为模板的地址，以及图片的地址########
def get_feature_bylark(gray, step_size, window_size=5):
    ds, unlark, LARK = lark._raw_larks(gray, wsize=window_size, interval=step_size);
    class p:
        pass
    p.x = math.ceil(gray.shape[0] / step_size)
    p.y = math.ceil(gray.shape[1] / step_size)
    x, y, z = LARK.T.shape
    transe_LARK = LARK.T.reshape(x * y, z, order='F').copy()
    p.descriptors = transe_LARK
    p.ds = ds
    p.unlark = unlark
    return p


def Get_location_postion(template_source, background_source):
    # 计算缩放的倍数
    zoom_scale = background_source.shape[1] / set_width_size
    background_zoom = dbtool.resize(background_source, width=set_width_size)
    template_zoom = dbtool.resize(template_source, width=set_width_size_1)

    ## 多尺度设定
    max_scale_1  =  template_zoom.shape[0]/(background_zoom.shape[0] * 0.8)
    max_scale_2  =  template_zoom.shape[1]/(background_zoom.shape[1] * 0.8)
    max_scale = max(max_scale_1, max_scale_2)
    min_scale_1 =  template_zoom.shape[0]/ (background_zoom.shape[0] * 0.1)
    min_scale_2 =  template_zoom.shape[1] / (background_zoom.shape[1] * 0.1)
    min_scale = min(min_scale_1, min_scale_2)
    range_sizes = [i/10 for i in range(math.ceil(max_scale*10), math.floor(min_scale*10)+1,1)]
    # print(range_sizes)
    background_gary = cv2.cvtColor(background_zoom, cv2.COLOR_BGR2GRAY)
    template_gary = cv2.cvtColor(template_zoom, cv2.COLOR_BGR2GRAY)
    # 提取LARK特征
    background_lark = get_feature_bylark(background_gary, step_size, win_size)
    template_lark = get_feature_bylark(template_gary, step_size, win_size)
    # 降维
    reducer.fit(template_lark.descriptors)
    background_reduce_lark = reducer.transform(background_lark.descriptors).reshape([background_lark.x,background_lark.y, -1])
    template_reduce_lark = reducer.transform(template_lark.descriptors).reshape([template_lark.x, template_lark.y, -1])
    # 卷积 输出概率图 多尺度的问题
    # 这里进行多尺度缩放
    background_reduce_lark_list  = [dbtool.resize(background_reduce_lark, width=background_reduce_lark.shape[1]*itemnum) for itemnum in range_sizes]
    dis_list = [dbtool.fftconvle(inlarks, template_reduce_lark) for inlarks in background_reduce_lark_list]
    dis_list_probality = [item/(1- item) for item in dis_list]

    # dis_list_probality_resize = [dbtool.resize(dis_list_probality_item, width=dis_list_probality_item.shape[1]*range_item) for (dis_list_probality_item, range_item) in zip(dis_list_probality, range_sizes)]
    # 找到最大的值所对应的位置，以及所对应的矩形
    # TODO 列表的长度处理
    apoint = []
    retangle = []
    for item, sizes in zip(dis_list_probality, range_sizes):
        temp_v = find_area(item,(template_gary.shape[0], template_gary.shape[1]),step_size,zoom_scale/sizes)
        if (temp_v != None):
            apoint.extend(temp_v)
    if (len(apoint) != 0):
        ndarray_apoint = np.array(apoint)
        ndarray_apoint = ndarray_apoint[np.lexsort(-ndarray_apoint.T)]
        retangle = dbtool.py_cpu_nms(ndarray_apoint, 0.1)
    return retangle, apoint

def find_area(probability_map, box_shape, step_size, zoom_scale):
    max_value = np.max(probability_map)
    logger.debug('max_value is %s', np.max(max_value))
    if (max_value < t) :
        return None
    current_retangle_num = math.ceil(probability_map.shape[0] * probability_map.shape[1] * (1 - alpha))
    current_retangle_num = min(current_retangle_num, max_rectangle_num)
    dis_one = np.reshape(probability_map, -1)
    result = dis_one.argsort()[-current_retangle_num:]
    apoint = []
    for i in result:
        x_i = int(i % probability_map.shape[1]) * step_size
        y_i = int(i / probability_map.shape[1]) * step_size
        apoint.append(list(map(lambda x: int(x * zoom_scale),
                               [x_i, y_i, (x_i + box_shape[1]),
                                (y_i + box_shape[0]), dis_one[i]])))
    return apoint

if __name__ =='__main__':
    data_dir = 'data/point_instrument'
    # python 找到文件夹下面的文件夹
    for filedir in os.listdir(data_dir):
        template_url = os.path.join(data_dir, filedir, 'template/template.jpg')
        if os.path.isdir(os.path.join(data_dir, filedir)):
            for pic in os.listdir(os.path.join(data_dir, filedir, 'source')):
                p = os.path.join(data_dir, filedir, 'source', pic)
                if os.path.isfile(p):
                    background_url = p
                    template_source = cv2.imread(template_url)
                    background_source = cv2.imread(background_url)
                    logger.info("===========start==============")
                    logger.info("template url: %s, currrent url: %s", template_url, background_url)
                    start = time.clock()
                    result, apoint = Get_location_postion(template_source, background_source)
                    end = time.clock()
                    logger.debug("location_cost: %s", end-start)
                    logger.debug(len(result))
                    if (len(apoint)!=0):
                        # for i in range(0, len(apoint)):  #这里是输出测试非极大值抑制
                        for i in result:
                            dbtool.drwabox(background_source, apoint[i], (255, 255, 255), 10)
                        logger.info("Find %d object", len(result))
                    else :
                        logger.info("Not find object")
                    cv2.imwrite(os.path.join(data_dir, filedir, 'result', pic), background_source)