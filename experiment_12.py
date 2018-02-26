import numpy as np
import scipy
import time
from scipy import signal
object_shpae = ((32,32),(64,64),(128,128))
bg_shpae = ((354,243),(547,364),(912,608))
conv_fun = {
    "FFT":lambda  feature_object_r,feature_bg :scipy.signal.fftconvolve(feature_object_r, feature_bg, mode='valid'),
    "Direct":lambda  feature_object_r,feature_bg :scipy.signal.convolve(feature_object_r, feature_bg, mode='valid',method='auto')
}
if __name__ =='__main__':
    for data_name, conv in conv_fun.items():
        for object_item in object_shpae:
            object_ones = np.ones(object_item,'float')
            for bg_item in bg_shpae:
                bg_ones = np.ones(bg_item, 'float')
                start = time.clock()
                conv(object_ones,bg_ones)
                elapsed = (time.clock() - start)
                print("use %s in (%d,%d) object shape and (%d,%d) bg shape cost %f"%
                      (data_name, object_item[0], object_item[1], bg_item[0], bg_item[1], elapsed))