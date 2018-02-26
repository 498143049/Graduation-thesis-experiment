import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale
import sys
sys.path.append('''F:\python_learning\pythonMylib''')
import dbtool
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import  numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.externals import joblib
from matplotlib import rc
from skimage import feature
import time
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
model_dir = 'model/experiment7'
pic_dir = 'pic/experiment7'
data_dir = 'data/experiment7'
txt_data = 'correct.txt'

def deal_randon(mat):
    canny = feature.canny(mat, sigma=3)
    point = 50;
    theta = np.linspace(-25., 0., point, endpoint=False)
    sinogram = radon(canny, theta=theta, circle=True)
    maxR = np.max(sinogram,axis=0)
    return canny, maxR, 90+np.argmax(maxR)*25.0/point-25
if __name__ =='__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)



    # 测试全部
    allError = 0
    sum = 0;
    with open(os.path.join(data_dir, txt_data)) as myfile:
        for i, item in enumerate(myfile.readlines()):
            image = imread(os.path.join(data_dir,"test_correct"+"_"+str(i)+".jpg"), as_grey=True)
            image = dbtool.resize(image, 50);

            # image = rescale(image, scale=1, mode='reflect')
            # if(image.shape[1]>50):
            #     image = dbtool.resize(image, 50);
            start = time.clock()
            canny,sinogram,ans2 = deal_randon(image)
            end = time.clock()
            terror = abs(ans2 - float(item))
            allError+=terror
            stime = (end-start)*1000
            sum+=stime
            print("the %d differance %f %f %f ;cost time: %f;" %(i, ans2, float(item),terror, stime ))
        allError = allError/20
    print("all allError is %f cost %f" %(allError, sum/20))
    tem_num = 1
    image = imread(os.path.join(data_dir,"test_correct"+"_"+str(tem_num)+".jpg"), as_grey=True)
    print(image.shape)
    # shape[0] height shape[1] width
    # image = rescale(image, scale=1, mode='reflect')
    image = dbtool.resize(image, 40);
    canny,max_R,ans2 = deal_randon(image)
    plt.plot(max_R)
    print(ans2)
    plt.show()
