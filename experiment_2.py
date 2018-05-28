## d读取数据验证 RGB hsv YIQ空间的泛化能力
import sys
sys.path.append('''.\lib''')
import dbtool
import cv2
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
import os
def extra(fileName, templet):
    imgsrc =  cv2.imread(fileName);
    imgsrc = dbtool.resize(imgsrc, imgsrc.shape[1] / 16);
    # equ_src = dbtool.hisEqulColor(imgsrc.copy())
    equ_src = imgsrc
    HSV_src = cv2.cvtColor(equ_src, cv2.COLOR_BGR2HSV)
    YUV_src = cv2.cvtColor(equ_src, cv2.COLOR_BGR2YCrCb)
    return equ_src.reshape(-1, 3)[templet != 0], HSV_src.reshape(-1, 3)[templet != 0], YUV_src.reshape(-1, 3)[templet != 0]

def extra_bg(fileName):
    img_bg = cv2.imread(fileName, 0);
    img_bg = dbtool.resize(img_bg, img_bg.shape[1] / 16);
    mask = img_bg.reshape(-1);
    return mask

list_img = [['data/experiment2/src.jpg','data/experiment2/bg.jpg'],[ 'data/experiment2/src2.jpg','data/experiment2/bg2.jpg']]
list_title = ["HS","UV"]

list_color = ['r', 'g', 'b']
list_name = ['Nomal', 'light', 'black']
list_resutl = [ extra(fileName[0], extra_bg(fileName[1])) for fileName in list_img]

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
model_dir = 'model/experiment2'
pic_dir = 'pic/experiment2'
data_dir = 'data/experiment2'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)


##开始画图
plt.figure(figsize=(6, 2.8))
plt.rcParams.update(params)
ax = plt.subplot(121);
ax.set_title("HS")
x1 = ax.scatter(list_resutl[0][1][:,0], list_resutl[0][1][:,1], c='r',label = "src",s=25,alpha=0.4,marker='o')  # 绘制数据点
x2 = ax.scatter(list_resutl[1][1][:,0], list_resutl[1][1][:,1], c='b',label = "src1",s=25,alpha=0.4,marker='o')  # 绘制数据点
ax.set_xlim(0,255)
ax.set_ylim(0,255)
ax.set_xlabel('H')
ax.set_ylabel('S')
ax.legend(["src","src1"])
ax = plt.subplot(122);
ax.set_title("UV")
ax.scatter(list_resutl[0][2][:,1], list_resutl[0][2][:,2], c='r',label = "src",s=25,alpha=0.4,marker='o')  # 绘制数据点
ax.scatter(list_resutl[1][2][:,1], list_resutl[1][2][:,2], c='b',label = "src1",s=25,alpha=0.4,marker='o')  # 绘制数据点
ax.set_xlim(0,255)
ax.set_ylim(0,255)
ax.set_xlabel('U')
ax.set_ylabel('V')
ax.legend(["src","src1"])
plt.subplots_adjust(right=0.9, left=0.1, bottom=0.15, top=0.9, wspace=0.2, hspace=0.3)
plt.savefig(os.path.join(pic_dir, "result2.png"))
plt.savefig(os.path.join(pic_dir, "result2.png"))
plt.show()

