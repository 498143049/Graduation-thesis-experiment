
## d读取数据验证 RGB hsv YIQ空间的稳定
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


list_img = ['data/experiment1/src.jpg', 'data/experiment1/light_bg.jpg', 'data/experiment1/black_bg.jpg']
list_color = ['r', 'g', 'b']
list_name = ['Nomal', 'light', 'black']
list_title = ["BGR","HSV","YUV"]
img_bg =  cv2.imread("data/experiment1/bg.jpg",0);
img_bg = dbtool.resize(img_bg, img_bg.shape[1] / 16);
mask = img_bg.reshape(-1);
list_resutl = [ extra(fileName, mask) for fileName in list_img]
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
model_dir = 'model/experiment1'
pic_dir = 'pic/experiment1'
data_dir = 'data/experiment1'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)


##开始画图
plt.figure(figsize=(6, 4))
plt.rcParams.update(params)
plt.grid(True)
ax = plt.subplot(221, projection='3d');
ax.set_title("Gather")
ax.scatter(list_resutl[0][0][:,0], list_resutl[0][0][:,1], list_resutl[0][0][:,2], c='r',label = "RGB",s=10,alpha=0.4,marker='o')  # 绘制数据点
ax.scatter(list_resutl[0][1][:,0], list_resutl[0][1][:,1], list_resutl[0][1][:,2], c='g',label = "HSV",s=10,alpha=0.4,marker='o')  # 绘制数据点
ax.scatter(list_resutl[0][2][:,0], list_resutl[0][2][:,1], list_resutl[0][2][:,2], c='b',label = "YUV",s=2,alpha=0.4,marker='o')  # 绘制数据点
ax.set_zlim3d(0,255)
ax.set_ylim3d(0,255)
ax.set_xlim3d(0,255)
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.legend(loc=1) # 展示图例

#
for j in range(0, 3):
    ax = plt.subplot(2,2,j+2, projection='3d');
    ax.set_title(list_title[j])
    ax.set_zlim3d(0, 255)
    ax.set_ylim3d(0, 255)
    ax.set_xlim3d(0, 255)
    ax.set_zlabel(list_title[j][2])  # 坐标轴
    ax.set_ylabel(list_title[j][1])
    ax.set_xlabel(list_title[j][0])
    for i,item in enumerate(list_resutl):
        ax.scatter(item[j][:, 0], item[j][:, 1], item[j][:, 2], c=list_color[i], label=list_name[i], s=10,alpha=0.4,marker='o')  # 绘制数据点
    ax.legend(loc=1)  # 展示图例

plt.subplots_adjust(right=0.9, left=0.02, bottom=0.1, top=0.95, wspace=0.2, hspace=0.3)
plt.savefig(os.path.join(pic_dir, "result2.png"))
plt.show()



# EQU_src, HSV_src, YUV_src = extra('data/experiment1/src.jpg', mask)
# a1,b1,c1 = extra('data/experiment1/light_bg.jpg', mask)
# a2,b2,c2 = extra('data/experiment1/black_bg.jpg', mask)
# # x, y, z = a[:,0], a[:,1], a[:,2]
# # x1,y1,z1 = a1[:,0], a1[:,1], a1[:,2]
# # x2,y2,z2 = a2[:,0], a2[:,1], a2[:,2]
#
# x, y, z = b[:,0], b[:,1], b[:,2]
# x1,y1,z1 = b1[:,0], b1[:,1], b1[:,2]
# x2,y2,z2 = b2[:,0], b2[:,1], b2[:,2]
#
# # x, y, z = c[:,0], c[:,1], c[:,2]
# # x1,y1,z1 = c1[:,0], c1[:,1], c1[:,2]
# # x2,y2,z2 = c2[:,0], c2[:,1], c2[:,2]
# #  将数据点分成三部分画，在颜色上有区分度
# ax = plt.subplot(221, projection='3d')  # 创建一个三维的绘图工程
# #  将数据点分成三部分画，在颜色上有区分度
# ax.scatter(x, y, z, c='r')  # 绘制数据点
# ax.scatter(x1, y1, z1, c='g')  # 绘制数据点
# ax.scatter(x2, y2, z2, c='b')  # 绘制数据点
# ax.set_zlabel('Z')  # 坐标轴
# ax.set_ylabel('Y')
# ax.set_xlabel('X')
# plt.show()
# plt.figure(figsize=(4, 7))
# plt.imshow(img_bg,cmap='gray')
# plt.show()



