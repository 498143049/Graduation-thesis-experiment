import sys
sys.path.append('''F:\python_learning\pythonMylib''')
import dbtool
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import  numpy as np
import matplotlib.font_manager
from scipy import stats
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.externals import joblib
from matplotlib import rc
import gabor
import cv2
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from skimage.feature import hog
from sklearn.decomposition import *
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
train_data_dir = ["data/train_svm/positive", "data/train_svm/negative"]
model_dir = 'model/experiment6'
pic_dir = 'pic/experiment6'
data_dir = 'data/experiment6'

# 定义三个特征提取函数 输入参数都是灰度图 返回的都是特征

def before_do(file_name):
    mat = cv2.imread(file_name, 0)
    mat = cv2.resize(mat, (32, 48))
    return mat

def get_LBP(mat):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(mat, n_points, radius, 'uniform')
    max_bins = int(lbp.max() + 1);
    feature_set, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
    return feature_set

filters = gabor.build_filters()
def get_Gabor(mat):
    feature_set = gabor.get_image_feature_vector(mat, filters, None)
    return feature_set

def get_hog(mat):
    result = hog(mat, 8, (4, 4), (2, 3) , block_norm='L2', visualise=False, transform_sqrt=True)
    return result

def load_data():
    list_p_files = [os.path.join(train_data_dir[0], p) for p in os.listdir(train_data_dir[0])]
    list_n_files = [os.path.join(train_data_dir[1], p) for p in os.listdir(train_data_dir[1])]
    return  list_p_files, list_n_files

feacture_extra = {
    "LBP":get_LBP,
    "Gabor":get_Gabor,
    "HOG":get_hog
}


if __name__ =='__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    n_file, p_file = load_data()
    n_mat = list(map(before_do, n_file))
    p_mat = list(map(before_do, p_file))
    plt.figure(figsize=(6, 4))
    plt.rcParams.update(params)
    plt.subplots_adjust(right=0.96, left=0.06, bottom=0.14, top=0.9, wspace=0.2, hspace=0.3)
    for i,(fun_name, fun) in enumerate(feacture_extra.items()):
        n_feature = [np.array(fun(item)) for item in n_mat]
        p_feature = [np.array(fun(item)) for item in p_mat]
        print(n_feature[0].shape)
        size_n = len(n_feature)
        all_feature = n_feature+p_feature
        pca =  TSNE(n_components=2)
        all_feature = np.array(all_feature)
        reduced_data_pca = pca.fit_transform(all_feature)
        colors = ['black', 'blue']
        subplot = plt.subplot(2, 2, i + 1)
        x = reduced_data_pca[:size_n, 0]
        y = reduced_data_pca[:size_n, 1]
        a = subplot.scatter(x, y,  c='blue', s=5)
        x1 = reduced_data_pca[size_n:, 0]
        y1 = reduced_data_pca[size_n:, 1]
        b = subplot.scatter(x1, y1,  c='red', s=5)
        subplot.legend([a, b], ['negative', 'positive'], prop=matplotlib.font_manager.FontProperties(size=10), loc='upper right')
        subplot.set_xlabel("%d. %s" % (i + 1, fun_name))
    plt.suptitle("tsne Analyse")
    plt.savefig(os.path.join(pic_dir,  "result.png"))
    plt.show()
