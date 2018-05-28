import sys
sys.path.append('''.\lib''')
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
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn import cross_validation
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
train_data_dir = ["data/train_svm/positive", "data/train_svm/negative"]
model_dir = 'model/train_svm'
pic_dir = 'pic/train_svm'
data_dir = 'data/train_svm'

# 定义三个特征提取函数 输入参数都是灰度图 返回的都是特征

def before_do(file_name):
    mat = cv2.imread(file_name, 0)
    mat = cv2.resize(mat, (32, 48))
    return mat

def get_LBP(mat):
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(mat, n_points, radius, method='uniform')
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
    return  list_n_files, list_p_files

feacture_extra = {
     # "LBP":get_LBP,
    # "Gabor":get_Gabor,
      "HOG":get_hog
}
#

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
    plt.subplots_adjust(right=0.96, left=0.1, bottom=0.14, top=0.9, wspace=0.2, hspace=0.3)
    for i,(fun_name, fun) in enumerate(feacture_extra.items()):
        n_feature = [np.array(fun(item)) for item in n_mat]
        p_feature = [np.array(fun(item)) for item in p_mat]
        X = np.array(p_feature+n_feature)
        Y = np.concatenate([np.ones(len(p_feature)),np.zeros(len(n_feature))])
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
        print(X_train.shape)
        svm = svm.SVC(C=0.1, kernel='linear', probability=True,random_state=42)

        y_score = svm.fit(X_train, y_train).decision_function(X_test)

        # Compute ROC curve and ROC area for each class
        fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值

        plt.plot(fpr, tpr, color='darkorange',
                 lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
    plt.savefig(os.path.join(pic_dir,  "result.png"))
    plt.show()