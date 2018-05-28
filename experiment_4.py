## 验证各种算法对闭合面的闭合情况，s
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
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
rng = np.random.RandomState(41)
outliers_fraction  = 0.1
classifiers = {
     "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05, kernel="rbf", gamma=0.1),
      "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
     "Isolation Forest": IsolationForest(n_estimators=4, max_samples=10000 ,n_jobs=4,contamination=outliers_fraction,random_state=rng),
     "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=outliers_fraction)
}
model_dir = 'model/experiment4'
pic_dir = 'pic/experiment4'
data_dir = 'data/experiment4'
list_img = ['src.jpg', 'light_bg.jpg', 'black_bg.jpg']
list_name = ['Nomal', 'light', 'black']

if __name__ =='__main__':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    img_bg =  cv2.imread(os.path.join(data_dir,"bg.jpg") ,0);
    img_bg = dbtool.resize(img_bg, img_bg.shape[1] / 4);
    mask = img_bg.reshape(-1);
    print('load mask ok!')
    list_resut = [ dbtool.extra(os.path.join(data_dir,fileName), mask, 4) for fileName in list_img]
    HSV_data = [dbtool.normalization_n(np.concatenate([item[1][:,0] for item in list_resut]),0,180),
                dbtool.normalization_n(np.concatenate([item[1][:,1] for item in list_resut]),0,255),
            dbtool.normalization_n(np.concatenate([item[1][:, 2] for item in list_resut]), 0, 255)]
    YUV_dat =  [dbtool.normalization_n(np.concatenate([item[2][:,0] for item in list_resut]),0,255),
            dbtool.normalization_n(np.concatenate([item[2][:,1] for item in list_resut]),0,255),
             dbtool.normalization_n(np.concatenate([item[2][:,2] for item in list_resut]),0,255)]
    data_source = { "HSV": HSV_data, "YUV": YUV_dat,}
    print('load data ok!')

    xx, yy, zz = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    for data_name, data in data_source.items():
        np.random.seed(42)
        X = np.array(data).T
        print(X.shape)
        plt.figure(figsize=(6, 4))
        plt.rcParams.update(params)
        plt.subplots_adjust(right=0.96, left=0.06, bottom=0.14, top=0.975, wspace=0.2, hspace=0.3)
        for i, (clf_name, clf) in enumerate(classifiers.items()):
            if clf_name == "Local Outlier Factor":
                y_pred = clf.fit_predict(X)
                scores_pred = clf.negative_outlier_factor_
            else:
                clf.fit(X)
                scores_pred = clf.decision_function(X)
                y_pred = clf.predict(X)
        #     保存模型
            joblib.dump(clf, os.path.join(model_dir, data_name + clf_name + ".model"))
            threshold = stats.scoreatpercentile(scores_pred, 5)
            if clf_name == "Local Outlier Factor":
                Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
                z_send = clf._predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
            else:
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
                z_send = clf.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
            z_send = z_send.reshape(xx.shape)
            subplot = plt.subplot(2, 2, i + 1, projection='3d')
            bol = z_send == 1;
            X_1,Y_1,Z_1 = xx[bol], yy[bol], zz[bol]
            subplot.scatter(X[:, 0], X[:, 1], X[:, 2], c='b', marker='^', label = "src")
            subplot.scatter(X_1, Y_1, Z_1, c='r', marker='o', label = "bound")
            subplot.set_zlabel(data_name[2])  # 坐标轴
            subplot.set_ylabel(data_name[1])
            subplot.set_xlabel(data_name[0])
            subplot.legend(loc='upper right')
            subplot.set_xlabel("%d. %s  Areasum: %f" % (i + 1, clf_name, (np.sum(z_send) + 27000) / 54000))
        plt.suptitle(data_name)
        plt.savefig(os.path.join(pic_dir, data_name+".png"))
    plt.show()
    print('print ok!')

# a -b =x
# a+b = 10000
# a = x+10000/200000
