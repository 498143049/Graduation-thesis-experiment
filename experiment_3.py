## 验证各种算法对闭合面的闭合情况，总共有2种算法
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
from sklearn.externals import joblib
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial'], 'size': 8})
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
rng = np.random.RandomState(41)
outliers_fraction  = 0.1
classifiers = {
     "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05, kernel="rbf",gamma=0.1),
      "Robust covariance": EllipticEnvelope(contamination=outliers_fraction),
     "Isolation Forest": IsolationForest(n_estimators=4, max_samples=10000,contamination=outliers_fraction,random_state=rng),
     "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=outliers_fraction)
}

list_img = ['src.jpg', 'light_bg.jpg', 'black_bg.jpg']
list_name = ['Nomal', 'light', 'black']
data_dir = 'data/experiment3'
model_dir = 'model/experiment3'
pic_dir = 'pic/experiment3'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)




img_bg =  cv2.imread(os.path.join(data_dir,"bg.jpg") ,0);
img_bg = dbtool.resize(img_bg, img_bg.shape[1] / 4);
mask = img_bg.reshape(-1);
print('load mask ok!')
list_resut = [dbtool.extra(os.path.join(data_dir,fileName), mask, 4) for fileName in list_img]
# [[HSV_H, HSV_S], [YUV_U, YUV_V]

HSV_data = [dbtool.normalization_n(np.concatenate([item[1][:,0] for item in list_resut]),0,180),
                dbtool.normalization_n(np.concatenate([item[1][:,1] for item in list_resut]),0,255)]
YUV_dat =  [dbtool.normalization_n(np.concatenate([item[2][:,1] for item in list_resut]),0,255),
             dbtool.normalization_n(np.concatenate([item[2][:,2] for item in list_resut]),0,255)]
data_source = {
    "HS": HSV_data,
    "UV": YUV_dat,
}
print('load data ok!')
#  训练并且画图
xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
for data_name, data in data_source.items():
    np.random.seed(42)
    X = np.array(data).T
    plt.figure(figsize=(6.3, 4.52))
    plt.rcParams.update(params)
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        if clf_name == "Local Outlier Factor":
            y_pred = clf.fit_predict(X)
            scores_pred = clf.negative_outlier_factor_
        else:
            clf.fit(X)
            scores_pred = clf.decision_function(X)
            y_pred = clf.predict(X)

        joblib.dump(clf,os.path.join(model_dir,data_name+clf_name+".model"))
        threshold = stats.scoreatpercentile(scores_pred, 5)
        if clf_name == "Local Outlier Factor":
            # decision_function is private for LOF
            Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])
            z_send = clf._predict(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            z_send = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        subplot = plt.subplot(2, 2,  i + 1)
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)
        a = subplot.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')
        b = subplot.scatter(X[:, 0], X[:, 1], c='white', s=5, edgecolor='k')
        subplot.axis('tight')
        subplot.legend(
            [a.collections[0], b],
            ['learned decision function', 'true ', 'true outliers'],
            prop=matplotlib.font_manager.FontProperties(size=10),
            loc='lower right')
        subplot.set_xlabel("%d. %s  Areasum: %f" % (i + 1, clf_name, (np.sum(z_send)+10000)/20000))
        subplot.set_xlim((0, 1))
        subplot.set_ylim((0, 1))

    plt.subplots_adjust(0.02, 0.1, 0.96, 0.94, 0.2, 0.3)
    plt.suptitle(data_name)
    plt.savefig(os.path.join(pic_dir, data_name+".png"))
plt.show()

print('print ok!')

# a -b =x
# a+b = 10000
# a = x+10000/200000
