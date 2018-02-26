import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy import stats
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

import test
rng = np.random.RandomState(41)
# Example settings
n_samples = 1750
outliers_fraction = 0.05
# 执行一次或者多次
clusters_separation = [2]
# define two outlier detection tools to be compared
classifiers = {
    "One-Class SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05, kernel="rbf", gamma=0.1),
    # "Robust covariance": EllipticEnvelope(contamination=0.1),
    "Isolation Forest": IsolationForest(n_estimators=10, max_samples=5000,contamination=0.1,random_state=rng),
    # "Local Outlier Factor": LocalOutlierFactor(n_neighbors=100, contamination=0.1)
}
# Compare given classifiers under given settings 诡异
xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
# Fit the problem with varying cluster separation
for i, offset in enumerate(clusters_separation):
    np.random.seed(42)
    # Data generation
    X=test.normalization(test.get_train())
    # Fit the model
    plt.figure(figsize=(9, 7))
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        threshold = stats.scoreatpercentile(scores_pred,5)
        x_test =  test.normalization(test.get_test())
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        ans = clf.decision_function(x_test)
        Z = Z.reshape(xx.shape)
        subplot = plt.subplot(2, 2, 2*i+1)
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
        a = subplot.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')
        b = subplot.scatter(X[:,0], X[:,1], c='white',s=5, edgecolor='k')
        subplot.axis('tight')
        subplot.legend(
            [a.collections[0],b],
            ['learned decision function', 'true ', 'true outliers'],
            prop=matplotlib.font_manager.FontProperties(size=10),
            loc='lower right')
        subplot.set_xlabel("%d. %s " % (i + 1, clf_name))
        subplot.set_xlim((-1, 1))
        subplot.set_ylim((-1, 1))
        ans = ans.reshape(test.test_shape.shape[0],test.test_shape.shape[1]);
        plt.subplot(2,2,2*i+2).imshow(ans)
    plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
    plt.suptitle("Outlier detection")
plt.show()

