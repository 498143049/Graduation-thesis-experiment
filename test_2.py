model_dir = 'model/experiment9'
pic_dir = 'pic/experiment9'
data_dir = 'data/experiment9'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(1000, n_features=300, centers=4,
                  cluster_std=8, random_state=42)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
rand = np.random.RandomState(42)

for axi in ax.flat:
    i, j = rand.randint(X.shape[1], size=2)
    axi.scatter(X[:, i], X[:, j], c=y)
    from lpproj import LocalityPreservingProjection

    lpp = LocalityPreservingProjection(n_components=2)

    X_2D = lpp.fit_transform(X)

from lpproj import LocalityPreservingProjection
lpp = LocalityPreservingProjection(n_components=2)

X_2D = lpp.fit_transform(X)

plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y)

plt.title("Projected from 500->2 dimensions");
from sklearn.decomposition import PCA
Xpca = PCA(n_components=2).fit_transform(X)
plt.scatter(Xpca[:, 0], Xpca[:, 1], c=y);
rand  = np.random.RandomState(42)
Xnoisy = X.copy()
Xnoisy[:10] += 1000 * rand.randn(10, X.shape[1])
Xpca = PCA(n_components=2).fit_transform(Xnoisy)
Xlpp = LocalityPreservingProjection(n_components=2).fit_transform(Xnoisy)

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
ax[0].scatter(Xlpp[:, 0], Xlpp[:, 1], c=y)
ax[0].set_title('LPP with outliers')
ax[1].scatter(Xpca[:, 0], Xpca[:, 1], c=y)
ax[1].set_title('PCA with outliers');
plt.show()