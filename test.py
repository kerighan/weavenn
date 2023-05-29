import matplotlib.pyplot as plt
import numpy as np
from hdbscan import HDBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             completeness_score, homogeneity_score)
from weavenn import WeaveNN
import time


def get_score(y, y_pred):
    # compute scores
    homogeneity = homogeneity_score(y, y_pred)
    completeness = completeness_score(y, y_pred)
    rand_score = adjusted_rand_score(y, y_pred)
    mutual_info_score = adjusted_mutual_info_score(y, y_pred)
    mean_score = (homogeneity + completeness +
                  rand_score + mutual_info_score) / 4
    return mean_score



N = np.random.randint(400, 5000)
n_c = np.random.randint(2, 20)
dim = 384
dispersion = np.random.uniform(.5, 1.3)
std = np.abs(dispersion * np.random.normal(size=n_c, scale=dim**.5))
print(N, n_c, dispersion)
print()
X, y_true = make_blobs(n_samples=N,
                       centers=n_c,
                       n_features=dim,
                       cluster_std=std)

# ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# ----
start = time.time()
y_pred = HDBSCAN(min_cluster_size=2, min_samples=50).fit_predict(X)
print(time.time() - start, "seconds")
n_outliers = np.sum(y_pred == -1)
print(n_outliers, "outliers")
print("hdbscan:", round(get_score(y_true, y_pred), 4))
alpha = [1 if y_pred[i] != -1 else 0.1 for i in range(len(y_pred))]
axes[0].scatter(X[:, 0], X[:, 1], c=y_pred, s=1.8, alpha=alpha)

# ----
print()
start = time.time()
weavenn = WeaveNN(k=None, score="silhouette", threshold=.0,
                  use_modularity=False, kernel="tanh", min_cluster_size=10)
y_pred = weavenn.fit_predict(X)
# set same number of outliers as hdbscan
# density = weavenn._density
# outliers = np.argsort(density)[:n_outliers]
# y_pred[outliers] = -1
print(time.time() - start, "seconds")
print(np.sum(y_pred == -1), "outliers")
print("weavenn-not mst :", round(get_score(y_true, y_pred), 4))

# ----
print()
start = time.time()
weavenn = WeaveNN(k=None, score="silhouette", threshold=.2,
                  ann_method="annoy", use_modularity=False,
                  kernel="tanh", min_cluster_size=10)
y_pred = weavenn.fit_predict(X)
# set same number of outliers as hdbscan
# density = weavenn._density
# outliers = np.argsort(density)[:n_outliers]
# y_pred[outliers] = -1
print(time.time() - start, "seconds")
print(np.sum(y_pred == -1), "outliers")
print("weavenn-not mst :", round(get_score(y_true, y_pred), 4))
alpha = [1 if y_pred[i] != -1 else 0.1 for i in range(len(y_pred))]
axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, s=1.8, alpha=alpha)

# ----
axes[2].scatter(X[:, 0], X[:, 1], c=y_true, s=1.8)
plt.show()