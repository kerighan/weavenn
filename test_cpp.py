import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from weavenn import WeaveNN, score

n_clusters = 15
X, y = make_blobs(
    30000, n_features=2,
    cluster_std=.4*np.random.random(size=(n_clusters,)), centers=n_clusters)

start = time.time()
y_pred = WeaveNN(k=75, prune=False, method="louvain").fit_predict(X)
print(time.time() - start)
y_pred_2 = WeaveNN(k=75, prune=False, method="optimal").fit_predict(X)
start = time.time()
y_pred_3 = WeaveNN(k=75, prune=True, method="optimal").fit_predict(X)
print(time.time() - start)
print()
A, _ = score(y, y_pred)
print(A)
print(len(set(y_pred)))
print()
B, _ = score(y, y_pred_2)
print(B)
print(len(set(y_pred_2)))
C, _ = score(y, y_pred_3)
print(C)
print(B >= A)
print(C >= B)

# plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.set_title('Louvain')
ax1.scatter(X[:, 0], X[:, 1], c=y_pred, s=1)

ax2.set_title('Louvain-optimal')
ax2.scatter(X[:, 0], X[:, 1], c=y_pred_2, s=1)

ax3.set_title('Louvain-optimal-prune')
ax3.scatter(X[:, 0], X[:, 1], c=y_pred_3, s=1)
plt.show()
