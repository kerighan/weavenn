import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.datasets import make_blobs
from weavenn import WeaveNN

N = 1000
N_out = 250
print(f"N={N}")
X, _ = make_blobs(n_samples=N, n_features=2, cluster_std=1.5)

x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
X_out = np.random.uniform(low=[x_min, y_min], high=[
                          x_max, y_max], size=(N_out, 2))
X = np.vstack([X, X_out])


clusterer = WeaveNN(min_sc=.5, min_sim=.1)
y = clusterer.fit_predict(X)
c = y.copy()
c[c >= 0] = 1

fig, axes = plt.subplots(1, 2)
axes[0].scatter(X[:, 0], X[:, 1], c=y, alpha=clusterer._sigma_count)
axes[1].bar(range(N + N_out), sorted(clusterer._sigma_count))

plt.show()
