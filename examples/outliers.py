import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.datasets import make_blobs
from weavenn import WeaveNN

N = 1000
print(f"N={N}")
X, _ = make_blobs(n_samples=N, n_features=2, cluster_std=1.25)

y = WeaveNN(k=10, min_sim=.01, min_sc=.3).fit_predict(X)
y[y >= 0] = 1

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
