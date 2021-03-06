import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.datasets import make_blobs
from weavenn import WeaveNN

N = 1000
print(f"N={N}")
X, _ = make_blobs(n_samples=N, n_features=2, cluster_std=1.25)

clusterer = WeaveNN(k=100, min_sim=.01)
y = clusterer.fit_predict(X)
sc = np.array(clusterer._sigma_count)
plt.scatter(X[:, 0], X[:, 1], alpha=sc * .25)
plt.show()
