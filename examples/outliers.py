import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.datasets import make_blobs
from weavenn import WeaveNN

N = 1000
print(f"N={N}")
X, _ = make_blobs(n_samples=N, n_features=2, cluster_std=1)

G = WeaveNN(k=10, min_sim=.7).fit_transform(X)
outliers = np.zeros(len(G.nodes))
for isolate in nx.isolates(G):
    outliers[isolate] = -1


plt.scatter(X[:, 0], X[:, 1], c=outliers)
plt.show()
