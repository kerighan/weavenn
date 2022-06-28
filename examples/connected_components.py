import random
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.datasets import make_blobs
from weavenn import WeaveNN

N = 500
print(f"N={N}")
X, y = make_blobs(n_samples=N, n_features=2, cluster_std=1.25)

clusterer = WeaveNN(k=20, min_sim=.25, method="mch")
y = clusterer.fit_predict(X)
# G = clusterer.fit_transform(X)
# e = [(a, b) for a, b, w in G.edges(data=True) if w["weight"] <= .9]
# G.remove_edges_from(e)
# pos = {node: X[node] for node in G.nodes}

plt.scatter(X[:, 0], X[:, 1], c=y)
# nx.draw(G, node_color=y, pos=pos)
plt.show()
