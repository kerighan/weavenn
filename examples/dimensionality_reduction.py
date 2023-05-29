import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.datasets import fetch_openml, make_blobs
from umap import UMAP
from weavenn import WeaveNN

mnist = fetch_openml("mnist_784", version=1)
X = np.array(mnist.data)
y = mnist.target.astype(int)

# X, y = make_blobs(1000, n_features=100, centers=5,
#                   cluster_std=13, random_state=1)

fig, axes = plt.subplots(1, 2)

X1 = WeaveNN(k=20, min_sim=0., reduce_dim=3).fit_reduce(
    X, walk_len=50, n_walks=50, verbose=True, init=None, epochs=20, corruption=.5, b=None)
X2 = UMAP(20).fit_transform(X)
axes[0].scatter(X1[:, 0], X1[:, 1], c=y)
axes[1].scatter(X2[:, 0], X2[:, 1], c=y)
plt.show()
