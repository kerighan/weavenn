import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from cylouvain import best_partition, modularity
from hdbscan import HDBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             completeness_score, homogeneity_score)
from weavenn import WeaveNN

# np.
N = np.random.randint(100, 4000)
X, y = make_blobs(n_samples=N, centers=4, n_features=2,
                  cluster_std=1, random_state=3)
print(f"N={N}")

y_hdbscan = HDBSCAN().fit_predict(X)

# compute scores
homogeneity = homogeneity_score(y, y_hdbscan)
completeness = completeness_score(y, y_hdbscan)
rand_score = adjusted_rand_score(y, y_hdbscan)
mutual_info_score = adjusted_mutual_info_score(y, y_hdbscan)
print("hdbscan:")
print(f"homogeneity     = {homogeneity:.3f}")
print(f"completeness    = {completeness:.3f}")
print(f"rand_score      = {rand_score:.3f}")
print(f"mutual_info     = {mutual_info_score:.3f}")

# create model and build graph
# use larger k_max for larger clusters
weave = WeaveNN(k=20, method="louvain")
# get graph from cloud points
# G = weave.fit_transform(X)
# get cluster labels from graph
y_weave = weave.fit_predict(X)

G = weave.fit_transform(X)

y_test = best_partition(G, resolution=2)
print(modularity(y_test, G))
y_test = [y_test[i] for i in range(len(X))]


# compute scores
homogeneity = homogeneity_score(y, y_weave)
completeness = completeness_score(y, y_weave)
rand_score = adjusted_rand_score(y, y_weave)
mutual_info_score = adjusted_mutual_info_score(y, y_weave)
print("\nweavenn:")
print(f"homogeneity     = {homogeneity:.3f}")
print(f"completeness    = {completeness:.3f}")
print(f"rand_score      = {rand_score:.3f}")
print(f"mutual_info     = {mutual_info_score:.3f}")

# plot results of hdbscan and weave
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_title('HDBSCAN')
ax1.scatter(X[:, 0], X[:, 1], c=y_hdbscan)
position = {i: X[i] for i in range(len(X))}
ax2.set_title('WeaveNN')
ax2.scatter(X[:, 0], X[:, 1], c=y_test)
# nx.draw(G, pos=position, node_size=10, node_color=y_weave, ax=ax2)
plt.show()
