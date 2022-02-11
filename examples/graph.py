import matplotlib.pyplot as plt
import networkx as nx
from hdbscan import HDBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             completeness_score, homogeneity_score)
from weavenn import WeaveNN

X, y = make_blobs(n_samples=500)

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
weave = WeaveNN(k_max=200)  # use larger k_max for larger clusters
G = weave.fit_transform(X)
# get cluster labels from graph
y_weave = weave.predict(G)

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
nx.draw(G, pos=position, node_size=10, node_color=y_weave, ax=ax2)
plt.show()
