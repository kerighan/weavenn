import matplotlib.pyplot as plt
from umap import UMAP
from weavenn import WeaveNN

from datasets import load

X, y = load("usps")

a = WeaveNN(k=10, min_sim=0).fit_reduce(X, corruption=.99)
print("done")
b = UMAP().fit_transform(X)
print("done")

fig, axes = plt.subplots(1, 2)
axes[0].scatter(a[:, 0], a[:, 1], c=y)
axes[1].scatter(b[:, 0], b[:, 1], c=y)
plt.show()
