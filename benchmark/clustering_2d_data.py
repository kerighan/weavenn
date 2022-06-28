import matplotlib.pyplot as plt
import numpy as np
from hdbscan import HDBSCAN
from sklearn.datasets import make_blobs
from tqdm import tqdm
from weavenn import WeaveNN, score

np.random.seed(0)
n_tests = 20
weavenn_scores = []
hdbscan_scores = []
isolation_scores = []
dim = 5

# hdbscan=0.8811025722276167
# weavenn=0.7963607844903734
# isolation=0.9030100245940667
for i in tqdm(range(n_tests)):
    n_clusters = np.random.randint(2, 20)
    N = np.random.randint(25, 1000, size=(n_clusters,))
    std = dim**.5
    cluster_std = np.random.random(size=n_clusters)*std

    X, y = make_blobs(N, cluster_std=cluster_std, n_features=dim)
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()

    y_isolation = WeaveNN(k=60, method="mch", reduce_dim=1,
                          min_sim=.1).fit_predict(X)
    y_weavenn = WeaveNN(k=60, method="calinski_harabasz",
                        reduce_dim=1, min_sim=.1).fit_predict(X)
    y_hdbscan = HDBSCAN().fit_predict(X)

    score_weavenn, _ = score(y, y_weavenn)
    score_isolation, _ = score(y, y_isolation)
    score_hdbscan, _ = score(y, y_hdbscan)
    weavenn_scores.append(score_weavenn)
    hdbscan_scores.append(score_hdbscan)
    isolation_scores.append(score_isolation)

weavenn_scores = np.mean(weavenn_scores)
hdbscan_scores = np.mean(hdbscan_scores)
isolation_scores = np.mean(isolation_scores)
print(f"hdbscan={hdbscan_scores}")
print(f"weavenn={weavenn_scores}")
print(f"isolation={isolation_scores}")
