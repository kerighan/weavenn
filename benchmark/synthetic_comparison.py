import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from tqdm import tqdm
from weavenn import WeaveNN, score

np.random.seed(0)
N = 1000
n_states = 10
random_states = np.random.randint(100, 1000000, size=n_states)

for dim in range(20, 520, 20):
    # dim = 100
    std = dim**.5
    res = []
    for state in random_states:
        X, y = make_blobs(n_samples=N, n_features=dim,
                          cluster_std=std, random_state=state)
        y_hdbscan = HDBSCAN().fit_predict(X)
        y_weavenn = WeaveNN(k=150, method="louvain").fit_predict(X)

        score_AMI_hdbscan, _ = score(y, y_hdbscan)
        score_AMI_weavenn, _ = score(y, y_weavenn)
        res.append({
            "hdbscan_AMI": score_AMI_hdbscan,
            "weavenn_AMI": score_AMI_weavenn,
        })

    res = pd.DataFrame(res)
    print(dim)
    print(res.mean(axis=0))
    print()
# print(res)

# print(f"N={N}")
