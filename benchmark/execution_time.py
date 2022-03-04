import time

import numpy as np
import oodles as oo
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from tqdm import tqdm
from weavenn import WeaveNN, score

np.random.seed(0)
N = 1000
n_states = 10
random_state = np.random.randint(100, 1000000)
tables = oo.Sheets("14LllQq7uNV_YExQxEEEXdhnJEpF5RC4QhQ_rI4skvAQ")

dim = 10
res = []
for N in range(1000, 10000, 1000):

    X, y = make_blobs(n_samples=N, n_features=dim,
                      cluster_std=2, random_state=random_state)

    start = time.time()
    y_hdbscan = HDBSCAN().fit_predict(X)
    elapsed_hdbscan = time.time() - start

    start = time.time()
    y_weavenn = WeaveNN(k=100, method="louvain").fit_predict(X)
    elapsed_weavenn = time.time() - start

    print(N)
    print(elapsed_weavenn)
    print(elapsed_hdbscan)
    res.append({
        "N": N,
        "elapsed_weavenn": elapsed_weavenn,
        "elapsed_hdbscan": elapsed_hdbscan,
    })

res = pd.DataFrame(res)
tables["execution_time"] = res
# print(res)

# print(f"N={N}")
