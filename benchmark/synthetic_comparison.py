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
random_states = np.random.randint(100, 1000000, size=n_states)
tables = oo.Sheets("14LllQq7uNV_YExQxEEEXdhnJEpF5RC4QhQ_rI4skvAQ")

scores = []
for dim in tqdm([2, 4, 6, 8, 10, 12, 14, 16, 18] + list(range(20, 520, 20))):
    std = dim**.5
    res = []
    for state in random_states:
        X, y = make_blobs(n_samples=N, n_features=dim, centers=5,
                          cluster_std=std, random_state=state)
        y_hdbscan = HDBSCAN().fit_predict(X)
        y_weavenn = WeaveNN(k=100, method="louvain").fit_predict(X)

        score_AMI_hdbscan, _ = score(y, y_hdbscan)
        score_AMI_weavenn, _ = score(y, y_weavenn)
        res.append({
            "hdbscan_AMI": score_AMI_hdbscan,
            "weavenn_AMI": score_AMI_weavenn,
        })

    res = pd.DataFrame(res).mean(axis=0)
    scores.append(
        {"hdbscan_AMI": res["hdbscan_AMI"],
         "weavenn_AMI": res["weavenn_AMI"]})
scores = pd.DataFrame(scores)
tables["synthetic_blobs"] = scores
