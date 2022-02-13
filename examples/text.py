from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from weavenn import WeaveNN
from hdbscan import HDBSCAN
from DBSCANPP import DBSCANPP
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS
from tqdm import tqdm


newsgroups_train = fetch_20newsgroups(subset='train')

pipe = Pipeline([
    ("vectorizer", TfidfVectorizer(max_features=100000)),
    ("reducer", TruncatedSVD(n_components=200))
])
X = pipe.fit_transform(newsgroups_train.data)
y = newsgroups_train.target


clustering_algorithms = [
    ("weavenn", WeaveNN()),
    ("OPTICS", OPTICS(min_samples=2)),
    ("dbscan", DBSCAN()),
    ("dbscanpp", DBSCANPP(p=0.1, eps_density=5.0, eps_clustering=5.0, minPts=10)),
    ("hdbscan", HDBSCAN()),
]

results = []
for name, algorithm in tqdm(clustering_algorithms):
    if name == "dbscanpp":
        y_pred = algorithm.fit_predict(X, init="k-centers", cluster_outliers=True)
    else:
        y_pred = algorithm.fit_predict(X)

    adj_rand_score = adjusted_rand_score(y, y_pred)
    adj_mutual_info_score = adjusted_mutual_info_score(y, y_pred)
    results.append({
        "algorithm": name,
        "adjusted_rand_score": adj_rand_score,
        "adjusted_mutual_info_score": adj_mutual_info_score
    })
results = pd.DataFrame(results).T
print(results)
