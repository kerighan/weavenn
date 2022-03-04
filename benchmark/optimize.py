import time

import numpy as np
import oodles as oo
import pandas as pd
from tqdm import tqdm
from weavenn.weavenn import WeaveNN, predict_knnl, score

from datasets import load

# iris, mobile, zoo, wine, glass, seeds, dates, raisin, phonemes
# v     x       v    v     v      v      v      x       v
# stellar, 20newsgroups, fashion, letters, mnist
# v        v             x        v        v
dataset = "fashion"
X, y = load(dataset)


def compare_weavenn_with_knnl():
    sheets = oo.Sheets("1JqzQUWP5UQ40Q5iLrs4vVPKqiMVAp4pOJfHNNfUWfzQ")
    ks = range(20, 160, 10)
    data = []
    start = time.time()
    for k in tqdm(ks):
        tmp = {"k": k}
        y_pred = predict_knnl(X, k)
        score_1, score_2 = score(y, y_pred)
        tmp["knnl_AMI"] = score_1
        tmp["knnl_RAND"] = score_2

        y_pred = WeaveNN(k=k, min_sim=0.01).fit_predict(X)
        score_1, score_2 = score(y, y_pred)
        tmp["weavenn_AMI"] = score_1
        tmp["weavenn_RAND"] = score_2
        print(tmp)

        data.append(tmp)
    print(time.time() - start)
    data = pd.DataFrame(data)[
        ["k", "knnl_AMI", "weavenn_AMI", "knnl_RAND", "weavenn_RAND"]]

    print(data["knnl_AMI"].mean(), data["knnl_AMI"].std())
    print(data["weavenn_AMI"].mean(), data["weavenn_AMI"].std())
    print()
    print(data["knnl_RAND"].mean(), data["knnl_RAND"].std())
    print(data["weavenn_RAND"].mean(), data["weavenn_RAND"].std())

    print()
    print(data["knnl_AMI"].max())
    print(data["weavenn_AMI"].max())
    print()
    print(data["knnl_RAND"].max())
    print(data["weavenn_RAND"].max())
    sheets[f"{dataset}"] = data


def optimize_hdbscan():
    from hdbscan import HDBSCAN
    sheets = oo.Sheets("1XevPDv28MdeVsgxxi1vUSZV9pRRXl8VevB6ds6L2rLY")

    min_cluster_size = [2, 5, 10, 15, 20, 30, 50, 60, 80, 100]
    min_samples = [1, 5, 10, 15, 20, 30, 50, 60, 80, 100]
    scores = []
    for mcs in tqdm(min_cluster_size):
        for ms in min_samples:
            clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=ms)
            y_pred = clusterer.fit_predict(X)
            score_1, score_2 = score(y, y_pred)
            scores.append(
                {"mcs": mcs, "ms": ms, "AMI": score_1, "rand": score_2})

    scores = pd.DataFrame(scores)
    sheets[dataset] = scores
    print(scores.max(axis=0))


def optimize_optics():
    from sklearn.cluster import OPTICS
    sheets = oo.Sheets("1NN_441RDAIjNnNXbXbiTZH1G6mfXN0rveKmDCnEvd_E")

    min_samples = range(2, 30)
    scores = []
    for ms in tqdm(min_samples):
        clusterer = OPTICS(metric="euclidean", min_samples=ms)
        y_pred = clusterer.fit_predict(X)
        score_1, score_2 = score(y, y_pred)
        scores.append(
            {"ms": ms, "AMI": score_1, "rand": score_2})

    scores = pd.DataFrame(scores)
    sheets[dataset] = scores
    print(scores.max(axis=0))


def optimize_dbscanpp():
    from DBSCANPP import DBSCANPP
    sheets = oo.Sheets("1bUc5b25wnX9av-p3gco4pKNppep_J7gmfv5gtwGx08s")

    values = {
        "iris": (1, 4),
        "mobile": (1, 800),
        "zoo": (.1, 4),
        "wine": (.1, 500),
    }

    min_eps, max_eps = values[dataset]
    epsilons = np.linspace(min_eps, max_eps, 20)
    scores = []
    for epsilon in tqdm(epsilons):
        for p in [.1, .2, .3]:
            # try:
            dbscanpp = DBSCANPP(
                p=p,
                eps_density=epsilon,
                eps_clustering=epsilon*2, minPts=4)
            y_pred = dbscanpp.fit_predict(X, init="k-centers")
            score_1, score_2 = score(y, y_pred)
            scores.append(
                {"eps": epsilon, "AMI": score_1, "rand": score_2})
            # except ValueError:
            #     scores.append({"eps": epsilon, "AMI": 0, "rand": 0})

    scores = pd.DataFrame(scores)
    sheets[dataset] = scores
    print(scores.max(axis=0))


def optimize_affinity_propagation():
    from sklearn.cluster import AffinityPropagation
    sheets = oo.Sheets("1i59jamowuVSOjr6xzQ9TB24K_f5vHTHqiIXT1Ab3rIE")

    dampings = np.linspace(0.5, .9, 10)
    scores = []
    for d in tqdm(dampings):
        clusterer = AffinityPropagation(damping=d)
        y_pred = clusterer.fit_predict(X)
        score_1, score_2 = score(y, y_pred)
        scores.append(
            {"damping": d, "AMI": score_1, "rand": score_2})

    scores = pd.DataFrame(scores)
    sheets[dataset] = scores
    print(scores.max(axis=0))


if __name__ == "__main__":
    # optimize_dbscanpp()
    optimize_optics()
    # optimize_affinity_propagation()
    # compare_weavenn_with_knnl()
