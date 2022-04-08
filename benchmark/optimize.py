import time

import numpy as np
import oodles as oo
import pandas as pd
from tqdm import tqdm
from weavenn.weavenn import WeaveNN, predict_knnl, score

from datasets import load

# iris, mobile, zoo, wine, glass, seeds, dates, raisin, phonemes
# v     v       x    v     v      x      x      v       v
# stellar, 20newsgroups, fashion, letters, mnist
# v        v             v        v        v
dataset = "mobile"
X, y = load(dataset)


def optimize_weavenn():
    ks = range(20, 160, 10)
    data = []
    start = time.time()
    for k in tqdm(ks):
        tmp = {"k": k}

        y_pred = WeaveNN(k=k).fit_predict(X, resolution=1)
        score_1, score_2 = score(y, y_pred)
        tmp["weavenn_AMI"] = score_1
        tmp["weavenn_RAND"] = score_2
        print(tmp)

        data.append(tmp)
    print(time.time() - start)
    data = pd.DataFrame(data)[
        ["k", "weavenn_AMI", "weavenn_RAND"]]

    print(data["weavenn_AMI"].mean(), data["weavenn_AMI"].std())
    print(data["weavenn_AMI"].max())

    # sheets = oo.Sheets("1_Aqv7TP6WxfL3WXP17H_PF-fa8FR-DMycrIATjXRxlo")
    # sheets[dataset] = data


def optimize_weavenn_lpa():
    from nodl.community import label_propagation
    ks = range(20, 160, 10)
    data = []
    start = time.time()
    for k in tqdm(ks):
        tmp = {"k": k}

        G = WeaveNN(k=k).fit_transform(X)
        y_pred = label_propagation(G)
        y_pred = [y_pred[i] for i in range(len(G.nodes))]
        # y_pred = WeaveNN(k=k).fit_predict(X)
        score_1, score_2 = score(y, y_pred)
        tmp["weavenn_AMI"] = score_1
        tmp["weavenn_RAND"] = score_2
        print(tmp)

        data.append(tmp)
    print(time.time() - start)
    data = pd.DataFrame(data)[
        ["k", "weavenn_AMI", "weavenn_RAND"]]

    print(data["weavenn_AMI"].mean(), data["weavenn_AMI"].std())
    print(data["weavenn_AMI"].max())

    # sheets = oo.Sheets("1_Aqv7TP6WxfL3WXP17H_PF-fa8FR-DMycrIATjXRxlo")
    # sheets[dataset] = data


def optimize_weavenn_full():
    ks = range(20, 160, 10)
    data = []
    start = time.time()
    for k in tqdm(ks):
        for min_sim in [.0001, .001, .01, .1, .25, .5, .8]:
            tmp = {"k": k, "min_sim": min_sim}

            y_pred = WeaveNN(k=k, min_sim=min_sim, prune=False).fit_predict(X)
            score_1, score_2 = score(y, y_pred)
            tmp["weavenn_AMI"] = score_1
            tmp["weavenn_RAND"] = score_2
            print(tmp)

            data.append(tmp)
    print(time.time() - start)
    data = pd.DataFrame(data)[
        ["k", "min_sim", "weavenn_AMI", "weavenn_RAND"]]

    print(data["weavenn_AMI"].mean(), data["weavenn_AMI"].std())
    print(data["weavenn_AMI"].max())

    sheets = oo.Sheets("1jYHWTxRoahdrT06ofOoqmUMLgnRxy0OxppsFAoB_e60")
    sheets[dataset] = data


def optimize_knnl():
    sheets = oo.Sheets("1lY3ekInWwIO3KNWgab9Y7QhzswcJE1iGV0n1j3oQuag")
    ks = range(20, 160, 10)
    data = []
    start = time.time()
    for k in tqdm(ks):
        tmp = {"k": k}
        y_pred = predict_knnl(X, k)
        score_1, score_2 = score(y, y_pred)
        tmp["knnl_AMI"] = score_1
        tmp["knnl_RAND"] = score_2
        data.append(tmp)
    data = pd.DataFrame(data)[
        ["k", "knnl_AMI", "knnl_RAND"]]

    print(data["knnl_AMI"].mean(), data["knnl_AMI"].std())
    print(data["knnl_AMI"].max())
    sheets[dataset] = data


def optimize_hdbscan():
    from hdbscan import HDBSCAN

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

    sheets = oo.Sheets("1XevPDv28MdeVsgxxi1vUSZV9pRRXl8VevB6ds6L2rLY")
    sheets[dataset] = scores
    print(scores.max(axis=0))


def optimize_optics():
    from sklearn.cluster import OPTICS

    min_samples = range(2, 30)
    scores = []
    for ms in tqdm(min_samples):
        clusterer = OPTICS(metric="euclidean", min_samples=ms)
        y_pred = clusterer.fit_predict(X)
        score_1, score_2 = score(y, y_pred)
        scores.append(
            {"ms": ms, "AMI": score_1, "rand": score_2})

    scores = pd.DataFrame(scores)
    sheets = oo.Sheets("1NN_441RDAIjNnNXbXbiTZH1G6mfXN0rveKmDCnEvd_E")
    sheets[dataset] = scores
    print(scores.max(axis=0))


def optimize_dbscanpp():
    from DBSCANPP import DBSCANPP
    sheets = oo.Sheets("1bUc5b25wnX9av-p3gco4pKNppep_J7gmfv5gtwGx08s")

    values = {
        "iris": (1, 4),
        "mobile": (1, 800),
        "zoo": (.1, 3),
        "wine": (.1, 10),
        "glass": (.1, 10),
        "seeds": (.1, 5),
        "dates": (1, 5),
        "raisin": (.1, 1),
        "phonemes": (.1, 100),
        "stellar": (.2, 3),
        "20newsgroups": (.1, .4),
        "fashion": (1, 4),
        "letters": (.1, .5),
        "mnist": (200, 1200),
        "usps": (2.368, 6.8),
    }

    min_eps, max_eps = values[dataset]
    epsilons = np.linspace(min_eps, max_eps, 20)
    scores = []
    for epsilon in tqdm(epsilons):
        for p in [.1, .2, .3]:
            try:
                dbscanpp = DBSCANPP(
                    p=p,
                    eps_density=epsilon,
                    eps_clustering=epsilon, minPts=4)
                y_pred = dbscanpp.fit_predict(X, init="k-centers")
                score_1, score_2 = score(y, y_pred)
                scores.append(
                    {"eps": epsilon, "p": p, "AMI": score_1, "rand": score_2})
                print(epsilon, score_1)
            except ValueError as e:
                # print(e)
                scores.append({"eps": epsilon, "p": p, "AMI": 0, "rand": 0})

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
    # optimize_knnl()
    optimize_weavenn()
    # optimize_hdbscan()
    # optimize_optics()
    # optimize_dbscanpp()
    # optimize_weavenn_lpa()
    # optimize_affinity_propagation()
    # optimize_weavenn_full()
