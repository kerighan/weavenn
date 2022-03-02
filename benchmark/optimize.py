import time

import oodles as oo
import pandas as pd
from tqdm import tqdm
from weavenn.weavenn import WeaveNN, predict_knnl, score

from datasets import load

# iris, mobile, zoo, wine, glass, seeds, 20newsgroups, fashion, letters, phones
# x     x       v    v     v      v
dataset = "iris"
X, y = load(dataset)

sheets = oo.Sheets("13N_pbxrKk-C-Pr-xVnyPVbmhu_KzYYsWcm0gJqDMb5Y")


def optimize_knnl():

    ks = range(20, 160, 10)
    data = []
    start = time.time()
    for k in tqdm(ks):
        tmp = {"k": k}
        y_pred = predict_knnl(X, k)
        score_1, _ = score(y, y_pred)
        tmp["knnl_AMI"] = score_1

        y_pred = WeaveNN(k=k, min_sim=0.01).fit_predict(X)
        score_1, _ = score(y, y_pred)
        tmp["weavenn_AMI"] = score_1
        print(tmp)

        data.append(tmp)
    print(time.time() - start)
    data = pd.DataFrame(data)[
        ["k", "knnl_AMI", "weavenn_AMI"]]

    print(data["knnl_AMI"].mean(), data["knnl_AMI"].std())
    print(data["weavenn_AMI"].mean(), data["weavenn_AMI"].std())

    print()
    print(data["knnl_AMI"].max())
    print(data["weavenn_AMI"].max())
    sheets[f"{dataset}_test"] = data


def optimize_weavenn():
    from weavenn import WeaveNN

    ks = range(20, 160, 10)
    scores = []
    for k in tqdm(ks):
        weave = WeaveNN(k=k)
        y_pred = weave.fit_predict(X)
        score_1, score_2 = score(y, y_pred)
        scores.append(
            {"k": k, "AMI": score_1, "rand": score_2})

    scores = pd.DataFrame(scores)
    sheets[f"weavenn_{dataset}"] = scores


def optimize_hdbscan():
    from hdbscan import HDBSCAN

    # ks = range(20, 160, 10)
    min_cluster_size = range(5, 25, 5)
    min_samples = range(1, 20)
    scores = []
    for mcs in tqdm(min_cluster_size):
        for ms in min_samples:
            clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=ms)
            y_pred = clusterer.fit_predict(X)
            score_1, score_2 = score(y, y_pred)
            scores.append(
                {"mcs": mcs, "ms": ms, "AMI": score_1, "rand": score_2})

    scores = pd.DataFrame(scores)
    sheets[f"hdbscan_{dataset}"] = scores


if __name__ == "__main__":
    # optimize_hdbscan()
    # optimize_weavenn()
    optimize_knnl()
