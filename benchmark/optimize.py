from pprint import pprint

import knnl
import numpy as np
import oodles as oo
import pandas as pd
from tqdm import tqdm
from weavenn.weavenn import score

from datasets import load

# iris, mobile, zoo, wine, glass, seeds, 20newsgroups
# v   ,       , v  , v   , v    , v    ,
dataset = "seeds"
X, y = load(dataset)

sheets = oo.Sheets("13N_pbxrKk-C-Pr-xVnyPVbmhu_KzYYsWcm0gJqDMb5Y")


def optimize_knnl():
    from weavenn.weavenn import predict_knnl

    ks = range(20, 160, 10)
    data = []
    for k in tqdm(ks):
        tmp = {"k": k}
        y_pred = knnl.predict(X, k)
        score_1, score_2 = score(y, y_pred)
        tmp["knn"] = (score_2 + score_1) / 2
        y_pred = knnl.predict_2(X, k)
        score_1, score_2 = score(y, y_pred)
        tmp["weavenn"] = (score_2 + score_1) / 2
        y_pred = knnl.predict_5(X, k)
        score_1, score_2 = score(y, y_pred)
        tmp["test"] = (score_2 + score_1) / 2
        data.append(tmp)
    data = pd.DataFrame(data)[["k", "knn", "weavenn", "test"]]
    print(data["knn"].mean())
    print(data["weavenn"].mean())
    print(data["test"].mean())
    print()
    print(data["knn"].max())
    print(data["weavenn"].max())
    print(data["test"].max())
    sheets[dataset] = data

    # k = 50
    # y_pred = knnl.predict(X, k)
    # score_1, score_2 = score(y, y_pred)
    # print(score_1, score_2)

    # y_pred = knnl.predict_5(X, 100)
    # score_1, score_2 = score(y, y_pred)
    # print(score_1, score_2)


def optimize_weavenn():
    from weavenn import WeaveNN

    ks = [80, 90, 100, 110]
    ks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    min_sims = np.linspace(1e-3, .5, 5)

    scores = []
    for k in tqdm(ks):
        # k += 50
        # weave = WeaveNN(k_max=k, min_sim=.001, threshold=.1)
        # y_pred = weave.fit_predict(X)
        # score_1, score_2 = score(y, y_pred)
        # scores.append(({"k": k, "min_sim": 0.001}, (score_1, score_2)))

        for sim in min_sims:
            weave = WeaveNN(k_max=k, min_sim=sim, threshold=.0)
            y_pred = weave.fit_predict(X)
            score_1, score_2 = score(y, y_pred)
            scores.append(({"k": k, "min_sim": sim}, (score_1, score_2)))

    scores = sorted(scores, key=lambda x: 0.5 *
                    x[1][0]+0.5*x[1][1], reverse=True)
    pprint(scores[:5])


if __name__ == "__main__":
    # optimize_weavenn()
    optimize_knnl()
