from pprint import pprint

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from tqdm import tqdm
from weavenn import WeaveNN

from datasets import classification_datasets, load

results = {
    "iris": [.76, .74],
    "wine": [.36, .34],
    "mobile": [.42, .24],
    "zoo": [.5, .21],
    "seeds": [.69, .74],
    "heart": [.16352, .1563],
    "20newsgroups": [.25, .1],
    "stellar": [.212, .059],
    "mnist": [.853, .819],
    "fashion": [.600, .381]
}


datasets = [(dataset, load(dataset))
            for dataset in classification_datasets
            if dataset not in ["mnist", "fashion", "stellar", "zoo"]]

scores = []
best_score = None
for min_sim in np.linspace(1e-3, 5e-1, 10):
    min_sim = round(min_sim, 3)
    # for threshold in np.linspace(1e-6, .7, 10):
    clusterer = WeaveNN(
        k_max=100,
        threshold=0.05,
        max_sim=0.99,
        min_sim=min_sim,
        kernel="tanh")
    score = []
    for name, (X, y) in datasets:
        y_pred = clusterer.fit_predict(X)
        score_1 = adjusted_mutual_info_score(y, y_pred)
        score_2 = adjusted_rand_score(y, y_pred)

        val = results[name]
        # score_1 -= val[0]
        # score_2 -= val[1]

        score.append(score_1)
        score.append(score_2)
    score = np.mean(score)
    scores.append((min_sim, score))

    tmp = sorted(scores, key=lambda x: x[1], reverse=True)[0]
    if best_score != tmp:
        print(tmp)
        best_score = tmp
scores = sorted(scores, key=lambda x: x[1], reverse=True)
pprint(scores[:25])
