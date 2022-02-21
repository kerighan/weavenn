import time

import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from tqdm import tqdm

from algorithms import algorithms, predict
from datasets import classification_datasets, load

results = []
# for name in tqdm(["stellar"]):
for name in tqdm(classification_datasets):
    if name in ["fashion", "stellar", "mnist"]:
        continue

    X, y_true = load(name)
    # for algorithm in ["AffinityPropagation"]:
    for algorithm in ["weavenn"]:
        start_time = time.time()
        y_pred = predict(algorithm, X)
        if y_pred is None:
            adj_mutual_info = None
            adj_rand = None
        else:
            adj_mutual_info = adjusted_mutual_info_score(y_true, y_pred)
            adj_rand = adjusted_rand_score(y_true, y_pred)

        elapsed_time = time.time() - start_time

        results.append({
            "dataset": name,
            "algorithm": algorithm,
            "adjusted_mutual_info_score": adj_mutual_info,
            "adjusted_rand_score": adj_rand,
            "elapsed_time": elapsed_time
        })
        print(results)
results = pd.DataFrame(results)
print(results)
