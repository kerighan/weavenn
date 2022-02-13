from datasets import classification_datasets, load
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score)
from algorithms import algorithms, predict
import pandas as pd
from tqdm import tqdm
import time


results = []
# for name in tqdm(["stellar"]):
for name in tqdm(classification_datasets):
    # if name in ["mnist", "fashion", "stellar", "20newsgroups"]:
    #     continue

    X, y_true = load(name)
    # for algorithm in ["AffinityPropagation"]:
    for algorithm in algorithms[:2]:
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
