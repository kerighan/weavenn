from pprint import pprint

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from tqdm import tqdm
from weavenn import WeaveNN

from datasets import classification_datasets, load

dataset = "wine"
X, y = load(dataset)

# k_max = [100]
# threshold = np.linspace(.001, .9, 10)
k_max = [50, 75, 100, 125, 150]
threshold = [.001, .005, .01, .05, .1, .15, .2]
# max_sim = np.linspace(.5, 1, 5)
# min_sim = [.001, .005, .01, .015, .05, .1, .15, .2]
min_sim = [0.01]
# min_sim = np.linspace(.001, .5, 10)

best = None

# runs = []
# for k in k_max:
#     for t in threshold:
#             for m in min_sim:
#                 clusterer = WeaveNN(k_max=k, threshold=t, max_sim=M, min_sim=m)
#                 y_pred = clusterer.fit_predict(X)
#                 mutual_info = adjusted_mutual_info_score(y, y_pred)
#                 rand_score = adjusted_rand_score(y, y_pred)
#                 score = (mutual_info + rand_score) / 2
#                 del clusterer

#                 runs.append(({
#                     "k_max": k,
#                     "threshold": t,
#                     "max_sim": M,
#                     "min_sim": m
#                 }, score))

#                 best_run = sorted(runs, key=lambda x: x[1], reverse=True)[0]
#                 if best_run != best:
#                     print(best_run)
#                     best = best_run

# best_runs = sorted(runs, key=lambda x: x[1], reverse=True)
# print()
# pprint(best_runs[:10])

clusterer = WeaveNN(min_sim=.1, max_sim=.99, threshold=.0)
# clusterer = WeaveNN()

# best_run = best_runs[0][0]
# clusterer = WeaveNN(
#     k_max=best_run["k_max"], threshold=best_run["threshold"], max_sim=.99, min_sim=best_run["min_sim"])

y_pred = clusterer.fit_predict(X)
mutual_info = adjusted_mutual_info_score(y, y_pred)
rand_score = adjusted_rand_score(y, y_pred)
print(f"mutual_info={mutual_info:.3f}")
print(f"rand_score={rand_score:.3f}")
