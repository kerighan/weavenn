from sklearn import datasets
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             completeness_score, homogeneity_score)
from weavenn import WeaveNN

# load dataset
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

# create weavenn model
y_pred = WeaveNN().fit_predict(X)

# compute scores
homogeneity = homogeneity_score(y, y_pred)
completeness = completeness_score(y, y_pred)
rand_score = adjusted_rand_score(y, y_pred)
mutual_info_score = adjusted_mutual_info_score(y, y_pred)

print(f"homogeneity     = {homogeneity:.3f}")
print(f"completeness    = {completeness:.3f}")
print(f"rand_score      = {rand_score:.3f}")
print(f"mutual_info     = {mutual_info_score:.3f}")
