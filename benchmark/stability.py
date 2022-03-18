import oodles as oo
import pandas as pd
from tqdm import tqdm

from datasets import classification_datasets

tables = oo.Sheets("14LllQq7uNV_YExQxEEEXdhnJEpF5RC4QhQ_rI4skvAQ")

res = []
n_above = 0
n_total = 0
for dataset in classification_datasets:
    knnl = pd.read_excel("results/knnl.xlsx", sheet_name=dataset)
    weavenn = pd.read_excel("results/weavenn.xlsx", sheet_name=dataset)

    knnl_mean, knnl_std = knnl["knnl_AMI"].agg(["mean", "std"])
    knnl_cv = knnl_std / knnl_mean
    knnl_range = knnl["knnl_AMI"].max() - knnl["knnl_AMI"].min()
    weavenn_mean, weavenn_std = weavenn["weavenn_AMI"].agg(["mean", "std"])
    weavenn_cv = weavenn_std / weavenn_mean
    weavenn_range = weavenn["weavenn_AMI"].max() - weavenn["weavenn_AMI"].min()

    n_above += (weavenn["weavenn_AMI"] > knnl["knnl_AMI"]).sum()
    n_total += knnl.shape[0]

    res.append({
        "dataset": dataset,
        "weavenn_mean": weavenn_mean,
        "knnl_mean": knnl_mean,
        "weavenn_cv": weavenn_cv,
        "knnl_cv": knnl_cv,
        "weavenn_range": weavenn_range,
        "knnl_range": knnl_range
    })

res = pd.DataFrame(res)
tables["stability"] = res

mean_above = (res["knnl_mean"] > res["weavenn_mean"]).sum()
cv_above = (res["knnl_cv"] > res["weavenn_cv"]).sum()
print(n_above / n_total)
