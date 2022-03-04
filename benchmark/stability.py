import oodles as oo
import pandas as pd
from tqdm import tqdm

from datasets import classification_datasets

sheets = oo.Sheets("1JqzQUWP5UQ40Q5iLrs4vVPKqiMVAp4pOJfHNNfUWfzQ")
tables = oo.Sheets("14LllQq7uNV_YExQxEEEXdhnJEpF5RC4QhQ_rI4skvAQ")
res = []
for dataset in tqdm(classification_datasets):
    df = sheets[dataset].values()
    for column in df.columns:
        df[column] = df[column].apply(lambda x: float(x.replace(",", ".")))

    weavenn_mean, weavenn_std = df["weavenn_AMI"].agg(["mean", "std"])
    knnl_mean, knnl_std = df["knnl_AMI"].agg(["mean", "std"])

    res.append({
        "dataset": dataset,
        "weavenn_mean": weavenn_mean,
        "knnl_mean": knnl_mean,
        "weavenn_std": weavenn_std,
        "knnl_std": knnl_std,
    })
df = pd.DataFrame(res)
print(df)
tables["stability"] = df
