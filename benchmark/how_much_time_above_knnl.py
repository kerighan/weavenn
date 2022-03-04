import oodles as oo
import pandas as pd
from tqdm import tqdm

from datasets import classification_datasets

sheets = oo.Sheets("1JqzQUWP5UQ40Q5iLrs4vVPKqiMVAp4pOJfHNNfUWfzQ")
tables = oo.Sheets("14LllQq7uNV_YExQxEEEXdhnJEpF5RC4QhQ_rI4skvAQ")
n_above = 0
n_total = 0
for dataset in tqdm(classification_datasets):
    df = sheets[dataset].values()
    for column in df.columns:
        df[column] = df[column].apply(lambda x: float(x.replace(",", ".")))
    n_above += (df["weavenn_AMI"] > df["knnl_AMI"]).sum()
    n_total += df.shape[0]
print(n_above / n_total)
