import numpy as np
import pandas as pd
from sklearn.datasets import (fetch_20newsgroups, fetch_openml, fetch_rcv1,
                              load_iris, load_wine)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, minmax_scale, robust_scale

# =============================================================================
# Classification datasets
# =============================================================================


classification_datasets = [
    "iris", "wine", "mobile", "glass", "zoo", "seeds", "letters", "phonemes",
    "20newsgroups", "stellar", "mnist", "fashion"
]


def load(name):
    if name == "iris":
        res = load_iris()
        return res.data, res.target
    elif name == "wine":
        res = load_wine()
        return res.data, res.target
    elif name == "mobile":
        return load_mobile()
    elif name == "glass":
        return load_glass()
    elif name == "zoo":
        return load_zoo()
    elif name == "letters":
        return load_letters()
    elif name == "phonemes":
        return load_phonemes()
    elif name == "seeds":
        return load_seeds()
    elif name == "mnist":
        return load_mnist()
    elif name == "fashion":
        return load_fashion()
    elif name == "stellar":
        return load_stellar()
    elif name == "20newsgroups":
        return load_20newsgroups()
    elif name == "reuters":
        return load_reuters()
    elif name == "mall":
        return load_mall()


def load_zoo():
    df = pd.concat([
        pd.read_csv("datasets/zoo.csv"),
        pd.read_csv("datasets/zoo2.csv"),
        pd.read_csv("datasets/zoo3.csv")])
    y = df["class_type"].values
    X = minmax_scale(df.iloc[:, 1:-1].values)
    return X, y


def load_letters():
    with open("datasets/letter-recognition.data") as f:
        data = f.read().splitlines()
    X = []
    y = []
    for line in data:
        splits = line.split(",")
        y.append(splits[0])
        tmp = []
        for s in splits[1:]:
            tmp.append(int(s))
        X.append(tmp)
    X = np.array(X)
    X = minmax_scale(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y


def load_phonemes():
    df = pd.read_csv("datasets/Phoneme Recognition.txt")
    le = LabelEncoder()
    y = le.fit_transform(df.g)
    del df["g"], df["row.names"], df["speaker"]
    X = df.values
    # X = PCA(n_components=20).fit_transform(X)
    # X = robust_scale(X)
    return X, y


def load_seeds():
    with open("datasets/seeds_dataset.txt") as f:
        data = f.read().splitlines()
    data = [[float(it) for it in line.split()] for line in data]
    data = np.array(data)
    X = robust_scale(data[:, :-1])
    y = data[:, -1].astype(int)
    return X, y


def load_mobile():
    df = pd.read_csv("datasets/mobile/train.csv")
    y = df["price_range"].values
    X = df.iloc[:, :-1].values
    return X, y


def load_glass():
    df = pd.read_csv("datasets/glass.csv")
    y = df["Type"].values
    X = df.iloc[:, :-1].values
    return X, y


def load_mnist():
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data
    y = mnist.target.astype(int)
    X = PCA(n_components=20).fit_transform(X)

    return X, y


def load_fashion():
    from keras.datasets import fashion_mnist
    (X, y), _ = fashion_mnist.load_data()
    X = X.astype(float) / 255.
    X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
    X = PCA(n_components=20).fit_transform(X)

    return X, y


def load_cifar10():
    from keras.datasets import cifar10
    (X, y), _ = cifar10.load_data()
    X = X.astype(float)
    print(X.shape)
    return X, y


def load_stellar():
    df = pd.read_csv("datasets/star_classification.csv")
    le = LabelEncoder()
    y = le.fit_transform(df["class"])
    del df["class"]
    X = robust_scale(df.values)

    return X, y


def load_20newsgroups():
    newsgroups_train = fetch_20newsgroups(subset='train')

    pipe = Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=100000)),
        ("reducer", TruncatedSVD(n_components=50))
    ])
    X = pipe.fit_transform(newsgroups_train.data)
    y = newsgroups_train.target
    return X, y


# =============================================================================
# Clustering datasets
# =============================================================================


clustering_datasets = ["mall"]


def load_mall():
    df = pd.read_csv("datasets/Mall_Customers.csv")
    le = LabelEncoder()
    gender = le.fit_transform(df["Gender"])[:, None]
    del df["CustomerID"]
    del df["Gender"]
    X = np.concatenate([gender, df.values], axis=-1)
    return robust_scale(X)


def load_reuters():
    data = fetch_rcv1()
    pipe = Pipeline([
        ("reducer", TruncatedSVD(n_components=50))
    ])
    X = pipe.fit_transform(data.data[:10000])
    return X


if __name__ == "__main__":
    print(load_phonemes())
