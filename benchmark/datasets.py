from sklearn.datasets import load_wine, load_iris, fetch_openml, fetch_rcv1
from sklearn.preprocessing import robust_scale, LabelEncoder, minmax_scale
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np


# =============================================================================
# Classification datasets
# =============================================================================


classification_datasets = [
    "iris", "wine", "mobile", "zoo", "seeds", "heart",
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
    elif name == "zoo":
        return load_zoo()
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
    elif name == "heart":
        return load_heart()
    elif name == "mall":
        return load_mall()


def load_heart():
    df = pd.read_csv("datasets/heart.csv")
    X = robust_scale(df.iloc[:, :-1].values)
    y = df["target"].values
    return X, y


def load_zoo():
    df = pd.concat([
        pd.read_csv("datasets/zoo.csv"),
        pd.read_csv("datasets/zoo2.csv"),
        pd.read_csv("datasets/zoo3.csv")])
    y = df["class_type"].values
    X = minmax_scale(df.iloc[:, 1:-1].values)
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


def load_mnist():
    from sklearn.decomposition import PCA
    mnist = fetch_openml("mnist_784", version=1)
    X = mnist.data
    y = mnist.target.astype(int)
    X = PCA(n_components=20).fit_transform(X)
    return X, y


def load_fashion():
    from sklearn.decomposition import PCA
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
    # index = np.arange(0, 100000)
    # np.random.shuffle(index)
    # print(index)
    # index = index[:30000]

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
        ("reducer", TruncatedSVD(n_components=200))
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
        ("reducer", TruncatedSVD(n_components=200))
    ])
    X = pipe.fit_transform(data.data[:50000])
    return X


if __name__ == "__main__":
    print(load_reuters())
