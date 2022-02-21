from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, OPTICS, AffinityPropagation
from weavenn import WeaveNN

algorithms = [
    "weavenn", "knn-L", "OPTICS", "DBSCAN", "HDBSCAN", "AffinityPropagation"
]


def predict(name, X):
    if name == "weavenn":
        return predict_weavenn(X)
    elif name == "knn-L":
        return predict_knnl(X)
    elif name == "AffinityPropagation":
        return predict_affinity(X)
    elif name == "OPTICS":
        return predict_optics(X)
    elif name == "DBSCAN":
        return predict_dbscan(X)
    elif name == "DBSCAN++":
        return predict_dbscanpp(X)
    elif name == "HDBSCAN":
        return predict_hdbscan(X)


def predict_affinity(X):
    return AffinityPropagation().fit_predict(X)


def predict_weavenn(X):
    return WeaveNN(verbose=False, k=70).fit_predict(X)


def predict_hdbscan(X):
    return HDBSCAN().fit_predict(X)


def predict_dbscan(X):
    D = X.shape[1]
    return DBSCAN(min_samples=2*D).fit_predict(X)


def predict_optics(X):
    return OPTICS(min_samples=2, metric="euclidean").fit_predict(X)


def predict_dbscanpp(X):
    from DBSCANPP import DBSCANPP
    try:
        return DBSCANPP(
            p=0.1, eps_density=5.0,
            eps_clustering=5.0, minPts=10
        ).fit_predict(X, init="k-centers", cluster_outliers=True)
    except ValueError:
        return None


def predict_knnl(X):
    from weavenn.weavenn import predict_knnl as pknnl
    return pknnl(X, k=30)
