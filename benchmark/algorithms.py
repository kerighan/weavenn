from weavenn import WeaveNN
from hdbscan import HDBSCAN
from DBSCANPP import DBSCANPP
from sklearn.cluster import DBSCAN, OPTICS, AffinityPropagation


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
    elif name == "ROCK":
        return predict_rock(X)


def predict_affinity(X):
    return AffinityPropagation().fit_predict(X)


def predict_weavenn(X):
    return WeaveNN(verbose=True).fit_predict(X)


def predict_hdbscan(X):
    return HDBSCAN().fit_predict(X)


def predict_dbscan(X):
    D = X.shape[1]
    return DBSCAN(min_samples=2*D).fit_predict(X)


def predict_optics(X):
    return OPTICS(min_samples=2, metric="euclidean").fit_predict(X)


def predict_dbscanpp(X):
    try:
        return DBSCANPP(
            p=0.1, eps_density=5.0,
            eps_clustering=5.0, minPts=10
        ).fit_predict(X, init="k-centers", cluster_outliers=True)
    except ValueError:
        return None


def predict_knnl(X):
    from weavenn.ann import get_hnswlib_nns_function
    from weavenn.weavenn import get_louvain_communities
    import networkx as nx

    get_nns = get_hnswlib_nns_function("l2")
    labels, dists = get_nns(X, k=40)
    
    edges = set()
    for i, row in enumerate(labels):
        for j in row:
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            edges.add((i, j))
    G = nx.Graph()
    G.add_nodes_from(range(len(X)))
    G.add_edges_from(edges)
    return get_louvain_communities(G)
