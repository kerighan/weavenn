import time

import networkx as nx
import numpy as np

from .ann import get_ann_algorithm


class WeaveNN:
    def __init__(
        self,
        k=100,
        ann_algorithm="hnswlib",
        clustering_algorithm="louvain",
        metric="l2",
        verbose=False
    ):
        self.k = k
        self._get_nns = get_ann_algorithm(ann_algorithm, metric)
        self._clustering = get_clustering_algorithm(clustering_algorithm)
        self.verbose = verbose

    def fit_predict(self, X, resolution=1., random_state=123):
        G = self.fit_transform(X)
        return self.predict(
            G,
            resolution=resolution,
            random_state=random_state)

    def predict(self, G, resolution=1., random_state=123):
        res = self._clustering(G,
                               resolution=resolution,
                               random_state=random_state)
        return res

    def fit_transform(self, X):
        labels, distances = self._get_nns(X, min(len(X), self.k))
        G = self._build_graph(labels, distances)
        return G

    def _build_graph(self, labels, distances):
        candidates = {}
        local_scaling = distances[:, -1]
        for i, (neighbors, dists) in enumerate(zip(labels, distances)):
            nns_i = set(neighbors)

            dists = (dists**2)/local_scaling[i]
            for index, j in enumerate(neighbors):
                if i == j:
                    continue
                pair = (i, j) if i < j else (j, i)
                if pair in candidates:
                    continue

                nns_j = set(labels[j])
                nn_count = np.log(len(nns_i.intersection(nns_j)) + 1)
                candidates[pair] = 1 - np.tanh(
                    dists[index] * nn_count / local_scaling[j])

        edges = []
        for (i, j), weight in candidates.items():
            if weight <= 0.0:
                continue
            edges.append((i, j, weight))

        # create graph
        G = nx.Graph()
        G.add_nodes_from(range(len(labels)))
        G.add_weighted_edges_from(edges)
        return G


# =============================================================================
# Clustering functions
# =============================================================================


def get_clustering_algorithm(algorithm):
    if algorithm == "louvain":
        return get_louvain_communities
    elif algorithm == "combo":
        return get_pycombo_communities
    elif algorithm == "lpa":
        return get_lpa_communities


def get_louvain_communities(G, resolution=1., random_state=123, **kwargs):
    n = len(G.nodes)
    try:
        from cylouvain import best_partition
        if nx.is_weighted(G):
            A = nx.adjacency_matrix(G).astype(float)
            node_to_com = best_partition(
                A, resolution=resolution)
        else:
            node_to_com = best_partition(
                G, resolution=resolution)

    except ImportError:
        from community import best_partition

        node_to_com = best_partition(
            G, resolution=resolution, random_state=random_state)
    coms = np.zeros(n, dtype=int)
    for node, com in node_to_com.items():
        coms[node] = com
    return coms


def get_lpa_communities(G, **kwargs):
    from networkx.algorithms.community.label_propagation import \
        asyn_lpa_communities

    n = len(G.nodes)
    node_to_com = list(asyn_lpa_communities(G, weight="weight"))
    coms = np.zeros(n, dtype=int)
    for i, nodes in enumerate(node_to_com):
        for node in nodes:
            coms[node] = i
    return coms


def get_pycombo_communities(G, **kwargs):
    import pycombo

    n = len(G.nodes)
    node_to_com, _ = pycombo.execute(G)
    coms = np.zeros(n, dtype=int)
    for node, com in node_to_com.items():
        coms[node] = com
    return coms


# =============================================================================
# Baseline model
# =============================================================================


def predict_knnl(X, k=100):
    ann = get_ann_algorithm("hnswlib", "l2")
    clusterer = get_clustering_algorithm("louvain")

    labels, _ = ann(X, k)
    edges = set()
    for i, row in enumerate(labels):
        for j in row:
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            edges.add(pair)

    G = nx.Graph()
    G.add_edges_from(edges)
    return clusterer(G)


def score(y, y_pred):
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    adj_mutual_info = adjusted_mutual_info_score(y, y_pred)
    adj_rand = adjusted_rand_score(y, y_pred)
    return adj_mutual_info, adj_rand
