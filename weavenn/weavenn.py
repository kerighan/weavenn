import time

import networkx as nx
import numpy as np

from .ann import get_ann_algorithm


class WeaveNN:
    def __init__(
        self,
        k=100,
        ann_algorithm="hnswlib",
        method="louvain",
        prune=False,
        metric="l2",
        min_sim=.01,
        verbose=False
    ):
        self.k = k
        self._get_nns = get_ann_algorithm(ann_algorithm, metric)
        if method == "louvain":
            self._clustering = cluster_louvain
        else:
            self._clustering = cluster_louvain_opti
        self.prune = prune
        self.min_sim = min_sim
        self.verbose = verbose

    def fit_predict(self, X, resolution=1.):
        G = self.fit_transform(X)
        return self.predict(
            G, X,
            resolution=resolution)

    def predict(self, G, X=None, resolution=1.):
        res = self._clustering(G, X=X, prune=self.prune)
        return res

    def fit_transform(self, X):
        labels, distances = self._get_nns(X, min(len(X), self.k))
        G = self._build_graph(labels, distances)
        return G

    def _build_graph(self, labels, distances):
        candidates = set()
        local_scaling = distances[:, -1]
        edges = []
        for i, (neighbors, dists, node_scaling) in enumerate(
                zip(labels, distances, local_scaling)):

            # get node scalings
            node_scaling = node_scaling.clip(1e-6, None)
            neighbor_scaling = local_scaling[neighbors].clip(1e-6, None)
            # normalize distances
            dists = dists**2/(node_scaling * neighbor_scaling)

            nns_i = set(neighbors)
            for index, j in enumerate(neighbors):
                if i == j:
                    continue
                pair = (i, j) if i < j else (j, i)
                if pair in candidates:
                    continue

                nns_j = set(labels[j])
                nn_count = len(nns_i.intersection(nns_j)) + 1

                candidates.add(pair)
                weight = 1 - np.tanh(dists[index]*np.log(nn_count))
                if weight < self.min_sim:
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

def cluster_louvain_opti(G, X=None, prune=False, resolution=1):
    from louvaincpp import metric_louvain
    return metric_louvain(G, X=X, resolution=resolution, prune=prune)


def cluster_louvain(G, X=None, prune=False, resolution=1):
    from louvaincpp import louvain
    partition = louvain(G, resolution=resolution, prune=prune)
    y = np.zeros(len(G.nodes))
    for i, v in partition.items():
        y[i] = v
    return y


# =============================================================================
# Baseline model
# =============================================================================


def predict_knnl(X, k=100):
    ann = get_ann_algorithm("hnswlib", "l2")
    clusterer = cluster_louvain

    labels, _ = ann(X, k)
    edges = set()
    for i, row in enumerate(labels):
        for j in row:
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            edges.add(pair)

    G = nx.Graph()
    G.add_nodes_from(range(len(X)))
    G.add_edges_from(edges)
    return clusterer(G)


def score(y, y_pred):
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    adj_mutual_info = adjusted_mutual_info_score(y, y_pred)
    adj_rand = adjusted_rand_score(y, y_pred)
    return adj_mutual_info, adj_rand
