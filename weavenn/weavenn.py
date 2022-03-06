import time

import networkx as nx
import numpy as np

from .ann import get_ann_algorithm
from _weavenn import get_partitions, generate_dendrogram


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
        self.method = method
        self.prune = prune
        self.min_sim = min_sim
        self.verbose = verbose

    def fit_predict(self, X, resolution=1.):
        labels, distances = self._get_nns(X, min(len(X), self.k))

        n_nodes = X.shape[0]
        local_scaling = np.array(distances[:, -1])

        if self.method == "louvain":
            partitions = get_partitions(labels, distances, local_scaling,
                                        self.min_sim, resolution,
                                        self.prune, False)
            y, _ = extract_partition(partitions, n_nodes, 1)
            return y
        else:
            from sklearn.metrics import davies_bouldin_score as scoring
            partitions = get_partitions(labels, distances, local_scaling,
                                        self.min_sim, resolution,
                                        self.prune, True)

            best_score = -float("inf")
            best_y = None
            last_score = -float("inf")
            for level in range(1, len(partitions) + 1):
                y, Q = extract_partition(partitions, n_nodes, level)
                try:
                    score = -scoring(X, y)
                except ValueError:
                    score = -float("inf")
                score = Q
                if score >= best_score:
                    best_y = y.copy()
                    best_score = score
                last_score = score
            return best_y

    def fit_transform(self, X):
        labels, distances = self._get_nns(X, min(len(X), self.k))
        graph_neighbors, graph_weights = self._build_graph(labels, distances)
        return graph_neighbors, graph_weights


# =============================================================================
# Baseline model
# =============================================================================


def predict_knnl(X, k=100):
    from louvaincpp import louvain

    ann = get_ann_algorithm("hnswlib", "l2")
    labels, _ = ann(X, k)

    # edges = set()
    # for i, row in enumerate(labels):
    #     for j in row:
    #         if i == j:
    #             continue
    #         pair = (i, j) if i < j else (j, i)
    #         edges.add(pair)
    # G = nx.Graph()
    # G.add_nodes_from(range(len(X)))
    # G.add_edges_from(edges)
    # return louvain(G)
    n_nodes = X.shape[0]
    graph_neighbors = [[] for _ in range(n_nodes)]
    graph_weights = [[] for _ in range(n_nodes)]
    visited = set()
    for i, row in enumerate(labels):
        for j in row:
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            if pair in visited:
                continue
            visited.add(pair)

            graph_neighbors[i].append(j)
            graph_neighbors[j].append(i)
            graph_weights[i].append(1.)
            graph_weights[j].append(1.)
    partitions = generate_dendrogram(
        graph_neighbors, graph_weights, 1., False, False)
    y, _ = extract_partition(partitions, n_nodes, 1)
    return y


def score(y, y_pred):
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    adj_mutual_info = adjusted_mutual_info_score(y, y_pred)
    adj_rand = adjusted_rand_score(y, y_pred)
    return adj_mutual_info, adj_rand


def extract_partition(dendrogram, n_nodes, level):
    partitions, Q = dendrogram[-level]
    partition = range(len(partitions))
    for i in range(level, len(dendrogram) + 1):
        new_partition = np.zeros(n_nodes, dtype=int)
        partitions, _ = dendrogram[-i]
        for j in range(len(partitions)):
            new_partition[j] = partition[partitions[j]]
        partition = new_partition
    return partition, Q


def get_outliers(partition):
    from collections import defaultdict

    n_nodes = defaultdict(int)
    for node, com in partition.items():
        pass

