import time

import networkx as nx
import numpy as np
from _weavenn import get_graph, get_partitions

from weavenn.score import get_scoring_function

from .ann import get_ann_algorithm


class WeaveNN:
    def __init__(
        self,
        k=50,
        ann_algorithm="hnswlib",
        method="auto",
        score="calinski_harabasz",
        prune=False,
        metric="l2",
        min_sim=.01,
        min_sc=None,
        z_modularity=False,
        verbose=False
    ):
        self.k = k
        self._get_nns = get_ann_algorithm(ann_algorithm, metric)
        self.method = method
        self.prune = prune
        self.min_sim = min_sim
        self.min_sc = min_sc
        self.verbose = verbose
        self.score = score
        self.z_modularity = z_modularity

    def fit_predict(self, X, resolution=1.):
        labels, distances = self._get_nns(X, min(len(X), self.k))
        if self.verbose:
            print("[*] Computed nearest neighbors")

        n_nodes, dim = X.shape
        local_scaling = np.array(distances[:, -1])

        method = "louvain"
        if self.method == "auto":
            if dim == 2:
                method = "optimal"
        elif self.method == "optimal":
            method = "optimal"

        if method == "louvain":
            partitions, sigma_count, _, _ = get_partitions(
                labels, distances, local_scaling,
                self.min_sim, resolution,
                self.prune, False, self.z_modularity)
            self._sigma_count = sigma_count

            y, _ = extract_partition(partitions, n_nodes, 1)

            if self.min_sc is None:
                min_sc = None
            else:
                min_sc = np.percentile(
                    sigma_count, int(round(self.min_sc * 100)))
            return relabel(
                y, sigma_count=sigma_count, min_sc=min_sc)
        else:
            partitions, sigma_count, _, _ = get_partitions(
                labels, distances, local_scaling,
                self.min_sim, resolution,
                self.prune, True, self.z_modularity)
            self._sigma_count = sigma_count
            # G = self.graph_from_neighbors(graph_neighbors, graph_weights)
            # from cdlib import evaluation
            # from community import best_partition
            # y_pred = best_partition(G)
            # ave = evaluation.avg_embeddedness(G, y_pred)
            # print(ave)

            # return [y_pred[i] for i in range(len(G.nodes))]

            y = extract_optimal_partition(
                X, partitions, n_nodes, self.score,
                labels, distances, sigma_count)

            if self.min_sc is None:
                min_sc = None
            else:
                min_sc = np.percentile(
                    sigma_count, int(round(self.min_sc * 100)))
            return relabel(
                y, sigma_count=sigma_count, min_sc=min_sc)

    def fit_transform(self, X):
        import networkx as nx
        labels, distances = self._get_nns(X, min(len(X), self.k))
        local_scaling = np.array(distances[:, -1])
        # get adjacency list
        graph_neighbors, graph_weights, _ = get_graph(
            labels, distances, local_scaling, self.min_sim)
        # build networkx graph

        return self.graph_from_neighbors(graph_neighbors, graph_weights)

    def graph_from_neighbors(self, graph_neighbors, graph_weights):
        visited = set()
        G = nx.Graph()
        for i in range(len(graph_neighbors)):
            G.add_node(i)
            neighbors = graph_neighbors[i]
            weights = graph_weights[i]
            for index in range(len(neighbors)):
                j = neighbors[index]
                pair = (i, j) if i < j else (j, i)
                if pair in visited:
                    continue
                visited.add(pair)

                G.add_edge(i, j, weight=weights[index])
        return G


# =============================================================================
# Baseline model
# =============================================================================


def predict_knnl(X, k=100):
    from cylouvain import best_partition

    ann = get_ann_algorithm("hnswlib", "l2")
    labels, _ = ann(X, k)

    n_nodes = X.shape[0]
    visited = set()
    edges = []
    for i, row in enumerate(labels):
        for j in row:
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            if pair in visited:
                continue
            visited.add(pair)
            edges.append((i, j))
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(edges)
    partition = best_partition(G)
    y = np.zeros(n_nodes, dtype=int)
    for i, val in partition.items():
        y[i] = val
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


def extract_optimal_partition(
        X, partitions, n_nodes, scoring, labels, distances, sigma_count):
    scoring = get_scoring_function(scoring)
    best_score = -float("inf")
    best_y = None
    for level in range(1, len(partitions) + 1):
        y, Q = extract_partition(partitions, n_nodes, level)
        try:
            score = scoring(X, y, Q, labels, distances, sigma_count)
        except ValueError:
            score = -float("inf")
        if score >= best_score:
            best_y = y.copy()
            best_score = score
    return best_y


def relabel(partition, sigma_count=None, min_sc=None):
    if min_sc is not None:
        for i, sc in enumerate(sigma_count):
            if sc < min_sc:
                partition[i] = -1

    cm_to_nodes = {}
    for node, com in enumerate(partition):
        cm_to_nodes.setdefault(com, []).append(node)

    cm_to_nodes = sorted(cm_to_nodes.items(),
                         key=lambda x: len(x[1]), reverse=True)

    index = 0
    for cm, nodes in cm_to_nodes:
        if cm == -1:
            for node in nodes:
                partition[node] = -1
        else:
            for node in nodes:
                partition[node] = index
            index += 1
    return partition
