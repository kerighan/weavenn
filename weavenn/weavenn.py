import time

import networkx as nx
import numpy as np
from _weavenn import get_graph, get_partitions

from .ann import get_ann_algorithm


class WeaveNN:
    def __init__(
        self,
        k=100,
        ann_algorithm="hnswlib",
        method="louvain",
        score="davies_bouldin",
        prune=False,
        metric="l2",
        min_sim=.01,
        min_cluster_size=2,
        z_modularity=False,
        verbose=False
    ):
        self.k = k
        self._get_nns = get_ann_algorithm(ann_algorithm, metric)
        self.method = method
        self.prune = prune
        self.min_sim = min_sim
        self.verbose = verbose
        self.score = score
        self.min_cluster_size = min_cluster_size
        self.z_modularity = z_modularity

    def fit_predict(self, X, resolution=1.):
        labels, distances = self._get_nns(X, min(len(X), self.k))
        if self.verbose:
            print("[*] Computed nearest neighbors")

        n_nodes = X.shape[0]
        local_scaling = np.array(distances[:, -1])

        if self.method == "louvain":
            partitions = get_partitions(labels, distances, local_scaling,
                                        self.min_sim, resolution,
                                        self.prune, False, self.z_modularity)
            y, _ = extract_partition(partitions, n_nodes, 1)
            return relabel(y, min_cluster_size=min_cluster_size)
        else:
            if self.score == "modularity":
                def scoring(X, y, Q):
                    return Q
            elif self.score == "davies_bouldin":
                from sklearn.metrics import davies_bouldin_score

                def scoring(X, y, Q):
                    return -davies_bouldin_score(X, y)
            elif self.score == "silhouette":
                from sklearn.metrics import silhouette_score

                def scoring(X, y, Q):
                    return silhouette_score(X, y)
            elif self.score == "calinski_harabasz":
                from sklearn.metrics import calinski_harabasz_score

                def scoring(X, y, Q):
                    return calinski_harabasz_score(X, y)

            partitions = get_partitions(labels, distances, local_scaling,
                                        self.min_sim, resolution,
                                        self.prune, True, self.z_modularity)

            best_score = -float("inf")
            best_y = None
            last_score = -float("inf")
            for level in range(1, len(partitions) + 1):
                y, Q = extract_partition(partitions, n_nodes, level)

                try:
                    score = scoring(X, y, Q)
                except ValueError:
                    score = -float("inf")
                if score >= best_score:
                    best_y = y.copy()
                    best_score = score
                last_score = score
            return relabel(best_y, min_cluster_size=min_cluster_size)

    def fit_transform(self, X):
        import networkx as nx
        labels, distances = self._get_nns(X, min(len(X), self.k))
        local_scaling = np.array(distances[:, -1])
        # get adjacency list
        graph_neighbors, graph_weights = get_graph(
            labels, distances, local_scaling, self.min_sim)
        # build networkx graph
        G = nx.Graph()
        for i in range(len(graph_neighbors)):
            G.add_node(i)
            neighbors = graph_neighbors[i]
            weights = graph_weights[i]
            for index in range(len(neighbors)):
                j = neighbors[index]
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


def relabel(partition, min_cluster_size=2):
    cm_to_nodes = {}
    for node, com in enumerate(partition):
        cm_to_nodes.setdefault(com, []).append(node)

    cm_to_nodes = sorted(cm_to_nodes.items(),
                         key=lambda x: len(x[1]), reverse=True)
    for i, (_, nodes) in enumerate(cm_to_nodes):
        if len(nodes) < min_cluster_size:
            for node in nodes:
                partition[node] = -1
        else:
            for node in nodes:
                partition[node] = i
    return partition
