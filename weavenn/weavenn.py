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
        method="mch",
        prune=False,
        metric="l2",
        min_sim=.25,
        min_sc=None,
        reduce_dim=3,
        max_clusters=None,
        use_percentile=False,
        verbose=False
    ):
        self.k = k
        self._get_nns = get_ann_algorithm(ann_algorithm, metric)
        self.method = method
        self.prune = prune
        self.min_sim = min_sim
        self.min_sc = min_sc if min_sc else 0
        self.use_percentile = use_percentile
        self.verbose = verbose
        self.max_clusters = max_clusters
        self.reduce_dim = reduce_dim

    def infer_min_sc(self, sigma_count):
        min_sc = self.min_sc
        if self.use_percentile:
            min_sc = np.percentile(sigma_count, int(round(min_sc * 100)))
        return min_sc

    def fit_predict(self, X, resolution=1.):
        import matplotlib.pyplot as plt

        labels, distances = self._get_nns(X, min(len(X), self.k))
        if self.verbose:
            print("[*] Computed nearest neighbors")

        avg_dist = distances.mean(axis=0)
        avg_dist /= avg_dist.max()
        I = avg_dist.sum()
        dim = I / (self.k - I)
        # print(f"dim={dim}")
        beta = dim / self.reduce_dim
        # plt.plot(range(self.k), avg_dist)
        # plt.plot(range(self.k), avg_dist**beta)
        # plt.show()

        n_nodes, _ = X.shape
        local_scaling = np.array(distances[:, -1])

        if self.method == "louvain":
            partitions, sigma_count, _, _ = get_partitions(
                labels, distances, local_scaling,
                self.min_sim, resolution,
                self.prune, False, beta, dim, self.min_sc, self.k)
            self._sigma_count = sigma_count

            y, _ = extract_partition(partitions, n_nodes, 1)
        else:
            partitions, sigma_count, _, _ = get_partitions(
                labels, distances, local_scaling,
                self.min_sim, resolution,
                self.prune, True, beta, dim, self.min_sc, self.k)
            self._sigma_count = sigma_count

            y = extract_partition_from_score(
                X, partitions, n_nodes, self.method,
                labels, distances, sigma_count,
                max_clusters=self.max_clusters)

        # order communities and infer outliers
        return relabel(X, y)

    def fit_transform(self, X):
        import networkx as nx
        labels, distances = self._get_nns(X, min(len(X), self.k))

        avg_dist = distances.mean(axis=0)
        avg_dist /= avg_dist.max()
        I = avg_dist.sum()
        dim = I / (self.k - I)
        beta = dim / self.reduce_dim

        local_scaling = np.array(distances[:, -1])
        # get adjacency list
        graph_neighbors, graph_weights, sigma_count = get_graph(
            labels, distances, local_scaling, self.min_sim, beta, dim)
        self._sigma_count = sigma_count
        # build networkx graph
        return self.graph_from_neighbors(graph_neighbors, graph_weights)

    def fit_reduce(
        self, D, n_components=2, walk_len=50, n_walks=25, corruption=.66,
        batch_size=400, epochs=10, b=None, init="spectral"
    ):
        from .reduce import reduce
        if isinstance(D, np.ndarray):
            G = self.fit_transform(D)
            return reduce(G, n_components, walk_len=walk_len, n_walks=n_walks,
                          corruption=corruption, batch_size=batch_size, epochs=epochs, b=b, init=init)
        else:
            return reduce(D, n_components, walk_len=walk_len, n_walks=n_walks,
                          corruption=corruption, batch_size=batch_size, epochs=epochs, b=b, init=init)

    def graph_from_neighbors(self, graph_neighbors, graph_weights):
        visited = set()
        G = nx.Graph()
        G.add_nodes_from(range(len(graph_neighbors)))
        for i in range(len(graph_neighbors)):
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
# Utilities functions
# =============================================================================


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


def extract_partition_from_score(
        X, partitions, n_nodes, scoring,
        labels, distances, sigma_count, max_clusters=None):
    scoring = get_scoring_function(scoring)
    best_score = -float("inf")
    best_y = None
    for level in range(1, len(partitions) + 1):
        y, Q = extract_partition(partitions, n_nodes, level)
        score = scoring(X, y, Q, labels, distances, sigma_count)
        if max_clusters is not None:
            n_clusters = np.unique(y).shape[0]
            if n_clusters > max_clusters:
                score *= 1e-3

        if score >= best_score:
            best_y = y.copy()
            best_score = score
    return best_y


def relabel(
    X, partition, group_duplicates=True, tol=4
):
    if group_duplicates:
        # assign same labels for same embedding
        X_round = X.round(decimals=tol)
        unq, count = np.unique(X_round, axis=0, return_counts=True)
        repeated_groups = unq[count > 1]
        for group in repeated_groups:
            idx = np.argwhere(np.all(X_round == group, axis=1)).ravel()
            # find first non-negative community
            for i in idx:
                cm = partition[i]
                if cm != -1:
                    break
            # set partition to all items
            for i in idx:
                partition[i] = cm

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
            if len(nodes) == 1:
                partition[nodes[0]] = -1
            else:
                for node in nodes:
                    partition[node] = index
                index += 1
    return partition


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


def lpa(G, sc):
    from collections import Counter
    from random import choice
    n_nodes = len(G.nodes)
    # sc = np.sort(sc)
    sc = np.array(sc)
    nodes = np.argsort(sc)
    # nodes = nodes[::-1]

    cm = np.arange(n_nodes)

    # one pass
    n_changes = -1
    while n_changes != 0:
        n_changes = 0
        np.random.shuffle(nodes)
        for node in nodes:
            current_cm = cm[node]
            node_sc = sc[node]

            count = Counter()

            nbs = set(G.neighbors(node))
            for neighbor in nbs:
                if sc[neighbor] >= node_sc:
                    continue

                target = set(G.neighbors(neighbor))
                common_neighbors = len(nbs.intersection(target))

                nb_cm = cm[neighbor]
                count[nb_cm] += sc[neighbor] * common_neighbors

            if len(count) == 0:
                continue

            _, max_count = count.most_common()[0]
            candidates = []
            for candidate_cm, c in count.most_common():
                if c < max_count:
                    break
                candidates.append(candidate_cm)
            new_cm = choice(candidates)
            if new_cm == current_cm:
                continue

            n_changes += 1
            cm[node] = new_cm
    return cm
