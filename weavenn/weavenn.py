import networkx as nx
import numpy as np
from numba import njit
from .louvain import get_partitions_from_graph
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
import logging


# Configure the logging module
logging.basicConfig(level=logging.INFO)  # Set the logging level to INFO


class WeaveNN:
    def __init__(
        self,
        k=None,
        kernel="exp",
        metric="l2",
        reduction_dim=3,
        min_cluster_size=2,
        max_n_clusters=None,
        score="silhouette",
        sample_size=3000,
        use_modularity=False,
        use_global_scale=False,
        threshold=.1,
        scale_free_factor=2,
        scale_free_gamma=2.5,
        verbose=False
    ):
        self.k = k
        self.kernel = kernel
        self.metric = metric
        self.reduction_dim = reduction_dim
        self.min_cluster_size = min_cluster_size
        self.max_n_clusters = max_n_clusters
        self.score = score
        self.sample_size = sample_size
        self.use_modularity = use_modularity
        self.threshold = threshold
        self.use_global_scale = use_global_scale
        # scale free parameters of the graph
        self.scale_free_factor = scale_free_factor
        self.scale_free_gamma = scale_free_gamma
        self.verbose = verbose

    def get_knns(self, X):
        import hnswlib

        n, dim = X.shape

        index = hnswlib.Index(space=self.metric, dim=dim)
        index.init_index(
            max_elements=n, ef_construction=3 * self.k,
            M=250, random_seed=100)
        index.add_items(X, range(n))

        labels, distances = index.knn_query(X, k=self.k)
        return labels, distances

    def get_reduced_distances(self):
        labels, distances = self._labels, self._distances
        distances = (distances / self._sigma[:, None]) ** self._beta
        return labels, distances

    def fit_transform(self, X, return_labels=False):
        # infer number of nearest neighbors if not specified
        if self.k is None:
            # approximate number of edges in a scale-free graph
            # in order to find the optimal k
            N = X.shape[0]
            m = N**(1 + 1 / self.scale_free_gamma)
            self.k = int(round(self.scale_free_factor * m / N))

            if self.verbose:
                logging.info(f"k={self.k}")

        # compute k-nearest-neighbors
        labels, distances = self.get_knns(X)
        self._labels = labels
        self._distances = distances

        # dimensionality fingerprinting
        dim = self.infer_dimensionality(distances)
        if self.verbose:
            logging.info(f"empirical_dim={dim:.2f}")

        # dimensionality reduction exponent
        if self.reduction_dim is not None:
            beta = dim / self.reduction_dim
        else:
            beta = 1
        beta = 1e-3 if beta == 0 else beta
        self._beta = beta

        # avoid null distances
        if not self.use_global_scale:
            sigma = distances[:, -1].clip(1e-9)
        else:
            sigma = np.array([self._global_scaling] * len(X))
        self._sigma = sigma

        # get edges from k-nearest neighbors
        edges = self.get_edges(labels, distances, sigma, beta)
        if len(edges) == 0:
            return [-1] * len(X)

        # change edges labels to integers (numba changes them to floats)
        edges = [(int(u), int(v), w) for u, v, w in edges if not np.isnan(w)]

        # build networkx graph
        G = nx.Graph()
        G.add_nodes_from(range(len(X)))
        G.add_weighted_edges_from(edges)

        # recompute density after edge removal and edge reweighting
        self._density = self.compute_density(G)

        if return_labels:
            return G, labels
        return G

    def fit_predict(self, X, return_graph=False):
        G, labels = self.fit_transform(X, return_labels=True)
        # infer the best partition from score
        y = self.get_best_score(G, X, self.score, labels)
        y = relabel(y, self.min_cluster_size)
        if return_graph:
            return G, y
        return y

    def compute_cluster_dispersion(self, X, labels):
        unique_labels = np.unique(labels)
        dispersion = []
        dispersion_keys = []

        for label in unique_labels:
            cluster_points = X[labels == label]
            variances = np.var(cluster_points, axis=0)
            dispersion.append(np.mean(variances))
            dispersion_keys.append(label)

        # normalize dispersion
        dispersion = np.array(dispersion)
        dispersion = (dispersion - dispersion.mean()) / dispersion.std()
        dispersion = dict(zip(dispersion_keys, dispersion))
        return dispersion

    def compute_density(self, G):
        density = np.array([G.degree(node, "weight") / self.k
                            for node in range(len(G.nodes))])
        return density

    def get_best_score(self, G, X, score, labels):
        if score == "calinski_harabasz":
            scoring = calinski_harabasz_score
        elif score == "silhouette":
            def scoring(X, y):
                return silhouette_score(X, y, sample_size=self.sample_size)
        elif score == "davies_bouldin":
            scoring = davies_bouldin_score

        best_score = -float("inf")
        best_y = None
        for partition in get_partitions_from_graph(G):
            y, Q, _ = partition
            try:
                if self.use_modularity:
                    score = Q
                else:
                    score = scoring(X, y)
            except ValueError:
                continue
            # penalize if number of clusters too high
            if self.max_n_clusters is not None:
                n_clusters = len(set(y))
                if n_clusters > self.max_n_clusters:
                    score *= 1e-6
            # keep the best score
            if score >= best_score:
                best_score = score
                best_y = y
        return best_y

    def infer_dimensionality(self, distances):
        dist_median = np.median(distances, axis=0)
        self._global_scaling = dist_median[-1].clip(1e-9)

        # infer dimensionality
        dist_median /= self._global_scaling
        area = dist_median.sum()
        dim = area / (self.k - area)
        return dim

    def get_edges(self, labels, distances, sigma, beta):
        if self.kernel == "exp":
            scale = -np.log(np.log2(self.k) / self.k)
            return get_edges_exp(
                labels, distances, sigma, beta, scale, self.threshold)
        elif self.kernel == "tanh":
            scale = np.arctanh(1 - np.log2(self.k) / self.k)
            return get_edges_tanh(
                labels, distances, sigma, beta, scale, self.threshold)


def get_func(c, k):
    dists = np.linspace(0, 1, k)**.33

    def evaluate(alpha):
        res = k - np.sum(np.tanh(alpha * dists))
        loss = (c - res)**2
        return loss
    return evaluate


def get_scale(c, k):
    from scipy.optimize import minimize
    return minimize(get_func(c, k), x0=1).x[0]


@njit
def isolation_score(labels, y, density):
    res = 0
    for i, neighbors in enumerate(labels):
        y_src = y[i]
        num = 0
        denom = 0
        for neighbor in neighbors:
            if neighbor == i:
                continue
            d_neighbor = density[neighbor]
            denom += d_neighbor
            num += (y_src == y[neighbor]) * d_neighbor
        res += num / denom
    res /= len(labels)
    return res


@njit
def compute_power(a, b):
    return np.exp(b * np.log(a + 1e-6))


@njit
def get_edges_exp(labels, distances, sigma, beta, scale, threshold):
    edges = []
    visited = set()
    for i, (neighbors, dists_i, sigma_i) in enumerate(
            zip(labels, distances, sigma)):

        # transform distance in the right dimensionality
        squashed_i = compute_power(dists_i / sigma_i, beta)
        sims = []
        for (j, dist, d_i) in zip(neighbors, dists_i, squashed_i):
            if j == i:
                continue
            if i < j:
                edge = (i, j)
            else:
                edge = (j, i)
            if dist == 0:
                edges.append((*edge, 1.))
            if edge in visited:
                continue

            sigma_j = sigma[j]
            # transform distance in the right dimensionality
            d_j = compute_power(dist / sigma_j, beta)

            # combine the two distances
            d_ij = d_i * d_j

            sim = np.exp(-scale * d_ij)
            sims.append(sim)
            if sim < threshold:
                continue
            edges.append((*edge, sim))
            visited.add(edge)
    return edges


@njit
def get_edges_tanh(labels, distances, sigma, beta, scale, threshold):
    edges = []
    visited = set()
    for i, (neighbors, dists_i, sigma_i) in enumerate(
            zip(labels, distances, sigma)):

        # transform distance in the right dimensionality
        squashed_i = compute_power(dists_i / sigma_i, beta)
        sims = []
        for (j, dist, d_i) in zip(neighbors, dists_i, squashed_i):
            if j == i:
                continue
            if i < j:
                edge = (i, j)
            else:
                edge = (j, i)
            if dist == 0:
                edges.append((*edge, 1))
            if edge in visited:
                continue

            sigma_j = sigma[j]
            # transform distance in the right dimensionality
            d_j = compute_power(dist / sigma_j, beta)

            # combine the two distances
            d_ij = d_i * d_j

            sim = 1 - np.tanh(scale * d_ij)
            sims.append(sim)
            if sim < threshold:
                continue
            edges.append((*edge, sim))
            visited.add(edge)
    return edges


def relabel(y, min_cluster_size):
    cm2nodes = {}
    for node, cm in enumerate(y):
        cm2nodes.setdefault(cm, []).append(node)
    cm2nodes = sorted(cm2nodes.items(), key=lambda x: len(x[1]), reverse=True)
    y_new = np.ones(shape=len(y), dtype=int) * -1
    index = 0
    for cm, nodes in cm2nodes:
        if cm == -1:
            continue
        if len(nodes) < min_cluster_size:
            break
        for node in nodes:
            y_new[node] = index
        index += 1
    return y_new
