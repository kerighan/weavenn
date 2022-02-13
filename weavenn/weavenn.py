import networkx as nx
import numpy as np
import time

from .ann import get_ann_algorithm


class WeaveNN:
    def __init__(
        self,
        k_max=100,
        min_sim=.001,
        max_sim=.99,
        kernel="tanh",
        ann_algorithm="hnswlib",
        clustering_algorithm="louvain",
        metric="l2",
        threshold=.05,
        centrifugal=False,
        verbose=False
    ):
        self.k_max = k_max
        self.min_sim = min_sim
        self.max_sim = max_sim
        self.threshold = threshold
        self._get_similarities = get_kernel(kernel)
        self._get_nns = get_ann_algorithm(ann_algorithm, metric)
        self._clustering = get_clustering_algorithm(clustering_algorithm)
        self._compare = get_compare(centrifugal)
        self.verbose = verbose

    def _info(self, msg, elapsed_time):
        if self.verbose:
            print(f"[*] {msg} - T={elapsed_time:.2f}")

    def fit_predict(self, X, resolution=1., random_state=123):
        G = self.fit_transform(X)
        return self.predict(
            G,
            resolution=resolution,
            random_state=random_state)

    def predict(self, G, resolution=1., random_state=123):
        start_time = time.time()
        res = self._clustering(
            G, resolution=resolution, random_state=random_state)
        elapsed_time = time.time() - start_time
        self._info("detect communities", elapsed_time)
        return res

    def fit_transform(self, X):
        # get k-nearest neighbors
        start_time = time.time()
        labels, distances = self._get_nns(X, min(len(X), self.k_max))
        elapsed_time = time.time() - start_time
        self._info("find k-nearest neighbors", elapsed_time)

        # get candidates among all edges
        start_time = time.time()
        candidates, density = self._get_candidates(labels, distances)
        elapsed_time = time.time() - start_time
        self._info("extract candidates edges", elapsed_time)

        # elect edges among the candidates
        start_time = time.time()
        edges = self._get_edges_from_candidates(candidates, density)
        # create networkx graph
        G = nx.Graph()
        G.add_nodes_from(range(len(X)))
        G.add_weighted_edges_from(edges)
        elapsed_time = time.time() - start_time
        self._info("create graph", elapsed_time)
        return G

    def _get_candidates(self, labels, distances):
        # get minimum and maximum similarity
        min_sim, max_sim = self.min_sim, self.max_sim

        # scaling factor to normalize distances
        # and avoid exploding values
        scale = 1 / (1 + np.max(distances))

        # find candidates among possible edges
        density = {}  # density
        candidates = []  # candidate edges
        for i, (nns, dists) in enumerate(zip(labels, distances)):
            dists *= scale
            _sims, _density = self._get_similarities(dists, min_sim, max_sim)
            density[i] = _density
            for j, _similarity in zip(nns, _sims):
                if i == j:  # avoid self loops
                    continue
                candidates.append((i, j, _similarity))
        return candidates, density

    def _get_edges_from_candidates(self, candidates, density):
        _compare = self._compare

        edges = []
        for i, j, sim in candidates:
            density_i = density[i]
            density_j = density[j]

            # weight similarity by density
            sim *= 2 * density_j / (density_i + density_j)

            # if similarity is too small, ignore candidate
            if sim < self.threshold:
                continue

            # only allow centrifugal or centripete edges
            if _compare(density_i, density_j):
                continue
            edges.append((i, j, sim))
        return edges


# =============================================================================
# Kernel functions
# =============================================================================


def get_kernel(kernel):
    if kernel == "tanh":
        return get_tanh_similarities
    elif kernel == "exp":
        return get_exp_similarities
    else:
        raise ValueError(f"Kernel {kernel} not found")


def get_tanh_similarities(dists, min_sim, max_sim):
    d_k = dists[-1]
    d_1 = dists[1]
    if d_k == d_1:  # only return ones
        return dists**0, 10

    # avoid exploding values
    d_k = max(d_k, 1e-12)
    d_1 = max(d_1, 1e-12)

    # compute density factor b
    numerator = np.log(np.arctanh(1 - max_sim) /
                       np.arctanh(1 - min_sim))
    denominator = np.log(d_1 / d_k)
    b = numerator / denominator

    if d_k**b == 0:
        return dists**0, b

    # compute scaling factor a
    a = np.arctanh(1 - min_sim) / (d_k**b)

    # return similarities and density factor
    similarities = 1 - np.tanh(a*dists**b)
    return similarities, b


def get_exp_similarities(dists, min_sim, max_sim):
    d_k = max(dists[-1], 1e-12)
    d_1 = max(dists[1], 1e-12)

    # compute density factor b
    log_m = np.log(min_sim)
    log_M = np.log(max_sim)
    numerator = np.log(log_m / log_M)
    b = numerator / np.log(d_k / d_1)

    # compute scaling factor a
    a = -log_m/(d_k**b)

    # return similarities and density factor
    similarities = np.exp(-a*dists**b)
    return similarities, b


# =============================================================================
# Comparison functions
# =============================================================================


def get_compare(centrifugal):
    if centrifugal:
        return is_centrifugal
    return is_centripete


def is_centrifugal(i, j):
    return j > i


def is_centripete(i, j):
    return i > j


# =============================================================================
# Clustering functions
# =============================================================================


def get_clustering_algorithm(algorithm):
    if algorithm == "louvain":
        return get_louvain_communities
    elif algorithm == "combo":
        return get_pycombo_communities


def get_louvain_communities(G, resolution=1., random_state=123, **kwargs):
    n = len(G.nodes)
    try:
        from cylouvain import best_partition
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


def get_pycombo_communities(G, **kwargs):
    import pycombo

    n = len(G.nodes)
    node_to_com, _ = pycombo.execute(G)
    coms = np.zeros(n, dtype=int)
    for node, com in node_to_com.items():
        coms[node] = com
    return coms
