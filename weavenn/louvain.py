import networkx as nx
import numpy as np
from _weavenn import generate_dendrogram


def get_partitions_from_graph(G):
    try:
        A = nx.to_scipy_sparse_matrix(G)
    except AttributeError:
        A = nx.to_scipy_sparse_array(G)
    data = A.data
    indices = A.indices
    indptr = A.indptr

    graph_neighbors = [[] for i in range(len(G.nodes))]
    graph_weights = [[] for i in range(len(G.nodes))]
    for i in range(len(G.nodes)):
        start, end = indptr[i:i + 2]
        for j, w in zip(indices[start:end], data[start:end]):
            graph_neighbors[i].append(j)
            graph_weights[i].append(w)
    return get_partitions_from_connectivity(graph_neighbors, graph_weights)


def get_partitions_from_connectivity(
    graph_neighbors, graph_weights, resolution=1.
):
    partitions = generate_dendrogram(
        graph_neighbors, graph_weights, resolution, False, True)
    return iter_partitions(partitions, len(graph_neighbors))


def iter_partitions(dendrogram, n_nodes):
    for level in range(len(dendrogram)):
        partitions, Q = dendrogram[-level]
        partition = range(len(partitions))
        for i in range(level, len(dendrogram) + 1):
            new_partition = np.zeros(n_nodes, dtype=int)
            partitions, _ = dendrogram[-i]
            for j in range(len(partitions)):
                new_partition[j] = partition[partitions[j]]
            partition = new_partition
        yield partition, Q, len(np.unique(partition))
