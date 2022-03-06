#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "louvain.hpp"

namespace py = pybind11;

struct hash_pair
{
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2> &p) const
    {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 + 1759 * hash2;
    }
};

std::pair<GraphNeighbors, GraphWeights> get_graph(
    py::array_t<uint64_t> _labels,
    py::array_t<float> _distances,
    py::array_t<float> _local_scaling,
    float min_sim)
{
    // get data buffers
    py::buffer_info labelsBuf = _labels.request();
    uint64_t *labels = (uint64_t *)labelsBuf.ptr;

    py::buffer_info distancesBuf = _distances.request();
    float *distances = (float *)distancesBuf.ptr;

    py::buffer_info local_scalingBuf = _local_scaling.request();
    float *local_scaling = (float *)local_scalingBuf.ptr;

    size_t n_nodes = local_scalingBuf.shape[0];
    size_t k = labelsBuf.shape[1];

    std::unordered_set<std::pair<uint64_t, uint64_t>, hash_pair> visited;

    GraphNeighbors graph_neighbors;
    GraphWeights graph_weights;
    graph_neighbors.resize(n_nodes);
    graph_weights.resize(n_nodes);

    for (uint64_t i = 0; i < n_nodes; i++)
    {
        float node_scaling = local_scaling[i];
        if (node_scaling < 0.000001)
            node_scaling = 0.000001;

        for (size_t index = 0; index < k; index++)
        {
            uint64_t j = labels[i * k + index];
            if (i == j)
                continue;

            std::pair<uint64_t, uint64_t> pair;
            if (i < j)
                pair = std::make_pair(i, j);
            else
                pair = std::make_pair(j, i);

            if (visited.find(pair) != visited.end())
                continue;
            visited.insert(pair);

            float neighbor_scaling = local_scaling[j];
            if (neighbor_scaling < 0.000001)
                neighbor_scaling = 0.000001;

            double dist = distances[i * k + index];
            dist = (dist * dist) / (node_scaling * neighbor_scaling);
            double weight = 1. - tanh(dist);

            // double a = dist/node_scaling;
            // double b = dist/neighbor_scaling;
            // double weight_a = 1. - tanh(a*a);
            // double weight_b = 1. - tanh(b*b);
            // double weight = (weight_a + weight_b)/2;

            if (weight < min_sim)
                continue;

            graph_neighbors[i].push_back(j);
            graph_neighbors[j].push_back(i);
            graph_weights[i].push_back(weight);
            graph_weights[j].push_back(weight);
        }
    }
    return std::make_pair(graph_neighbors, graph_weights);
}

std::vector<std::pair<Nodes, float>> get_partitions(
    py::array_t<uint64_t> _labels,
    py::array_t<float> _distances,
    py::array_t<float> _local_scaling,
    float min_sim, float resolution,
    bool prune, bool full)
{
    auto [graph_neighbors, graph_weights] = get_graph(_labels, _distances, _local_scaling, min_sim);
    return generate_dendrogram(graph_neighbors, graph_weights, resolution, prune, full);
}
