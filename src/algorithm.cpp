#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "louvain.hpp"
#include <algorithm>

namespace py = pybind11;

float MIN_VALUE = 0.0001;

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

std::tuple<GraphNeighbors, GraphWeights, Weights> filter_edges(
    GraphNeighbors const &graph_neighbors,
    GraphWeights const &graph_weights,
    Weights const &sigma_count,
    float min_sc, float k)
{

    size_t n_nodes = graph_neighbors.size();
    GraphNeighbors gn;
    GraphWeights gw;
    Weights sc;
    gn.resize(n_nodes);
    gw.resize(n_nodes);
    sc.resize(n_nodes);
    for (Node i = 0; i < n_nodes; i++)
    {
        if (sigma_count[i] < min_sc)
            continue;

        for (size_t index = 0; index < graph_neighbors[i].size(); index++)
        {
            Node j = graph_neighbors[i][index];
            if (sigma_count[j] < min_sc)
                continue;
            Weight w = graph_weights[i][index];
            gn[i].push_back(j);
            gw[i].push_back(w);
            sc[i] += w / k;
        }
    }
    return std::make_tuple(gn, gw, sc);
}

std::tuple<GraphNeighbors, GraphWeights, Weights> get_graph(
    py::array_t<uint64_t> _labels,
    py::array_t<float> _distances,
    py::array_t<float> _local_scaling,
    float min_sim, float beta, float dim)
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
    Weights sigma_count;
    graph_neighbors.resize(n_nodes);
    graph_weights.resize(n_nodes);
    sigma_count.resize(n_nodes);

    float scale = atanh(1 - log2(k) / k);
    // float scale = 1;
    // float scale = atanh(1. - pow(1. / k, 1. / dim));

    for (uint64_t i = 0; i < n_nodes; i++)
    {
        float sigma_i = std::max(local_scaling[i], MIN_VALUE);

        for (size_t index = 0; index < k; index++)
        {
            uint64_t j = labels[i * k + index];
            if (i == j)
                continue;

            // add pair to visited edges
            std::pair<uint64_t, uint64_t> pair;
            if (i < j)
                pair = std::make_pair(i, j);
            else
                pair = std::make_pair(j, i);
            if (visited.find(pair) != visited.end())
                continue;
            visited.insert(pair);

            float sigma_j = std::max(local_scaling[j], MIN_VALUE);

            float dist = distances[i * k + index];
            float weight;
            if (dist == 0)
                weight = 1;
            else
            {
                dist = pow((dist * dist) / (sigma_i * sigma_j), beta);
                weight = 1. - tanh(dist * scale);
            }

            if (weight < min_sim)
                continue;

            graph_neighbors[i].push_back(j);
            graph_neighbors[j].push_back(i);
            graph_weights[i].push_back(weight);
            graph_weights[j].push_back(weight);
            sigma_count[i] += weight / k;
            sigma_count[j] += weight / k;
        }
    }
    return std::make_tuple(graph_neighbors, graph_weights, sigma_count);
}

std::tuple<std::vector<std::pair<Nodes, float>>, Weights, GraphNeighbors, GraphWeights> get_partitions(
    py::array_t<uint64_t> _labels,
    py::array_t<float> _distances,
    py::array_t<float> _local_scaling,
    float min_sim, float resolution,
    bool prune, bool full, float beta, float dim, float min_sc, size_t k)
{
    auto [graph_neighbors, graph_weights, sigma_count] = get_graph(
        _labels, _distances, _local_scaling, min_sim, beta, dim);
    if (min_sc > 0)
    {
        auto [gn_, gw_, sc_] = filter_edges(
            graph_neighbors, graph_weights, sigma_count, min_sc, k);
        graph_neighbors = gn_;
        graph_weights = gw_;
        sigma_count = sc_;
    }

    GraphNeighbors gn = graph_neighbors;
    GraphWeights gw = graph_weights;
    return std::make_tuple(
        generate_dendrogram(
            graph_neighbors, graph_weights,
            resolution, prune, full),
        sigma_count, gn, gw);
}
