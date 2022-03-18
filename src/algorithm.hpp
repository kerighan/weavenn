#include <iostream>
#include <stdlib.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "louvain.hpp"

namespace py = pybind11;

std::pair<GraphNeighbors, GraphWeights> get_graph(
    py::array_t<uint64_t> _labels,
    py::array_t<float> _distances,
    py::array_t<float> _local_scaling,
    float min_sim);

std::vector<std::pair<Nodes, float>> get_partitions(
    py::array_t<uint64_t> _labels,
    py::array_t<float> _distances,
    py::array_t<float> _local_scaling,
    float min_sim, float resolution,
    bool prune, bool full, bool z_modularity);
