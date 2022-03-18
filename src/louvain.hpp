#include <stdlib.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using Node = int64_t;
using Nodes = std::vector<Node>;
using GraphNeighbors = std::vector<Nodes>;
using Weight = float;
using Weights = std::vector<Weight>;
using GraphWeights = std::vector<Weights>;
using WeightMap = std::unordered_map<Node, float>;

std::tuple<Nodes, Weights, Weights, Weights, Weights, float> init_status(
    GraphNeighbors const &graph_neighbors,
    GraphWeights const &graph_weights);

float modularity(
    Weights const &internals,
    Weights const &degrees,
    float total_weight,
    float resolution,
    bool z_modularity);

WeightMap neighcom(
    GraphNeighbors const &graph_neighbors,
    GraphWeights const &graph_weights,
    Nodes const &node2com,
    Node node);

void one_level(
    GraphNeighbors const &graph_neighbors,
    GraphWeights const &graph_weights,
    Nodes &node2com,
    Weights &internals,
    Weights const &loops,
    Weights &degrees,
    Weights const &gdegrees,
    float total_weight,
    float resolution,
    bool z_modularity);

void one_level_prune(
    GraphNeighbors const &graph_neighbors,
    GraphWeights const &graph_weights,
    Nodes &node2com,
    Weights &internals,
    Weights const &loops,
    Weights &degrees,
    Weights const &gdegrees,
    float total_weight,
    float resolution,
    bool z_modularity);

void one_step(
    GraphNeighbors const &graph_neighbors,
    GraphWeights const &graph_weights,
    Nodes &node2com,
    Weights &internals,
    Weights const &loops,
    Weights &degrees,
    Weights const &gdegrees,
    float total_weight,
    float resolution,
    bool z_modularity);

std::tuple<GraphNeighbors, Nodes> renumber(
    GraphNeighbors const &graph_neighbors,
    GraphWeights const &graph_weights,
    Nodes const &node2com);

std::tuple<GraphNeighbors, GraphWeights, size_t> induced_graph(
    GraphNeighbors const &graph_neighbors,
    GraphWeights const &graph_weights,
    GraphNeighbors const &communities,
    Nodes const &node2com);

Nodes get_partition(Nodes const &node2com, size_t n_nodes);

std::vector<std::pair<Nodes, float>> generate_dendrogram(
    GraphNeighbors &graph_neighbors,
    GraphWeights &graph_weights,
    float resolution,
    bool prune,
    bool full,
    bool z_modularity);