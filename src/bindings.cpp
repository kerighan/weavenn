#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "algorithm.hpp"
#include "louvain.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_weavenn, m)
{
    m.def("get_graph", &get_graph);
    m.def("get_partitions", &get_partitions);
    m.def("generate_dendrogram", &generate_dendrogram);
}
