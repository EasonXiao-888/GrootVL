#pragma once
#include <torch/extension.h>

#include "bfs.h"
#include "mst.h"
#include "rst.h"
#include "refine.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    /* build trees */
    m.def("bfs_forward", &bfs_forward, "bfs_forward");
    m.def("mst_forward", &mst_forward, "mst_forward");
    m.def("rst_forward", &rst_forward, "rst_forward");
    /* scan */
    m.def("tree_scan_refine_forward", &tree_scan_refine_forward, "tree_scan_refine_forward");
    m.def("tree_scan_refine_backward_feature", &tree_scan_refine_backward_feature, "tree_scan_refine_backward_feature");
    m.def("tree_scan_refine_backward_edge_weight", &tree_scan_refine_backward_edge_weight, "tree_scan_refine_backward_edge_weight");
    m.def("tree_scan_refine_backward_self_weight", &tree_scan_refine_backward_self_weight, "tree_scan_refine_backward_self_weight");
}
