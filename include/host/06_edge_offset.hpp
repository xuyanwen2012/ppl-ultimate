#pragma once

#include <future>
#include <numeric>

#include "third-party/BS_thread_pool.hpp"

namespace cpu {

// ---------------------------------------------------------------------
// Edge Offset (prefix sum, for CPU, should use a single thread)
// this will return the number of octree nodes
// ---------------------------------------------------------------------

[[nodiscard]] inline std::future<int> dispatch_edge_offset(
    BS::thread_pool& pool,
    const int n_brt_nodes,
    const std::vector<int>& edge_count,
    std::vector<int>& edge_offset) {
  // should not use ".end()", we allocated more than actual data
  // we need to use our own "n_brt_nodes"
  return pool.submit_task([&] {
    std::partial_sum(edge_count.begin(),
                     edge_count.begin() + n_brt_nodes,
                     edge_offset.begin());
    return edge_offset[n_brt_nodes - 1];
  });
}

}  // namespace cpu