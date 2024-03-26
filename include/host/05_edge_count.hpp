#pragma once

#include "shared/edge_func.h"
#include "shared/structures.h"
#include "third-party/BS_thread_pool.hpp"

namespace cpu {

// ---------------------------------------------------------------------
// Edge Count (1->1 relation)
// ---------------------------------------------------------------------

[[nodiscard]] inline BS::multi_future<void> dispatch_edge_count(
    BS::thread_pool& pool,
    const size_t n_desired_threads,
    const std::unique_ptr<radix_tree>& radix_tree,
    std::vector<int>& edge_count) {
  return pool.submit_blocks(
      0,
      radix_tree->n_nodes(),
      [&](const int start, const int end) {
        for (auto i = start; i < end; ++i) {
          shared::process_edge_count_i(i,
                                       radix_tree->u_prefix_n,
                                       radix_tree->u_parent,
                                       edge_count.data());
        }
      },
      n_desired_threads);
}

};  // namespace cpu