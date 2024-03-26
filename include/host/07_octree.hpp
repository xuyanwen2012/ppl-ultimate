#pragma once

#include "shared/oct_func.h"
#include "shared/structures.h"
#include "third-party/BS_thread_pool.hpp"

namespace cpu {

// ---------------------------------------------------------------------
// Octree (1->1 relation, but has a lot of input)
// ---------------------------------------------------------------------

[[nodiscard]] inline BS::multi_future<void> dispatch_build_octree(
    BS::thread_pool& pool,
    const size_t n_desired_threads,
    const std::vector<int>& edge_count,
    const std::vector<int>& edge_offset,
    const std::vector<morton_t>& sorted_unique_morton,
    const std::unique_ptr<radix_tree>& radix_tree,
    const std::unique_ptr<octree>& octree,  // output
    const float min_coord,
    const float range) {
  return pool.submit_blocks(
      0,
      radix_tree->n_nodes(),
      [&](const int start, const int end) {
        for (auto i = start; i < end; ++i) {
          shared::process_oct_node(i,
                                   octree->u_children,
                                   octree->u_corner,
                                   octree->u_cell_size,
                                   octree->u_child_node_mask,
                                   edge_offset.data(),
                                   edge_count.data(),
                                   sorted_unique_morton.data(),
                                   radix_tree->u_prefix_n,
                                   radix_tree->u_parents,
                                   min_coord,
                                   range);
        }
      },
      n_desired_threads);
}

}  // namespace cpu