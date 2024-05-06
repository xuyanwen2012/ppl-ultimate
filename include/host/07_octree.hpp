#pragma once

#include "shared/oct_func.h"
#include "shared/structures.h"
#include "third-party/BS_thread_pool.hpp"

namespace cpu {

// ---------------------------------------------------------------------
// Octree (1->1 relation, but has a lot of input)
// ---------------------------------------------------------------------

[[nodiscard]] inline BS::multi_future<void> dispatch_make_oct_node(
    BS::thread_pool& pool,
    const size_t n_desired_threads,
    const int* edge_offset,
    const int* edge_count,
    const morton_t* sorted_unique_morton,
    const radix_tree& radix_tree,
    const octree& octree,  // output
    const float min_coord,
    const float range) {
  std::cerr << "\nBEFORE edge_count: ";
  for (int i = 0; i < 50; i++) {
    std::cerr << edge_count[i] << " ";
  }
  std::cerr << "\nBEFORE edge_offset: ";
  for (int i = 0; i < 20; i++) {
    std::cerr << edge_offset[i] << " ";
  }
  std::cerr << "\nBEFORE sorted_uniqe_morton: ";
  for (int i = 0; i < 20; i++) {
    std::cerr << sorted_unique_morton[i] << " ";
  }
  std::cerr << "\nBEFORE radix_tree.u_prefix_n: ";
  for (int i = 0; i < 20; i++) {
    std::cerr << static_cast<int>(radix_tree.u_prefix_n[i]) << " ";
  }
  std::cerr << "\nBEFORE radix_tree.u_parents: ";
  for (int i = 0; i < 20; i++) {
    std::cerr << radix_tree.u_parents[i] << " ";
  }
  std::cerr << "\nBEFORE min_coord: " << min_coord;
  std::cerr << "\nBEFORE range: " << range;

  return pool.submit_blocks(
      1,  // making sure for cpu, skip the root node
      radix_tree.n_nodes(),
      [=,
       &n_desired_threads,
       &edge_count,
       &sorted_unique_morton,
       &radix_tree,
       &octree,
       &min_coord,
       &range](const int start, const int end) {
        std::cerr << "\nedge_count: ";
        for (int i = 0; i < 50; i++) {
          std::cerr << edge_count[i] << " ";
        }
        std::cerr << "\nedge_offset: ";
        for (int i = 0; i < 20; i++) {
          std::cerr << edge_offset[i] << " ";
        }
        std::cerr << "\nsorted_uniqe_morton: ";
        for (int i = 0; i < 20; i++) {
          std::cerr << sorted_unique_morton[i] << " ";
        }
        std::cerr << "\nradix_tree.u_prefix_n: ";
        for (int i = 0; i < 20; i++) {
          std::cerr << static_cast<int>(radix_tree.u_prefix_n[i]) << " ";
        }
        std::cerr << "\nradix_tree.u_parents: ";
        for (int i = 0; i < 20; i++) {
          std::cerr << radix_tree.u_parents[i] << " ";
        }
        std::cerr << "\nmin_coord: " << min_coord;
        std::cerr << "\nrange: " << range;
        for (auto i = start; i < end; ++i) {
          shared::process_oct_node(i,
                                   octree.u_children,
                                   octree.u_corner,
                                   octree.u_cell_size,
                                   octree.u_child_node_mask,
                                   edge_offset,
                                   edge_count,
                                   sorted_unique_morton,
                                   radix_tree.u_prefix_n,
                                   radix_tree.u_parents,
                                   min_coord,
                                   range);
        }
      },
      n_desired_threads);
}

[[nodiscard]] inline BS::multi_future<void> dispatch_link_leaf(
    BS::thread_pool& pool,
    const size_t n_desired_threads,
    const int* edge_offset,
    const int* edge_count,
    const morton_t* sorted_unique_morton,
    const radix_tree& radix_tree,
    const octree& octree  // output
) {
  std::cerr << "\n\nSTARTING edge_count: ";
  return pool.submit_blocks(
      0,
      radix_tree.n_nodes(),
      [&](const int start, const int end) {
        for (auto i = start; i < end; ++i) {
          shared::process_link_leaf(i,
                                    octree.u_children,
                                    octree.u_child_leaf_mask,
                                    edge_offset,
                                    edge_count,
                                    sorted_unique_morton,
                                    radix_tree.u_has_leaf_left,
                                    radix_tree.u_has_leaf_right,
                                    radix_tree.u_prefix_n,
                                    radix_tree.u_parents,
                                    radix_tree.u_left_child);
        }
      },
      n_desired_threads);
}

}  // namespace cpu