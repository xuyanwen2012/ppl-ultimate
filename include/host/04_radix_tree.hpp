#pragma once

#include "brt_func.hpp"
#include "third-party/BS_thread_pool.hpp"

namespace cpu {

// ---------------------------------------------------------------------
// Radix Tree (1->1 relation)
// ---------------------------------------------------------------------

[[nodiscard, deprecated]] inline BS::multi_future<void>
dispatch_build_radix_tree(BS::thread_pool& pool,
                          const size_t n_desired_threads,
                          const std::vector<morton_t>& sorted_morton,
                          radix_tree* radix_tree) {
  return pool.submit_blocks(
      0,
      radix_tree->n_nodes(),
      [&](const int start, const int end) {
        for (auto i = start; i < end; ++i) {
          process_radix_tree_i(
              i, radix_tree->n_nodes(), sorted_morton.data(), radix_tree);
        }
      },
      n_desired_threads);
}

[[nodiscard]] inline BS::multi_future<void> dispatch_build_radix_tree(
    BS::thread_pool& pool,
    const size_t n_desired_threads,
    const morton_t* sorted_morton,
    radix_tree* radix_tree) {
  return pool.submit_blocks(
      0,
      radix_tree->n_nodes(),
      [&](const int start, const int end) {
        for (auto i = start; i < end; ++i) {
          process_radix_tree_i(
              i, radix_tree->n_nodes(), sorted_morton, radix_tree);
        }
      },
      n_desired_threads);
}

}  // namespace cpu