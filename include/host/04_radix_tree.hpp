#pragma once

#include "brt_func.hpp"
#include "third-party/BS_thread_pool.hpp"

namespace cpu {

// ---------------------------------------------------------------------
// Radix Tree (1->1 relation)
// ---------------------------------------------------------------------

[[nodiscard]] inline BS::multi_future<void> dispatch_build_radix_tree(
    BS::thread_pool& pool,
    const size_t n_desired_threads,
    const morton_t* sorted_morton,
    radix_tree* radix_tree) {
  // print sorted morton
  // std::cerr << "\nsorted morton: \n";
  // for (int i = 0; i < 100; i++) {
  //   std::cerr << sorted_morton[i] << " ";
  // }
  // std::cerr << std::endl;
  return pool.submit_blocks(
      0,
      radix_tree->n_nodes(),
      [=, &radix_tree](const int start, const int end) {
        // std::cerr << "\n\nsorted morton in thread pool: \n";
        // for (int i = 0; i < 100; i++) {
        //   std::cerr << sorted_morton[i] << " ";
        // }
        // std::cerr << std::endl;
        for (auto i = start; i < end; ++i) {
          process_radix_tree_i(
              i, radix_tree->n_nodes(), sorted_morton, radix_tree);
        }
      },
      n_desired_threads);
}

}  // namespace cpu