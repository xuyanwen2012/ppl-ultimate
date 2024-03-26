#pragma once

#include "shared/morton_func.h"
#include "third-party/BS_thread_pool.hpp"

namespace cpu {

// ---------------------------------------------------------------------
// Morton encoding (1->1 relation)
// ---------------------------------------------------------------------

// raw pointer version

[[nodiscard]] inline BS::multi_future<void> dispatch_morton_code(
    BS::thread_pool &pool,
    const size_t desired_n_threads,
    const int n,
    const glm::vec4 *u_points,
    morton_t *u_morton,
    const float min_coord,
    const float range) {
  return pool.submit_blocks(
      0,
      n,
      [&](const int start, const int end) {
        std::transform(u_points + start,
                       u_points + end,
                       u_morton + start,
                       [min_coord, range](const glm::vec4 &xyz) {
                         return shared::xyz_to_morton32(xyz, min_coord, range);
                       });
      },
      desired_n_threads);
}

}  // namespace cpu