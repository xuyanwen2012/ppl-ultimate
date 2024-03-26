#pragma once

#include <future>

#include "shared/morton_func.h"
#include "third-party/BS_thread_pool.hpp"

namespace cpu {

// ---------------------------------------------------------------------
// Unique (for CPU, should only use a single thread), we have small problem size
//
// And this future will return the number of unique elements
//
// ---------------------------------------------------------------------

[[nodiscard]] inline std::future<int> dispatch_unique(BS::thread_pool& pool,
                                                      const int n,
                                                      const morton_t* u_morton,
                                                      morton_t* u_morton_alt) {
  return pool.submit_task([&] {
    const auto last = std::unique_copy(u_morton, u_morton + n, u_morton_alt);
    const auto n_unique = std::distance(u_morton_alt, last);
    return static_cast<int>(n_unique);
  });
}

}  // namespace cpu