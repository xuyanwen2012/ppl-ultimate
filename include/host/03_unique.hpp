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

[[nodiscard, deprecated]] inline std::future<int> dispatch_unique(
    BS::thread_pool& pool,
    const std::vector<morton_t>& u_morton,
    std::vector<morton_t>& u_morton_alt) {
  return pool.submit_task([&] {
    // DEBUG_PRINT("[tid ", std::this_thread::get_id(), "] started. (unique)");

    const auto last = std::unique_copy(
        u_morton.begin(), u_morton.end(), u_morton_alt.begin());

    const auto n_unique = std::distance(u_morton_alt.begin(), last);

    // DEBUG_PRINT("[tid ",
    //             std::this_thread::get_id(),
    //             "] ended. (unqiue = ",
    //             n_unique,
    //             ")");

    return static_cast<int>(n_unique);
  });
}

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