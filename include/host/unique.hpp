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

[[nodiscard]]
inline std::future<int> dispatch_unique(BS::thread_pool& pool,
                                        const std::vector<morton_t>& u_sort,
                                        std::vector<morton_t>& u_sort_unique) {
  return pool.submit_task([&] {
    // DEBUG_PRINT("[tid ", std::this_thread::get_id(), "] started. (unique)");

    const auto last =
        std::unique_copy(u_sort.begin(), u_sort.end(), u_sort_unique.begin());

    const auto n_unique = std::distance(u_sort_unique.begin(), last);

    // DEBUG_PRINT("[tid ",
    //             std::this_thread::get_id(),
    //             "] ended. (unqiue = ",
    //             n_unique,
    //             ")");

    return static_cast<int>(n_unique);
  });
}

}  // namespace cpu