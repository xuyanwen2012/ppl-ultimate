#pragma once

#include "barrier.hpp"
#include "shared/morton_func.h"
#include "third-party/BS_thread_pool.hpp"

namespace cpu {

// ---------------------------------------------------------------------
// Radix Sort (challenging)
// ---------------------------------------------------------------------

[[nodiscard]] BS::multi_future<void> dispatch_binning_pass(
    BS::thread_pool& pool,
    const size_t n_threads,
    barrier& barrier,
    const int n,
    const morton_t* u_sort,
    morton_t* u_sort_alt,
    const int shift);

}  // namespace cpu