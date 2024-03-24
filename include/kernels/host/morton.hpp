#pragma once

#include "shared/morton_func.h"
#include "third-party/BS_thread_pool.hpp"

namespace cpu {

[[nodiscard]]
BS::multi_future<void> dispatch_morton_code(
    BS::thread_pool &pool,
    const size_t desired_n_threads,
    const std::vector<glm::vec4> &u_points,
    std::vector<morton_t> &u_morton,
    const float min_coord,
    const float range);

}  // namespace cpu