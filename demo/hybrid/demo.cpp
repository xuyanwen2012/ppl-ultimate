#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include "cuda/dispatcher.h"
#include "host/all.hpp"
#include "host/barrier.hpp"
#include "shared/morton_func.h"
#include "shared/structures.h"

// third-party
#include "third-party/BS_thread_pool.hpp"
#include "third-party/BS_thread_pool_utils.hpp"

int demo(const int n_threads) {
  // Problem size
  // constexpr auto n = 1920 * 1080;  // ~2M
  constexpr auto n = 1024;  // ~2M

  constexpr auto min_coord = 0.0f;
  constexpr auto range = 1024.0f;
  constexpr auto seed = 114514;

  auto pipe_ptr = std::make_unique<pipe>(n, min_coord, range, seed);

  std::mt19937 gen(seed);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution dis(min_coord, min_coord + range);
  std::generate_n(pipe_ptr->u_points, n, [&dis, &gen] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });

  gpu::initialize_dispatcher(1);

  // gpu::dispatch_ComputeMorton(1, 0, *pipe_ptr);

  std::generate_n(pipe_ptr->u_morton, n, [i = n]() mutable { return --i; });

  gpu::dispatch_RadixSort(1, 0, *pipe_ptr);
  gpu::sync_device();

  // gpu::sync_stream(0);

  const auto is_sorted =
      std::is_sorted(pipe_ptr->u_morton, pipe_ptr->u_morton + n);
  std::cout << "Is sorted: " << std::boolalpha << is_sorted << '\n';

  for (auto i = 0; i < n; ++i) {
    std::cout << i << ":\t" << pipe_ptr->u_morton[i] << '\n';
  }

  gpu::release_dispatcher();

  return 0;
}