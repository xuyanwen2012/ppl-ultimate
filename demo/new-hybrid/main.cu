#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

#include "host_code.hpp"
#include "shared/structures.h"
#include "third-party/BS_thread_pool_utils.hpp"

// Problem size
constexpr auto n = 1920 * 1080;  // ~2M
constexpr auto min_coord = 0.0f;
constexpr auto range = 1024.0f;
constexpr auto seed = 114514;

void init_data(const std::unique_ptr<pipe>& gpu_pip,
               const std::unique_ptr<pipe>& cpu_pip) {
  std::mt19937 gen(seed);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution dis(min_coord, min_coord + range);
  std::generate_n(gpu_pip->u_points, n, [&dis, &gen] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });

  std::copy(gpu_pip->u_points, gpu_pip->u_points + n, cpu_pip->u_points);
}

int main() {
  // config
  constexpr auto n_threads = 2;

  auto gpu_pip = std::make_unique<pipe>(n, min_coord, range, seed);
  auto cpu_pip = std::make_unique<pipe>(n, min_coord, range, seed);

  init_data(gpu_pip, cpu_pip);

  BS::timer t;

  t.start();

  cpu::dispatch_ComputeMorton(n_threads, cpu_pip.get());
  cpu::dispatch_RadixSort(n_threads, cpu_pip.get());

  t.stop();

  const auto is_sorted =
      std::is_sorted(cpu_pip->u_morton, cpu_pip->u_morton + n);
  std::cout << "CPU: " << (is_sorted ? "Sorted" : "Not sorted") << '\n';

  std::cout << "Time: " << t.ms() << "ms\n";

  return 0;
}