#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

#include "cuda/dispatcher.cuh"
#include "host_code.hpp"
#include "shared/structures.h"
#include "third-party/BS_thread_pool_utils.hpp"
#include "third-party/CLI11.hpp"

// Problem size
constexpr auto n = 1920 * 1080;  // ~2M
constexpr auto min_coord = 0.0f;
constexpr auto range = 1024.0f;
constexpr auto seed = 114514;

// demo config
constexpr auto n_iterations = 30;

void gen_data(const std::unique_ptr<pipe>& p) {
  std::mt19937 gen(seed);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution dis(min_coord, min_coord + range);
  std::generate_n(p->u_points, n, [&dis, &gen] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });
}

void init_data(const std::unique_ptr<pipe>& gpu_pip,
               const std::unique_ptr<pipe>& cpu_pip) {
  std::mt19937 gen(seed);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution dis(min_coord, min_coord + range);
  std::generate_n(gpu_pip->u_points, n, [&dis, &gen] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });

  std::copy(gpu_pip->u_points, gpu_pip->u_points + n, cpu_pip->u_points);
}

void demo_cpu_only(int n_threads) {
  auto p = std::make_unique<pipe>(n, min_coord, range, seed);

  gen_data(p);

  BS::timer t;

  t.start();

  std::cout << "staring CPU only demo" << std::endl;

  for (auto i = 0; i < n_iterations; ++i) {
    cpu::dispatch_ComputeMorton(n_threads, p.get());
    cpu::dispatch_RadixSort(n_threads, p.get());
    cpu::dispatch_RemoveDuplicates(n_threads, p.get());
    cpu::dispatch_BuildRadixTree(n_threads, p.get());
    cpu::dispatch_EdgeCount(n_threads, p.get());
    cpu::dispatch_EdgeOffset(n_threads, p.get());
    cpu::dispatch_BuildOctree(n_threads, p.get());

    std::cout << '.' << std::flush;
  }
  std::cout << '\n';

  t.stop();

  // print total time and average time, 't.ms()'

  std::cout << "CPU only Total: " << t.ms() << "ms" << std::endl;
  std::cout << "Average: " << t.ms() / n_iterations << "ms" << std::endl;
}

void demo_gpu_only(int grid_size) {
  auto p = std::make_unique<pipe>(n, min_coord, range, seed);

  constexpr auto n_streams = 1;
  gpu::initialize_dispatcher(n_streams);

  gen_data(p);

  BS::timer t;
  t.start();

  std::cout << "staring GPU only demo" << std::endl;
  for (auto i = 0; i < n_iterations; ++i) {
    constexpr auto stream_id = 0;
    gpu::dispatch_ComputeMorton(grid_size, stream_id, p.get());
    gpu::dispatch_RadixSort(grid_size, stream_id, p.get());
    gpu::dispatch_RemoveDuplicates_sync(grid_size, stream_id, p.get());
    gpu::dispatch_BuildRadixTree(grid_size, stream_id, p.get());
    gpu::dispatch_EdgeCount(grid_size, stream_id, p.get());
    gpu::dispatch_EdgeOffset(grid_size, stream_id, p.get());
    gpu::dispatch_BuildOctree(grid_size, stream_id, p.get());

    std::cout << '.' << std::flush;
  }
  std::cout << '\n';

  t.stop();

  // print total time and average time, 't.ms()'

  std::cout << "GPU only Total: " << t.ms() << "ms" << std::endl;
  std::cout << "Average: " << t.ms() / n_iterations << "ms" << std::endl;

  void release_dispatcher();
}
