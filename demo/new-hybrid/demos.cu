#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

#include "cuda/dispatcher.cuh"
#include "host/dispatcher.hpp"
#include "shared/structures.h"
#include "third-party/BS_thread_pool_utils.hpp"
#include "third-party/CLI11.hpp"

// Problem size
// constexpr auto n = 1920 * 1080;  // ~2M
constexpr auto n = 640 * 480;
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

// for ~2M inputs
// 1: ~259 ms
// 2: ~134 ms
// 3: ~95 ms
// 4: ~79 ms
// 5: ~74 ms
// 6: ~72 ms
void demo_cpu_only(const int n_threads) {
  auto p = std::make_unique<pipe>(n, min_coord, range, seed);

  gen_data(p);

  BS::timer t;

  t.start();

  std::cout << "staring CPU only demo" << '\n';

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

  std::cout << "CPU only Total: " << t.ms() << "ms" << '\n';
  std::cout << "Average: " << t.ms() / n_iterations << "ms" << '\n';
}

// for ~2M inputs
// 1: ~187 ms
// 2: ~104 ms
// 4: ~62 ms
// 8: ~41 ms
// 16: ~34 ms
void demo_gpu_only(const int grid_size) {
  auto p = std::make_unique<pipe>(n, min_coord, range, seed);

  constexpr auto n_streams = 1;
  gpu::initialize_dispatcher(n_streams);

  gen_data(p);

  BS::timer t;
  t.start();

  std::cout << "staring GPU only demo" << '\n';
  for (auto i = 0; i < n_iterations; ++i) {
    constexpr auto stream_id = 0;
    gpu::dispatch_ComputeMorton(grid_size, stream_id, p.get());
    gpu::dispatch_RadixSort(grid_size, stream_id, p.get());
    gpu::dispatch_RemoveDuplicates_sync(grid_size, stream_id, p.get());
    gpu::dispatch_BuildRadixTree(grid_size, stream_id, p.get());
    gpu::dispatch_EdgeCount(grid_size, stream_id, p.get());
    gpu::dispatch_EdgeOffset(grid_size, stream_id, p.get());
    gpu::dispatch_BuildOctree(grid_size, stream_id, p.get());
    gpu::sync_stream(stream_id);

    std::cout << '.' << std::flush;
  }
  std::cout << '\n';

  t.stop();

  // print total time and average time, 't.ms()'

  std::cout << "GPU only Total: " << t.ms() << "ms" << '\n';
  std::cout << "Average: " << t.ms() / n_iterations << "ms" << '\n';

  gpu::release_dispatcher();
}

void demo_cpu_gpu_independent(int n_threads, int grid_size) {
  // basically,
  // during CPU iteration, we can do 2 gpu iterations.
  // this give us additional 2x speedup.
}

void demo_simple_cpu_gpu_coarse(const int n_threads, const int grid_size) {
  auto cpu_p = std::make_unique<pipe>(n, min_coord, range, seed);
  auto gpu_p = std::make_unique<pipe>(n, min_coord, range, seed);

  constexpr auto n_streams = 1;
  gpu::initialize_dispatcher(n_streams);

  gen_data(cpu_p);
  std::copy_n(cpu_p->u_points, n, gpu_p->u_points);

  // BS::timer t;
  // t.start();

  // std::cout << "staring 2-stage CPU/GPU coarse-grained demo" << std::endl;
  // for (auto i = 0; i < n_iterations; ++i) {
  //   constexpr auto stream_id = 0;
  //   gpu::dispatch_ComputeMorton(grid_size, stream_id, p.get());
  //   gpu::dispatch_RadixSort(grid_size, stream_id, p.get());
  //   gpu::dispatch_RemoveDuplicates_sync(grid_size, stream_id, p.get());
  //   gpu::dispatch_BuildRadixTree(grid_size, stream_id, p.get());
  //   gpu::dispatch_EdgeCount(grid_size, stream_id, p.get());
  //   gpu::dispatch_EdgeOffset(grid_size, stream_id, p.get());
  //   gpu::dispatch_BuildOctree(grid_size, stream_id, p.get());
  //   gpu::sync_stream(stream_id);

  //   std::cout << '.' << std::flush;
  // }
  // std::cout << '\n';

  // t.stop();

  // // print total time and average time, 't.ms()'

  // std::cout << "GPU only Total: " << t.ms() << "ms" << std::endl;
  // std::cout << "Average: " << t.ms() / n_iterations << "ms" << std::endl;

  // void release_dispatcher();
}
