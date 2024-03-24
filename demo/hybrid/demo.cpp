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
  constexpr auto n = 1920 * 1080;  // ~2M
  // constexpr auto n = 1024;

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

  BS::timer timer;

  timer.start();

  const auto grid_size = 16;

  gpu::dispatch_ComputeMorton(grid_size, 0, *pipe_ptr);

  gpu::dispatch_RadixSort(grid_size, 0, *pipe_ptr);

  gpu::dispatch_RemoveDuplicates_sync(grid_size, 0, *pipe_ptr);

  gpu::dispatch_BuildRadixTree(grid_size, 0, *pipe_ptr);

  gpu::dispatch_EdgeCount(grid_size, 0, *pipe_ptr);
  gpu::dispatch_EdgeOffset_async(grid_size, 0, *pipe_ptr);
  gpu::sync_stream(0);

  // todo: hide this
  pipe_ptr->oct.set_n_nodes(
      pipe_ptr->u_edge_offset[pipe_ptr->n_brt_nodes() - 1]);

  gpu::dispatch_BuildOctree(grid_size, 0, *pipe_ptr);

  // gpu::sync_stream(0);
  gpu::sync_device();

  timer.stop();

  const auto is_sorted =
      std::is_sorted(pipe_ptr->u_morton, pipe_ptr->u_morton + n);
  std::cout << "Is sorted: " << std::boolalpha << is_sorted << '\n';

  std::cout << "==========================\n";
  std::cout << " Total Time spent: " << timer.ms() << " ms\n";
  std::cout << " n_unique = " << pipe_ptr->n_unique_mortons() << " ("
            << static_cast<double>(pipe_ptr->n_unique_mortons()) / n * 100.0
            << "%)\n";
  std::cout << " n_brt_nodes = " << pipe_ptr->n_brt_nodes() << '\n';
  std::cout << " n_octree_nodes = " << pipe_ptr->n_oct_nodes() << " ("
            << static_cast<double>(pipe_ptr->n_oct_nodes()) / n * 100.0
            << "%)\n";
  std::cout << "--------------------------\n";

  gpu::release_dispatcher();

  return 0;
}