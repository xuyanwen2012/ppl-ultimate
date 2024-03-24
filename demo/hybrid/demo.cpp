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

  auto gpu_pip = std::make_unique<pipe>(n, min_coord, range, seed);

  auto cpu_pip = std::make_unique<pipe>(n, min_coord, range, seed);

  std::mt19937 gen(seed);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution dis(min_coord, min_coord + range);
  std::generate_n(gpu_pip->u_points, n, [&dis, &gen] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });

  std::copy(gpu_pip->u_points, gpu_pip->u_points + n, cpu_pip->u_points);

  gpu::initialize_dispatcher(1);

  BS::timer timer;

  BS::thread_pool pool(n_threads);

  barrier sort_barrier(n_threads);

  cpu::dispatch_morton_code(pool,
                            n_threads,
                            n,
                            cpu_pip->u_points,
                            cpu_pip->u_morton,
                            min_coord,
                            range)
      .wait();
  cpu::dispatch_binning_pass(pool,
                             n_threads,
                             sort_barrier,
                             n,
                             cpu_pip->u_morton,
                             cpu_pip->u_morton_alt,
                             0)
      .wait();
  cpu::dispatch_binning_pass(pool,
                             n_threads,
                             sort_barrier,
                             n,
                             cpu_pip->u_morton_alt,
                             cpu_pip->u_morton,
                             8)
      .wait();
  cpu::dispatch_binning_pass(pool,
                             n_threads,
                             sort_barrier,
                             n,
                             cpu_pip->u_morton,
                             cpu_pip->u_morton_alt,
                             16)
      .wait();
  cpu::dispatch_binning_pass(pool,
                             n_threads,
                             sort_barrier,
                             n,
                             cpu_pip->u_morton_alt,
                             cpu_pip->u_morton,
                             24)
      .wait();

  // using 'u_morton_alt' as 'sorted morton'
  auto unique_future =
      cpu::dispatch_unique(pool, n, cpu_pip->u_morton, cpu_pip->u_morton_alt);
  auto n_unique = unique_future.get();
  cpu_pip->set_n_unique(n_unique);
  cpu_pip->brt.set_n_nodes(n_unique - 1);

  cpu::dispatch_build_radix_tree(
      pool, n_threads, cpu_pip->u_morton_alt, &cpu_pip->brt)
      .wait();

  // cpu::dispatch_edge_count(pool, n_threads, brt, u_edge_count).wait();

  // auto offset_future = cpu::dispatch_edge_offset(
  //     pool, brt->n_nodes(), u_edge_count, u_edge_offset);
  // auto n_octree_nodes = offset_future.get();
  // oct->set_n_nodes(n_octree_nodes);

  // cpu::dispatch_build_octree(pool,
  //                            n_threads,
  //                            u_edge_count,
  //                            u_edge_offset,
  //                            u_morton,
  //                            brt,
  //                            oct,
  //                            min_coord,
  //                            range)
  //     .wait();

  timer.start();

  const auto grid_size = 16;
  const auto stream_id = 0;

  gpu::dispatch_ComputeMorton(grid_size, stream_id, *gpu_pip);
  gpu::dispatch_RadixSort(grid_size, stream_id, *gpu_pip);

  {
    gpu::sync_device();
    auto is_equal =
        std::equal(cpu_pip->u_morton, cpu_pip->u_morton + n, gpu_pip->u_morton);
    std::cout << "Is equal: " << std::boolalpha << is_equal << '\n';
  }

  gpu::dispatch_RemoveDuplicates_sync(grid_size, stream_id, *gpu_pip);

  {
    gpu::sync_device();
    std::cout << "gpu n_unique = " << gpu_pip->n_unique_mortons() << '\n';
    std::cout << "cpu n_unique = " << cpu_pip->n_unique_mortons() << '\n';
    const auto is_equal =
        std::equal(cpu_pip->u_morton_alt,
                   cpu_pip->u_morton_alt + gpu_pip->n_unique_mortons(),
                   gpu_pip->u_morton_alt);
    std::cout << "Is equal: " << std::boolalpha << is_equal << '\n';
    // print mistmatches

    for (auto i = 0; i < gpu_pip->n_unique_mortons(); ++i) {
      if (cpu_pip->u_morton_alt[i] != gpu_pip->u_morton_alt[i]) {
        std::cout << "mismatch at " << i << ": " << cpu_pip->u_morton_alt[i]
                  << " != " << gpu_pip->u_morton_alt[i] << '\n';
      }
    }
  }

  // std::cout << "Is equal: " << std::boolalpha << is_equal << '\n';

  // gpu::dispatch_BuildRadixTree(grid_size, stream_id, *gpu_pip);
  // gpu::dispatch_EdgeCount(grid_size, stream_id, *gpu_pip);
  // gpu::dispatch_EdgeOffset(grid_size, stream_id, *gpu_pip);

  // // print each brt node
  // for (auto i = 0; i < gpu_pip->n_brt_nodes(); ++i) {
  //   std::cout << "brt[" << i << "] = " << gpu_pip->brt.u_prefix_n[i] << '\n';
  // }

  // gpu::dispatch_BuildOctree(grid_size, stream_id, *gpu_pip);

  // gpu::sync_stream(0);
  gpu::sync_device();

  // timer.stop();

  // gpu_pip->oct.set_n_nodes(gpu_pip->u_edge_offset[gpu_pip->n_brt_nodes() -
  // 1]);

  // const auto is_sorted =
  //     std::is_sorted(gpu_pip->u_morton, gpu_pip->u_morton + n);
  // std::cout << "Is sorted: " << std::boolalpha << is_sorted << '\n';

  // std::cout << "==========================\n";
  // std::cout << " Total Time spent: " << timer.ms() << " ms\n";
  // std::cout << " n_unique = " << gpu_pip->n_unique_mortons() << " ("
  //           << static_cast<double>(gpu_pip->n_unique_mortons()) / n * 100.0
  //           << "%)\n";
  // std::cout << " n_brt_nodes = " << gpu_pip->n_brt_nodes() << '\n';
  // std::cout << " n_octree_nodes = " << gpu_pip->n_oct_nodes() << " ("
  //           << static_cast<double>(gpu_pip->n_oct_nodes()) / n * 100.0
  //           << "%)\n";
  // std::cout << "--------------------------\n";

  gpu::release_dispatcher();

  return 0;
}