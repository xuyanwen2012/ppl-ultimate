#include <memory>
#include <random>
#include <vector>

#include "host/all.hpp"
#include "host/barrier.hpp"
#include "host/sort.hpp"
#include "shared/morton_func.h"
#include "shared/structures.h"

// third-party
#include "third-party/BS_thread_pool.hpp"
#include "third-party/BS_thread_pool_utils.hpp"
#include "third-party/CLI11.hpp"

int main(const int argc, const char *argv[]) {
  const auto max_cpu_cores = std::thread::hardware_concurrency();

  int n_threads = 1;

  CLI::App app{"Demo"};

  app.add_option("-t,--threads", n_threads, "Number of threads")
      ->check(CLI::Range(1u, max_cpu_cores));

  CLI11_PARSE(app, argc, argv);

  // Problem size
  constexpr auto n = 1920 * 1080;  // ~2M
  constexpr auto min_coord = 0.0f;
  constexpr auto range = 1024.0f;
  constexpr auto educated_guess = 0.55;
  constexpr auto seed = 114514;

  std::vector<glm::vec4> u_points(n);
  std::vector<morton_t> u_morton(n);
  std::vector<morton_t> u_morton_alt(n);
  auto brt = std::make_unique<radix_tree>(n);
  std::vector<int> u_edge_count(n);
  std::vector<int> u_edge_offset(n);
  auto oct = std::make_unique<octree>(n * educated_guess);

  std::mt19937 gen(seed);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution dis(min_coord, min_coord + range);
  std::generate(u_points.begin(), u_points.end(), [&dis, &gen] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });

  BS::thread_pool pool;
  barrier sort_barrier(n_threads);

  BS::timer timer;

  timer.start();

  cpu::dispatch_morton_code(
      pool, n_threads, u_points, u_morton, min_coord, range)
      .wait();
  cpu::dispatch_binning_pass(
      pool, n_threads, sort_barrier, u_morton, u_morton_alt, 0)
      .wait();
  cpu::dispatch_binning_pass(
      pool, n_threads, sort_barrier, u_morton_alt, u_morton, 8)
      .wait();
  cpu::dispatch_binning_pass(
      pool, n_threads, sort_barrier, u_morton, u_morton_alt, 16)
      .wait();
  cpu::dispatch_binning_pass(
      pool, n_threads, sort_barrier, u_morton_alt, u_morton, 24)
      .wait();

  // using 'u_morton_alt' as 'sorted morton'
  auto unique_future = cpu::dispatch_unique(pool, u_morton, u_morton_alt);
  auto n_unique = unique_future.get();
  brt->set_n_nodes(n_unique - 1);

  cpu::dispatch_build_radix_tree(pool, n_threads, u_morton_alt, brt.get())
      .wait();

  cpu::dispatch_edge_count(pool, n_threads, brt, u_edge_count).wait();

  auto offset_future = cpu::dispatch_edge_offset(
      pool, brt->n_nodes(), u_edge_count, u_edge_offset);
  auto n_octree_nodes = offset_future.get();
  oct->set_n_nodes(n_octree_nodes);

  cpu::dispatch_build_octree(pool,
                             n_threads,
                             u_edge_count,
                             u_edge_offset,
                             u_morton,
                             brt,
                             oct,
                             min_coord,
                             range)
      .wait();

  timer.stop();

  // ---------------------------------------------------------------------
  // Validation
  // ---------------------------------------------------------------------

  std::cout << "==========================\n";
  std::cout << " Total Time spent: " << timer.ms() << " ms\n";
  std::cout << " n_unique = " << n_unique << '\n';
  std::cout << " n_brt_nodes = " << brt->n_nodes() << '\n';
  std::cout << " n_octree_nodes = " << n_octree_nodes << " ("
            << static_cast<double>(n_octree_nodes) / n * 100.0 << "%)\n";
  std::cout << "--------------------------\n";
  // std::cout << " Morton: " << morton_timestamp << " ms\n";
  // std::cout << " Sort: " << sort_timestamp - morton_timestamp << " ms\n";
  // std::cout << " Unique: " << unique_timestamp - sort_timestamp << " ms\n";
  // std::cout << " BRT: " << brt_timestamp - unique_timestamp << " ms\n";
  // std::cout << " Edge Count && Offset: " << edge_offset_timestamp -
  // brt_timestamp << " ms\n"; std::cout << " Octree: " << timer.current_ms() -
  // edge_offset_timestamp << " ms\n";
  std::cout << "==========================\n";

  const auto is_sorted = std::is_sorted(u_morton.begin(), u_morton.end());
  std::cout << "Is sorted: " << std::boolalpha << is_sorted << '\n';

  return 0;
}