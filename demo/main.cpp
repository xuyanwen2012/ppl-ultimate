#include <memory>
#include <random>
#include <vector>

#include "kernels/host/all.hpp"
#include "shared/morton_func.h"
#include "shared/structures.h"

// third-party
#include "third-party/BS_thread_pool.hpp"
#include "third-party/BS_thread_pool_utils.hpp"
#include "third-party/CLI11.hpp"

// ---------------------------------------------------------------------
// Morton encoding (1->1 relation)
// ---------------------------------------------------------------------

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

  constexpr auto threads = 10;

  BS::thread_pool pool;

  BS::timer timer;

  timer.start();

  cpu::dispatch_morton_code(pool, threads, u_points, u_morton, min_coord, range)
      .wait();

  timer.stop();

  // peek 32 morton codes
  for (auto i = 0; i < 32; ++i) {
    std::cout << u_morton[i] << std::endl;
  }

  std::cout << "Elapsed time: " << timer.ms() << " ms" << std::endl;

  return 0;
}