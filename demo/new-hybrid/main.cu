#include <iostream>
#include <thread>

#include "demos.cuh"
#include "third-party/CLI11.hpp"

int main(const int argc, const char* argv[]) {
  int n_threads = 1;
  int n_grid_size = 4;
  int demo_id = 0;

  const auto max_threads = static_cast<int>(std::thread::hardware_concurrency());

  CLI::App app{"Demo for new hybrid version"};

  app.add_option("-t,--threads", n_threads, "Number of threads")
      ->check(CLI::Range(1, max_threads));
  app.add_option("-d,--demo", demo_id, "Demo id");
  app.add_option("-g,--grid", n_grid_size, "Grid size");

  CLI11_PARSE(app, argc, argv);

  std::cout << "Threads: " << n_threads << '\n';
  std::cout << "Grid size: " << n_grid_size << '\n';
  std::cout << "Demo id: " << demo_id << '\n';

  switch (demo_id) {
    case 0:
      demo_cpu_only(n_threads);
      break;
    case 1:
      demo_gpu_only(n_grid_size);
      break;
    case 2:
      demo_simple_cpu_gpu_coarse(n_threads, n_grid_size);
      break;
    default:
      std::cerr << "Invalid demo id" << '\n';
      break;
  }

  return 0;
}