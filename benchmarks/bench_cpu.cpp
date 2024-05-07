#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <thread>

#include "host/dispatcher.hpp"
#include "host_code.hpp"
#include "third-party/CLI11.hpp"

// Problem size
constexpr auto n = 640 * 480;  // ~300k
// constexpr auto n = 1920 * 1080;  // ~2M
constexpr auto min_coord = 0.0f;
constexpr auto range = 1024.0f;
constexpr auto seed = 114514;

// Max threads
unsigned max_threads;

// Bench mark config
constexpr auto n_iterations = 50;

void gen_data(const std::unique_ptr<pipe>& p) {
  std::mt19937 gen(seed);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution dis(min_coord, min_coord + range);
  std::generate_n(p->u_points, n, [&dis, &gen] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });
}

class CPU : public benchmark::Fixture {
 public:
  explicit CPU() : p(std::make_unique<pipe>(n, min_coord, range, seed)) {
    gen_data(p);

    // basically pregenerate the data
    cpu::dispatch_ComputeMorton(max_threads, p.get());
    cpu::dispatch_RadixSort(max_threads, p.get());
    cpu::dispatch_RemoveDuplicates(max_threads, p.get());
    cpu::dispatch_BuildRadixTree(max_threads, p.get());
    cpu::dispatch_EdgeCount(max_threads, p.get());
    cpu::dispatch_EdgeOffset(max_threads, p.get());
    cpu::dispatch_BuildOctree(max_threads, p.get());
  }

  std::unique_ptr<pipe> p;
};

// --------------------------------------------------
// Morton
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_Morton)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_ComputeMorton(n_threads, p.get());
  }
}

// --------------------------------------------------
// Sort
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_Sort)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_RadixSort(n_threads, p.get());
  }
}

// --------------------------------------------------
// Unique
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_RemoveDup)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_RemoveDuplicates(n_threads, p.get());
  }
}

// --------------------------------------------------
// Radix Tree
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_RadixTree)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_BuildRadixTree(n_threads, p.get());
  }
}

// --------------------------------------------------
// Edge Count
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_EdgeCount)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_EdgeCount(n_threads, p.get());
  }
}

// --------------------------------------------------
// Edge Offset
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_EdgeOffset)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_EdgeOffset(n_threads, p.get());
  }
}

// --------------------------------------------------
// Octree
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_Octree)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_BuildOctree(n_threads, p.get());
  }
}

void register_benchmarks() {
  BENCHMARK_REGISTER_F(CPU, BM_Morton)
      ->RangeMultiplier(2)
      ->Range(1, max_threads)
      ->ArgName("Threads")
      ->Unit(benchmark::kMillisecond)
      ->Iterations(n_iterations);

  BENCHMARK_REGISTER_F(CPU, BM_Sort)
      ->RangeMultiplier(2)
      ->Range(1, max_threads)
      ->ArgName("Threads")
      ->Unit(benchmark::kMillisecond)
      ->Iterations(n_iterations);

  BENCHMARK_REGISTER_F(CPU, BM_RemoveDup)
      ->RangeMultiplier(2)
      ->Range(1, max_threads)
      ->ArgName("Threads")
      ->Unit(benchmark::kMillisecond)
      ->Iterations(n_iterations);

  BENCHMARK_REGISTER_F(CPU, BM_RadixTree)
      ->RangeMultiplier(2)
      ->Range(1, max_threads)
      ->ArgName("Threads")
      ->Unit(benchmark::kMillisecond)
      ->Iterations(n_iterations);

  BENCHMARK_REGISTER_F(CPU, BM_EdgeCount)
      ->RangeMultiplier(2)
      ->Range(1, max_threads)
      ->ArgName("Threads")
      ->Unit(benchmark::kMillisecond)
      ->Iterations(n_iterations);

  BENCHMARK_REGISTER_F(CPU, BM_EdgeOffset)
      ->RangeMultiplier(2)
      ->Range(1, max_threads)
      ->ArgName("Threads")
      ->Unit(benchmark::kMillisecond)
      ->Iterations(n_iterations);

  BENCHMARK_REGISTER_F(CPU, BM_Octree)
      ->RangeMultiplier(2)
      ->Range(1, max_threads)
      ->ArgName("Threads")
      ->Unit(benchmark::kMillisecond)
      ->Iterations(n_iterations);
}

int main(int argc, char** argv) {
  CLI::App app{"CPU Benchmark"};

  // Determine the maximum number of threads supported by the hardware
  max_threads = std::thread::hardware_concurrency();
  std::vector<int> cores = {};

  // Add options for custom core setting
  app.add_option(
         "--cores", cores, "Set specific cores to run the benchmarks on")
      ->expected(-1);  // Allows an unlimited number of arguments

  CLI11_PARSE(app, argc, argv);

  // Adjust max_threads based on the number of specified cores
  if (!cores.empty()) {
    max_threads = cores.size();
  }

  cpu::start_thread_pool(max_threads, cores);
  register_benchmarks();

  // Initialize Google Benchmark
  benchmark::Initialize(&argc, argv);

  // Optionally, check if Google Benchmark should run
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;  // Help or error message already displayed
  }

  // Register and run benchmarks
  std::cout << "Running benchmarks..." << std::endl;
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
