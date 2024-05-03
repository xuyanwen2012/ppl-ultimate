#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <thread>

#include "host/dispatcher.hpp"

// Problem size
// constexpr auto n = 640 * 480;  // ~300k
constexpr auto n = 1920 * 1080;  // ~2M
constexpr auto min_coord = 0.0f;
constexpr auto range = 1024.0f;
constexpr auto seed = 114514;

// Bench mark config
constexpr auto n_iterations = 5;
auto max_threads = 1;  // std::thread::hardware_concurrency();

void gen_data(const std::unique_ptr<pipe>& p) {
  std::mt19937 gen(seed);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution dis(min_coord, min_coord + range);
  std::generate_n(p->u_points, n, [&dis, &gen] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });
}

// ---------------------------------------------------------------------
static void BM_ComputeMorton(benchmark::State& state) {
  const auto n_threads = state.range(0);
  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);

  for (auto _ : state) {
    cpu::dispatch_ComputeMorton(n_threads, p.get());
  }
}

BENCHMARK(BM_ComputeMorton)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, max_threads)
    ->Iterations(n_iterations)
    ->ArgName("Threads");

// ---------------------------------------------------------------------

static void BM_RadixSort(benchmark::State& state) {
  const auto n_threads = state.range(0);
  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);
  cpu::dispatch_ComputeMorton(n_threads, p.get());

  for (auto _ : state) {
    cpu::dispatch_RadixSort(n_threads, p.get());
  }
}

BENCHMARK(BM_RadixSort)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, max_threads)
    ->Iterations(n_iterations)
    ->ArgName("Threads");

// ---------------------------------------------------------------------

static void BM_RemoveDuplicates(benchmark::State& state) {
  const auto n_threads = state.range(0);
  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);
  cpu::dispatch_ComputeMorton(n_threads, p.get());
  cpu::dispatch_RadixSort(n_threads, p.get());

  for (auto _ : state) {
    cpu::dispatch_RemoveDuplicates(n_threads, p.get());
  }
}

BENCHMARK(BM_RemoveDuplicates)
    //->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, max_threads)
    ->Iterations(n_iterations)
    ->ArgName("Threads");

// ---------------------------------------------------------------------

static void BM_BuildRadixTree(benchmark::State& state) {
  const auto n_threads = state.range(0);
  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);
  cpu::dispatch_ComputeMorton(n_threads, p.get());
  cpu::dispatch_RadixSort(n_threads, p.get());
  cpu::dispatch_RemoveDuplicates(n_threads, p.get());

  for (auto _ : state) {
    cpu::dispatch_BuildRadixTree(n_threads, p.get());
  }
}

BENCHMARK(BM_BuildRadixTree)
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, max_threads)
    ->Iterations(n_iterations)
    ->ArgName("Threads");

// ---------------------------------------------------------------------

static void BM_EdgeCount(benchmark::State& state) {
  const auto n_threads = state.range(0);
  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);
  cpu::dispatch_ComputeMorton(n_threads, p.get());
  cpu::dispatch_RadixSort(n_threads, p.get());
  cpu::dispatch_RemoveDuplicates(n_threads, p.get());
  cpu::dispatch_BuildRadixTree(n_threads, p.get());

  for (auto _ : state) {
    cpu::dispatch_EdgeCount(n_threads, p.get());
  }
}

BENCHMARK(BM_EdgeCount)
    //->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, max_threads)
    ->Iterations(n_iterations)
    ->ArgName("Threads");

// ---------------------------------------------------------------------

// static void BM_EdgeOffset(benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   auto p = std::make_unique<pipe>(n, min_coord, range, seed);
//   gen_data(p);
//   cpu::dispatch_ComputeMorton(n_threads, p.get());
//   cpu::dispatch_RadixSort(n_threads, p.get());
//   cpu::dispatch_RemoveDuplicates(n_threads, p.get());
//   cpu::dispatch_BuildRadixTree(n_threads, p.get());
//   cpu::dispatch_EdgeCount(n_threads, p.get());

//   for (auto _ : state) {
//     cpu::dispatch_EdgeOffset(n_threads, p.get());
//   }
// }

// BENCHMARK(BM_EdgeOffset)
//     ->Unit(benchmark::kMillisecond)
//     ->RangeMultiplier(2)
//     ->Range(1, max_threads)
//     ->Iterations(n_iterations)
//     ->ArgName("Threads");

// ---------------------------------------------------------------------

static void BM_BuildOctree(benchmark::State& state) {
  const auto n_threads = state.range(0);
  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);
  cpu::dispatch_ComputeMorton(n_threads, p.get());
  cpu::dispatch_RadixSort(n_threads, p.get());
  cpu::dispatch_RemoveDuplicates(n_threads, p.get());
  cpu::dispatch_BuildRadixTree(n_threads, p.get());
  cpu::dispatch_EdgeCount(n_threads, p.get());
  // cpu::dispatch_EdgeOffset(n_threads, p.get());

  for (auto _ : state) {
    cpu::dispatch_BuildOctree(n_threads, p.get());
  }
}

BENCHMARK(BM_BuildOctree)
    //->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, max_threads)
    ->Iterations(n_iterations)
    ->ArgName("Threads");

// ---------------------------------------------------------------------

BENCHMARK_MAIN();

// int main(int argc, char** argv) {
//   // benchmark::Initialize(&argc, argv);
//   // benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, {});
//   auto p = std::make_unique<pipe>(n, min_coord, range, seed);

//   gen_data(p);

//   // basically pregenerate the data
//   const auto max_thread = std::thread::hardware_concurrency();
//   cpu::dispatch_ComputeMorton(max_thread, p.get());
//   cpu::dispatch_RadixSort(max_thread, p.get());
//   cpu::dispatch_RemoveDuplicates(max_thread, p.get());
//   cpu::dispatch_BuildRadixTree(max_thread, p.get());
//   cpu::dispatch_EdgeCount(max_thread, p.get());
//   std::cout << "Starting EdgeOffset" << std::endl;
//   // cpu::dispatch_EdgeOffset(max_thread, p.get());
//   cpu::dispatch_BuildOctree(max_thread, p.get());
// }