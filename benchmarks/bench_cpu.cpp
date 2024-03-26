#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <thread>

#include "host_code.hpp"

// Problem size
constexpr auto n = 1920 * 1080;  // ~2M
constexpr auto min_coord = 0.0f;
constexpr auto range = 1024.0f;
constexpr auto seed = 114514;

// Bench mark config
constexpr auto n_iterations = 20;

void gen_data(const std::unique_ptr<pipe>& p) {
  std::mt19937 gen(seed);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution dis(min_coord, min_coord + range);
  std::generate_n(p->u_points, n, [&dis, &gen] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });
}

class MyFixture : public benchmark::Fixture {
 public:
  explicit MyFixture() : p(std::make_unique<pipe>(n, min_coord, range, seed)) {
    gen_data(p);

    // basically pregenerate the data
    const auto max_thread = std::thread::hardware_concurrency();
    cpu::dispatch_ComputeMorton(max_thread, p.get());
    cpu::dispatch_RadixSort(max_thread, p.get());
    cpu::dispatch_RemoveDuplicates(max_thread, p.get());
    cpu::dispatch_BuildRadixTree(max_thread, p.get());
    cpu::dispatch_EdgeCount(max_thread, p.get());
    cpu::dispatch_EdgeOffset(max_thread, p.get());
    cpu::dispatch_BuildOctree(max_thread, p.get());
  }

  std::unique_ptr<pipe> p;
};

// --------------------------------------------------
// Morton
// --------------------------------------------------

BENCHMARK_DEFINE_F(MyFixture, BM_Morton)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_ComputeMorton(n_threads, p.get());
  }
}

BENCHMARK_REGISTER_F(MyFixture, BM_Morton)
    ->DenseRange(1, 6, 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(n_iterations);

// --------------------------------------------------
// Sort
// --------------------------------------------------

BENCHMARK_DEFINE_F(MyFixture, BM_Sort)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_RadixSort(n_threads, p.get());
  }
}

BENCHMARK_REGISTER_F(MyFixture, BM_Sort)
    ->DenseRange(1, 6, 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(n_iterations);

// --------------------------------------------------
// Unique
// --------------------------------------------------

BENCHMARK_DEFINE_F(MyFixture, BM_RemoveDup)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_RemoveDuplicates(n_threads, p.get());
  }
}

BENCHMARK_REGISTER_F(MyFixture, BM_RemoveDup)
    ->DenseRange(1, 6, 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(n_iterations);

// --------------------------------------------------
// Radix Tree
// --------------------------------------------------

BENCHMARK_DEFINE_F(MyFixture, BM_RadixTree)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_BuildRadixTree(n_threads, p.get());
  }
}

BENCHMARK_REGISTER_F(MyFixture, BM_RadixTree)
    ->DenseRange(1, 6, 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(n_iterations);

// --------------------------------------------------
// Edge Count
// --------------------------------------------------

BENCHMARK_DEFINE_F(MyFixture, BM_EdgeCount)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_EdgeCount(n_threads, p.get());
  }
}

BENCHMARK_REGISTER_F(MyFixture, BM_EdgeCount)
    ->DenseRange(1, 6, 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(n_iterations);

// --------------------------------------------------
// Edge Offset
// --------------------------------------------------

BENCHMARK_DEFINE_F(MyFixture, BM_EdgeOffset)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_EdgeOffset(n_threads, p.get());
  }
}

BENCHMARK_REGISTER_F(MyFixture, BM_EdgeOffset)
    ->DenseRange(1, 6, 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(n_iterations);

// --------------------------------------------------
// Octree
// --------------------------------------------------

BENCHMARK_DEFINE_F(MyFixture, BM_Octree)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_BuildOctree(n_threads, p.get());
  }
}

BENCHMARK_REGISTER_F(MyFixture, BM_Octree)
    ->DenseRange(1, 6, 1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(n_iterations);

BENCHMARK_MAIN();
