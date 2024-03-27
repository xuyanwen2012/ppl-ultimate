#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>

#include "cuda/dispatcher.cuh"
#include "cuda_bm_timer.cuh"

// Problem size
// constexpr auto n = 1920 * 1080;  // ~2M
constexpr auto n = 640 * 480;
constexpr auto min_coord = 0.0f;
constexpr auto range = 1024.0f;
constexpr auto seed = 114514;

// Bench mark config
constexpr auto n_iterations = 20;

void gen_data(const std::unique_ptr<struct pipe>& p) {
  std::mt19937 gen(seed);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution dis(min_coord, min_coord + range);
  std::generate_n(p->u_points, n, [&dis, &gen] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });
}

// --------------------------------------------------
// Morton
// --------------------------------------------------

void BM_GPU_Morton(benchmark::State& st) {
  const auto grid_size = st.range(0);
  const auto block_size = 768;

  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);

  for (auto _ : st) {
    CudaEventTimer timer(st, true);
    gpu::dispatch_ComputeMorton(grid_size, 0, p.get());
  }
}

BENCHMARK(BM_GPU_Morton)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->Iterations(n_iterations)
    ->ArgName("GridSize");

// --------------------------------------------------
// Sort
// --------------------------------------------------

void BM_GPU_Sort(benchmark::State& st) {
  const auto grid_size = st.range(0);

  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);
  gpu::dispatch_ComputeMorton(grid_size, 0, p.get());

  for (auto _ : st) {
    CudaEventTimer timer(st, true);
    gpu::dispatch_RadixSort(grid_size, 0, p.get());
  }
}

BENCHMARK(BM_GPU_Sort)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->Iterations(n_iterations)
    ->ArgName("GridSize");

// --------------------------------------------------
// Unique
// --------------------------------------------------

void BM_GPU_Unique(benchmark::State& st) {
  const auto grid_size = st.range(0);

  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);
  gpu::dispatch_ComputeMorton(grid_size, 0, p.get());
  gpu::dispatch_RadixSort(grid_size, 0, p.get());

  for (auto _ : st) {
    CudaEventTimer timer(st, true);
    gpu::dispatch_RemoveDuplicates_sync(grid_size, 0, p.get());
  }
}

BENCHMARK(BM_GPU_Unique)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->Iterations(n_iterations)
    ->ArgName("GridSize");

// --------------------------------------------------
// Radix Tree
// --------------------------------------------------

void BM_GPU_RadixTree(benchmark::State& st) {
  const auto grid_size = st.range(0);

  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);
  gpu::dispatch_ComputeMorton(grid_size, 0, p.get());
  gpu::dispatch_RadixSort(grid_size, 0, p.get());
  gpu::dispatch_RemoveDuplicates_sync(grid_size, 0, p.get());

  for (auto _ : st) {
    CudaEventTimer timer(st, true);
    gpu::dispatch_BuildRadixTree(grid_size, 0, p.get());
  }
}

BENCHMARK(BM_GPU_RadixTree)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->Iterations(n_iterations)
    ->ArgName("GridSize");

// --------------------------------------------------
// Edge Count
// --------------------------------------------------

void BM_GPU_EdgeCount(benchmark::State& st) {
  const auto grid_size = st.range(0);

  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);
  gpu::dispatch_ComputeMorton(grid_size, 0, p.get());
  gpu::dispatch_RadixSort(grid_size, 0, p.get());
  gpu::dispatch_RemoveDuplicates_sync(grid_size, 0, p.get());
  gpu::dispatch_BuildRadixTree(grid_size, 0, p.get());

  for (auto _ : st) {
    CudaEventTimer timer(st, true);
    gpu::dispatch_EdgeCount(grid_size, 0, p.get());
  }
}

BENCHMARK(BM_GPU_EdgeCount)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->Iterations(n_iterations)
    ->ArgName("GridSize");

// --------------------------------------------------
// Edge Offset
// --------------------------------------------------

void BM_GPU_EdgeOffset(benchmark::State& st) {
  const auto grid_size = st.range(0);

  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);
  gpu::dispatch_ComputeMorton(grid_size, 0, p.get());
  gpu::dispatch_RadixSort(grid_size, 0, p.get());
  gpu::dispatch_RemoveDuplicates_sync(grid_size, 0, p.get());
  gpu::dispatch_BuildRadixTree(grid_size, 0, p.get());
  gpu::dispatch_EdgeCount(grid_size, 0, p.get());

  for (auto _ : st) {
    CudaEventTimer timer(st, true);
    gpu::dispatch_EdgeOffset(grid_size, 0, p.get());
  }
}

BENCHMARK(BM_GPU_EdgeOffset)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->Iterations(n_iterations)
    ->ArgName("GridSize");

// --------------------------------------------------
// Octree
// --------------------------------------------------

void BM_GPU_Octree(benchmark::State& st) {
  const auto grid_size = st.range(0);

  auto p = std::make_unique<pipe>(n, min_coord, range, seed);
  gen_data(p);
  gpu::dispatch_ComputeMorton(grid_size, 0, p.get());
  gpu::dispatch_RadixSort(grid_size, 0, p.get());
  gpu::dispatch_RemoveDuplicates_sync(grid_size, 0, p.get());
  gpu::dispatch_BuildRadixTree(grid_size, 0, p.get());
  gpu::dispatch_EdgeCount(grid_size, 0, p.get());
  gpu::dispatch_EdgeOffset(grid_size, 0, p.get());

  for (auto _ : st) {
    CudaEventTimer timer(st, true);
    gpu::dispatch_BuildOctree(grid_size, 0, p.get());
  }
}

BENCHMARK(BM_GPU_Octree)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(2)
    ->Range(1, 32)
    ->Iterations(n_iterations)
    ->ArgName("GridSize");

// --------------------------------------------------
// Modified GPU bench main
// --------------------------------------------------

int main(int argc, char** argv) {
  int device_count;
  CHECK_CUDA_CALL(cudaGetDeviceCount(&device_count));

  CHECK_CUDA_CALL(cudaSetDevice(0));

  for (auto device_id = 0; device_id < device_count; ++device_id) {
    cudaDeviceProp device_prop;
    CHECK_CUDA_CALL(cudaGetDeviceProperties(&device_prop, device_id));

    std::cout << "Device ID: " << device_id << '\n';
    std::cout << "Device name: " << device_prop.name << '\n';
    std::cout << "Compute capability: " << device_prop.major << "."
              << device_prop.minor << '\n';
    std::cout << "Total global memory: "
              << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << '\n';
    std::cout << "Number of multiprocessors: "
              << device_prop.multiProcessorCount << '\n';
    std::cout << "Max threads per block: " << device_prop.maxThreadsPerBlock
              << '\n';
    std::cout << "Max threads per multiprocessor: "
              << device_prop.maxThreadsPerMultiProcessor << '\n';
    std::cout << "Warp size: " << device_prop.warpSize << '\n';
    std::cout << '\n';
  }

  gpu::initialize_dispatcher(1);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  gpu::release_dispatcher();
}