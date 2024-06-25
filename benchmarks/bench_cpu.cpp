#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
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

unsigned max_threads;
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
    cpu::dispatch_ComputeMorton(max_threads, p.get(), 0);
    cpu::dispatch_RadixSort(max_threads, p.get(), 0);
    cpu::dispatch_RemoveDuplicates(max_threads, p.get(), 0);
    cpu::dispatch_BuildRadixTree(max_threads, p.get(), 0);
    cpu::dispatch_EdgeCount(max_threads, p.get(), 0);
    cpu::dispatch_EdgeOffset(max_threads, p.get(), 0);
    cpu::dispatch_BuildOctree(max_threads, p.get(), 0);
  }

  std::unique_ptr<pipe> p;
};

// --------------------------------------------------
// Morton
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_Morton)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_ComputeMorton(n_threads, p.get(), 0);
  }
}

// --------------------------------------------------
// Sort
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_Sort)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_RadixSort(n_threads, p.get(), 0);
  }
}

// --------------------------------------------------
// Unique
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_RemoveDup)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_RemoveDuplicates(n_threads, p.get(), 0);
  }
}

// --------------------------------------------------
// Radix Tree
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_RadixTree)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_BuildRadixTree(n_threads, p.get(), 0);
  }
}

// --------------------------------------------------
// Edge Count
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_EdgeCount)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_EdgeCount(n_threads, p.get(), 0);
  }
}

// --------------------------------------------------
// Edge Offset
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_EdgeOffset)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_EdgeOffset(n_threads, p.get(), 0);
  }
}

// --------------------------------------------------
// Octree
// --------------------------------------------------

BENCHMARK_DEFINE_F(CPU, BM_Octree)(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    cpu::dispatch_BuildOctree(n_threads, p.get(), 0);
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

void printCpuInfo() {
  std::ifstream cpuInfo("/proc/cpuinfo");
  if (!cpuInfo.is_open()) {
    std::cerr << "Unable to open file" << std::endl;
    return;
  }

  std::string content((std::istreambuf_iterator<char>(cpuInfo)),
                      std::istreambuf_iterator<char>());
  cpuInfo.close();

  // New regex pattern to handle newlines correctly
  std::regex pattern(R"(processor\s*:\s*(\d+)[\s\S]*?BogoMIPS\s*:\s*([\d\.]+))",
                     std::regex::ECMAScript);
  std::map<float, std::vector<int>> coreBogoMIPS;

  auto begin = std::sregex_iterator(content.begin(), content.end(), pattern);
  auto end = std::sregex_iterator();

  for (auto i = begin; i != end; ++i) {
    std::smatch match = *i;
    int processorId = std::stoi(match[1].str());
    float bogoMIPS = std::stof(match[2].str());
    coreBogoMIPS[bogoMIPS].push_back(processorId);
  }

  // Print header
  std::cout << "------------------------------------------" << std::endl;
  std::cout << "| Core Type       | BogoMIPS | Cores      |" << std::endl;
  std::cout << "------------------------------------------" << std::endl;

  int typeCount = 0;
  for (const auto& entry : coreBogoMIPS) {
    std::cout << "| Core Type " << ++typeCount << "     | " << entry.first
              << "    | ";
    for (auto it = entry.second.begin(); it != entry.second.end(); ++it) {
      if (it != entry.second.begin()) {
        std::cout << ", ";
      }
      std::cout << *it;
    }
    std::cout << " |" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
  }
}

int main(int argc, char** argv) {
  CLI::App app{"UCSC Redwood: CPU Tree Construction/Traversal Pipeline"};
  app.allow_extras();

  bool list_cores = false;
  app.add_flag("-i,--info", list_cores, "Get the CPU information");

  bool benchmark_mode = false;
  app.add_flag("-b,--benchmark", benchmark_mode, "Run a pipeline benchmark");

  std::vector<int> core_ids;
  app.add_option("-c,--cores",
                 core_ids,
                 "Specific core IDs to run the benchmark on. ex: 0 1 2 3 4 5");

  bool pipeline_mode = false;
  app.add_flag(
      "-p,--pipeline", pipeline_mode, "Run a single pipeline instance");

  std::vector<std::string> core_groups_str;
  app.add_option("-g,--groups",
                 core_groups_str,
                 "Thread pool core groups for pipeline stages (comma-separated "
                 "core IDs) ex: 0,1,2 3,4,5");

  CLI11_PARSE(app, argc, argv);

  // Determine the maximum number of threads supported by the hardware
  max_threads = std::thread::hardware_concurrency();

  if (list_cores) {
    printCpuInfo();
    return 0;
  }

  if (pipeline_mode) {
    if (core_groups_str.empty()) {
    }
    //  Parse the core groups
    std::vector<std::vector<int>> core_groups;
    for (const auto& group : core_groups_str) {
      std::vector<int> core_group;
      std::stringstream ss(group);
      int core_id;
      while (ss >> core_id) {
        core_group.push_back(core_id);
        if (ss.peek() == ',') {
          ss.ignore();
        }
      }
      core_groups.push_back(core_group);
    }
    // print core_groups
    for (const auto& group : core_groups) {
      std::cout << "Core group: ";
      for (int id : group) {
        std::cout << id << " ";
      }
      std::cout << std::endl;
    }
    // TODO: Validate the core groups

    // TODO: Run the pipeline
  } else {
    // Declare a 2D vector to store the core IDs
    std::vector<std::vector<int>> core_ids_2d_vec;
    // Adjust max_threads based on the number of specified cores
    if (!core_ids.empty()) {
      max_threads = core_ids.size();
      // Print the benchmark information
      std::cout << "Running benchmark on cores:";
      for (int core_id : core_ids) {
        std::cout << core_id << " ";
      }
      core_ids_2d_vec.push_back(core_ids);
    } else {
      // Print the benchmark information
      std::cout << "Running benchmark on all cores";
    }
    std::cout << std::endl;

    // Start the thread manager
    cpu::start_thread_manager(core_ids_2d_vec);
    // Register benchmarks
    register_benchmarks();
    // Initialize Google Benchmark
    benchmark::Initialize(&argc, argv);
    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
  }
  return 0;
}
