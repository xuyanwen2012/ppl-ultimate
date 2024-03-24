#include <iostream>
#include <thread>

#include "main.h"
#include "third-party/CLI11.hpp"

void printCudaDeviceInfo() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    std::cout << "No CUDA devices found" << std::endl;
  } else {
    std::cout << "CUDA Device(s) Information:" << std::endl;
    for (int i = 0; i < deviceCount; ++i) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      std::cout << "Device " << i << ": " << prop.name << std::endl;
      std::cout << "  Compute Capability: " << prop.major << "." << prop.minor
                << std::endl;
      std::cout << "  Total Global Memory: "
                << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
      std::cout << "  Multiprocessors: " << prop.multiProcessorCount
                << std::endl;
      std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock
                << std::endl;
      std::cout << "  Max Threads per Multiprocessor: "
                << prop.maxThreadsPerMultiProcessor << std::endl;
      std::cout << "  GPU Clock Rate: " << prop.clockRate / 1000 << " MHz"
                << std::endl;
      std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000
                << " MHz" << std::endl;
      std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits"
                << std::endl;
      std::cout << "  L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB"
                << std::endl;
      std::cout << std::endl;
    }
  }
}

int main(const int argc, const char *argv[]) {
  printCudaDeviceInfo();

  const auto max_cpu_cores = std::thread::hardware_concurrency();

  int n_threads = 1;

  CLI::App app{"Demo"};

  app.add_option("-t,--threads", n_threads, "Number of threads")
      ->check(CLI::Range(1u, max_cpu_cores));

  CLI11_PARSE(app, argc, argv);

  demo(n_threads);

  return 0;
}