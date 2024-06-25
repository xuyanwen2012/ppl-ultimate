#pragma once

#include <sched.h>

#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "third-party/BS_thread_pool.hpp"

class ThreadPoolManager {
 public:
  ThreadPoolManager() = default;

  ThreadPoolManager(const std::vector<std::vector<int>>& core_groups) {
    initialize(core_groups);
  }

  void initialize(const std::vector<std::vector<int>>& core_groups) {
    thread_pools.clear();
    core_list.clear();

    if (core_groups.empty()) {
      thread_pools.emplace_back(std::make_unique<BS::thread_pool>());
    } else {
      for (const auto& cores : core_groups) {
        // Make sure that the core list is empty before adding new cores
        while (!core_list.empty()) {
        }
        // Store the cores in core_lists
        core_list.insert(core_list.end(), cores.begin(), cores.end());
        // Create thread pool
        thread_pools.emplace_back(std::make_unique<BS::thread_pool>(
            cores.size(), [this]() { pin_thread(); }));
      }
    }
    // Make sure that the core list is empty before ending
    while (!core_list.empty()) {
    }
  }

  // Function to get a thread pool based on the core group index
  BS::thread_pool& get_thread_pool(size_t core_group_index) {
    if (core_group_index >= thread_pools.size() || core_group_index < 0) {
      throw std::out_of_range("Invalid core group index");
    }
    return *thread_pools[core_group_index];
  }

  ~ThreadPoolManager() {
    for (auto& pool : thread_pools) {
      pool->~thread_pool();
    }
  }

 private:
  // Function to pin a thread to a specific core
  void pin_thread() {
    std::lock_guard<std::mutex> lock(core_list_mutex);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    int core = core_list.back();
    core_list.pop_back();
    CPU_SET(core, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
      perror("sched_setaffinity failed");
    } else {
      printf("Thread %u pinned to Core %d\n", std::this_thread::get_id(), core);
    }
  }

  std::vector<std::unique_ptr<BS::thread_pool>> thread_pools;
  std::vector<int> core_list;  // Store core lists
  std::mutex core_list_mutex;
};
