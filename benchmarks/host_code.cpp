

#include "host/manager.hpp"
#include "third-party/BS_thread_pool.hpp"
namespace cpu {

ThreadPoolManager manager;

void start_thread_manager(const std::vector<std::vector<int>>& core_groups) {
  new (&manager) ThreadPoolManager(core_groups);
}

}  // namespace cpu
