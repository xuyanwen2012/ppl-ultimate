
#include <sched.h>

#include "third-party/BS_thread_pool.hpp"
namespace cpu {

std::vector<int> cores;
std::mutex core_mutex;

void pin_thread() {
  std::lock_guard<std::mutex> lock(core_mutex);
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  int core = cores.back();
  cores.pop_back();
  CPU_SET(core, &cpuset);
  if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
    perror("sched_setaffinity failed");
  } else {
    printf("Thread %u pinned to Core %d\n", std::this_thread::get_id(), core);
  }
}

BS::thread_pool pool; // By default, use all cores, but can be changed by start_thread_pool

void start_thread_pool(int n_threads, std::vector<int> cores) {
  if (cores.size() > 0) {
    cpu::cores = cores;
    pool.~thread_pool();
    new (&pool) BS::thread_pool(n_threads, pin_thread);
  }
}

}  // namespace cpu
