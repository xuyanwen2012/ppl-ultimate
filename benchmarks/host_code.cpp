
#include <sched.h>

#include "third-party/BS_thread_pool.hpp"
namespace cpu {

std::vector<int> cores = {0};
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
    printf("Thread %u set to CPU %d\n", std::this_thread::get_id(), core);
  }
  if (cores.empty()) {
    cores = {0, 7};
  }
}

// void pin_thread() {
//   cpu_set_t cpuset;
//   CPU_ZERO(&cpuset);
//   CPU_SET(5, &cpuset);
//   if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
//     perror("sched_setaffinity failed");
//   } else {
//     printf("Thread %u set to CPU %d\n", std::this_thread::get_id(), 5);
//   }
// }
// by default it uses maximum number of threads on the System, great!
BS::thread_pool pool;

}  // namespace cpu
