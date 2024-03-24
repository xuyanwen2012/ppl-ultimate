
#include <algorithm>
#include <numeric>
#include <vector>

#include "barrier.hpp"
#include "shared/morton_func.h"

constexpr int BASE_BITS = 8;
constexpr int BASE = (1 << BASE_BITS);  // 256
constexpr int MASK = (BASE - 1);        // 0xFF

constexpr int DIGITS(const unsigned int v, const int shift) {
  return (v >> shift) & MASK;
}

struct {
  std::mutex mtx;
  int bucket[BASE] = {};  // shared among threads
  std::condition_variable cv;
  size_t current_thread = 0;
} sort;

void k_binning_pass(const size_t tid,
                    barrier& barrier,
                    const morton_t* u_sort_begin,
                    const morton_t* u_sort_end,
                    std::vector<morton_t>& u_sort_alt,  // output
                    const int shift) {
  // DEBUG_PRINT("[tid ", tid, "] started. (Binning, shift=", shift, ")");

  int local_bucket[BASE] = {};

  // compute histogram (local)
  std::for_each(
      u_sort_begin, u_sort_end, [shift, &local_bucket](const morton_t& code) {
        ++local_bucket[DIGITS(code, shift)];
      });

  std::unique_lock lck(sort.mtx);

  // update to shared bucket
  for (auto i = 0; i < BASE; ++i) {
    sort.bucket[i] += local_bucket[i];
  }

  lck.unlock();

  barrier.wait();

  if (tid == 0) {
    std::partial_sum(std::begin(sort.bucket),
                     std::end(sort.bucket),
                     std::begin(sort.bucket));
  }

  barrier.wait();

  lck.lock();
  sort.cv.wait(lck, [&] { return tid == sort.current_thread; });

  // update the local_bucket from the shared bucket
  for (auto i = 0; i < BASE; i++) {
    sort.bucket[i] -= local_bucket[i];
    local_bucket[i] = sort.bucket[i];
  }

  --sort.current_thread;
  sort.cv.notify_all();

  lck.unlock();

  std::for_each(u_sort_begin,
                u_sort_end,
                [shift, &local_bucket, &u_sort_alt](const morton_t& code) {
                  u_sort_alt[local_bucket[DIGITS(code, shift)]++] = code;
                });

  // DEBUG_PRINT("[tid ", tid, "] ended. (Binning, shift=", shift, ")");
}
