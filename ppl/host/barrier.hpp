#pragma once

#include <condition_variable>
#include <mutex>

class barrier {
 public:
  explicit barrier(const size_t count)
      : thread_count_(count), count_(count), generation_(0) {}

  void wait() {
    std::unique_lock lock(mutex_);
    int gen = generation_;
    if (--count_ == 0) {
      generation_++;
      count_ = thread_count_;
      cond_.notify_all();
    } else {
      cond_.wait(lock, [this, gen] { return gen != generation_; });
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable cond_;
  size_t thread_count_;
  size_t count_;
  int generation_;
};
