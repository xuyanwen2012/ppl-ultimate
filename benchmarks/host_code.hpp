#pragma once

#include <vector>

namespace cpu {

void start_thread_pool(int n_threads, std::vector<int> cores);

}