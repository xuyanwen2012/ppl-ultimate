

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "shared/structures.h"
#include "third-party/BS_thread_pool.hpp"

void gen_data(const struct pipe* p) {
  std::mt19937 gen(p->seed);
  std::uniform_real_distribution dis(p->min_coord, p->min_coord + p->range);
  std::generate_n(p->u_points, p->n_input(), [&] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });
}

BS::thread_pool pool;

int main() {
  constexpr auto n = 640 * 480;  // ~300k

  std::vector<std::unique_ptr<pipe>> pipes;
  for (auto i = 0; i < 10; ++i) {
    pipes.push_back(std::make_unique<pipe>(n));
    gen_data(pipes.back().get());
  }

  std::cout << "done\n";
  return 0;
}