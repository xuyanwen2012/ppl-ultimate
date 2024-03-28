

#include <algorithm>
#include <cstddef>
#include <glm/fwd.hpp>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "block.hpp"
#include "host/barrier.hpp"
#include "host/brt_func.hpp"
#include "shared/edge_func.h"
#include "shared/oct_func.h"
#include "shared/structures.h"
#include "third-party/BS_thread_pool.hpp"
#include "third-party/BS_thread_pool_utils.hpp"

void gen_data(const struct pipe* p) {
  std::mt19937 gen(p->seed);
  std::uniform_real_distribution dis(p->min_coord, p->min_coord + p->range);
  std::generate_n(p->u_points, p->n_input(), [&] {
    return glm::vec4(dis(gen), dis(gen), dis(gen), 1.0f);
  });
}

BS::thread_pool pool;

// ----------------------------------------------------------------------------
// Sort
// ----------------------------------------------------------------------------

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

void reset_sort(const size_t n_threads) {
  std::fill_n(sort.bucket, BASE, 0);
  sort.current_thread = n_threads - 1;
}

void k_binning_pass(const size_t tid,
                    barrier& barrier,
                    const morton_t* u_sort_begin,
                    const morton_t* u_sort_end,
                    morton_t* u_sort_alt,  // output
                    const int shift) {
  int local_bucket[BASE] = {};

  // compute histogram (local)
  std::for_each(u_sort_begin, u_sort_end, [&](const morton_t& code) {
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

  std::for_each(u_sort_begin, u_sort_end, [&](auto code) {
    u_sort_alt[local_bucket[DIGITS(code, shift)]++] = code;
  });
}

// ----------------------------------------------------------------------------
// Wrappers (passing all arguments)
// ----------------------------------------------------------------------------

void wrapper_morton(pipe* p, int start, int end) {
  std::transform(
      p->u_points + start,
      p->u_points + end,
      p->u_morton + start,
      [min_coord = p->min_coord, range = p->range](const glm::vec4& xyz) {
        return shared::xyz_to_morton32(xyz, min_coord, range);
      });
}

void wrapper_bin_pass(pipe* p,
                      barrier& bar,
                      int start,
                      int end,
                      int bin_pass /* 0..3 */,
                      int tid,
                      int n_threads) {
  if (tid == 0) reset_sort(n_threads);
  bar.wait();
  if (bin_pass % 2 == 0) {
    k_binning_pass(tid,
                   bar,
                   p->u_morton + start,
                   p->u_morton + end,
                   p->u_morton_alt,
                   bin_pass * 8);
  } else {
    k_binning_pass(tid,
                   bar,
                   p->u_morton_alt + start,
                   p->u_morton_alt + end,
                   p->u_morton,
                   bin_pass * 8);
  }
}

void wrapper_remove_dups(pipe* p, const int tid) {
  if (tid == 0) {
    const auto last = std::unique_copy(
        p->u_morton, p->u_morton + p->n_input(), p->u_morton_alt);
    const auto n_unique = std::distance(p->u_morton_alt, last);

    p->set_n_unique(static_cast<int>(n_unique));
    p->brt.set_n_nodes(n_unique - 1);
  }
}

void wrapper_radix_tree(pipe* p, int start, int end) {
  for (auto i = start; i < end; ++i) {
    // boundary check
    if (i < p->n_brt_nodes()) {
      cpu::process_radix_tree_i(i, p->n_brt_nodes(), p->u_morton_alt, &p->brt);
    }
  }
}

void wrapper_edge_count(pipe* p, int start, int end) {
  // Kernel E and F (1/2)
  for (auto i = start; i < end; ++i) {
    if (i < p->n_brt_nodes()) {
      shared::process_edge_count_i(
          i, p->brt.u_prefix_n, p->brt.u_parents, p->u_edge_counts);
    }
  }
}

void wrapper_edge_offset(pipe* p, int tid) {
  if (tid == 0) {
    std::partial_sum(p->u_edge_counts,
                     p->u_edge_counts + p->n_brt_nodes(),
                     p->u_edge_offsets);
    const auto n_oct_nodes = p->u_edge_offsets[p->n_brt_nodes() - 1];
    p->oct.set_n_nodes(n_oct_nodes);
  }
}

void wrapper_make_oct_nodes(pipe* p, int start, int end) {
  for (auto i = start; i < end; ++i) {
    if (i < p->n_brt_nodes()) {
      shared::process_oct_node(i,
                               p->oct.u_children,
                               p->oct.u_corner,
                               p->oct.u_cell_size,
                               p->oct.u_child_node_mask,
                               p->u_edge_offsets,
                               p->u_edge_counts,
                               p->u_morton_alt,  // sorted unique morton
                               p->brt.u_prefix_n,
                               p->brt.u_parents,
                               p->min_coord,
                               p->range);
    }
  }
}

void wrapper_link_oct_nodes(pipe* p, int start, int end) {
  for (auto i = start; i < end; ++i) {
    if (i < p->n_brt_nodes()) {
      shared::process_link_leaf(i,
                                p->oct.u_children,
                                p->oct.u_child_leaf_mask,
                                p->u_edge_offsets,
                                p->u_edge_counts,
                                p->u_morton_alt,  // sorted unique morton
                                p->brt.u_has_leaf_left,
                                p->brt.u_has_leaf_right,
                                p->brt.u_prefix_n,
                                p->brt.u_parents,
                                p->brt.u_left_child);
    }
  }
}

// ----------------------------------------------------------------------------
// CPU only, maximum threads, do everything
// ----------------------------------------------------------------------------

// let's say we have 6 threads.
// should be
// 0.771 + 2.10 + 0.511 + 4.20 + 0.208 + 0.020 + 4.54 = ~12.359 ms
void example_pipe(std::unique_ptr<pipe>& p) {
  const auto desired_n_threads = 6;

  // partition the input data [0, n) into blocks
  const my_blocks blks(0, p->n_input(), desired_n_threads);

  BS::multi_future<void> future;
  future.reserve(blks.get_num_blocks());
  barrier bar(desired_n_threads);

  BS::timer t;
  t.start();

  for (size_t tid = 0; tid < blks.get_num_blocks(); ++tid) {
    future.push_back(pool.submit_task(
        [start = blks.start(tid), end = blks.end(tid), tid, &bar, p = p.get()] {
          wrapper_morton(p, start, end);
          bar.wait();

          wrapper_bin_pass(p, bar, start, end, 0, tid, desired_n_threads);
          bar.wait();
          wrapper_bin_pass(p, bar, start, end, 1, tid, desired_n_threads);
          bar.wait();
          wrapper_bin_pass(p, bar, start, end, 2, tid, desired_n_threads);
          bar.wait();
          wrapper_bin_pass(p, bar, start, end, 3, tid, desired_n_threads);
          bar.wait();

          wrapper_remove_dups(p, tid);
          bar.wait();

          wrapper_radix_tree(p, start, end);
          bar.wait();

          wrapper_edge_count(p, start, end);
          bar.wait();

          wrapper_edge_offset(p, tid);
          bar.wait();

          wrapper_make_oct_nodes(p, start, end);
          bar.wait();

          wrapper_link_oct_nodes(p, start, end);
          // bar.wait();
        }));
  }

  future.wait();

  t.stop();

  std::cout << "Time: " << t.ms() << "ms\n";

  auto is_sorted = std::is_sorted(p->u_morton, p->u_morton + p->n_input());
  std::cout << "is_sorted: " << std::boolalpha << is_sorted << '\n';

  std::cout << "n_unique: " << p->n_unique_mortons() << '\n';
  std::cout << "n_nodes: " << p->brt.n_nodes() << '\n';
}

int main() {
  constexpr auto n = 640 * 480;  // ~300k

  // std::vector<std::unique_ptr<pipe>> pipes;
  // for (auto i = 0; i < 4; ++i) {
  //   pipes.push_back(std::make_unique<pipe>(n));
  //   gen_data(pipes.back().get());
  // }

  std::unique_ptr<pipe> p = std::make_unique<pipe>(n);

  gen_data(p.get());

  example_pipe(p);

  std::cout << "done\n";
  return 0;
}