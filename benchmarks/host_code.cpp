#include "host_code.hpp"

#include "host/all.hpp"
#include "host/barrier.hpp"
#include "third-party/BS_thread_pool.hpp"

namespace cpu {
// by default it uses maximum number of threads on the System, great!
BS::thread_pool pool;

void init(int n_threads) { pool.reset(n_threads); }

void dispatch_ComputeMorton(const int n_threads, pipe* p) {
  cpu::dispatch_morton_code(pool,
                            n_threads,
                            p->n_input(),
                            p->u_points,
                            p->u_morton,
                            p->min_coord,
                            p->range)
      .wait();
}

void dispatch_RadixSort(const int n_threads, pipe* p) {
  barrier bar(n_threads);
  dispatch_binning_pass(
      pool, n_threads, bar, p->n_input(), p->u_morton, p->u_morton_alt, 0)
      .wait();
  dispatch_binning_pass(
      pool, n_threads, bar, p->n_input(), p->u_morton_alt, p->u_morton, 8)
      .wait();
  dispatch_binning_pass(
      pool, n_threads, bar, p->n_input(), p->u_morton, p->u_morton_alt, 16)
      .wait();
  dispatch_binning_pass(
      pool, n_threads, bar, p->n_input(), p->u_morton_alt, p->u_morton, 24)
      .wait();
}

void dispatch_RemoveDuplicates(const int n_threads, pipe* p) {
  auto unique_future =
      dispatch_unique(pool, p->n_input(), p->u_morton, p->u_morton_alt);
  auto n_unique = unique_future.get();
  p->set_n_unique(n_unique);
  p->brt.set_n_nodes(n_unique - 1);
}

void dispatch_BuildRadixTree(const int n_threads, pipe* p) {
  dispatch_build_radix_tree(pool, n_threads, p->u_morton_alt, &p->brt).wait();
}

void dispatch_EdgeCount(const int n_threads, pipe* p) {
  dispatch_edge_count(pool, n_threads, &p->brt, p->u_edge_counts).wait();
}

void dispatch_EdgeOffset(const int n_threads, pipe* p) {
  dispatch_edge_offset(pool, n_threads, p->u_edge_counts, p->u_edge_offsets)
      .wait();
}

void dispatch_BuildOctree(const int n_threads, pipe* p) {
  dispatch_make_oct_node(pool,
                         n_threads,
                         p->u_edge_offsets,
                         p->u_edge_counts,
                         p->getUniqueKeys(),
                         p->brt,
                         p->oct,
                         p->min_coord,
                         p->range)
      .wait();
  dispatch_link_leaf(pool,
                     n_threads,
                     p->u_edge_offsets,
                     p->u_edge_counts,
                     p->getUniqueKeys(),
                     p->brt,
                     p->oct)
      .wait();
}
}  // namespace cpu