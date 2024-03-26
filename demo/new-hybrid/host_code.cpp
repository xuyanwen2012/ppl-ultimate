#include "host_code.hpp"

#include "host/all.hpp"
#include "host/barrier.hpp"
#include "third-party/BS_thread_pool.hpp"

namespace cpu {

// by default it uses maximum number of threads on the System, great!
BS::thread_pool pool;
barrier* sort_barrier = nullptr;  // need to be initialized

void dispatch_ComputeMorton(int n_threads, struct pipe* p) {
  cpu::dispatch_morton_code(pool,
                            n_threads,
                            p->n_input(),
                            p->u_points,
                            p->u_morton,
                            p->min_coord,
                            p->range)
      .wait();
}

void dispatch_RadixSort(int n_threads, struct pipe* p) {
  if (sort_barrier == nullptr) {
    sort_barrier = new barrier(n_threads);
  }

  cpu::dispatch_binning_pass(pool,
                             n_threads,
                             *sort_barrier,
                             p->n_input(),
                             p->u_morton,
                             p->u_morton_alt,
                             0)
      .wait();
  cpu::dispatch_binning_pass(pool,
                             n_threads,
                             *sort_barrier,
                             p->n_input(),
                             p->u_morton_alt,
                             p->u_morton,
                             8)
      .wait();
  cpu::dispatch_binning_pass(pool,
                             n_threads,
                             *sort_barrier,
                             p->n_input(),
                             p->u_morton,
                             p->u_morton_alt,
                             16)
      .wait();
  cpu::dispatch_binning_pass(pool,
                             n_threads,
                             *sort_barrier,
                             p->n_input(),
                             p->u_morton_alt,
                             p->u_morton,
                             24)
      .wait();
}

void dispatch_RemoveDuplicates(int n_threads, struct pipe* p);

void dispatch_BuildRadixTree(int n_threads, struct pipe* p);

void dispatch_EdgeCount(int n_threads, struct pipe* p);

void dispatch_EdgeOffset(int n_threads, struct pipe* p);

void dispatch_BuildOctree(int n_threads, struct pipe* p);

}  // namespace cpu