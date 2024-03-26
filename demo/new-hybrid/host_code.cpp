#include "host_code.hpp"

#include "host/all.hpp"
#include "host/barrier.hpp"
#include "third-party/BS_thread_pool.hpp"

namespace cpu {

BS::thread_pool pool;

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

void dispatch_RadixSort(int n_threads, struct pipe* p);

void dispatch_RemoveDuplicates(int n_threads, struct pipe* p);

void dispatch_BuildRadixTree(int n_threads, struct pipe* p);

void dispatch_EdgeCount(int n_threads, struct pipe* p);

void dispatch_EdgeOffset(int n_threads, struct pipe* p);

void dispatch_BuildOctree(int n_threads, struct pipe* p);

}  // namespace cpu