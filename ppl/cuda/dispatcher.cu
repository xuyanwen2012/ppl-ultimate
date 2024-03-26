#include <numeric>
#include <vector>

#include "cuda/01_morton.cuh"
#include "cuda/02_sort.cuh"
#include "cuda/03_unique.cuh"
#include "cuda/04_radix_tree.cuh"
#include "cuda/05_edge_count.cuh"
#include "cuda/06_prefix_sum.cuh"
#include "cuda/07_octree.cuh"
#include "cuda/agents/prefix_sum_agent.cuh"
#include "cuda/agents/unique_agent.cuh"
#include "cuda/helper.cuh"
#include "shared/morton_func.h"
#include "shared/structures.h"

namespace gpu {

std::vector<cudaStream_t> streams;  // need to initialize

void initialize_dispatcher(const int n_streams) {
  streams.resize(n_streams);
  for (int i = 0; i < n_streams; ++i) {
    CHECK_CUDA_CALL(cudaStreamCreate(&streams[i]));
  }
}

void release_dispatcher() {
  for (auto stream : streams) {
    CHECK_CUDA_CALL(cudaStreamDestroy(stream));
  }
}

void sync_device() { SYNC_DEVICE(); }

void sync_stream(const int stream_id) {
  CHECK_CUDA_CALL(cudaStreamSynchronize(streams[stream_id]));
}

void dispatch_ComputeMorton(const int grid_size,
                            const int stream_id,
                            pipe& pipe) {
  constexpr auto block_size = 768;

  k_ComputeMortonCode<<<grid_size, block_size, 0, streams[stream_id]>>>(
      pipe.u_points, pipe.u_morton, pipe.n_input(), pipe.min_coord, pipe.range);
}

void dispatch_RadixSort(const int grid_size, const int stream_id, pipe& pipe) {
  const auto n = pipe.n_input();

  static_assert(sizeof(morton_t) == sizeof(unsigned int));

  pipe.clearSmem();

  const auto& stream = streams[stream_id];

  k_GlobalHistogram<<<grid_size, pipe::GLOBAL_HIST_THREADS, 0, stream>>>(
      pipe.u_morton, pipe.im_storage.d_global_histogram, n);

  k_Scan<<<pipe::RADIX_PASSES, pipe::RADIX, 0, stream>>>(
      pipe.im_storage.d_global_histogram,
      pipe.im_storage.d_first_pass_histogram,
      pipe.im_storage.d_second_pass_histogram,
      pipe.im_storage.d_third_pass_histogram,
      pipe.im_storage.d_fourth_pass_histogram);

  k_DigitBinningPass<<<grid_size,
                       pipe::BINNING_THREADS,
                       0,
                       stream>>>(pipe.u_morton,  // <---
                                 pipe.u_morton_alt,
                                 pipe.im_storage.d_first_pass_histogram,
                                 pipe.im_storage.d_index,
                                 n,
                                 0);

  k_DigitBinningPass<<<grid_size, pipe::BINNING_THREADS, 0, stream>>>(
      pipe.u_morton_alt,
      pipe.u_morton,  // <---
      pipe.im_storage.d_second_pass_histogram,
      pipe.im_storage.d_index,
      n,
      8);

  k_DigitBinningPass<<<grid_size,
                       pipe::BINNING_THREADS,
                       0,
                       stream>>>(pipe.u_morton,  // <---
                                 pipe.u_morton_alt,
                                 pipe.im_storage.d_third_pass_histogram,
                                 pipe.im_storage.d_index,
                                 n,
                                 16);

  k_DigitBinningPass<<<grid_size, pipe::BINNING_THREADS, 0, stream>>>(
      pipe.u_morton_alt,
      pipe.u_morton,  // <---
      pipe.im_storage.d_fourth_pass_histogram,
      pipe.im_storage.d_index,
      n,
      24);

  sync_device();
}

// void dispatch_RemoveDuplicates_async(const int grid_size,
//                                      const int stream_id,
//                                      pipe& pipe) {}

// void RemoveDuplicates_on_complete(const int grid_size,
//                                   const int stream_id,
//                                   pipe& pipe) {
//   SYNC_STREAM(streams[stream_id]);
//   pipe.set_n_unique(pipe.im_storage.u_flag_heads[pipe.n_input() - 1]);
//   pipe.brt.set_n_nodes(pipe.n_unique_mortons() - 1);
// }

void dispatch_RemoveDuplicates_sync(const int grid_size,
                                    const int stream_id,
                                    pipe& pipe) {
  constexpr auto unique_block_size = UniqueAgent::n_threads;  // 256
  constexpr auto prefix_block_size =
      PrefixSumAgent<unsigned int>::n_threads;  // 128

  const auto& stream = streams[stream_id];

  k_FindDups<<<grid_size, unique_block_size, 0, stream>>>(
      pipe.getSortedKeys(),
      pipe.im_storage.u_flag_heads,  // <-- output
      pipe.n_input());

  // k_SingleBlockExclusiveScan<<<1, prefix_block_size, 0, stream>>>(
  //     pipe.im_storage.u_flag_heads,
  //     pipe.im_storage.u_flag_heads,  // <-- output
  //     pipe.n_input());

  SYNC_STREAM(stream);
  std::partial_sum(pipe.im_storage.u_flag_heads,
                   pipe.im_storage.u_flag_heads + pipe.n_input(),
                   pipe.im_storage.u_flag_heads);

  k_MoveDups<<<grid_size, unique_block_size, 0, stream>>>(
      pipe.getSortedKeys(),
      pipe.im_storage.u_flag_heads,
      pipe.n_input(),
      pipe.getUniqueKeys(),  // <-- output
      nullptr);
  SYNC_STREAM(stream);

  // last element of flag_heads(prefix summed) is the number of unique elements
  // AND Plus 1!!!
  const auto n_unique = pipe.im_storage.u_flag_heads[pipe.n_input() - 1] + 1;
  pipe.set_n_unique(n_unique);
  pipe.brt.set_n_nodes(n_unique - 1);
}

void dispatch_BuildRadixTree(const int grid_size,
                             const int stream_id,
                             pipe& pipe) {
  constexpr auto n_threads = 512;
  const auto& stream = streams[stream_id];

  k_BuildRadixTree<<<grid_size, n_threads, 0, stream>>>(
      pipe.n_unique_mortons(),
      pipe.getUniqueKeys(),
      pipe.brt.u_prefix_n,
      pipe.brt.u_has_leaf_left,
      pipe.brt.u_has_leaf_right,
      pipe.brt.u_left_child,
      pipe.brt.u_parent);
}

void dispatch_EdgeCount(const int grid_size, const int stream_id, pipe& pipe) {
  constexpr auto block_size = 512;
  const auto& stream = streams[stream_id];

  k_EdgeCount<<<grid_size, block_size, 0, stream>>>(pipe.brt.u_prefix_n,
                                                    pipe.brt.u_parent,
                                                    pipe.u_edge_count,
                                                    pipe.n_brt_nodes());
}

void dispatch_EdgeOffset(const int grid_size, const int stream_id, pipe& pipe) {
  constexpr auto n_threads = PrefixSumAgent<int>::n_threads;
  const auto& stream = streams[stream_id];

  // has to be single
  k_SingleBlockExclusiveScan<<<1, n_threads, 0, stream>>>(
      pipe.u_edge_count,
      pipe.u_edge_offset,  // <-- output
      pipe.n_brt_nodes());
}

void dispatch_BuildOctree(const int grid_size,
                          const int stream_id,
                          pipe& pipe) {
  constexpr auto block_size = 512;
  const auto& stream = streams[stream_id];

  // for (auto i = 0; i < pipe.n_brt_nodes(); ++i) {
  //   std::cout << "parent: " << pipe.brt.u_parent[i]
  //             << " prefix_n: " << (int)pipe.brt.u_prefix_n[i] << std::endl;
  // }

  k_MakeOctNodes<<<grid_size, block_size, 0, stream>>>(
      pipe.oct.u_children,
      pipe.oct.u_corner,
      pipe.oct.u_cell_size,
      pipe.oct.u_child_node_mask,
      pipe.u_edge_count,
      pipe.u_edge_offset,
      pipe.getUniqueKeys(),
      pipe.brt.u_prefix_n,
      pipe.brt.u_parent,
      pipe.min_coord,
      pipe.range,
      pipe.n_brt_nodes());

  std::cout << "octree built\n";
}

}  // namespace gpu
