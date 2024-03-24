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

// const auto n = pipe.n_input();

// pipe.clearSmem();

// k_GlobalHistogram<<<grid_size,
//                     pipe::GLOBAL_HIST_THREADS,
//                     0,
//                     streams[stream_id]>>>(
//     pipe.u_morton, pipe.im_storage.d_global_histogram, n);

// k_Scan<<<pipe::RADIX_PASSES, pipe::RADIX, 0, streams[stream_id]>>>(
//     pipe.im_storage.d_global_histogram,
//     pipe.im_storage.d_first_pass_histogram,
//     pipe.im_storage.d_second_pass_histogram,
//     pipe.im_storage.d_third_pass_histogram,
//     pipe.im_storage.d_fourth_pass_histogram);

// k_DigitBinningPass<<<grid_size,
//                      pipe::BINNING_THREADS,
//                      0,
//                      streams[stream_id]>>>(
//     pipe.u_morton,  // <---
//     pipe.u_morton_alt,
//     pipe.im_storage.d_first_pass_histogram,
//     pipe.im_storage.d_index,
//     n,
//     0);

// k_DigitBinningPass<<<grid_size,
//                      pipe::BINNING_THREADS,
//                      0,
//                      streams[stream_id]>>>(
//     pipe.u_morton_alt,
//     pipe.u_morton,  // <---
//     pipe.im_storage.d_second_pass_histogram,
//     pipe.im_storage.d_index,
//     n,
//     8);

// k_DigitBinningPass<<<grid_size,
//                      pipe::BINNING_THREADS,
//                      0,
//                      streams[stream_id]>>>(
//     pipe.u_morton,  // <---
//     pipe.u_morton_alt,
//     pipe.im_storage.d_third_pass_histogram,
//     pipe.im_storage.d_index,
//     n,
//     16);

// k_DigitBinningPass<<<grid_size,
//                      pipe::BINNING_THREADS,
//                      0,
//                      streams[stream_id]>>>(
//     pipe.u_morton_alt,
//     pipe.u_morton,  // <---
//     pipe.im_storage.d_fourth_pass_histogram,
//     pipe.im_storage.d_index,
//     n,
//     24);

void dispatch_RemoveDuplicates(const int grid_size,
                               const int stream_id,
                               pipe& pipe);

void dispatch_BuildRadixTree(const int grid_size,
                             const int stream_id,
                             pipe& pipe);

void dispatch_EdgeCount(const int grid_size, const int stream_id, pipe& pipe);

void dispatch_EdgeOffset(const int grid_size, const int stream_id, pipe& pipe);

void dispatch_BuildOctree(const int grid_size, const int stream_id, pipe& pipe);

}  // namespace gpu
