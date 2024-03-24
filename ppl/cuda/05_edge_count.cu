#include <device_launch_parameters.h>

#include "shared/edge_func.h"

namespace gpu {

__global__ void k_EdgeCount(const uint8_t* prefix_n,
                            const int* parents,
                            int* edge_count,
                            const int n_brt_nodes) {
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  for (auto i = idx; i < n_brt_nodes; i += stride) {
    shared::process_edge_count_i(i, prefix_n, parents, edge_count);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    edge_count[0] = 0;
  }
}

}  // namespace gpu
