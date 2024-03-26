#include <device_launch_parameters.h>

#include "cuda/07_octree.cuh"
#include "shared/oct_func.h"

__global__ void gpu::k_MakeOctNodes(int (*oct_children)[8],
                                    glm::vec4* oct_corner,
                                    float* oct_cell_size,
                                    int* oct_child_node_mask,
                                    const int* edge_offsets,  // prefix sum
                                    const int* edge_counts,   // edge count
                                    const unsigned int* codes,
                                    const uint8_t* rt_prefix_n,
                                    const int* rt_parents,
                                    const float min_coord,
                                    const float range,
                                    const int n_brt_nodes) {
  // do the initial setup on 1 thread
  if (threadIdx.x == 0) {
    const auto root_level = rt_prefix_n[0] / 3;
    const auto root_prefix = codes[0] >> (morton_bits - (3 * root_level));

    // compute root's corner
    shared::morton32_to_xyz(&oct_corner[0],
                            root_prefix << (morton_bits - (3 * root_level)),
                            min_coord,
                            range);
    oct_cell_size[0] = range;
  }

  __syncthreads();

  const auto n = static_cast<unsigned>(n_brt_nodes);
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  // all threads participate in the main work
  // i > 0 && i < N
  for (auto i = idx; i < n; i += stride) {
    if (i == 0) {
      continue;
    }
    // printf("i: %d\n", i);
    shared::process_oct_node(static_cast<int>(i),
                             oct_children,
                             oct_corner,
                             oct_cell_size,
                             oct_child_node_mask,
                             edge_offsets,
                             edge_counts,
                             codes,
                             rt_prefix_n,
                             rt_parents,
                             min_coord,
                             range);
  }
}

__global__ void gpu::k_LinkLeafNodes(int (*oct_children)[8],
                                     int* oct_child_leaf_mask,
                                     const int* edge_offsets,
                                     const int* edge_counts,
                                     const unsigned int* codes,
                                     const bool* rt_has_leaf_left,
                                     const bool* rt_has_leaf_right,
                                     const uint8_t* rt_prefix_n,
                                     const int* rt_parents,
                                     const int* rt_left_child,
                                     const int n_brt_nodes) {
  const auto n = static_cast<unsigned int>(n_brt_nodes);
  const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  // all threads participate in the main work
  // i < N
  for (auto i = idx; i < n; i += stride) {
    shared::process_link_leaf(static_cast<int>(i),
                              oct_children,
                              oct_child_leaf_mask,
                              edge_offsets,
                              edge_counts,
                              codes,
                              rt_has_leaf_left,
                              rt_has_leaf_right,
                              rt_prefix_n,
                              rt_parents,
                              rt_left_child);
  }
}
